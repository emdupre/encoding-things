import re
from pathlib import Path

import click
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import maskers, masking
from nilearn.datasets import load_mni152_gm_mask
from nilearn.image import index_img, new_img_like, resample_to_img


def _threshold_roi_file(roi_nii, threshold=0.8, output_dir=None):
    roi_nii = Path(roi_nii)
    pseg = nib.load(roi_nii)
    data = pseg.get_fdata()
    print(f"{roi_nii.name}: max probability = {data.max():.3f}")

    # Parse the ROI name (e.g. "EBA") from the BIDS-like filename
    match = re.search(r"roi-([^_]+)", roi_nii.name)
    if match is None:
        raise ValueError(
            f"Could not find 'roi-<label>' in filename: {roi_nii.name}"
        )
    roi = match.group(1)

    # Binary mask (uint8 keeps the file small and unambiguous)
    mask_data = (data > threshold).astype(np.uint8)
    if mask_data.sum() == 0:
        raise ValueError(
            f"No voxels above threshold {threshold} for {roi} "
            f"(max is {data.max():.3f}). Try a lower threshold."
        )
    print(
        f"{roi}: {int(mask_data.sum())} voxels survive threshold {threshold}"
    )

    mask = new_img_like(pseg, mask_data, copy_header=True)
    mask.header.set_data_dtype(np.uint8)

    # Save in the requested format
    roi_fname = f"parcel-kanwisher_space-MNI152T1_roi-{roi}_cut-{threshold}_mask.nii.gz"
    output_dir = Path(output_dir) if output_dir is not None else roi_nii.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / roi_fname
    nib.save(mask, output_file)

    return mask, output_file


def _find_or_build_roi_nii(roi, roi_dir, threshold=0.8):
    roi_dir = Path(roi_dir)
    roi_fname = f"parcel-kanwisher_space-MNI152T1_roi-{roi}_*_mask.nii.gz"
    mask_matches = sorted(roi_dir.glob(roi_fname))
    if mask_matches:
        # If several thresholds exist, prefer the requested one, else newest
        preferred = roi_dir.glob(  # no-op, glob already consumed above
            f"parcel-kanwisher_space-MNI152T1_roi-{roi}_cut-{threshold}_mask.nii.gz"
        )
        try:
            return Path(next(preferred))
        except StopIteration:
            return mask_matches[-1]
    # No mask yet: try to generate one from the probability map
    prob_fname = (
        f"parcel-kanwisher_space-MNI152T1_*_roi-{roi}_desc-bilat_pseg.nii.gz"
    )
    prob_matches = sorted(roi_dir.glob(prob_fname))
    if not prob_matches:
        raise FileNotFoundError(
            f"Could not find mask matching {roi_fname} or "
            f"probability map matching {prob_fname} in {roi_dir}"
        )

    _, roi_from_prob = _threshold_roi_file(
        prob_matches[-1], threshold=threshold, output_dir=roi_dir
    )
    return roi_from_prob


def _load_stim_arrays_nsd(data_dir, model, sub_name="subj01"):
    """
    Parameters
    ----------
    data_dir : str
    model : str
        Name of the feature model, e.g. "clip", "clip-all-layers" or "dino".
    """
    feats_dir = Path(data_dir, f"{model}-features")
    feats = np.load(
        feats_dir / f"{sub_name}_shared1000_{model}-trialFeatures.npy"
    )
    fnames = np.genfromtxt(
        feats_dir / f"{sub_name}_shared1000_trialStimuli.txt", dtype=str
    )
    fnames = [Path(f).stem for f in fnames]

    return feats, fnames


def _load_brain_arrays_nsd(
    sub_name, roi, data_dir, roi_dir=None, roi_threshold=0.8
):
    """
    Parameters
    ----------
    sub_name : str
    roi : str
    space : str
    data_dir : str
    roi_dir : str, optional
        Directory containing the thresholded ROI masks. Unthresholded
        probability maps are also looked up here if a mask is missing.
    roi_threshold : float, optional
        Threshold passed to _threshold_roi_file when the mask has to be
        generated on the fly.
    """

    annot_fname = f"{sub_name}_shared1000_trialSessions.txt"
    beta_fname = f"{sub_name}_res-3mm_shared1000_trialBetas.nii.gz"
    beta_nii = nib.load(Path(data_dir, "betas", beta_fname))
    ref_img = index_img(beta_nii, 0)

    gm_mask = load_mni152_gm_mask(resolution=3)
    resampled_mask = resample_to_img(
        gm_mask,
        ref_img,
        interpolation="nearest",
        copy_header=True,
        force_resample=True,
    )

    if roi is not None:
        if roi_dir is None:
            roi_dir = data_dir
            print(f"roi_dir not provided, looking for ROI file in {data_dir}")
        roi_nii = _find_or_build_roi_nii(roi, roi_dir, threshold=roi_threshold)

        mask = resample_to_img(
            roi_nii,
            ref_img,
            interpolation="nearest",
            copy_header=True,
            force_resample=True,
        )
    else:
        mask = resampled_mask

    masker = maskers.NiftiMasker(
        mask_img=mask, standardize="zscore_sample"
    ).fit()

    # Assuming one sessions ID per trial
    annot = np.genfromtxt(Path(data_dir, "annot", annot_fname), dtype=str)
    session_labels = [Path(f).stem for f in np.atleast_1d(annot)]

    y_vals = masker.transform(beta_nii)  # (n_trials, n_voxels)

    assert y_vals.shape[0] == len(session_labels), (
        f"Trial mismatch: {y_vals.shape[0]} betas vs "
        f"{len(session_labels)} annotation entries"
    )

    return y_vals, session_labels, masker.mask_img_


def _clean_inputs_nsd(y_arr, y_labels, x_arr, x_labels, n_repeats=3):
    """
    Keep only stimuli shown exactly `n_repeats` times and sort by stimulus ID
    so repetitions are contiguous. Assumes all inputs are per-trial and
    row-aligned (row i of every array refers to the same trial).

    Parameters
    ----------
    y_arr : array-like, shape (n_trials, n_voxels)
    y_labels : array-like, shape (n_trials,)
        Per-trial labels (e.g. session IDs).
    x_arr : array-like, shape (n_trials, dim_embed)
        Per-trial embeddings (the same image has the same embedding).
    x_labels : array-like, shape (n_trials,)
        Image ID of each trial.

    Returns
    -------
    sorted_stim : np.ndarray, shape (n_kept_trials,)
    incl_y_arr : np.ndarray, shape (n_kept_trials, n_voxels)
    incl_y_labels : np.ndarray, shape (n_kept_trials,)
    incl_x_arr : np.ndarray, shape (n_kept_trials, dim_embed)
    """
    # type coercion
    y_arr = np.asarray(y_arr)
    y_labels = np.asarray(y_labels)
    x_arr = np.asarray(x_arr)
    stim_ids = np.asarray(x_labels)

    # all arrays must be row-aligned
    n_trials = len(stim_ids)
    assert len(y_arr) == len(y_labels) == len(x_arr) == n_trials, (
        f"length mismatch: y_arr={len(y_arr)}, y_labels={len(y_labels)}, "
        f"x_arr={len(x_arr)}, x_labels={n_trials}"
    )

    # keep only images shown exactly n_repeats times
    labels, counts = np.unique(stim_ids, return_counts=True)
    incl_labels = labels[counts == n_repeats]
    keep = np.flatnonzero(np.isin(stim_ids, incl_labels))

    # stable sort: repetitions stay in original (chronological) order
    order = keep[np.argsort(stim_ids[keep], kind="stable")]
    sorted_stim = stim_ids[order]

    # sanity checks
    unique_sorted, first_idx, reps = np.unique(
        sorted_stim, return_index=True, return_counts=True
    )
    assert np.array_equal(unique_sorted, incl_labels)
    assert np.all(reps == n_repeats)
    # contiguity: image i occupies rows [i*n_repeats, (i+1)*n_repeats)
    assert np.array_equal(first_idx, np.arange(len(unique_sorted)) * n_repeats)

    incl_y_arr = y_arr[order]
    incl_y_labels = y_labels[order]
    incl_x_arr = x_arr[order]

    # embeddings should be identical across the repetitions of an image
    x_blocks = incl_x_arr.reshape(len(unique_sorted), n_repeats, -1)
    assert np.allclose(x_blocks, x_blocks[:, :1, :]), (
        "embeddings differ across repetitions of the same image: "
        "betas and labels may not be aligned"
    )
    assert incl_x_arr.shape[0] == incl_y_arr.shape[0] == len(sorted_stim)

    return sorted_stim, incl_y_arr, incl_y_labels, incl_x_arr


def gen_inputs_nsd(sub_name, roi, data_dir, feat_model, roi_dir=None):
    """
    Parameters
    ----------
    sub_name : str
    roi : str
    space : str
    data_dir : str
    """
    y_vals, session_labels, mask = _load_brain_arrays_nsd(
        sub_name, roi, data_dir, roi_dir
    )
    embd_feats, embd_fnames = _load_stim_arrays_nsd(
        data_dir, feat_model, sub_name
    )

    stim_vec, y_matrix, y_sessions, X_matrix = _clean_inputs_nsd(
        y_vals,
        session_labels,
        embd_feats,
        embd_fnames,
    )

    return stim_vec, y_matrix, y_sessions, X_matrix, mask


@click.command()
@click.option("--sub_name", default="subj01", help="Subject name.")
@click.option("--roi", default="EBA", help="Region-of-interest")
@click.option(
    "--data_dir",
    default="/data/parietal/store3/data/nsd-shared1000",
    help="Data directory.",
)
@click.option(
    "--roi_dir",
    default="/data/parietal/store3/work/aponcema/encoding-things/floc.rois/standard_masks/standard_rois",
    help="ROIs directory.",
)
@click.option(
    "--feat_model",
    default="clip",
    help="Model used to extract image features. Must be in ['clip', 'clip-all-layers', 'dino']",
)
def main(sub_name, roi, data_dir, roi_dir, feat_model):
    """
    Create trialwise inputs for voxelwise encoding models on NSD-shared1000
    data using existing CLIP embeddings (previously generated using
    thingsvision).
    """
    rois = [None, "EBA", "FFA", "OFA", "pSTS", "MPA", "OPA", "PPA", "V1", "V2"]
    if roi not in rois:
        err_msg = f"Unrecognized ROI {roi}"
        raise ValueError(err_msg)

    sub_names = ["subj01", "subj02", "subj05", "subj07"]
    if sub_name not in sub_names:
        err_msg = f"Unrecognized subject {sub_name}"
        raise ValueError(err_msg)

    if feat_model not in ["clip", "clip-all-layers", "dino"]:
        err_msg = f"Unrecognized feature model {feat_model}"
        raise ValueError(err_msg)

    stim_vec, y_matrix, y_sessions, X_matrix, mask = gen_inputs_nsd(
        sub_name, roi, data_dir, feat_model, roi_dir
    )

    out_stim = Path(
        data_dir,
        "encoding-inputs",
        f"{feat_model}-embeddings",
        f"{sub_name}_stim_labels.txt",
    )
    if not out_stim.is_file():
        out_stim.parent.mkdir(exist_ok=True, parents=True)
        np.savetxt(out_stim, stim_vec, fmt="%s")

    out_y_sessions = Path(
        data_dir,
        "encoding-inputs",
        f"{feat_model}-embeddings",
        f"{sub_name}_session_labels.txt",
    )
    if not out_y_sessions.is_file():
        out_y_sessions.parent.mkdir(exist_ok=True, parents=True)
        np.savetxt(out_y_sessions, y_sessions, fmt="%s")

    out_X_matrix = Path(
        data_dir,
        "encoding-inputs",
        f"{feat_model}-embeddings",
        f"{sub_name}_stim_features.npy",
    )
    if not out_X_matrix.is_file():
        out_X_matrix.parent.mkdir(exist_ok=True, parents=True)
        np.save(out_X_matrix, X_matrix)

    if roi is not None:
        out_y_matrix = Path(
            data_dir,
            "encoding-inputs",
            f"{feat_model}-embeddings",
            f"{sub_name}_roi-{roi}_brain_responses.npy",
        )
    else:
        out_y_matrix = Path(
            data_dir,
            "encoding-inputs",
            f"{feat_model}-embeddings",
            f"{sub_name}_brain_responses.npy",
        )
    if not out_y_matrix.is_file():
        out_y_matrix.parent.mkdir(exist_ok=True, parents=True)
        np.save(out_y_matrix, y_matrix)
    print(out_y_matrix)
    print(y_matrix.shape)

    out_mask = Path(
        data_dir,
        "encoding-inputs",
        f"{feat_model}-embeddings",
        f"{sub_name}_brain_mask.nii.gz",
    )
    if not out_mask.is_file():
        out_mask.parent.mkdir(exist_ok=True, parents=True)
        nib.save(mask, out_mask)


if __name__ == "__main__":
    main()
