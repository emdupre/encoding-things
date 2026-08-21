import json
from pathlib import Path

import click
import h5py
import nibabel as nib
import numpy as np
import pandas as pd


def _load_stim_arrays(data_dir):
    """
    Parameters
    ----------
    data_dir : str
    """
    clip_feats = np.load(Path(data_dir, "clip-features", "features.npy"))
    clip_fnames = np.genfromtxt(
        Path(data_dir, "clip-features", "file_names.txt"), dtype=str
    )
    clip_fnames = [Path(f).stem for f in clip_fnames]

    return clip_feats, clip_fnames


def _category_mapping(sub_name, data_dir):
    """
    Parameters
    ----------
    sub_name : str
    data_dir : str
    """
    annot_fname = f"{sub_name}_task-things_desc-perTrial_annotation.tsv"

    annot_df = pd.read_csv(Path(data_dir, "annot", annot_fname), sep="\t")
    annot_df = annot_df.loc[annot_df["atypical"] == False]

    cat53_mask = annot_df[annot_df["highercat53_names"] != "[]"].index
    image_names = annot_df["image_name"][cat53_mask]

    cat53_names = annot_df["highercat53_names"][cat53_mask].str.replace(
        r"'|\]|\[", "", regex=True
    )
    sanitized_ = []
    for cat in cat53_names.values:
        list_labels = cat.split(",")
        sanitized_.append([l.strip() for l in list_labels])

    # NOTE : this is consolidating duplicate keys
    cat_dict = dict(zip(image_names, pd.Series(sanitized_)))

    return cat_dict


def gen_inputs_image(sub_name, roi, space, data_dir):
    """
    Parameters
    ----------
    sub_name : str
    roi : str
    space : str
    data_dir : str
    """

    # Load the stimuli names
    stim_vec = np.loadtxt(
        Path(
            data_dir,
            "encoding-inputs",
            "trial",
            space,
            f"{sub_name}_stim_labels.txt",
        ),
        dtype=np.str_,
    )
    stim_vec = np.unique(stim_vec)

    # Load sorted image betas
    beta_fname = f"{sub_name}_task-things_space-{space}_model-fitHrfGLMdenoiseRR_stats-imageBetas_desc-zscore_statseries.h5"
    beta_h5 = h5py.File(Path(data_dir, "betas", beta_fname), "r")
    mask = nib.nifti1.Nifti1Image(
        np.array(beta_h5["mask_array"]), affine=np.array(beta_h5["mask_affine"])
    )

    stim_mask = np.ones(len(stim_vec), dtype=bool)
    rows = []

    # Select images shown to subject
    for i, stim_name in enumerate(stim_vec):
        try:
            rows.append(np.array(beta_h5[stim_name]["betas"]).flatten())
        except KeyError:
            stim_mask[i] = False
    stim_vec = stim_vec[stim_mask]
    y_matrix = np.vstack(rows)

    # Load and sort clip features
    clip_feats, clip_fnames = _load_stim_arrays(data_dir)
    X_matrix = np.vstack(
        [
            clip_feats[np.where(np.array(clip_fnames) == str(stim_name))[0]]
            for stim_name in stim_vec
        ]
    )

    # Category mapping (similar to gen-inputs)
    cat_dict = _category_mapping(sub_name, data_dir)

    return stim_vec, y_matrix, X_matrix, mask, cat_dict


@click.command()
@click.option("--sub_name", default="sub-01", help="Subject name.")
@click.option("--roi", default=None, help="Region-of-interest")
@click.option(
    "--space",
    default="T1w",
    help="Space in which brain responses were registered during preprocessing. Must be in ['MNI152NLin2009cAsym', 'T1w']",
)
@click.option(
    "--data_dir",
    # default="/home/emdupre/links/projects/rrg-pbellec/emdupre/things.betas",
    default="/Users/emdupre/Desktop/things-encode/",
    help="Data directory.",
)
def main(sub_name, roi, space, data_dir):
    """
    Create imagewise inputs for voxelwise encoding models on THINGS data using
    existing CLIP embeddings (previously generated using thingsvision).
    """
    rois = [None, "EBA", "FFA", "OFA", "pSTS", "MPA", "OPA", "PPA"]
    if roi not in rois:
        err_msg = f"Unrecognized ROI {roi}"
        raise ValueError(err_msg)

    sub_names = ["sub-01", "sub-02", "sub-03", "sub-06"]
    if sub_name not in sub_names:
        err_msg = f"Unrecognized subject {sub_name}"
        raise ValueError(err_msg)

    if space not in ["MNI152NLin2009cAsym", "T1w"]:
        err_msg = f"Unrecognized space {space}"
        raise ValueError(err_msg)

    stim_vec, y_matrix, X_matrix, mask, cat_dict = gen_inputs_image(
        sub_name, roi, space, data_dir
    )

    out_stim = Path(
        data_dir,
        "encoding-inputs",
        "image",
        space,
        f"{sub_name}_stim_labels.txt",
    )
    if not out_stim.is_file():
        out_stim.parent.mkdir(exist_ok=True, parents=True)
        np.savetxt(out_stim, stim_vec, fmt="%s")

    out_X_matrix = Path(
        data_dir,
        "encoding-inputs",
        "image",
        space,
        f"{sub_name}_stim_features.npy",
    )
    if not out_X_matrix.is_file():
        out_X_matrix.parent.mkdir(exist_ok=True, parents=True)
        np.save(out_X_matrix, X_matrix)

    if roi is not None:
        out_y_matrix = Path(
            data_dir,
            "encoding-inputs",
            "image",
            space,
            f"{sub_name}_space-{space}_roi-{roi}_brain_responses.npy",
        )
    else:
        out_y_matrix = Path(
            data_dir,
            "encoding-inputs",
            "image",
            space,
            f"{sub_name}_space-{space}_brain_responses.npy",
        )
    if not out_y_matrix.is_file():
        out_y_matrix.parent.mkdir(exist_ok=True, parents=True)
        np.save(out_y_matrix, y_matrix)

    print(out_y_matrix)
    print(y_matrix.shape)

    out_mask = Path(
        data_dir,
        "encoding-inputs",
        "image",
        space,
        f"{sub_name}_space-{space}_brain_mask.nii.gz",
    )
    if not out_mask.is_file():
        out_mask.parent.mkdir(exist_ok=True, parents=True)
        nib.save(mask, out_mask)

    out_dict = Path(
        data_dir,
        "encoding-inputs",
        "image",
        space,
        f"{sub_name}_category53_mapping.json",
    )
    if not out_dict.is_file():
        out_dict.parent.mkdir(exist_ok=True, parents=True)
        with open(out_dict, "w", encoding="utf8") as f:
            json.dump(cat_dict, f)


if __name__ == "__main__":
    main()
