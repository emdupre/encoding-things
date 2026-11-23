from pathlib import Path

import numpy as np
from nilearn import datasets, image, surface
from numpy.lib.format import open_memmap
from tqdm.contrib.concurrent import thread_map

roi_dir = "/scratch/emdupre/fsaverage_rois/floc.rois/standard_masks/standard_rois"
rois = list(Path(roi_dir).glob("*desc-bilat_pseg.nii.gz"))
subj_id = "subj07"
pred_dir = Path("/scratch/emdupre/NSD/stimuli.algonauts-predictions")
pred_paths = list(pred_dir.glob(f"brain-synthesizer-{subj_id}-0*prediction.npy"))

roi_id = "pSTS"
roi_paths = list(Path(roi_dir).glob(f"*roi-{roi_id}*.npy"))
roi_arr = np.load(roi_paths[0]).astype(bool)


def project_roi_to_fsaverage(roi):
    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage")
    name = str(roi.stem).split("_")[3]
    thr_img = image.math_img("img > 0.5", img=roi)
    left_surf = surface.vol_to_surf(
        thr_img,
        surf_mesh=fsaverage["pial_left"],
        inner_mesh=fsaverage["white_left"],
        interpolation="nearest_most_frequent",
    )
    right_surf = surface.vol_to_surf(
        thr_img,
        surf_mesh=fsaverage["pial_right"],
        inner_mesh=fsaverage["white_right"],
        interpolation="nearest_most_frequent",
    )
    thr_surf = np.hstack([left_surf, right_surf])
    np.save(
        file=f"parcel-kanwisher_space-fsaverage_{name}_desc-bilateral_mask.npy",
        arr=thr_surf,
    )


def mask_predictions(pred_path):
    stim_name = str(pred_path.stem).split("-")[3]
    pred_arr = open_memmap(pred_path).squeeze()
    return stim_name, pred_arr[roi_arr]


if __name__ == "__main__":
    result = thread_map(mask_predictions, pred_paths, max_workers=5)
    stim_names, roi_predictions = zip(*result)

    # Save prediction to .npy file
    np.save(
        Path(
            pred_dir,
            f"brain-synthesizer-{subj_id}-{roi_id}-prediction.npy",
        ),
        np.asarray(roi_predictions),
    )

    np.savetxt(
        Path(pred_dir, f"brain-synthesizer-{subj_id}-{roi_id}-file-names.npy"),
        np.array(stim_names),
        fmt="%s",
    )
