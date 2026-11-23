import time
from pathlib import Path

import click
import nibabel as nib
import numpy as np
from joblib import Memory, Parallel, delayed
from nilearn import datasets, surface


def project_lh(img, fsaverage):
    project_img = surface.vol_to_surf(
        img,
        surf_mesh=fsaverage["pial_left"],
        inner_mesh=fsaverage["white_left"],
        interpolation="linear",
    )
    return project_img


def project_rh(img, fsaverage):
    project_img = surface.vol_to_surf(
        img,
        surf_mesh=fsaverage["pial_right"],
        inner_mesh=fsaverage["white_right"],
        interpolation="linear",
    )
    return project_img


@click.command()
@click.option(
    "--subj_id",
    type=click.Choice(["subj01", "subj02", "subj05", "subj07"]),
    default="subj01",
    help="Subject name.",
)
@click.option(
    "--encoding_dir",
    default="/scratch/emdupre/NSD/encoding.inputs/",
    help="Encoding inputs directory.",
)
def main(subj_id, encoding_dir):
    location = "./cachedir"
    memory = Memory(location, verbose=10)
    project_lh_cached = memory.cache(project_lh)
    project_rh_cached = memory.cache(project_rh)

    encoding_dir = Path(encoding_dir)
    beta_fname = list(
        encoding_dir.rglob(f"{subj_id}*res-3mm_shared1000_trialBetas.nii.gz")
    )[0]
    print(f"Working on {beta_fname.stem}...")

    img = nib.load(beta_fname)
    n_timepoints = img.shape[-1]

    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage")

    start = time.time()
    left_surf = Parallel(n_jobs=50, prefer="threading", verbose=100)(
        delayed(project_lh_cached)(img.slicer[..., nt], fsaverage)
        for nt in range(n_timepoints)
    )
    stop = time.time()
    print("Finished left hemisphere projection in {:.2f} s".format(stop - start))

    start = time.time()
    right_surf = Parallel(n_jobs=50, prefer="threading", verbose=100)(
        delayed(project_rh_cached)(img.slicer[..., nt], fsaverage)
        for nt in range(n_timepoints)
    )
    stop = time.time()
    print("Finished right hemisphere projection in {:.2f} s".format(stop - start))

    beta_surf = np.hstack([np.asarray(left_surf), np.asarray(right_surf)])
    print(f"Projected shape : {beta_surf.shape}")

    np.save(
        file=f"{subj_id}-projected-fsaverage-shared1000-trialBetas.npy", arr=beta_surf
    )
    return


if __name__ == "__main__":
    main()
