import cortex
import matplotlib.pyplot as plt
import numpy as np
from nilearn import masking

# os.environ["PATH"] += ":/Applications/Inkscape.app/Contents/MacOS/"


def plot_flatmap(
    best_scores,
    sub_name,
    mask_img,
    cv_strategy,
    scoring_metric="r2_score",
    average=False,
):
    """
    Parameters
    ----------
    nii : nib.Nifti
        voxel-wise data to project to flatmap
    sub_name : str
    """
    lh, rh = cortex.get_hemi_masks(subject=sub_name, xfmname="align_auto")

    avg_best_score = np.mean(best_scores, axis=0)  # TODO: FIXME
    nii = masking.unmask(avg_best_score, mask_img)

    # https://gallantlab.org/pycortex/auto_examples/datasets/plot_vertex.html
    vol = cortex.Volume(
        data=np.swapaxes(nii.get_fdata(), 0, -1),
        subject=sub_name,
        xfmname="align_auto",
        mask=mask_img.get_fdata(),
        vmin=0,
        vmax=0.30,
        cmap="magma",
    )

    if average:
        out_name = (
            f"{sub_name}_{cv_strategy}-average_encoding_{scoring_metric}_flatmap.png"
        )
    else:
        out_name = f"{sub_name}_{cv_strategy}_encoding_{scoring_metric}_flatmap.png"

    # fig = cortex.quickshow(nii_vol, sampler="nearest")
    cortex.quickflat.make_png(
        out_name,
        vol,
        sampler="trilinear",
        curv_brightness=1.0,
        with_colorbar=True,
        colorbar_location="left",
        with_curvature=True,
        with_labels=False,
        with_rois=True,
        dpi=300,
        height=2048,
    )
    return


def plot_alphas_diagnostic(best_alphas, alphas, cv_fold=None, ax=None):
    """
    Adapted from gallantlab/himalaya
    BSD 3-Clause License
    Copyright (c) 2020, the himalaya developers All rights reserved.

    Plot a diagnostic plot for the selected alphas during cross-validation.

    To figure out whether to increase the range of alphas.

    Parameters
    ----------
    best_alphas : array of shape (n_targets, )
        Alphas selected during cross-validation for each target.
    alphas : array of shape (n_alphas)
        Alphas used while fitting the model.
    cv_fold : int or None
        Outer cross-validation fold, for labelling
    ax : None or figure axis

    Returns
    -------
    ax : figure axis
    """
    alphas = np.sort(alphas)
    n_alphas = len(alphas)
    indices = np.argmin(np.abs(best_alphas[None] - alphas[:, None]), 0)
    hist = np.bincount(indices, minlength=n_alphas)

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    log10alphas = np.log(alphas) / np.log(10)
    ax.plot(log10alphas, hist, ".-", markersize=12, label=f"Outer-CV fold {cv_fold}")
    ax.set_ylabel("Number of targets")
    ax.set_xlabel("log10(alpha)")
    if cv_fold is not None:
        ax.legend()
    ax.grid("on")
    return ax


def plot_voxel_hist(
    sub_name, expl_var, best_scores, scoring_metric="r2_score", ax=None
):
    r"""
    Adapted from the following examples :
    - https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
    - https://gallantlab.org/voxelwise_tutorials/notebooks/shortclips/03_compute_explainable_variance.html

    Parameters
    ----------
    sub_name : str
        Subject name
    expl_var : np.arr
        Explainable variance, as calculated using
        .. math::
            \\frac{1}{N}\\sum_{i=1}^N\\text{Var}(y_i) - \\frac{N}{N-1}\\sum_{i=1}^N\\text{Var}(r_i)
    scores : np.arr
        Scores from the encoding model calculated using the scoring metric
    scoring_metric : str
        Scoring metric used in encoding model scoring, must be 'r2_score' or 'correlation_score'

    Returns
    -------
    ax : figure axis
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    bins = np.linspace(0, 1, 100)
    ax.hist(
        expl_var,
        bins=bins,
        log=True,
        histtype="step",
        label="Explainable variance",
    )

    mean_score = np.mean(best_scores, axis=0)
    std_score = np.std(best_scores, axis=0)
    score_upper = mean_score + std_score
    score_lower = mean_score - std_score
    upper_ci, _ = np.histogram(score_upper, bins=bins, density=False)
    lower_ci, _ = np.histogram(score_lower, bins=bins, density=False)

    ax.hist(
        mean_score,
        bins=bins,
        log=True,
        histtype="step",
        label=(
            "$R^2$ values" if (scoring_metric == "r2_score") else "Correlation values"
        ),
    )
    ax.fill_between(
        bins[:-1],
        lower_ci,
        upper_ci,
        color="grey",
        alpha=0.2,
        step="post",
        label=r"$\pm$ 1 std. dev.",
    )

    if scoring_metric == "r2_score":
        ax.set_title(
            f"Histogram of explainable variance and average $R^2$ for {sub_name}"
        )
    else:
        ax.set_title(
            f"Histogram of explainable variance and average correlation for {sub_name}"
        )

    ax.set_ylabel("Number of voxels")
    ax.grid("on")
    ax.legend()
    return fig
