import json
import pickle
from pathlib import Path

import click
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy
from himalaya.scoring import correlation_score
from nilearn.maskers import NiftiMasker
from nilearn.plotting import plot_stat_map
from sklearn.metrics import r2_score
from sklearn.preprocessing import MultiLabelBinarizer

from src.plotting import plot_alphas_diagnostic, plot_flatmap, plot_voxel_hist
from src.solvers import (
    ompCV_sklearn,
    orthogonal_mp_sklearn,
    ridgeCV_himalaya,
    ridgeCV_rrr,
    ridgeCV_sklearn,
)


def THINGSPlus_logo(cat53_X, cat53_y):
    """
    Parameters
    ----------
    sub_name : str
        Subject name
    data_dir : str

    Returns
    -------
    X : np.arr
    y : np.arr
    groups : np.arr

    Note
    ----
    The resulting train, test splits will be of unequal sizes ;
    that is, images that are not labelled "animal" may also be not labelled
    "breakfast food," and so assigned to the training split multiple times.
    The resulting distribution of labels is known to have a significant
    rightward-skew given the pre-existing label distribution (i.e., the category
    "animal" is more likely to occur overall).
    """
    groups = []
    X = []
    y_idx = []
    for grp_lbl in range(53):
        for idx, (y_, X_) in enumerate(zip(cat53_y, cat53_X)):
            if y_[grp_lbl] == 1:
                X.append(X_)
                y_idx.append(idx)
                groups.append(grp_lbl + 1)

    X = np.asarray(X)
    y_idx = np.asarray(y_idx)
    groups = np.asarray(groups)

    return X, y_idx, groups


def explainable_variance(y_matrix, bias_correction=True, do_zscore=True):
    """
    Adapted from gallantlab/himalaya
    BSD 3-Clause License
    Copyright (c) 2020, the himalaya developers All rights reserved.

    Compute explainable variance for a set of voxels.

    Parameters
    ----------
    y_matrix : array of shape (n_repeats * n_stimuli, n_voxels)
        fMRI responses of the repeated test set.
    bias_correction: bool
        Perform bias correction based on the number of repetitions.
    do_zscore: bool
        z-score the data in time. Only set to False if your data time courses
        are already z-scored.

    Returns
    -------
    ev : array of shape (n_voxels, )
        Explainable variance per voxel.
    """
    n_repeats = 3  # NOTE : Hard-coded for THINGS dataset
    n_stimuli, n_voxels = y_matrix.shape
    data = y_matrix.reshape((n_stimuli // n_repeats, n_repeats, n_voxels)).swapaxes(
        0, 1
    )

    if do_zscore:
        data = scipy.stats.zscore(data, axis=1)

    mean_var = data.var(axis=1, dtype=np.float64, ddof=1).mean(axis=0)
    var_mean = data.mean(axis=0).var(axis=0, dtype=np.float64, ddof=1)
    expl_var = var_mean / mean_var

    if bias_correction:
        n_repeats = data.shape[0]
        expl_var = expl_var - (1 - expl_var) / (n_repeats - 1)
    return expl_var


@click.command()
@click.option(
    "--sub_name",
    type=click.Choice(["sub-01", "sub-02", "sub-03", "sub-06"]),
    default="sub-01",
    help="Subject identifier.",
)
@click.option(
    "--roi",
    default=None,
    type=click.Choice([None, "EBA", "FFA", "OFA", "pSTS", "MPA", "OPA", "PPA"]),
    help="Region-of-interest",
)
@click.option(
    "--cv_strategy",
    type=click.Choice(["image", "category", "kfold", "multilabel"]),
    default="image",
    help="Cross-validation strategy",
)
@click.option(
    "--scoring_metric",
    type=click.Choice(["r2_score", "correlation_score"]),
    default="r2_score",
    help="Desired scoring metric. Currently only 'r2_score' and 'correlation_score' "
    "are supported.",
)
@click.option(
    "--average",
    is_flag=True,
    help="Average repeat image presentations before encoding. "
    "Note that this is incompatible with the 'image' cv_strategy",
)
@click.option(
    "--data_dir",
    default="/home/emdupre/links/projects/rrg-pbellec/emdupre/things.betas",
    help="Data directory.",
)
@click.option(
    "--engine",
    default="himalaya",
    type=click.Choice(["himalaya", "sklearn", "rrr", "omp", "ompCV"]),
    help="Engine for running encoding analyses. Must be either 'sklearn' "
    "'rrr', 'omp', 'ompCV' or 'himalaya'. Note only the latter is GPU "
    "compatiable.",
)
@click.option(
    "--space",
    default="T1w",
    help="Space in which to run encoding analyses. Must be either 'MNI152NLin2009cAsym' or 'T1w'.",
)
def main(sub_name, roi, cv_strategy, scoring_metric, average, data_dir, engine, space):
    """ """
    # conditional argument parsing
    if average and (cv_strategy == "image"):
        err_msg = (
            f"Cross-validation strategy {cv_strategy} is not compatible with 'average'"
        )
        raise ValueError(err_msg)

    # set scoring function callable from string
    if scoring_metric == "r2_score":
        scoring = r2_score
    if scoring_metric == "correlation_score":
        scoring = correlation_score

    # load data
    X_matrix = np.load(
        Path(data_dir, "encoding-inputs", space, f"{sub_name}_stim_features.npy")
    )
    mask = nib.load(
        Path(
            data_dir,
            "encoding-inputs",
            space,
            f"{sub_name}_space-{space}_brain_mask.nii.gz",
        )
    )

    if roi is not None:
        y_matrix = np.load(
            Path(
                data_dir,
                "encoding-inputs",
                space,
                f"{sub_name}_space-{space}_roi-{roi}_brain_responses.npy",
            )
        )

        roi_fname = (
            f"{sub_name}_task-floc_space-{space}*_roi-{roi}_*_desc-smooth_mask.nii.gz"
        )
        try:
            roi_mask = nib.load(next(Path(data_dir, "rois", sub_name).glob(roi_fname)))
        except StopIteration:
            raise FileNotFoundError(f"Could not find ROI file matching {roi_fname}")

    else:
        y_matrix = np.load(
            Path(
                data_dir,
                "encoding-inputs",
                space,
                f"{sub_name}_space-{space}_brain_responses.npy",
            )
        )

    expl_var = explainable_variance(y_matrix)

    if cv_strategy == "kfold":
        groups = None
    else:
        # Note that "category" will return `incl_labels` corresponding
        # to image categories (e.g., 'acorn')
        # and "image" will return `incl_labels` corresponding
        # to image identities (e.g., 'acorn_01b').
        groups = np.loadtxt(
            Path(data_dir, "encoding-inputs", space, f"{sub_name}_stim_labels.txt"),
            dtype=np.str_,
        )
        if cv_strategy == "category":
            groups = np.asarray([g.rsplit("_", 1)[0] for g in groups])

        if cv_strategy == "multilabel":
            # NOTE : this is consolidating duplicate keys
            with open(
                Path(
                    data_dir,
                    "encoding-inputs",
                    space,
                    f"{sub_name}_category53_mapping.json",
                )
            ) as f:
                cat_dict = json.load(f)

            cat53_stim_mask_ = [True if g in cat_dict.keys() else False for g in groups]
            cat53_X = X_matrix[cat53_stim_mask_]

            cat53_dense_labels_ = []
            for sv in groups[cat53_stim_mask_]:
                cat53_dense_labels_.append(cat_dict.get(sv))

            mlb = MultiLabelBinarizer().fit(cat53_dense_labels_)
            cat53_y = mlb.transform(cat53_dense_labels_)

            X_matrix, y_idx, groups = THINGSPlus_logo(cat53_X, cat53_y)
            y_matrix = y_matrix[cat53_stim_mask_][y_idx]
    ####################################
    # FIXME
    inner_groups = np.loadtxt(
        Path(data_dir, "encoding-inputs", space, f"{sub_name}_session_labels.txt"),
        dtype=np.str_,
    )
    ####################################
    if average:
        # NOTE: shapes hard-coded for three repetitions, 4174 images, THINGS dataset
        if groups is not None:
            groups = groups[::3]
        X_matrix = X_matrix[::3]
        y_matrix = np.mean(
            y_matrix.reshape(len(X_matrix), 3, y_matrix.shape[-1]), axis=1
        )

    if engine == "sklearn":
        scores = ridgeCV_sklearn(
            X_matrix,
            y_matrix,
            groups=groups,
            scoring=scoring,
            cv_strategy=cv_strategy,
        )
        best_alphas = [estim.alpha_ for estim in scores["estimator"]]
        best_scores = [estim.best_score_ for estim in scores["estimator"]]
    elif engine == "rrr":
        scores = ridgeCV_rrr(
            X_matrix,
            y_matrix,
            ranks=[2**i for i in range(8)],
            groups=groups,
            scoring=scoring,
            cv_strategy=cv_strategy,
        )
        best_alphas = [estim.alpha_ for estim in scores["estimator"]]
        best_scores = [estim.best_score_ for estim in scores["estimator"]]
    elif engine == "himalaya":
        scores = ridgeCV_himalaya(
            X_matrix,
            y_matrix,
            groups=groups,
            scoring=scoring,
            cv_strategy=cv_strategy,
        )
        try:
            best_alphas = [best_alpha_.cpu() for best_alpha_ in scores["best_alphas"]]
            best_scores = [best_score_.cpu() for best_score_ in scores["best_scores"]]
        except AttributeError:
            best_alphas = scores["best_alphas"]
            best_scores = scores["best_scores"]

    elif engine == "omp":
        scores = orthogonal_mp_sklearn(
            X_matrix,
            y_matrix,
            groups=groups,
            scoring=scoring,
            cv_strategy=cv_strategy,
            max_nonzero_coefs=1000,
            inner_cv=5,
        )
        best_scores = scores["per_target_test_scores"]

    elif engine == "ompCV":
        scores = ompCV_sklearn(
            X_matrix,
            y_matrix,
            groups=groups,
            scoring=scoring,
            cv_strategy=cv_strategy,
            max_nonzero_coefs=100,
            inner_cv=5,
        )
        best_scores = scores["per_target_test_scores"]

    if roi is None:
        roi = "wholebrain"
    if average:
        out_file = Path(
            data_dir,
            "encoding-inputs",
            f"{sub_name}_space-{space}_roi-{roi}_cv-{cv_strategy}-average_{engine}_scores.pkl",
        )
    else:
        out_file = Path(
            data_dir,
            "encoding-inputs",
            f"{sub_name}_space-{space}_roi-{roi}_cv-{cv_strategy}_{engine}_scores.pkl",
        )

    if not out_file.is_file():
        with open(out_file, "wb") as f:
            pickle.dump(scores, f)

    # to un-pickle
    # with open(out_file, 'rb') as f:
    #     check = pickle.load(f)

    if roi == "wholebrain":
        # plot histogram of explainable var and scores across cortex
        fig_hist = plot_voxel_hist(
            sub_name, expl_var, best_scores, scoring_metric=scoring_metric
        )
        if average:
            fig_hist.savefig(
                f"{sub_name}_space-{space}_roi-{roi}_{cv_strategy}-average_{scoring_metric}_{engine}_expl_var_hist.png"
            )
        else:
            fig_hist.savefig(
                f"{sub_name}_space-{space}_roi-{roi}_{cv_strategy}_{scoring_metric}_{engine}_expl_var_hist.png"
            )
        plt.close(fig_hist)

        # plot diagnostic of voxelwise best alphas ; QC for two clear peaks
        fig_alphas, ax = plt.subplots(1, 1)
        for i, b_alpha in enumerate(best_alphas):
            plot_alphas_diagnostic(
                best_alphas=b_alpha, alphas=np.logspace(1, 20, 20), cv_fold=i, ax=ax
            )
        if average:
            fig_alphas.savefig(
                f"{sub_name}_space-{space}_roi-{roi}_{cv_strategy}-average_{scoring_metric}_{engine}_alphas.png"
            )
        else:
            fig_alphas.savefig(
                f"{sub_name}_space-{space}_roi-{roi}_{cv_strategy}_{scoring_metric}_{engine}_alphas.png"
            )
        plt.close(fig_alphas)

        # plot flatmap of scores across cortex
        plot_flatmap(
            best_scores,
            sub_name,
            mask,
            cv_strategy,
            scoring_metric=scoring_metric,
            average=average,
        )
    elif roi in ["EBA", "FFA", "OFA", "pSTS", "MPA", "OPA", "PPA"]:
        # plot stat map of scores across ROI
        if engine not in ("rrr", "omp"):
            masker = NiftiMasker(mask_img=roi_mask).fit()
            fig = plot_stat_map(
                masker.inverse_transform(np.mean(best_scores, axis=0)),
                display_mode="z",
                colorbar=True,
                symmetric_cbar=False,
                vmax=0.25,
                vmin=0,
                cmap="PuRd",
            )
            fig.savefig(
                f"{sub_name}_space-{space}_roi-{roi}_{cv_strategy}_{scoring_metric}_{engine}_statmap.png"
            )


if __name__ == "__main__":
    main()
