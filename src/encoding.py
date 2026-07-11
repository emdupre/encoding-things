import json
import pickle
from collections import defaultdict
from pathlib import Path

import click
import cortex
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy
import sklearn
from himalaya.scoring import correlation_score
from nilearn import masking, plotting
from nilearn.maskers import NiftiMasker
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
    cross_validate,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from rrr import ReducedRankRidgeRegressionCV

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


def ridgeCV_sklearn(
    X_matrix, y_matrix, groups=None, scoring=r2_score, cv_strategy="image"
):
    """
    Parameters
    ----------
    X_matrix : np.arr
        Training data for stimulus embeddings.
        Expected shape (n_samples, n_features)
    y_matrix : np.arr
        Training data for brain responses
        Expected shape (n_samples, n_features, n_repeats)
    groups : np.arr
        Group labels for outer_cv, should correspond to image
        identity or image categor(ies).
        Expected shape (n_samples, )
    scoring : Callable
        Scoring function for estimator predictions.
    cv_strategy : str
    """
    from sklearn.linear_model import RidgeCV

    scaler = StandardScaler(with_mean=True, with_std=False)
    scaler.fit_transform(X_matrix)
    scaler.fit_transform(y_matrix)

    if groups is None:
        outer_cv = KFold(shuffle=True, random_state=0)
    else:
        if cv_strategy == "image":
            outer_cv = GroupKFold(shuffle=True, random_state=0)
        elif cv_strategy == "multilabel":
            outer_cv = LeaveOneGroupOut()
    alphas = np.logspace(1, 20, 20)
    estimator = RidgeCV(
        alphas=alphas,
        alpha_per_target=True,
        cv=None,
    )
    scorer = make_scorer(scoring)
    sklearn.set_config(enable_metadata_routing=True)

    scores = cross_validate(
        estimator,
        X_matrix,
        y=y_matrix,
        cv=outer_cv,
        scoring=scorer,
        params={"groups": groups} if groups is not None else None,
        return_estimator=True,
        return_indices=True,
        error_score="raise",
    )
    return scores


def ridgeCV_rrr(
    X_matrix, y_matrix, ranks, groups=None, scoring=r2_score, cv_strategy="image"
):
    """
    Parameters
    ----------
    X_matrix : np.arr
        Training data for stimulus embeddings.
        Expected shape (n_samples, n_features)
    y_matrix : np.arr
        Training data for brain responses
        Expected shape (n_samples, n_features, n_repeats)
    ranks : int or list of int
        Rank(s) for the reduced-rank ridge regression estimator.
    groups : np.arr
        Group labels for outer_cv, should correspond to image
        identity or image categor(ies).
        Expected shape (n_samples, )
    scoring : Callable
        Scoring function for estimator predictions.
    cv_strategy : str
    """
    scaler = StandardScaler(with_mean=True, with_std=False)
    scaler.fit_transform(X_matrix)
    scaler.fit_transform(y_matrix)

    if groups is None:
        outer_cv = KFold(shuffle=True, random_state=0)
    else:
        if cv_strategy in ["category", "image"]:
            outer_cv = GroupKFold(shuffle=True, random_state=0)
        elif cv_strategy == "multilabel":
            outer_cv = LeaveOneGroupOut()
    alphas = np.logspace(1, 20, 20)
    estimator = ReducedRankRidgeRegressionCV(
        alphas=alphas,
        ranks=ranks,
    )
    scorer = make_scorer(scoring)
    sklearn.set_config(enable_metadata_routing=True)

    scores = cross_validate(
        estimator,
        X_matrix,
        y=y_matrix,
        cv=outer_cv,
        scoring=scorer,
        params={"groups": groups} if groups is not None else None,
        return_estimator=True,
        return_indices=True,
        error_score="raise",
    )
    return scores


def ridgeCV_himalaya(
    X_matrix, y_matrix, groups=None, scoring=r2_score, cv_strategy="image"
):
    """
    Parameters
    ----------
    X_matrix : np.arr
        Training data for stimulus embeddings.
        Expected shape (n_samples, n_features)
    y_matrix : np.arr
        Training data for brain responses
        Expected shape (n_samples, n_features, n_repeats)
    groups : np.arr
        Group labels for outer_cv, should correspond to image
        identity or image category.
        Expected shape (n_samples, )
    scoring : Callable
        Scoring function for estimator predictions.
    cv_strategy : str
    """
    from himalaya.backend import set_backend
    from himalaya.ridge import RidgeCV

    backend = set_backend("torch_cuda", on_error="warn")

    scores = defaultdict()
    train_indices, test_indices = [], []
    best_scores = []
    best_alphas = []

    if groups is None:
        outer_cv = KFold(shuffle=True, random_state=0)
    else:
        if cv_strategy in ["category", "image"]:
            outer_cv = GroupKFold(shuffle=True, random_state=0)
        elif cv_strategy == "multilabel":
            outer_cv = LeaveOneGroupOut()

    alphas = np.logspace(1, 20, 20)
    pl = make_pipeline(
        StandardScaler(with_mean=True, with_std=False),
        RidgeCV(
            alphas=alphas,
            solver_params=dict(
                n_targets_batch=500, n_alphas_batch=5, n_targets_batch_refit=100
            ),
        ),
    )

    for train_index, test_index in outer_cv.split(X_matrix, y_matrix, groups):
        train_indices.append(train_index)
        test_indices.append(test_index)

        pl.fit(X_matrix[train_index], y_matrix[train_index])

        if scoring is correlation_score:
            y_pred = pl.predict(X_matrix[test_index])
            best_scores.append(correlation_score(y_matrix[test_index], y_pred))
        else:
            best_scores.append(pl.score(X_matrix[test_index], y_matrix[test_index]))

        best_alphas.append(pl[-1].best_alphas_)

    scores["best_alphas"] = best_alphas
    scores["best_scores"] = best_scores
    scores["indices"] = {"train": train_indices, "test": test_indices}

    return scores


def ompCV_sklearn(
    X_matrix,
    y_matrix,
    groups=None,
    scoring=r2_score,
    cv_strategy="image",
    max_nonzero_coefs=None,
    inner_cv=5,
    n_jobs_outer=-1,
    n_jobs_inner=1,
):
    """
    Parameters
    ----------
    X_matrix : np.arr
        Training data for stimulus embeddings.
        Expected shape (n_samples, n_features)
    y_matrix : np.arr
        Training data for brain responses
        Expected shape (n_samples, n_features, n_repeats)
    groups : np.arr
        Group labels for outer_cv, should correspond to image
        identity or image categor(ies).
        Expected shape (n_samples, )
    scoring : Callable
        Scoring function for estimator predictions.
    cv_strategy : str
    max_nonzero_coefs : int or None
        Upper bound on sparsity level OMPCV searches over (analogous
        to the `alphas` grid in ridge). Passed as `max_iter` to
        OrthogonalMatchingPursuitCV. If None, defaults to sklearn's
        default: 10% of n_features, or 5, whichever is larger.
    inner_cv : int, cross-validation generator, iterable, or None
        Number of folds (or splitter/iterable) used *inside*
        OrthogonalMatchingPursuitCV to pick n_nonzero_coefs per target.
        If None, sklearn defaults to KFold (n_splits=5). This inner CV
        is separate from, and not group-aware with respect to,
        outer_cv/groups.
    n_jobs_outer : int
        Number of jobs to run in parallel passed to MultiOutputRegressor
        for the outer cross-validation. Parallelizes across targets
        (voxels) for each outer fold. If -1, uses all available cores.
    n_jobs_inner : int
        Number of jobs to run in parallel passed to
        OrthogonalMatchingPursuitCV for the inner cross-validation.
        Parallelizes across the estimator's inner CV folds within a
        single target fit. Defaults to 1 (sequential).
    """
    from sklearn.linear_model import OrthogonalMatchingPursuitCV
    from sklearn.multioutput import MultiOutputRegressor

    scaler = StandardScaler(with_mean=True, with_std=False)
    scaler.fit_transform(X_matrix)
    scaler.fit_transform(y_matrix)

    if groups is None:
        outer_cv = KFold(shuffle=True, random_state=0)
    else:
        if cv_strategy == "image":
            outer_cv = GroupKFold(shuffle=True, random_state=0)
        elif cv_strategy == "multilabel":
            outer_cv = LeaveOneGroupOut()

    base_estimator = OrthogonalMatchingPursuitCV(
        max_iter=max_nonzero_coefs,
        cv=inner_cv,
        n_jobs=n_jobs_inner,
    )
    # OMPCV only supports single-target regression, so we wrap it to get
    # per-target n_nonzero_coefs selection, same behavior as
    # RidgeCV(alpha_per_target=True). Note that this is not group-aware with
    # respect to outer_cv/groups.
    estimator = MultiOutputRegressor(base_estimator, n_jobs=n_jobs_outer)

    scorer = make_scorer(scoring)
    sklearn.set_config(enable_metadata_routing=True)

    scores = cross_validate(
        estimator,
        X_matrix,
        y=y_matrix,
        cv=outer_cv,
        scoring=scorer,
        params={"groups": groups} if groups is not None else None,
        return_estimator=True,
        return_indices=True,
        error_score="raise",
    )
    return scores


def orthogonal_mp_sklearn(
    X_matrix,
    y_matrix,
    groups=None,
    scoring=r2_score,
    cv_strategy="image",
    max_nonzero_coefs=None,
    n_inner_splits=5,
):
    """
    Nested cross-validation for Orthogonal Matching Pursuit with
    an independetly chosen sparsity level for each target.

    Parameters
    ----------
    X_matrix : np.arr
        Training data for stimulus embeddings.
        Expected shape (n_samples, n_features)
    y_matrix : np.arr
        Training data for brain responses
        Expected shape (n_samples, n_features, n_repeats)
    groups : np.arr
        Group labels for outer_cv, should correspond to image
        identity or image category.
        Expected shape (n_samples, )
    scoring : Callable
        Scoring function for estimator predictions.
    cv_strategy : str
    max_nonzero_coefs : int or None
        Maximum number of non-zero coefficients to consider for the
        OMP estimator. If None, defaults to the number of features in
        X_matrix.
    n_inner_splits : int
        Number of folds for the inner cross-validation to select
        the best sparsity level. Only used if n_nonzero_coefs is None.
    """
    from sklearn.linear_model import orthogonal_mp

    scaler = StandardScaler(with_mean=True, with_std=False)
    scaler.fit_transform(X_matrix)
    scaler.fit_transform(y_matrix)

    if max_nonzero_coefs is None:
        max_nonzero_coefs = X_matrix.shape[1]

    # Outer CV:
    if groups is None:
        outer_cv = KFold(shuffle=True, random_state=0)
    else:
        if cv_strategy in ["category", "image"]:
            outer_cv = GroupKFold(shuffle=True, random_state=0)
        elif cv_strategy == "multilabel":
            outer_cv = LeaveOneGroupOut()

    scores = defaultdict(list)

    # Outer loop
    for outer_train_index, outer_test_index in outer_cv.split(
        X_matrix, y_matrix, groups
    ):
        X_train = X_matrix[outer_train_index]
        y_train = y_matrix[outer_train_index]
        X_test = X_matrix[outer_test_index]
        y_test = y_matrix[outer_test_index]

        n_targets = y_train.shape[1]

        # Inner cv
        if groups is None:
            inner_cv = KFold(
                n_splits=n_inner_splits, shuffle=True, random_state=0
            )
            inner_groups = None
        else:
            inner_groups = groups[outer_train_index]
            if cv_strategy in ["category", "image"]:
                inner_cv = GroupKFold(
                    n_splits=n_inner_splits, shuffle=True, random_state=0
                )
            elif cv_strategy == "multilabel":
                inner_cv = LeaveOneGroupOut()

        # Per target tracking
        validation_scores = np.zeros((max_nonzero_coefs, n_targets))
        n_inner_folds_used = np.zeros((max_nonzero_coefs, n_targets))

        # Inner loop
        for inner_train_index, inner_val_index in inner_cv.split(
            X_train, y_train, inner_groups
        ):
            X_inner_train = X_train[inner_train_index]
            y_inner_train = y_train[inner_train_index]
            X_inner_val = X_train[inner_val_index]
            y_inner_val = y_train[inner_val_index]

            coef_path = orthogonal_mp(
                X_inner_train,
                y_inner_train,
                n_nonzero_coefs=max_nonzero_coefs,
                return_path=True,
                precompute=True,
            )
            n_path = coef_path.shape[-1]

            for k in range(n_path):
                coef_k = coef_path[:, :, k]  # (n_features, n_targets)
                y_pred_k = X_inner_val @ coef_k  # (n_val_samples, n_targets)
                for t in range(n_targets):
                    validation_scores[k, t] += scoring(
                        y_inner_val[:, t], y_pred_k[:, t]
                    )
                    n_inner_folds_used[k, t] += 1

        # Average only over folds that reached each k (guards against
        # premature stopping)
        with np.errstate(invalid="ignore"):
            validation_scores = np.divide(
                validation_scores,
                n_inner_folds_used,
                out=np.full_like(validation_scores, -np.inf),
                where=n_inner_folds_used > 0,
            )
        best_k_per_target = (
            validation_scores.argmax(axis=0) + 1
        )  # (n_targets,)

        # Refit, one call covers every target's own best k
        refit_ceiling = int(best_k_per_target.max())
        coef_path_refit = orthogonal_mp(
            X_train,
            y_train,
            n_nonzero_coefs=refit_ceiling,
            return_path=True,
            precompute=True,
        )
        final_coefs = np.stack(
            [
                coef_path_refit[:, t, best_k_per_target[t] - 1]
                for t in range(n_targets)
            ],
            axis=1,
        )  # (n_features, n_targets)

        y_pred = X_test @ final_coefs
        per_target_test_scores = np.array(
            [scoring(y_test[:, t], y_pred[:, t]) for t in range(n_targets)]
        )

        scores["per_target_test_scores"].append(per_target_test_scores)
        scores["best_k_per_target"].append(best_k_per_target)
        scores["validation_scores"].append(validation_scores)
        scores["indices"].append(
            dict(train=outer_train_index, test=outer_test_index)
        )

    return scores


def omp_fixed_k_sklearn(
    X_matrix,
    y_matrix,
    groups=None,
    scoring=r2_score,
    cv_strategy="image",
    k_grid=None,
    inner_cv=5,
    group_aware_inner=False,
    n_jobs_grid=-1
):
    """
    Parameters
    ----------
    X_matrix : np.arr
        Training data for stimulus embeddings.
        Expected shape (n_samples, n_features)
    y_matrix : np.arr
        Training data for brain responses
        Expected shape (n_samples, n_features, n_repeats)
    groups : np.arr
        Group labels for outer_cv, should correspond to image
        identity or image categor(ies).
        Expected shape (n_samples, )
    scoring : Callable
        Scoring function for estimator predictions.
    cv_strategy : str
    k_grid : array-like or None
        Candidate values for n_nonzero_coefs, shared across all targets.
        Defaults to a log spread from 1 to n_features if None.
    inner_cv : int
        Number of folds for the inner grid search over k, used when
        group_aware_inner=False.
    group_aware_inner : bool
        If True, use a group-aware splitter (GroupKFold) for the inner
        k-search as well, with `groups` routed through via metadata
        routing. If False, inner search uses plain KFold(inner_cv),
        matching the same rigor level as RidgeCV(cv=None).
    n_jobs_grid : int
        Number of jobs to run in parallel for the inner GridSearchCV
        over k. Defaults to -1 (all available cores).
    """
    from sklearn.linear_model import OrthogonalMatchingPursuit
    from sklearn.model_selection import GridSearchCV

    n_features = X_matrix.shape[1]
    if k_grid is None:
        k_grid = np.unique(np.linspace(1, n_features, 20, dtype=int))

    scaler = StandardScaler(with_mean=True, with_std=False)
    scaler.fit_transform(X_matrix)
    scaler.fit_transform(y_matrix)

    if groups is None:
        outer_cv = KFold(shuffle=True, random_state=0)
    else:
        if cv_strategy == "image":
            outer_cv = GroupKFold(shuffle=True, random_state=0)
        elif cv_strategy == "multilabel":
            outer_cv = LeaveOneGroupOut()

    scorer = make_scorer(scoring)
    sklearn.set_config(enable_metadata_routing=True)

    # Inner search over n_nonzero_coefs (k), analogous to RidgeCV's alpha
    # search). A single OMP fit handles all targets, with one shared k
    # across targets per fit.
    param_grid = {"n_nonzero_coefs": k_grid}
    if group_aware_inner:
        inner_cv_splitter = GroupKFold(
            n_splits=inner_cv, shuffle=True, random_state=0
        )
        inner_cv_splitter.set_split_request(groups=True)
    else:
        inner_cv_splitter = KFold(
            n_splits=inner_cv, shuffle=True, random_state=0
        )

    estimator = GridSearchCV(
        OrthogonalMatchingPursuit(),
        param_grid=param_grid,
        cv=inner_cv_splitter,
        scoring=scorer,
        n_jobs=n_jobs_grid,
        error_score="raise",
    )

    scores = cross_validate(
        estimator,
        X_matrix,
        y=y_matrix,
        cv=outer_cv,
        scoring=scorer,
        params={"groups": groups} if groups is not None else None,
        return_estimator=True,
        return_indices=True,
        error_score="raise",
    )
    return scores


@click.command()
@click.option("--sub_name", default="sub-01", help="Subject name.")
@click.option("--roi", default=None, help="Region-of-interest")
@click.option("--cv_strategy", default="image", help="Cross-validation strategy")
@click.option(
    "--scoring_metric",
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
    help="Engine for running encoding analyses. Must be either 'sklearn' "
    "'rrr' or 'himalaya'. Note only the latter is GPU compatiable.",
)
@click.option(
    "--space",
    default="T1w",
    help="Space in which to run encoding analyses. Must be either 'MNI152NLin2009cAsym' or 'T1w'.",
)
def main(sub_name, roi, cv_strategy, scoring_metric, average, data_dir, engine, space):
    """ """
    rois = [None, "EBA", "FFA", "OFA", "pSTS", "MPA", "OPA", "PPA"]
    if roi not in rois:
        err_msg = f"Unrecognized ROI {roi}"
        raise ValueError(err_msg)

    sub_names = ["sub-01", "sub-02", "sub-03", "sub-06"]
    if sub_name not in sub_names:
        err_msg = f"Unrecognized subject {sub_name}"
        raise ValueError(err_msg)

    cv_strategies = ["image", "category", "kfold", "multilabel"]
    if cv_strategy not in cv_strategies:
        err_msg = f"Unrecognized cross-validation strategy {cv_strategy}"
        raise ValueError(err_msg)

    if average and (cv_strategy == "image"):
        err_msg = (
            f"Cross-validation strategy {cv_strategy} is not compatible with 'average'"
        )
        raise ValueError(err_msg)

    scoring_metrics = ["r2_score", "correlation_score"]
    if scoring_metric not in scoring_metrics:
        err_msg = f"Unrecognized scoring metric {scoring_metric}"
        raise ValueError(err_msg)

    engines = ["himalaya", "sklearn", "rrr"]
    if engine not in engines:
        err_msg = f"Unrecognized engine {engine}"
        raise ValueError(err_msg)

    if scoring_metric == "r2_score":
        scoring = r2_score
    if scoring_metric == "correlation_score":
        scoring = correlation_score

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
        if engine != "rrr":
            masker = NiftiMasker(mask_img=roi_mask).fit()
            fig = plotting.plot_stat_map(
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
