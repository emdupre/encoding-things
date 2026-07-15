from collections import defaultdict

import numpy as np
import sklearn
from sklearn.linear_model import (
    OrthogonalMatchingPursuit,
    OrthogonalMatchingPursuitCV,
    orthogonal_mp,
)
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
    cross_validate,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


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
            inner_cv = KFold(n_splits=n_inner_splits, shuffle=True, random_state=0)
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
        best_k_per_target = validation_scores.argmax(axis=0) + 1  # (n_targets,)

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
            [coef_path_refit[:, t, best_k_per_target[t] - 1] for t in range(n_targets)],
            axis=1,
        )  # (n_features, n_targets)

        y_pred = X_test @ final_coefs
        per_target_test_scores = np.array(
            [scoring(y_test[:, t], y_pred[:, t]) for t in range(n_targets)]
        )

        scores["per_target_test_scores"].append(per_target_test_scores)
        scores["best_k_per_target"].append(best_k_per_target)
        scores["validation_scores"].append(validation_scores)
        scores["indices"].append(dict(train=outer_train_index, test=outer_test_index))

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
    n_jobs_grid=-1,
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
        inner_cv_splitter = GroupKFold(n_splits=inner_cv, shuffle=True, random_state=0)
        inner_cv_splitter.set_split_request(groups=True)
    else:
        inner_cv_splitter = KFold(n_splits=inner_cv, shuffle=True, random_state=0)

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
