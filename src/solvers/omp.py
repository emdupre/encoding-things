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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def ompCV_sklearn(
    X_matrix,
    y_matrix,
    groups=None,
    scoring=r2_score,
    cv_strategy="image",
    max_nonzero_coefs=None,
    inner_cv=5,
    n_jobs_outer=10,
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
        Scoring function for estimator predictions. Used only for the
        outer cross_validate reporting. The inner CV selects n_nonzero_coefs
        per target by minimizing MSE along its precomputed path.
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
        (voxels) for each outer fold. Defaults to 10.
    n_jobs_inner : int
        Number of jobs to run in parallel passed to
        OrthogonalMatchingPursuitCV for the inner cross-validation.
        Parallelizes across the estimator's inner CV folds within a
        single target fit. Defaults to 1 (sequential).
    """
    if groups is None:
        outer_cv = KFold(shuffle=True, random_state=0)
    else:
        if cv_strategy == "image":
            outer_cv = GroupKFold(shuffle=True, random_state=0)
        elif cv_strategy == "multilabel":
            outer_cv = LeaveOneGroupOut()

    base_estimator = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        OrthogonalMatchingPursuitCV(
            max_iter=max_nonzero_coefs,
            cv=inner_cv,
            n_jobs=n_jobs_inner,
        ),
    )
    # OMPCV only supports single-target regression, so we wrap it to get
    # per-target n_nonzero_coefs selection. Note that this is not group-aware
    # with respect to outer_cv/groups.
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

    # Reconstruct per-target scores from the fitted estimators and indices,
    # since cross_validate only returns a single score per fold.
    best_scores = []
    for fold_idx, fitted_estimator in enumerate(scores["estimator"]):
        test_idx = scores["indices"]["test"][fold_idx]
        X_test_fold = X_matrix[test_idx]
        y_test_fold = y_matrix[test_idx]
        y_pred_fold = fitted_estimator.predict(X_test_fold)
        fold_scores = np.array(
            [
                scoring(y_test_fold[:, t], y_pred_fold[:, t])
                for t in range(y_test_fold.shape[1])
            ]
        )
        best_scores.append(fold_scores)
    scores.update({"per_target_test_scores": best_scores})
    return scores


def _center_and_normalize(X_fit, y_fit, *others_X):
    """
    Fits centering + unit-norm column scaling on X_fit/y_fit only,
    then applies the same transform to X_fit and any other arrays
    passed in `others_X` (e.g. held-out X). orthogonal_mp has no
    intercept term and assumes unit-norm columns.
    """
    X_mean = X_fit.mean(axis=0)
    X_centered = X_fit - X_mean
    col_norms = np.linalg.norm(X_centered, axis=0)
    col_norms[col_norms == 0] = 1.0  # guard constant columns
    X_scaled = X_centered / col_norms
    y_mean = y_fit.mean(axis=0)
    y_centered = y_fit - y_mean
    others_scaled = [(Xo - X_mean) / col_norms for Xo in others_X]
    return X_scaled, y_centered, others_scaled


def orthogonal_mp_sklearn(
    X_matrix,
    y_matrix,
    groups=None,
    scoring=r2_score,
    cv_strategy="image",
    max_nonzero_coefs=None,
    inner_cv_splits=5,
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
        Scoring function for the final per-target scoring -n the outer loop.
        Inner loop is set to compute vectorized R^2 for each candidate k.
    cv_strategy : str
    max_nonzero_coefs : int or None
        Maximum number of non-zero coefficients to consider for the
        OMP estimator. If None, defaults to 10% of n_features (at
        least 5, capped at n_features)
    n_inner_splits : int
        Number of folds for the inner cross-validation to select
        the best sparsity level.
    """
    if max_nonzero_coefs is None:
        n_features = X_matrix.shape[1]
        max_nonzero_coefs = min(max(int(n_features * 0.1), 5), n_features)

    # Define outer CV:
    if groups is None:
        outer_cv = KFold(shuffle=True, random_state=0)
    else:
        if cv_strategy in ["category", "image"]:
            outer_cv = GroupKFold(shuffle=True, random_state=0)
        elif cv_strategy == "multilabel":
            outer_cv = LeaveOneGroupOut()

    scores = defaultdict(list)

    # ---- Outer loop ----
    for outer_train_index, outer_test_index in outer_cv.split(
        X_matrix, y_matrix, groups
    ):
        X_train_raw = X_matrix[outer_train_index]
        y_train_raw = y_matrix[outer_train_index]
        X_test_raw = X_matrix[outer_test_index]
        y_test_raw = y_matrix[outer_test_index]

        # orth_mp requires centered and unit-norm columns. Center/normalize
        # are fit on the outer-train set only, then applied to outer-test.
        X_train, y_train, (X_test,) = _center_and_normalize(
            X_train_raw, y_train_raw, X_test_raw
        )
        y_test = y_test_raw - y_train_raw.mean(axis=0)

        n_targets = y_train.shape[1]

        # ---- Inner cv ----
        if groups is None:
            inner_cv = KFold(n_splits=inner_cv, shuffle=True, random_state=0)
            inner_groups = None
        else:
            inner_groups = groups[outer_train_index]
            if cv_strategy in ["category", "image"]:
                inner_cv = GroupKFold(
                    n_splits=inner_cv, shuffle=True, random_state=0
                )
            elif cv_strategy == "multilabel":
                inner_cv = LeaveOneGroupOut()

        # Accumulators for the inner search: rows = candidate k (1..max),
        # columns = targets. validation_scores -> average R^2 across inner
        # folds for each k/target pair, n_inner_folds_used -> how many folds
        # contributed to each k/target pair.
        validation_scores = np.zeros((max_nonzero_coefs, n_targets))
        n_inner_folds_used = np.zeros((max_nonzero_coefs, n_targets))

        # ---- Inner loop: ----
        # Split on the raw outer-train arrays, but use the preprocessed version
        # for fitting and scoring in the inner loop.
        for inner_train_index, inner_val_index in inner_cv.split(
            X_train_raw, y_train_raw, inner_groups
        ):
            X_inner_train_raw = X_train_raw[inner_train_index]
            y_inner_train_raw = y_train_raw[inner_train_index]
            X_inner_val_raw = X_train_raw[inner_val_index]
            y_inner_val_raw = y_train_raw[inner_val_index]

            # Fresh preprocessing per inner fold -- fit on inner-train only,
            # to avoid inner-val rows leaking into the centering/scaling stats
            X_inner_train, y_inner_train, (X_inner_val,) = (
                _center_and_normalize(
                    X_inner_train_raw, y_inner_train_raw, X_inner_val_raw
                )
            )
            y_inner_val = y_inner_val_raw - y_inner_train_raw.mean(axis=0)

            # ONE call computes the entire path (k=1, 2, ..., max_nonzero_coefs)
            # for ALL targets at once.
            coef_path = orthogonal_mp(
                X_inner_train,
                y_inner_train,
                n_nonzero_coefs=max_nonzero_coefs,
                return_path=True,
                precompute=True,
            )  # shape: (n_features, n_targets, n_path_steps)
            n_path = coef_path.shape[-1]

            # Scoring computed for all targets at once
            y_val_mean = y_inner_val.mean(axis=0, keepdims=True)
            ss_tot = np.sum((y_inner_val - y_val_mean) ** 2, axis=0)
            # Targets with zero variance in THIS fold's validation split
            # contribute nothing usable, for every k
            valid_targets = ss_tot > 0  # boolean mask (n_targets,)
            # Predictions for ALL sparsity levels at once
            y_pred_all = np.einsum(
                "vf,ftp->vtp",
                X_inner_val,
                coef_path,
            )
            # Broadcast y_true over path dimension
            residuals = y_inner_val[:, :, None] - y_pred_all
            # Sum squared residuals
            ss_res = np.sum(residuals**2, axis=0)  # (n_targets, n_path)
            with np.errstate(divide="ignore", invalid="ignore"):
                r2 = 1 - ss_res / ss_tot[:, None]  # shape: (n_targets, n_path)

            validation_scores[:n_path, valid_targets] += r2[valid_targets].T
            n_inner_folds_used[:n_path, valid_targets] += 1

        # Average inner scores across folds, only divide where at least one
        # fold contributed to that k/target pair.
        with np.errstate(invalid="ignore"):
            validation_scores = np.divide(
                validation_scores,
                n_inner_folds_used,
                out=np.full_like(validation_scores, -np.inf),
                where=n_inner_folds_used > 0,
            )  # shape: (max_nonzero_coefs, n_targets)

        # Pick each target's own best k
        best_k_per_target = validation_scores.argmax(axis=0) + 1  # (n_targets)
        refit_ceiling = int(best_k_per_target.max())

        # Refit on full outer-train, each target at its own best k
        coef_path_refit = orthogonal_mp(
            X_train,
            y_train,
            n_nonzero_coefs=refit_ceiling,
            return_path=True,
            precompute=True,
        )  # shape: (n_features, n_targets, refit_ceiling)

        # For each target, get its own best k (indexing only)
        final_coefs = np.stack(
            [
                coef_path_refit[:, t, best_k_per_target[t] - 1]
                for t in range(n_targets)
            ],
            axis=1,
        )  # (n_features, n_targets)

        # Compute predictions on the outer-test set
        y_pred = X_test @ final_coefs
        per_target_test_scores = np.array(
            [scoring(y_test[:, t], y_pred[:, t]) for t in range(n_targets)]
        )

        # Store results for this outer fold
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
