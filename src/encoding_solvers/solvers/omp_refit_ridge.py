from warnings import warn

import numpy as np
import sklearn
from himalaya.scoring import correlation_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import OrthogonalMatchingPursuitCV, RidgeCV
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
    cross_validate,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class OMPSelectRidge(BaseEstimator, RegressorMixin):
    """
    Two-stage single-target estimator: OMPCV selects a sparse support of
    features, then RidgeCV is fit (and its own alpha chosen via CV)
    restricted to that support.
    Parameters
    ----------
    omp_max_iter : int or None
        Upper bound on sparsity level OMPCV searches over. Passed as
        `max_iter` to OrthogonalMatchingPursuitCV. If None, sklearn
        defaults to 10% of n_features.
    omp_cv : int, cross-validation generator, or iterable
        Inner CV used by OrthogonalMatchingPursuitCV to choose
        n_nonzero_coefs. Not group-aware with respect to outer_cv/groups.
    omp_n_jobs : int
        Number of jobs to run in parallel for ompCV. Defaults to 1.
    ridge_alphas : array-like or None
        Grid of alphas RidgeCV searches over. If None (default),
        np.logspace(1, 20, 20) is used.
    ridge_cv : int, cross-validation generator, iterable, or None
        CV used by RidgeCV to choose alpha. None uses sklearn's efficient
        generalized (leave-one-out) CV.
    """

    def __init__(
        self,
        omp_max_iter=None,
        omp_cv=5,
        omp_n_jobs=1,
        ridge_alphas=None,
        ridge_cv=None,
    ):
        self.omp_max_iter = omp_max_iter
        self.omp_cv = omp_cv
        self.omp_n_jobs = omp_n_jobs
        self.ridge_alphas = ridge_alphas
        self.ridge_cv = ridge_cv

    def fit(self, X, y):
        # Fit OMP to select features
        omp = OrthogonalMatchingPursuitCV(
            max_iter=self.omp_max_iter,
            cv=self.omp_cv,
            n_jobs=self.omp_n_jobs,
        )
        omp.fit(X, y)

        support = np.flatnonzero(omp.coef_)
        self.support_ = support
        self.omp_n_nonzero_coefs_ = omp.n_nonzero_coefs_

        ridge_alphas = (
            np.logspace(1, 20, 20)
            if self.ridge_alphas is None
            else self.ridge_alphas
        )
        ridge = RidgeCV(alphas=ridge_alphas, cv=self.ridge_cv)
        if support.size > 0:
            ridge.fit(X[:, support], y)
        else:
            # OMP selected nothing for this target; fall back to an
            # intercept-only model (To be discussed)
            ridge.fit(np.zeros((X.shape[0], 1)), y)
            warn(
                "OMP selected no features for this target; "
                "falling back to intercept-only RidgeCV."
            )
        self.ridge_ = ridge
        return self

    def predict(self, X):
        if self.support_.size == 0:
            return np.full(X.shape[0], self.ridge_.intercept_)
        return self.ridge_.predict(X[:, self.support_])


def ompCV_ridge_sklearn(
    X_matrix,
    y_matrix,
    groups=None,
    scoring=r2_score,
    cv_strategy="image",
    max_nonzero_coefs=None,
    inner_cv=5,
    ridge_alphas=np.logspace(1, 20, 20),
    ridge_cv=None,
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
        Training data for neural responses.
        Expected shape (n_samples, n_targets)
    groups : np.arr
        Group labels for outer_cv, should correspond to image identity
        or image categor(ies).
        Expected shape (n_samples, )
    scoring : Callable
        Scoring function for estimator predictions. Used only for the
        outer cross_validate reporting. The inner OMP CV selects
        n_nonzero_coefs per target by minimizing MSE along its
        precomputed path; the inner Ridge CV then selects alpha per
        target on that fixed support.
    cv_strategy : str
    max_nonzero_coefs : int or None
        Upper bound on sparsity level OMPCV searches over. Passed as
        `max_iter` to OrthogonalMatchingPursuitCV. If None, defaults to
        sklearn's default: 10% of n_features.
    inner_cv : int, cross-validation generator, iterable, or None
        Number of folds (or splitter/iterable) used *inside*
        OrthogonalMatchingPursuitCV to pick n_nonzero_coefs per target.
        If None, sklearn defaults to KFold (n_splits=5). This inner CV
        is separate from, and not group-aware with respect to,
        outer_cv/groups.
    ridge_alphas : array-like
        Grid of alphas RidgeCV searches over when refitting on the
        OMP-selected support, per target.
    ridge_cv : int, cross-validation generator, iterable, or None
        CV used by RidgeCV to pick alpha per target. None uses
        sklearn's efficient leave-one-out generalized CV.
    n_jobs_outer : int
        Number of jobs to run in parallel passed to MultiOutputRegressor
        for the outer cross-validation. Parallelizes across targets
        (voxels) for each outer fold. Defaults to 10.
    n_jobs_inner : int
        Number of jobs to run in parallel passed to
        OrthogonalMatchingPursuitCV for the inner CV that selects
        n_nonzero_coefs. Defaults to 1 (sequential). Note RidgeCV's
        inner alpha search does not take an n_jobs argument.
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
        OMPSelectRidge(
            omp_max_iter=max_nonzero_coefs,
            omp_cv=inner_cv,
            omp_n_jobs=n_jobs_inner,
            ridge_alphas=ridge_alphas,
            ridge_cv=ridge_cv,
        ),
    )
    # OMPSelectRidge only supports single-target regression, so we wrap it
    # to get per-target support-selection + ridge-refit. Note that the
    # inner selection/refit is not group-aware with respect to
    # outer_cv/groups.
    estimator = MultiOutputRegressor(base_estimator, n_jobs=n_jobs_outer)
    sklearn.set_config(enable_metadata_routing=True)

    if scoring is r2_score:
        scorer = make_scorer(scoring)
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
    elif scoring is correlation_score:
        scores = cross_validate(
            estimator,
            X_matrix,
            y=y_matrix,
            cv=outer_cv,
            params={"groups": groups} if groups is not None else None,
            return_estimator=True,
            return_indices=True,
            error_score="raise",
        )

    # Reconstruct per-target scores from the outer folds, since cross_validate
    # only returns a single score per fold.
    per_target_scores = []
    for fold_idx, fitted_estimator in enumerate(scores["estimator"]):
        test_idx = scores["indices"]["test"][fold_idx]
        X_test_fold = X_matrix[test_idx]
        y_test_fold = y_matrix[test_idx]
        y_pred_fold = fitted_estimator.predict(X_test_fold)

        if scoring is r2_score:
            fold_scores = r2_score(
                y_test_fold, y_pred_fold, multioutput="raw_values"
            )
        elif scoring is correlation_score:
            fold_scores = np.asarray(
                correlation_score(y_test_fold, y_pred_fold)
            )

        per_target_scores.append(fold_scores)

    per_target_scores = np.stack(
        per_target_scores, axis=0
    )  # (n_folds, n_targets)
    scores.update({"per_target_test_scores": per_target_scores})
    return scores
