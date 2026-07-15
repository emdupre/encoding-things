from collections import defaultdict

import numpy as np
import sklearn
from himalaya.scoring import correlation_score
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
    cross_validate,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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
