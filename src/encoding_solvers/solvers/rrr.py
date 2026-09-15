from collections import defaultdict

import numpy as np
from himalaya.scoring import correlation_score
from sklearn import base, metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
    # cross_validate,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class ReducedRankRidgeRegressionCV(base.BaseEstimator):
    """
    scikit-learn like estimator for the RRRR with built-in Cross-Validation

    Attributes
    ----------

    coef_ : ndarray of shape (n_features, n_targets)
    intercept_ : ndarray of shape (n_targets,)
        Estimated Coefficients & Intercept for the Reduced Rank Ridge Regression

    alpha_ : ndarray of shape (n_targets,)
        Estimated regularization parameter for each target for the Ridge Regression

    ranks_ : list of int
        Estimated Optimal Reduced Rank for the Projection of the Ridge Prediction

    best_score_ : float
        Optimal R2 score obtained on the inner cross-validation loops on 25% of the data

    """

    def __init__(self, alphas, ranks):
        self.alphas = alphas
        self.ranks = ranks
        self.n_alphas = len(alphas)
        self.n_ranks = len(ranks)

    def fit(self, X, Y):
        """
        Reduced Rank Ridge Regression following the paper from [Mukherjee2011]
        https://dept.stat.lsa.umich.edu/~jizhu/pubs/Mukherjee-SADM11.pdf

        Parameters
        ----------

        X : array, shape (n, p)
        Y : array, shape (n, q)

        """

        ridge = Ridge(fit_intercept=True)
        ## Remark: For RRRR, can't optimize one alpha per target (since only one rank for all targets)

        ## Splitting Data into train/validation sets for the rank grid search (75/25)
        n = len(X)
        Xt, Xv = X[: int(0.75 * n)], X[int(0.75 * n) :]
        Yt, Yv = Y[: int(0.75 * n)], Y[int(0.75 * n) :]

        ## Finding best rank hyperparameters (2D-GridSearch)
        score = np.zeros((self.n_ranks))

        ## fitting the ridge on the train set
        ridge.set_params(alpha=self.alphas[0])  # only one alpha for all targets
        ridge.fit(Xt, Yt)

        ## predicting the validation target with the trained ridge
        Yt_pred = ridge.predict(Xt)

        for i in range(self.n_ranks):
            svd = TruncatedSVD(n_components=self.ranks[i])
            svd.fit(Yt_pred)
            Vr = svd.components_.T  # shape: (q, rank)

            ## computing the validation score for every pair of (alpha, rank)
            Yv_pred = ridge.predict(Xv) @ Vr @ Vr.T
            score[i] = metrics.r2_score(Yv.ravel(), Yv_pred.ravel())

        idx_opt_rank = np.argmax(score)
        self.rank_ = self.ranks[idx_opt_rank]
        self.best_score_ = score[idx_opt_rank]

        ## Re-Computing the optimal regressions coefficents for the RRRR on full data
        ridge.fit(X, Y)
        svd = TruncatedSVD(n_components=self.rank_)
        svd.fit(ridge.predict(X))
        Vr = svd.components_.T
        self.coef_ = ridge.coef_.T @ Vr @ Vr.T  # shape: (p, q)
        self.intercept_ = ridge.intercept_  # shape: (q)

    def predict(self, X):
        return self.intercept_ + X @ self.coef_

    def score(self, X, y):
        y_pred = self.predict(X)
        return metrics.r2_score(y.ravel(), y_pred.ravel(), multioutput="raw_values")

    def eval(self, y_pred, y_true):
        return self.corr(y_pred, y_true)

    def corr(self, X, Y, axis=0):
        """
        Compute the pearson correlation between X and Y, handling the constant columns.

        Remark :
            All NaN values coming from constant columns are replaced by zeros.

        Parameters
        ----------
        X, Y: array, shape (n, p)

        """
        mX = X - np.mean(X, axis=axis, keepdims=True)
        mY = Y - np.mean(Y, axis=axis, keepdims=True)
        norm_mX = np.sqrt(np.sum(mX**2, axis=axis, keepdims=True))
        norm_mX[norm_mX == 0] = 1.0
        norm_mY = np.sqrt(np.sum(mY**2, axis=axis, keepdims=True))
        norm_mY[norm_mY == 0] = 1.0

        return np.sum(mX / norm_mX * mY / norm_mY, axis=axis)


def ridgeCV_rrr(
    X_matrix,
    y_matrix,
    ranks,
    groups=None,
    scoring=metrics.r2_score,
    cv_strategy="image",
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
    scores = defaultdict()
    train_indices, test_indices = [], []
    best_scores = []
    best_ranks = []

    if groups is None:
        outer_cv = KFold(shuffle=True, random_state=0)
    else:
        if cv_strategy in ["category", "image"]:
            outer_cv = GroupKFold(shuffle=True, random_state=0)
        elif cv_strategy == "multilabel":
            outer_cv = LeaveOneGroupOut()

    alphas = np.array([100])
    pl = make_pipeline(
        StandardScaler(with_mean=True, with_std=False),
        ReducedRankRidgeRegressionCV(
            alphas=alphas,
            ranks=ranks,
        ),
    )

    for train_index, test_index in outer_cv.split(X_matrix, y_matrix, groups):
        X_train, X_test = X_matrix[train_index], X_matrix[test_index]
        y_train, y_test = y_matrix[train_index], y_matrix[test_index]

        pl.fit(X_train, y_train)
        y_pred = pl.predict(X_test)
        y_true = StandardScaler(with_mean=True, with_std=False).fit_transform(
            y_matrix[test_index]
        )

        if scoring is metrics.r2_score:
            score = metrics.r2_score(y_true, y_pred, multioutput="raw_values")
        elif scoring is correlation_score:
            score = correlation_score(y_true, y_pred)

        best_scores.append(score)
        best_ranks.append(pl[-1].rank_)

    scores["best_ranks"] = best_ranks
    scores["best_scores"] = best_scores
    scores["indices"] = {"train": train_indices, "test": test_indices}

    return scores
