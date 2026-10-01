import pickle
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_pkl_keys(results_pkl):
    """
    Grab fields necessary for plotting from a given results.pkl file.

    Parameters
    ----------
    results_pkl : str
        Pathname

    Returns
    -------
    test_scores: list
    reg_params: list
    fit_times: list
    """
    with open(results_pkl, "rb") as p:
        scores = pickle.load(p)

    test_scores = scores["per_target_test_scores"]
    fit_times = scores["fit_time"]
    n_train_samples = [len(idx) for idx in scores["indices"]["train"]]
    explainable_variance = scores["explainable_variance"]

    # grab different regularization params for each of the solvers handle
    reg_params = []
    if "omp" in str(results_pkl):
        for fold in scores["estimator"]:
            n_coefs = []
            for feature_pl in fold.estimators_:
                n_coefs.append(
                    feature_pl.named_steps[
                        "orthogonalmatchingpursuitcv"
                    ].n_nonzero_coefs_
                )
            reg_params.append(np.asarray(n_coefs))
    elif "rrr" in str(results_pkl):
        reg_params = scores["ranks"]
    else:
        reg_params = scores["alphas"]

    return test_scores, reg_params, fit_times, n_train_samples, explainable_variance


def create_results_dict(sub_name, roi, solver, metric, data_path, percent=1.0):
    """
    Create a results dictionary for a given subject, ROI, and metric.

    Parameters
    ----------
    sub_name: str
    roi: str
    metric: str
    data_path: str of Path-like
    """
    res_dict = defaultdict()

    results_pkl = Path(
        data_path,
        "encoding-results",
        f"{sub_name}_space-T1w_roi-{roi}_dataPercent-{percent}_cv-kfold_{metric}_{solver}.pkl",
    )
    (test_scores, reg_params, fit_times, n_train_samples, explainable_variance) = (
        load_pkl_keys(results_pkl)
    )

    res_dict["test_scores"] = test_scores
    res_dict["regularization_params"] = reg_params
    res_dict["fit_times"] = fit_times
    res_dict["n_train_samples"] = n_train_samples
    res_dict["explainable_variance"] = explainable_variance

    return res_dict


def compute_quantile_and_std(test_scores, q=0.9, ddof=1):
    """
    From results_dict[solver] (list of n_folds arrays, each array =
    per-target scores for that fold), compute:
      - quantile_scores[solver]: mean across folds of the q-quantile
        of the per-target scores within each fold
      - std_scores[solver]: std across folds of those per-fold
        q-quantiles (used as the error bar on the bar plot)

    ddof=1 gives the sample std (usually preferable with few folds);
    use ddof=0 to match np.std's default.
    """
    folds = [np.asarray(fold) for fold in test_scores]
    fold_quantiles = np.array([np.quantile(f, q) for f in folds])

    quantile_scores = fold_quantiles.mean()
    std_scores = fold_quantiles.std(ddof=ddof)

    return quantile_scores, std_scores


def gather_test_scores(sub_name, roi, data_path, metric="r2_score"):
    """
    Parameters
    ----------
    sub_name: str
    roi: str
    data_path: str or Path-like
    metric: str

    Returns
    -------
    scores: dict
    highest_scores: dict
    std_scores: dict
    """
    solvers = ["sklearn", "ompCV", "rrr", "ompCV_ridge"]  # "himalaya",
    scores = defaultdict()
    highest_scores = defaultdict()
    std_scores = defaultdict()

    for solver in solvers:
        try:
            res_dict = create_results_dict(sub_name, roi, solver, metric, data_path)
            test_scores = res_dict["test_scores"]
            highest_test_scores, std_test_scores = compute_quantile_and_std(
                test_scores, q=0.9, ddof=1
            )

            scores[solver] = test_scores
            highest_scores[solver] = highest_test_scores
            std_scores[solver] = std_test_scores
        except FileNotFoundError:
            warnings.warn(
                f"{solver} results do not exist on {data_path}. "
                f"Verify that this solver has been run for {sub_name}, {roi} "
            )
            pass

    expl_var = res_dict["explainable_variance"]

    return scores, highest_scores, std_scores, expl_var
