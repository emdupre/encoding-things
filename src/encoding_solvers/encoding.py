import pickle
from pathlib import Path

import click
import numpy as np
import scipy
from himalaya.scoring import correlation_score
from sklearn.metrics import r2_score

from encoding_solvers.cross_validation import define_groups
from encoding_solvers.solvers import (
    ompCV_ridge_sklearn,
    ompCV_sklearn,
    ridgeCV_rrr,
    ridgeCV_sklearn,
)


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
    "--solver",
    default="himalaya",
    type=click.Choice(["sklearn", "rrr", "ompCV", "ompCV_ridge"]),
    help="Engine for running encoding analyses. Must be either 'sklearn' "
    "'rrr', 'ompCV', or 'ompCV_ridge'.",
)
@click.option(
    "--scoring_metric",
    type=click.Choice(["r2_score", "correlation_score"]),
    default="r2_score",
    help="Desired scoring metric. Currently only 'r2_score' and 'correlation_score' "
    "are supported.",
)
@click.option(
    "--data_dir",
    default="/home/emdupre/links/projects/rrg-pbellec/emdupre/things.betas",
    help="Data directory.",
)
@click.option(
    "--data_percentage",
    type=click.FloatRange(0, 1, min_open=True),
    default=1,
    help="Percentage of data to include in analysis. Must be between 0 and 1.",
)
def main(
    sub_name, roi, cv_strategy, solver, scoring_metric, data_dir, data_percentage=1
):
    """ """
    space = "T1w"

    # set scoring function callable from string
    if scoring_metric == "r2_score":
        scoring = r2_score
    if scoring_metric == "correlation_score":
        scoring = correlation_score

    # load data
    X_matrix = np.load(
        Path(
            data_dir,
            "encoding-inputs",
            space,
            f"{sub_name}_stim_features.npy",
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
    else:
        y_matrix = np.load(
            Path(
                data_dir,
                "encoding-inputs",
                space,
                f"{sub_name}_space-{space}_brain_responses.npy",
            )
        )

    # ignore inner groups for now
    groups, _ = define_groups(cv_strategy, sub_name, data_dir)

    # apply data_percentage, subset data if data_percentage < 1
    n_samples = int(data_percentage * len(X_matrix))

    print(
        f"Data percentage is {data_percentage}, "
        f"using {n_samples} samples for train and test."
    )
    X_matrix = X_matrix[:n_samples]
    y_matrix = y_matrix[:n_samples]
    if groups is not None:
        groups = groups[:n_samples]

    if solver == "sklearn":
        scores = ridgeCV_sklearn(
            X_matrix,
            y_matrix,
            groups=groups,
            scoring=scoring,
            cv_strategy=cv_strategy,
        )

    elif solver == "rrr":
        scores = ridgeCV_rrr(
            X_matrix,
            y_matrix,
            ranks=[2**i for i in range(7)],
            groups=groups,
            scoring=scoring,
            cv_strategy=cv_strategy,
        )

    elif solver == "ompCV":
        scores = ompCV_sklearn(
            X_matrix,
            y_matrix,
            groups=groups,
            scoring=scoring,
            cv_strategy=cv_strategy,
            max_nonzero_coefs=100,
            inner_cv=5,
        )

    elif solver == "ompCV_ridge":
        scores = ompCV_ridge_sklearn(
            X_matrix,
            y_matrix,
            groups=groups,
            scoring=scoring,
            cv_strategy=cv_strategy,
            max_nonzero_coefs=100,
            inner_cv=5,
        )

    expl_var = explainable_variance(y_matrix)
    scores["explainable_variance"] = expl_var

    # elif engine == "himalaya":
    #     scores = ridgeCV_himalaya(
    #         X_matrix,
    #         y_matrix,
    #         groups=groups,
    #         scoring=scoring,
    #         cv_strategy=cv_strategy,
    #     )
    #     try:
    #         best_alphas = [best_alpha_.cpu() for best_alpha_ in scores["best_alphas"]]
    #         best_scores = [best_score_.cpu() for best_score_ in scores["best_scores"]]
    #     except AttributeError:
    #         best_alphas = scores["best_alphas"]
    #         best_scores = scores["best_scores"]

    # elif engine == "omp":
    #     scores = orthogonal_mp_sklearn(
    #         X_matrix,
    #         y_matrix,
    #         groups=groups,
    #         scoring=scoring,
    #         cv_strategy=cv_strategy,
    #         max_nonzero_coefs=1000,
    #         inner_cv_splits=5,
    #     )
    #     best_scores = scores["per_target_test_scores"]

    if roi is None:
        roi = "wholebrain"

    out_file = Path(
        data_dir,
        "encoding-results",
        f"{sub_name}_space-{space}_roi-{roi}_dataPercent-{data_percentage}_cv-{cv_strategy}_{scoring_metric}_{solver}.pkl",
    )

    # if not out_file.is_file():
    with open(out_file, "wb") as f:
        pickle.dump(scores, f)

    # to un-pickle
    # with open(out_file, 'rb') as f:
    #     check = pickle.load(f)


if __name__ == "__main__":
    main()
