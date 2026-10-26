# %%
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# from .utils import _get_solver_colors, _get_solver_names, gather_test_scores


METRIC_NAMES = {
    "r2_score": "$R^2$ Score",
    "correlation_score": "Correlation Score",
}

PARAM_LABELS = {
    "sklearn": "Alpha",
    "rrr": "Rank",
    "ompCV": "n non-zero coefs",
    "ompCV_ridge": "n non-zero coefs",
}
LOG_SOLVERS = frozenset({"sklearn"})

DATA_PERCENTS = [f"{p / 10:.1f}" for p in range(1, 11)]  # "0.1" ... "1.0"
XTICKS = [0.2, 0.4, 0.6, 0.8, 1.0]

_KEY_PREFIX = "percent-"


# Loading ------------------------------------------------------------------ #
@dataclass
class RoiResults:
    """Results for one ROI, each dict keyed by ``"percent-<value>"``."""

    test_scores: dict = field(default_factory=dict)
    highest_scores: dict = field(default_factory=dict)
    std_scores: dict = field(default_factory=dict)
    expl_var: dict = field(default_factory=dict)
    reg_params: dict = field(default_factory=dict)


def _percent_key(percent):
    return f"{_KEY_PREFIX}{percent}"


def _percent_from_key(key):
    return float(key.removeprefix(_KEY_PREFIX))


def load_roi_results(sub, roi, data_path, metric, data_percents=DATA_PERCENTS):
    """Gather scores and regularization parameters for one ROI.

    Percentages with no data are skipped with a warning.
    """
    results = RoiResults()
    for percent in data_percents:
        key = _percent_key(percent)
        try:
            (
                test_scores,
                highest_scores,
                std_scores,
                expl_var,
                reg_params,
            ) = gather_test_scores(
                sub,
                roi=roi,
                data_path=data_path,
                metric=metric,
                data_percent=percent,
            )
        except UnboundLocalError:
            # TODO: make gather_test_scores proof against missing data
            warnings.warn(f"No data for {roi} at {percent} data percentage.")
            continue

        results.test_scores[key] = test_scores
        results.highest_scores[key] = highest_scores
        results.std_scores[key] = std_scores
        results.expl_var[key] = expl_var
        results.reg_params[key] = reg_params
    return results


# Regularization-parameter curves ------------------------------------------ #
def _iqr_errorbars(params):
    """Return (median, distance to 25th pct, distance to 75th pct)."""
    q25, med, q75 = np.quantile(params, [0.25, 0.5, 0.75])
    return med, med - q25, q75 - med


def _n_voxels(test_scores_roi):
    """Number of voxels, read from the largest data percentage available."""
    if not test_scores_roi:
        return None
    last_key = max(test_scores_roi, key=_percent_from_key)
    any_solver_scores = next(iter(test_scores_roi[last_key].values()))
    return len(any_solver_scores[0])


def _plot_params_curves(axes_col, roi, solvers, results, n_voxels):
    """Plot median (IQR) of the selected regularization parameter per solver."""
    # Sort by percentage so a missing percentage can't shift the x positions.
    keys = sorted(results.reg_params, key=_percent_from_key)

    for ax, solver in zip(axes_col, solvers):
        xs, medians, lower, upper = [], [], [], []
        for key in keys:
            per_solver = results.reg_params[key]
            if solver not in per_solver:
                continue  # solver not run at this percentage
            # concatenate all folds and all targets
            params = np.concatenate([np.ravel(p) for p in per_solver[solver]])
            med, err_low, err_high = _iqr_errorbars(params)
            xs.append(_percent_from_key(key))
            medians.append(med)
            lower.append(err_low)
            upper.append(err_high)

        if xs:
            ax.errorbar(
                xs,
                medians,
                yerr=[lower, upper],
                marker="o",
                color=_get_solver_colors(solver),
            )
        else:
            ax.text(
                0.5,
                0.5,
                "not run yet",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="grey",
            )

        if solver in LOG_SOLVERS:
            ax.set_yscale("log")
        else:
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.grid(True, linestyle="-", alpha=0.7)

        spec = ax.get_subplotspec()
        if spec.is_first_row():
            title = roi if n_voxels is None else f"{roi} - {n_voxels} voxels"
            ax.set_title(title)
        if spec.is_first_col():
            label = PARAM_LABELS.get(solver, "Parameter")
            ax.set_ylabel(f"{_get_solver_names(solver)}\n{label}")
        if spec.is_last_row():
            ax.set_xticks(XTICKS)
            ax.set_xticklabels([f"{t:.0%}" for t in XTICKS])
            ax.set_xlabel("Data percentage")


def plot_regularization_curves(
    sub, rois, results, solvers=("sklearn", "rrr", "ompCV")
):
    """Plot the median regularization parameter against the data percentage."""
    fig, axes = plt.subplots(
        len(solvers),
        len(rois),
        sharex=True,
        sharey="row",
        figsize=(4 * len(rois), 2.5 * len(solvers)),
        squeeze=False,
    )
    for j, roi in enumerate(rois):
        _plot_params_curves(
            axes[:, j],
            roi,
            solvers,
            results[roi],
            _n_voxels(results[roi].test_scores),
        )

    fig.suptitle(f"{sub} : Median regularization parameter")
    fig.tight_layout()
    return fig


# Score scaling curves ----------------------------------------------------- #
def plot_scaling_curves(
    ax,
    roi,
    scores_all,
    highest_scores_all,
    std_scores_all,
    expl_var_all,
):
    """Plot the 90th-percentile score per solver against the data percentage."""
    # sorted so that a missing percentage can't shift the x positions
    keys = sorted(scores_all[roi], key=_percent_from_key)
    if not keys:
        ax.text(
            0.5,
            0.5,
            "no data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="grey",
        )
        ax.set_title(roi)
        return ax

    # all solvers found at any percentage, in first-seen order
    solvers = list(dict.fromkeys(s for k in keys for s in scores_all[roi][k]))

    for solver in solvers:
        solver_keys = [k for k in keys if solver in highest_scores_all[roi][k]]
        ax.errorbar(
            [_percent_from_key(k) for k in solver_keys],
            [highest_scores_all[roi][k][solver] for k in solver_keys],
            yerr=[std_scores_all[roi][k][solver] for k in solver_keys],
            label=_get_solver_names(solver),
            marker="o",
            color=_get_solver_colors(solver),
        )

    ax.plot(
        [_percent_from_key(k) for k in keys],
        [np.quantile(expl_var_all[roi][k], 0.9) for k in keys],
        label="Noise ceiling",
        color="grey",
        marker="o",
    )

    ax.set_xticks(XTICKS)
    ax.set_xticklabels([f"{t:.0%}" for t in XTICKS])
    ax.set_xlabel("Data percentage")
    ax.yaxis.grid(True, linestyle="-", alpha=0.7)
    ax.set_title(f"{roi} - {_n_voxels(scores_all[roi])} voxels")
    return ax


# %%
def main():
    # data_path = "/data/parietal/store4/data/cneuromod/things.betas"
    data_path = "/Users/emdupre/Desktop/UdeM-Projects/things-encode/"
    subj_name = "sub-03"  # "sub-01", "sub-02",

    rois = ["EBA", "PPA", "pSTS"]

    metric = "r2_score"  # "correlation_score" or "r2_score"
    metric_name = METRIC_NAMES[metric]

    # Load data
    results = {
        roi: load_roi_results(subj_name, roi, data_path, metric)
        for roi in rois
    }

    # --- scaling curves ---
    fig, axes = plt.subplots(
        1, len(rois), sharey=True, figsize=(12, 6), squeeze=False
    )
    for roi, ax in zip(rois, axes[0]):
        plot_scaling_curves(
            ax,
            roi,
            {roi: results[roi].test_scores},
            {roi: results[roi].highest_scores},
            {roi: results[roi].std_scores},
            {roi: results[roi].expl_var},
        )

    # consolidate legend across all sub-plots
    by_label = {}
    for ax in axes[0]:
        handles, labels = ax.get_legend_handles_labels()
        by_label.update(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="center right")

    fig.suptitle(
        f"{subj_name} : 90th percentile {metric_name} (± std across folds)",
        fontsize=14,
    )

    # --- regularization-parameter curves ---
    plot_regularization_curves(subj_name, rois, results)

    plt.show()


# %%
if __name__ == "__main__":
    main()
