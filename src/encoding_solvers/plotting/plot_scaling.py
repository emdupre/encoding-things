# %%
import warnings
from dataclasses import dataclass, field

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
def _n_voxels(test_scores_roi):
    """Number of voxels, read from the largest data percentage available."""
    if not test_scores_roi:
        return None
    last_key = max(test_scores_roi, key=_percent_from_key)
    any_solver_scores = next(iter(test_scores_roi[last_key].values()))
    return len(any_solver_scores[0])


def _concat_params(per_solver_params):
    """Concatenate all folds and all targets into one flat array."""
    return np.concatenate([np.ravel(p) for p in per_solver_params])


def _plot_params_box(axes_col, roi, solvers, results, n_voxels):
    """Show the regularization-parameter distribution per solver.
    Values are pooled across all folds and all targets, one boxplot per data
    percentage."""
    # Sort by percentage so a missing percentage can't shift the x positions.
    keys = sorted(results.reg_params, key=_percent_from_key)

    for ax, solver in zip(axes_col, solvers):
        positions, distributions = [], []
        for key in keys:
            per_solver = results.reg_params[key]
            if solver not in per_solver:
                continue  # solver not run at this percentage
            distributions.append(_concat_params(per_solver[solver]))
            positions.append(_percent_from_key(key))

        if positions:
            color = _get_solver_colors(solver)
            ax.boxplot(
                distributions,
                positions=positions,
                widths=0.06,
                whis=(0, 100),  # show full range / (5,95)
                showfliers=False,
                manage_ticks=False,  # x ticks are set later
                patch_artist=True,
                boxprops=dict(facecolor=color, edgecolor=color, alpha=0.5),
                whiskerprops=dict(color=color),
                capprops=dict(color=color),
                medianprops=dict(color="black", linewidth=1.5),
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
        ax.yaxis.grid(True, linestyle="-", alpha=0.7)
        ax.set_xlim(0.05, 1.05)

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


def plot_regularization_params_distribution(
    sub, rois, results, solvers=("sklearn", "rrr", "ompCV")
):
    """Plot the regularization parameter distribution per data percentage."""

    fig, axes = plt.subplots(
        len(solvers),
        len(rois),
        sharex=True,
        sharey="row",
        figsize=(4 * len(rois), 2.5 * len(solvers)),
        squeeze=False,
        layout="constrained",
    )

    for j, roi in enumerate(rois):
        _plot_params_box(
            axes[:, j],
            roi,
            solvers,
            results[roi],
            _n_voxels(results[roi].test_scores),
        )

    # sharey="row": one call per row, after all columns are drawn
    for row_axes, solver in zip(axes, solvers):
        if solver in LOG_SOLVERS:
            continue
        ax = row_axes[0]
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        lo, hi = ax.get_ylim()
        if hi - lo < 3:  # e.g. a single selected rank: keep integer ticks
            mid = (lo + hi) / 2
            ax.set_ylim(mid - 1.5, mid + 1.5)

    fig.suptitle(f"{sub} : Regularization parameter distribution")
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
    plot_regularization_params_distribution(subj_name, rois, results)

    plt.show()


# %%
if __name__ == "__main__":
    main()
