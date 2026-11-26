# %%
from collections import Counter, defaultdict
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from utils import (
    Q,
    _get_solver_colors,
    _get_solver_names,
    _set_auto_yaxis,
    extract_solver_params,
    gather_test_scores,
)


def plot_ranks(
    default_dict, solver="rrr", scale="auto", min_size=60, max_size=600
):
    rois = [
        r for r in default_dict if default_dict[r]
    ]  # skip ROIs with no data
    fig, axes = plt.subplots(1, len(rois), sharey=True, figsize=(12, 6))
    axes = np.atleast_1d(axes)

    all_values = []
    for roi, ax in zip(rois, axes):
        # _plot_ranks_across_folds(ax, default_dict[roi], key=key)
        items = sorted(
            default_dict[roi].items(),
            key=lambda kv: float(kv[0].split("-")[-1]),
        )
        xs, ys, fracs = [], [], []
        for pct_key, models in items:
            pct = float(pct_key.split("-")[-1])
            values = [float(v) for v in models[solver]]
            for value, count in Counter(values).items():
                xs.append(pct)
                ys.append(value)
                fracs.append(count / len(values))

        sizes = min_size + (max_size - min_size) * np.array(fracs)
        ax.scatter(
            xs,
            ys,
            s=sizes,
            alpha=0.7,
            edgecolor="k",
            linewidth=0.5,
            color=_get_solver_colors(solver),
        )
        ax.set_xticks(sorted(set(xs)))
        ax.set_xlabel("Data percentage")
        ax.grid(alpha=0.3)
        ax.set_title(roi)
        for models in default_dict[roi].values():
            all_values.extend(float(v) for v in models[solver])

    _set_auto_yaxis(axes, all_values, scale=scale)
    axes[0].set_ylabel("Selected rank")

    for frac, label in [(0.2, "1/5"), (0.6, "3/5"), (1.0, "5/5")]:
        axes[-1].scatter(
            [],
            [],
            s=60 + 540 * frac,
            c="gray",
            alpha=0.7,
            edgecolor="k",
            linewidth=0.5,
            label=label,
        )
    axes[-1].legend(
        title="Folds in agreement",
        loc="upper right",
        labelspacing=1.5,
        borderpad=1,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig, axes


def plot_scaling_curves(
    ax,
    roi,
    scores_all,
    highest_scores_all,
    std_scores_all,
    expl_var_all,
):
    scaling_values_all = defaultdict()
    scaling_errs_all = defaultdict()

    x_ticks = np.linspace(0.1, 1.0, 10)

    # assume all rois have the same data percentages, solvers
    percents = scores_all[roi].keys()
    solvers = scores_all[roi][list(percents)[0]].keys()

    # first fold for full data for first solver
    n_voxels = len(scores_all[roi][list(percents)[-1]][list(solvers)[0]][0])

    scaling_values_all[roi] = defaultdict()
    scaling_errs_all[roi] = defaultdict()

    for j, solver in enumerate(solvers):
        scaling_values_all[roi][solver] = [
            highest_scores_all[roi][percent][solver] for percent in percents
        ]
        scaling_errs_all[roi][solver] = [
            std_scores_all[roi][percent][solver] for percent in percents
        ]

        ax.errorbar(
            x_ticks,
            scaling_values_all[roi][solver],
            yerr=scaling_errs_all[roi][solver],
            label=_get_solver_names(solver),
            marker="o",
            color=_get_solver_colors(solver),
        )

    upper_expl_var = [
        np.quantile(expl_var_all[roi][percent], float(f"0.{Q}"))
        for percent in percents
    ]
    ax.plot(
        x_ticks,
        upper_expl_var,
        label="Noise ceiling",
        color="grey",
        marker="o",
    )

    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels([f"{20*i}%" for i in range(1, 6)])
    ax.set_xlabel("Data percentage")

    ax.yaxis.grid(True, linestyle="-", alpha=0.7)
    ax.set_title(f"{roi} - {n_voxels} voxels")

    return ax


# %%
def main():
    data_path = (
        "/data/parietal/store3/work/aponcema/encoding-things/results_clip"
    )
    # data_path = "/Users/emdupre/Desktop/UdeM-Projects/things-encode/"
    sub_name = "sub-03"  # "sub-01", "sub-02",

    rois = ["EBA", "PPA", "pSTS"]  #
    n_rois = len(rois)

    metric_ = "r2_score"  # "correlation_score" or "r2_score"
    if metric_ == "r2_score":
        metric_name = "$R^2$ Score"
    elif metric_ == "correlation_score":
        metric_name = metric_.replace("_", " ").title()

    # scaling experiments
    scores_all = defaultdict()
    highest_scores_all = defaultdict()
    std_scores_all = defaultdict()
    reg_params_all = defaultdict()
    expl_var_all = defaultdict()

    data_percent = [
        "0.1",
        "0.2",
        "0.3",
        "0.4",
        "0.5",
        "0.6",
        "0.7",
        "0.8",
        "0.9",
        "1.0",
    ]

    fig, axes = plt.subplots(1, n_rois, sharey=True, figsize=(12, 6))
    for roi, ax in zip(rois, axes):
        scores_all[roi] = defaultdict()
        highest_scores_all[roi] = defaultdict()
        std_scores_all[roi] = defaultdict()
        reg_params_all[roi] = defaultdict()
        expl_var_all[roi] = defaultdict()

        for percent in data_percent:
            test_scores, highest_scores, std_scores, reg_params, expl_var = (
                gather_test_scores(
                    sub_name,
                    roi=roi,
                    data_path=data_path,
                    metric=metric_,
                    data_percent=percent,
                )
            )

            scores_all[roi][f"percent-{percent}"] = test_scores
            highest_scores_all[roi][f"percent-{percent}"] = highest_scores
            std_scores_all[roi][f"percent-{percent}"] = std_scores
            reg_params_all[roi][f"percent-{percent}"] = reg_params
            expl_var_all[roi][f"percent-{percent}"] = expl_var

            # print(f"percent-{percent}")
            # print(scores_all[roi][f"percent-{percent}"].keys())

        plot_scaling_curves(
            ax,
            roi,
            scores_all,
            highest_scores_all,
            std_scores_all,
            expl_var_all,
        )

    # consolidate legend across all sub-plots
    axes_legends = [ax.get_legend_handles_labels() for ax in axes]
    handles, labels = zip(*axes_legends)
    by_label = dict(
        zip(
            list(chain.from_iterable(labels)),
            list(chain.from_iterable(handles)),
        )
    )
    fig.legend(by_label.values(), by_label.keys(), loc="center right")

    # set overall figure title
    fig.suptitle(
        f"{sub_name} : "
        f"{Q}th percentile {metric_name} (± std across folds)",
        fontsize=14,
    )

    # ------------------
    # ranks across folds, using the regularization params already gathered
    rrr_params = extract_solver_params(reg_params_all, solver="rrr")
    fig_ranks, _ = plot_ranks(rrr_params, solver="rrr", scale="auto")
    fig_ranks.suptitle(
        f"{sub_name} : selected RRR rank across folds",
        fontsize=14,
    )

    plt.show()


# %%
if __name__ == "__main__":
    main()
