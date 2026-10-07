# %%
from collections import defaultdict
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np

# from .utils import _get_solver_colors, _get_solver_names, gather_test_scores


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
        np.quantile(expl_var_all[roi][percent], 0.9) for percent in percents
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
    # data_path = "/data/parietal/store4/data/cneuromod/things.betas"
    data_path = "/Users/emdupre/Desktop/UdeM-Projects/things-encode/"
    subj_name = "sub-03"  # "sub-01", "sub-02",

    rois = ["EBA", "PPA", "pSTS"]
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
        expl_var_all[roi] = defaultdict()

        for percent in data_percent:
            test_scores, highest_scores, std_scores, expl_var = gather_test_scores(
                "sub-03",
                roi=roi,
                data_path=data_path,
                metric=metric_,
                data_percent=percent,
            )

            scores_all[roi][f"percent-{percent}"] = test_scores
            highest_scores_all[roi][f"percent-{percent}"] = highest_scores
            std_scores_all[roi][f"percent-{percent}"] = std_scores

            expl_var_all[roi][f"percent-{percent}"] = expl_var

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
        zip(list(chain.from_iterable(labels)), list(chain.from_iterable(handles)))
    )
    fig.legend(by_label.values(), by_label.keys(), loc="center right")

    # set overall figure title
    fig.suptitle(
        f"{subj_name} : " f"90th percentile {metric_name} (± std across folds)",
        fontsize=14,
    )
    plt.show()


# %%
if __name__ == "__main__":
    main()
