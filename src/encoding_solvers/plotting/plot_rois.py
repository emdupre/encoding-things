# %%
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn.maskers import NiftiMasker
from nilearn.plotting import plot_stat_map

from .utils import gather_test_scores


def _get_solver_colors(n_solvers):
    colors = ["pink", "mediumorchid", "cornflowerblue", "mediumaquamarine", "grey"]
    return [colors[j % len(colors)] for j in range(n_solvers)]


def _plot_one_subject(
    ax_left,
    ax_right,
    sub,
    best_scores,
    highest_scores,
    std_scores,
    expl_var,
    rois,
    metric,
    metric_name,
):
    """Fill a pair of (left, right) axes with one subject's two subplots."""
    solvers = list(highest_scores.keys())

    plot_score_distribution(
        ax_left, rois, solvers, best_scores, expl_var, metric, metric_name
    )
    plot_quantile_scores_bar(
        ax_right, rois, solvers, highest_scores, std_scores, expl_var, metric_name
    )

    ax_left.set_title(f"{sub} — {ax_left.get_title()}")
    ax_right.set_title(f"{sub} — {ax_right.get_title()}")
    return


def plot_quantile_scores_bar(
    ax, rois, solvers, highest_scores, std_scores, expl_var, metric_name
):
    """
    Left subplot: grouped bar plot of the 90-quantile of metric per ROI/solver,
    with error bars = std of per-fold max scores across the 5 folds.
    """
    n_rois = len(rois)
    n_solvers = len(solvers)
    x = np.arange(n_rois)
    bar_width = 0.8 / n_solvers

    colors = _get_solver_colors(n_solvers)
    solver_plot_names = {
        "sklearn": "Ridge",
        "rrr": "Reduced Rank Ridge",
        "ompCV": "Orthogonal Matching Pursuit (OMP)",
        "ompCV_Ridge": "OMP-refined Ridge",
    }

    for i, roi in enumerate(rois):
        upper_expl_var = np.quantile(expl_var, q=0.9)
        ax.axhline(upper_expl_var, linestyle="-.", color="grey", label="Noise ceiling")

        for j, solver in enumerate(solvers):
            offset = (j - (n_solvers - 1) / 2) * bar_width
            score = highest_scores[solver]
            err = std_scores[solver]

            ax.bar(
                x[i] + offset,
                score,
                width=bar_width,
                yerr=err,
                capsize=3,
                label=solver_plot_names[solver] if i == 0 else "",
                color=colors[j],
            )

    ax.set_xticks(x)
    ax.set_xticklabels(rois)
    ax.set_xlabel("")

    ax.yaxis.grid(True, linestyle="-", alpha=0.7)
    ax.set_ylim(bottom=None, top=0.4)
    ax.set_ylabel(metric_name)

    ax.set_title(f"Highest {metric_name} (± std across folds)")
    ax.set_title(f"90th percentile {metric_name} (± std across folds)")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.8)

    return


def plot_score_distribution(
    ax, rois, solvers, best_scores, expl_var, metric, metric_name
):
    """
    Right subplot: distribution of all per-target, per-fold scores as
    violin plots, grouped by ROI/solver in the same layout as the bar plot.
    Y-axis is fixed to [-1, 1] for correlation metrics, [0, 1] for r2.
    """

    n_rois = len(rois)
    n_solvers = len(solvers)

    x = np.arange(n_rois)
    bar_width = 0.8 / n_solvers
    colors = _get_solver_colors(n_solvers)

    for i, roi in enumerate(rois):
        for j, solver in enumerate(solvers):
            offset = (j - (n_solvers - 1) / 2) * bar_width
            # flatten across folds and targets
            values = np.concatenate(
                [np.asarray(f).ravel() for f in best_scores[solver]]
            )

            parts = ax.violinplot(
                values,
                positions=[x[i] + offset],
                widths=bar_width * 0.9,
                showmeans=True,
                showextrema=True,
            )
            for body in parts["bodies"]:
                body.set_facecolor(colors[j])
                body.set_edgecolor("black")
                body.set_alpha(0.7)
            for key in ("cbars", "cmins", "cmaxes", "cmeans"):
                if key in parts:
                    parts[key].set_color("black")

        # plot noise ceiling
        offset = ((j + 1) - (n_solvers - 1) / 2) * bar_width
        parts = ax.violinplot(
            expl_var,
            positions=[x[i] + offset],
            widths=bar_width * 0.9,
            showmeans=True,
            showextrema=True,
        )
        for body in parts["bodies"]:
            body.set_facecolor("grey")
            body.set_edgecolor("black")
            body.set_alpha(0.7)
        for key in ("cbars", "cmins", "cmaxes", "cmeans"):
            if key in parts:
                parts[key].set_color("black")

    if metric == "r2_score":
        ax.set_ylim(0, 1)
    else:
        ax.set_ylim(-1, 1)

    ax.set_xticks(x)
    ax.set_xticklabels(rois)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(f"Distribution of {metric_name} (all folds/targets)")
    ax.axhline(0, color="black", linewidth=0.8)


def plot_max_metric_across_solvers(
    subjects,
    best_scores_all,
    highest_scores_all,
    std_scores_all,
    expl_var_all,
    rois,
    metric="r2_score",
    separate_figures=True,
):
    """
    Plot highest-metric (with std error bars) and score-distribution subplots
    for each subject.

    Parameters
    ----------
    separate_figures : bool, default True
        If True, produce one 2-column figure per subject (current/original
        behaviour). If False, produce a single figure with one row per
        subject and the same two columns.
    """
    n_rois = len(rois)
    if metric == "r2_score":
        metric_name = "$R^2$ Score"
    elif metric == "correlation_score":
        metric_name = metric.replace("_", " ").title()

    if separate_figures:
        for sub in subjects:
            fig, (ax_left, ax_right) = plt.subplots(
                1, 2, figsize=(2 * (2 * n_rois + 2), 5)
            )
            _plot_one_subject(
                ax_left,
                ax_right,
                sub,
                best_scores_all[sub],
                highest_scores_all[sub],
                std_scores_all[sub],
                expl_var_all[sub],
                rois,
                metric,
                metric_name,
            )
            fig.suptitle(f"{metric_name} — {sub}")
            plt.tight_layout()
            plt.show()
    else:
        n_subs = len(subjects)
        fig, axes = plt.subplots(
            n_subs,
            2,
            figsize=(3 * (2 * n_rois + 2), 4 * n_subs),
            squeeze=False,
        )
        for row, sub in enumerate(subjects):
            ax_left, ax_right = axes[row]
            _plot_one_subject(
                ax_left,
                ax_right,
                sub,
                best_scores_all[sub],
                highest_scores_all[sub],
                std_scores_all[sub],
                expl_var_all[sub],
                rois,
                metric,
                metric_name,
            )
        fig.suptitle(metric_name)
        plt.tight_layout()
        plt.show()


def load_roi_mask(sub, roi, data_dir):
    roi_fname = f"{sub}_task-floc_space-T1w*_roi-{roi}_*_desc-smooth_mask.nii.gz"
    roi_mask = nib.load(next(Path(data_dir, "encoding-inputs", "rois").glob(roi_fname)))
    return roi_mask


def plot_score_map(
    roi_mask,
    best_scores,
    axes=None,
    colorbar=True,
    cut_coords=None,
    display_mode="z",
    title=None,
    vmax=0.25,
    vmin=0,
    cmap="PuRd",
):
    """
    Same as before, but now accepts `axes` (to draw into an existing subplot),
    `colorbar` (so it can be switched off when sharing one colorbar across a
    figure), `cut_coords` (e.g. an int for N equally spaced z-cuts), and
    `title` (drawn on the slice display itself, not via ax.set_title, since
    nilearn subdivides `axes` internally). Calling it with no extra args
    reproduces the original standalone behaviour.
    """
    masker = NiftiMasker(mask_img=roi_mask).fit()
    display = plot_stat_map(
        masker.inverse_transform(np.mean(best_scores, axis=0)),
        display_mode=display_mode,
        cut_coords=cut_coords,
        colorbar=colorbar,
        symmetric_cbar=False,
        vmax=vmax,
        vmin=vmin,
        cmap=cmap,
        axes=axes,
        title=title,
    )
    return display


def plot_score_maps_grid(
    sub,
    rois,
    solvers,
    best_scores,
    data_path,
    metric,
    vmax=0.25,
    vmin=0,
    cmap="PuRd",
    n_cuts=2,
):
    """
    One figure per subject: rows = ROIs, columns = solvers. Each cell shows
    `n_cuts` z-axis slices of the mean (across folds) score map for that
    ROI/solver, sharing a single colorbar at the bottom of the figure.

    Text layout (top to bottom):
      - subject id: large, centered above the whole figure
      - ROI name: centered above each full row (spans all solver columns)
      - solver name: centered below the two z-cuts, only on the last row
    """
    n_rois = len(rois)
    n_solvers = len(solvers)

    fig, axes = plt.subplots(
        # n_rois, n_solvers, figsize=(4 * n_solvers, 2.5 * n_rois), squeeze=False
        n_rois,
        n_solvers,
        figsize=(3 * n_solvers, 2.5 * n_rois),
        squeeze=False,
    )

    # Fix the layout margins BEFORE plotting into the axes: plot_stat_map
    # subdivides each axes' bounding box into its slice panels at draw time,
    # so the axes need their final position first, or the slices and our
    # text labels (computed from axes.get_position() below) would drift
    # apart if we resized things afterwards (e.g. via tight_layout).
    # - `bottom` is raised to leave room for the solver labels + colorbar
    #   below the last row.
    # - `wspace` controls the horizontal gap between columns — lower it
    #   further (e.g. 0.05) for even tighter spacing.
    plt.subplots_adjust(top=0.85, bottom=0.16, hspace=0.5, wspace=0.1)

    for i, roi in enumerate(rois):
        roi_mask = load_roi_mask(sub, roi, data_path)
        for j, solver in enumerate(solvers):
            plot_score_map(
                roi_mask,
                best_scores[roi][solver],
                axes=axes[i, j],
                colorbar=False,
                cut_coords=n_cuts,
                display_mode="z",
                title=None,
                vmax=vmax,
                vmin=vmin,
                cmap=cmap,
            )

    # solver name, centered below the two z-cuts, only on the last row
    for j, solver in enumerate(solvers):
        bbox = axes[-1, j].get_position()
        fig.text(
            (bbox.x0 + bbox.x1) / 2,
            bbox.y0 - 0.02,
            solver,
            ha="center",
            va="top",
            fontsize=11,
        )

    # ROI name, centered above each full row (spans all solver columns)
    for i, roi in enumerate(rois):
        bbox_left = axes[i, 0].get_position()
        bbox_right = axes[i, -1].get_position()
        fig.text(
            (bbox_left.x0 + bbox_right.x1) / 2,
            bbox_left.y1 + 0.015,
            roi,
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    # subject id, large, at the very top of the figure
    metric_name = metric.replace("_", " ").title()
    fig.text(
        0.5,
        0.97,
        f"{sub} {metric_name} ",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
    )

    # single shared colorbar at the bottom, below the solver labels
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    )
    sm.set_array([])
    cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.015])  # [left, bottom, width, height]
    fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")

    plt.show()
    return fig


# %%
def main():
    # data_path = "/data/parietal/store4/data/cneuromod/things.betas"
    data_path = "/Users/emdupre/Desktop/UdeM-Projects/things-encode/"
    subjects = ["sub-03"]  # "sub-01", "sub-02",
    metric_ = "r2_score"  # "correlation_score" or "r2_score"

    scores_all = defaultdict()
    highest_scores_all = defaultdict()
    std_scores_all = defaultdict()
    expl_var_all = defaultdict()

    for sub in subjects:
        test_scores, highest_scores, std_scores, expl_var = gather_test_scores(
            sub, roi="EBA", data_path=data_path, metric=metric_
        )
        scores_all[sub] = test_scores
        highest_scores_all[sub] = highest_scores
        std_scores_all[sub] = std_scores
        expl_var_all[sub] = expl_var

    plot_max_metric_across_solvers(
        subjects,
        scores_all,
        highest_scores_all,
        std_scores_all,
        expl_var_all,
        rois=["EBA"],
        metric=metric_,
        separate_figures=False,
    )

    rois = ["EBA", "pSTS", "PPA"]
    solvers = ["sklearn", "ompCV", "rrr", "ompCV_ridge"]  # "himalaya",
    for sub in subjects:
        plot_score_maps_grid(
            sub, rois, solvers, scores_all[sub], data_path, metric=metric_
        )


# %%
if __name__ == "__main__":
    main()
