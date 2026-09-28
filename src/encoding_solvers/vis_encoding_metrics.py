# %%
import io
import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from nilearn.maskers import NiftiMasker
from nilearn.plotting import plot_stat_map


# %%
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(
                io.BytesIO(b), map_location="cpu", weights_only=False
            )
        else:
            return super().find_class(module, name)


def load_data(sub, roi, solver, metric="r2_score"):
    res_path = f"results/{sub}_space-T1w_roi-{roi}_cv-kfold-average_{metric}_{solver}_scores.pkl"
    out_file = res_path.format(sub=sub, roi=roi, solver=solver)

    if solver == "himalaya":
        with open(out_file, "rb") as f:
            scores = CPU_Unpickler(f).load()
    else:
        with open(out_file, "rb") as f:
            scores = pickle.load(f)
    return scores


def create_results_dict(sub, solvers, rois, metric="r2_score"):
    """
    Build best_scores[roi][solver] = list of folds (each fold = 1D array of
    per-target scores)
    """
    best_scores = {}

    for roi in rois:
        best_scores[roi] = {}

        scores = {
            solver: load_data(sub, roi, solver, metric) for solver in solvers
        }
        for solver in solvers:
            if solver in ["himalaya", "sklearn", "rrr"]:
                best_scores[roi][solver] = scores[solver]["best_scores"]
            elif solver in ["ompCV", "ompCV_ridge"]:
                best_scores[roi][solver] = scores[solver][
                    "per_target_test_scores"
                ]
            else:
                raise ValueError(f"Unknown solver: {solver}")

            arr = np.asarray(best_scores[roi][solver])
            print(
                f"[check] roi={roi} solver={solver} best_scores shape={arr.shape}"
            )

    return best_scores


def compute_quantile_and_std(best_scores, q=0.9, ddof=1):
    """
    From best_scores[roi][solver] (list of n_folds arrays, each array =
    per-target scores for that fold), compute:
      - quantile_scores[roi][solver]: mean across folds of the q-quantile
        of the per-target scores within each fold
      - std_scores[roi][solver]: std across folds of those per-fold
        q-quantiles (used as the error bar on the bar plot)

    ddof=1 gives the sample std (usually preferable with few folds);
    use ddof=0 to match np.std's default.
    """
    quantile_scores = {}
    std_scores = {}

    for roi in best_scores:
        quantile_scores[roi] = {}
        std_scores[roi] = {}
        for solver in best_scores[roi]:
            folds = [np.asarray(f) for f in best_scores[roi][solver]]
            fold_quantiles = np.array([np.quantile(f, q) for f in folds])

            quantile_scores[roi][solver] = fold_quantiles.mean()
            std_scores[roi][solver] = fold_quantiles.std(ddof=ddof)

    return quantile_scores, std_scores


def gather_results_for_subjects(subjects, solvers, rois, metric="r2_score"):
    """
    Loop over subjects and build best_scores / highest_scores / std_scores,
    each keyed by subject. This is the only place the per-subject loop
    lives now, so `main` can just call this once.
    """
    best_scores_all = {}
    highest_scores_all = {}
    std_scores_all = {}

    for sub in subjects:
        best_res = create_results_dict(sub, solvers, rois, metric=metric)
        # highest_res, std_res = compute_highest_and_std(best_res)
        highest_res, std_res = compute_quantile_and_std(
            best_res, q=0.9, ddof=1
        )
        best_scores_all[sub] = best_res
        highest_scores_all[sub] = highest_res
        std_scores_all[sub] = std_res

    return best_scores_all, highest_scores_all, std_scores_all


def _get_solver_colors(n_solvers):
    colors = ["pink", "mediumorchid", "cornflowerblue", "mediumaquamarine"]
    return [colors[j % len(colors)] for j in range(n_solvers)]


def _plot_one_subject(
    ax_left,
    ax_right,
    sub,
    best_scores,
    highest_scores,
    std_scores,
    metric,
    metric_name,
):
    """Fill a pair of (left, right) axes with one subject's two subplots."""
    rois = list(highest_scores.keys())
    solvers = list(highest_scores[next(iter(highest_scores))].keys())

    plot_quantile_scores_bar(
        ax_left, rois, solvers, highest_scores, std_scores, metric_name
    )
    plot_score_distribution(ax_right, rois, solvers, best_scores, metric)

    ax_left.set_title(f"{sub} — {ax_left.get_title()}")
    ax_right.set_title(f"{sub} — {ax_right.get_title()}")


def plot_quantile_scores_bar(
    ax, rois, solvers, highest_scores, std_scores, metric_name
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

    for i, roi in enumerate(rois):
        for j, solver in enumerate(solvers):
            offset = (j - (n_solvers - 1) / 2) * bar_width
            score = highest_scores[roi][solver]
            err = std_scores[roi][solver]
            ax.bar(
                x[i] + offset,
                score,
                width=bar_width,
                yerr=err,
                capsize=3,
                label=solver if i == 0 else "",
                color=colors[j],
            )

    ax.yaxis.grid(True, linestyle="-", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(rois)
    ax.set_xlabel("")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Highest {metric_name} (± std across folds)")
    ax.set_title(f"90th percentile {metric_name} (± std across folds)")
    ax.legend(title="Solver")
    ax.axhline(0, color="black", linewidth=0.8)


def plot_score_distribution(ax, rois, solvers, best_scores, metric):
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
                [np.asarray(f).ravel() for f in best_scores[roi][solver]]
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

    metric_name = metric.replace("_", " ").title()
    is_r2 = "r2" in metric.lower()
    ax.set_ylim(0, 1) if is_r2 else ax.set_ylim(-1, 1)

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
    metric_name = metric.replace("_", " ").title()
    n_rois = len(rois)

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
                metric,
                metric_name,
            )
        fig.suptitle(metric_name)
        plt.tight_layout()
        plt.show()


def load_roi_mask(sub, roi):
    roi_fname = (
        f"{sub}_task-floc_space-T1w*_roi-{roi}_*_desc-smooth_mask.nii.gz"
    )
    roi_mask = nib.load(
        next(Path(data_dir, "encoding-inputs", "rois").glob(roi_fname))
    )
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
        roi_mask = load_roi_mask(sub, roi)
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
    cbar_ax = fig.add_axes(
        [0.25, 0.02, 0.5, 0.015]
    )  # [left, bottom, width, height]
    fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")

    plt.show()
    return fig


# %%
data_dir = "/data/parietal/store4/data/cneuromod/things.betas"
rois = ["EBA", "pSTS", "PPA"]
subjects = ["sub-01", "sub-02", "sub-03"]
solvers = ["sklearn", "ompCV", "rrr", "ompCV_ridge"]  # "himalaya",
metric_ = "r2_score"  # "correlation_score" or "r2_score"

# %%
best_scores_all, highest_scores_all, std_scores_all = (
    gather_results_for_subjects(subjects, solvers, rois, metric=metric_)
)
plot_max_metric_across_solvers(
    subjects,
    best_scores_all,
    highest_scores_all,
    std_scores_all,
    metric=metric_,
    separate_figures=False,
)

for sub in subjects:
    plot_score_maps_grid(
        sub, rois, solvers, best_scores_all[sub], metric=metric_
    )
# %%
