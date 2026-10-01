from .cross_validation import define_groups, leave_one_THINGSplus_out
from .plotting.plot_wholebrain import (
    plot_alphas_diagnostic,
    plot_flatmap,
    plot_voxel_hist,
)
from .solvers import (
    ompCV_sklearn,
    orthogonal_mp_sklearn,
    ridgeCV_himalaya,
    ridgeCV_rrr,
    ridgeCV_sklearn,
)
