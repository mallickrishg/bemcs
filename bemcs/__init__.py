from importlib.metadata import PackageNotFoundError, version as _version

from .bemcs import (
    discretized_line,
    slip_functions,
    slip_functions_mean,
    slipgradient_functions,
    initialize_els,
    standardize_els_geometry,
    plot_els_geometry,
    get_strainenergy_from_stress,
    label_nodes,
    inpolygon,
    displacements_stresses_triangle_force_planestrain_farfield,
    displacements_stresses_triangle_force_planestrain_nearfield,
    displacements_stresses_triangle_force_planestrain,
    displacements_stresses_triangle_force_antiplane_farfield,
    displacements_stresses_triangle_force_antiplane_nearfield,
    displacements_stresses_triangle_force_antiplane,
    kelvin_point_source_disp,
    kelvin_point_source_stress,
    laplacian_pointforce_disp,
    laplacian_pointforce_sx,
    laplacian_pointforce_sy,
)

from . import bemAssembly
from .bemAssembly import (
    Mesh,
    KernelInfo,
    CoefficientLayout,
    get_kernels,
    get_slip_kernels,
    get_force_kernels,
    traction_kernels,
    kernel_matrix,
    solve,
    assemble,
    close_collocation_points,
    CollocationProximityWarning,
    interface_kappa,
    evaluate_fields,
    plot_fields,
    plot_mesh,
)

try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "bemAssembly",
    "Mesh",
    "KernelInfo",
    "CoefficientLayout",
    "get_kernels",
    "get_slip_kernels",
    "get_force_kernels",
    "traction_kernels",
    "kernel_matrix",
    "solve",
    "close_collocation_points",
    "CollocationProximityWarning",
    "assemble",
    "interface_kappa",
    "evaluate_fields",
    "plot_fields",
    "plot_mesh",
]
