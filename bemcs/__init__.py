from importlib.metadata import PackageNotFoundError, version as _version

from .bemcs import (
    discretized_line,
    constant_kernel,
    quadratic_kernel_farfield,
    quadratic_kernel_coincident,
    displacements_stresses_constant_no_rotation,
    displacements_stresses_quadratic_no_rotation,
    displacements_stresses_quadratic_slip_no_rotation_antiplane,
    displacements_stresses_linear_force_no_rotation_antiplane,
    displacements_stresses_linear_force_no_rotation_planestrain,
    f_slip_to_displacement_stress,
    get_quadratic_coefficients_for_linear_slip,
    phicoef,
    slip_functions,
    slip_functions_mean,
    slipgradient_functions,
    get_slip_slipgradient,
    rotate_displacement_stress,
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
    get_kernels,
    get_slip_kernels,
    get_force_kernels,
    traction_kernels,
    kernel_matrix,
    solve,
    assemble,
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
    "get_kernels",
    "get_slip_kernels",
    "get_force_kernels",
    "traction_kernels",
    "kernel_matrix",
    "solve",
    "assemble",
    "interface_kappa",
    "evaluate_fields",
    "plot_fields",
    "plot_mesh",
]
