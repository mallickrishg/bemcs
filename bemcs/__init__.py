import pkg_resources

from .bemcs import (
    discretized_line,
    constant_kernel,
    quadratic_kernel_farfield,
    quadratic_kernel_coincident,
    displacements_stresses_constant_no_rotation,
    displacements_stresses_quadratic_no_rotation,
    displacements_stresses_quadratic_slip_no_rotation_antiplane,
    displacements_stresses_linear_force_no_rotation_antiplane,
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
    get_matrices_slip_slip_gradient,
    get_matrices_slip_slip_gradient_antiplane,
    get_displacement_stress_kernel,
    get_displacement_stress_kernel_slip_antiplane,
    coeffs_to_disp_stress,
    plot_displacements_stresses_els,
    get_traction_kernels,
    get_traction_kernels_antiplane,
    get_strainenergy_from_stress,
    get_slipvector_on_fault,
    get_slipvector_on_fault_antiplane,
    label_nodes,
    construct_smoothoperator,
    inpolygon,
)

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:
    __version__ = "unknown"

__all__ = []
