import pkg_resources

from .bemcs import (
    discretized_line,
    constant_kernel,
    quadratic_kernel_farfield,
    quadratic_kernel_coincident,
    displacements_stresses_constant_no_rotation,
    displacements_stresses_quadratic_no_rotation,
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
    get_displacement_stress_kernel,
    coeffs_to_disp_stress,
    plot_displacements_stresses_els,
    get_traction_kernels,
    get_displacement_stress_kernel_constant,
    get_strain_from_stress,
    get_slipvector_on_fault,
)

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:
    __version__ = "unknown"

__all__ = []
