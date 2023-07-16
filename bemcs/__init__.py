import pkg_resources

from .bemcs import (
    plot_fields_no_elements,
    plot_fields,
    plot_element_geometry,
    plot_nine_fields,
    standardize_elements,
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
    get_individualdesignmatrix_3qn,
    get_designmatrix_xy_3qn,
    rotate_displacement_stress,
    get_quadratic_displacement_stress_kernel,
    compute_tractionkernels
)

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:
    __version__ = "unknown"

__all__ = []
