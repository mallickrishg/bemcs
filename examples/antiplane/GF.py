import numpy as np
import bemcs

"""
get_kernels_linforce(x_obs, y_obs, els, connect_matrix, mu=1):
    Build per-global-force antiplane displacement and stress kernels using linear (2-point) element coefficients.

get_kernels_trapezoidalforce(x_obs, y_obs, els, connect_matrix, mu=1):
    Build per-global-force antiplane displacement and stress kernels using trapezoidal (3-point) element coefficients.

coeffs_from_GFcoeffs(els, connect_matrix, f):
    Scatter global-force coefficients into per-node element coefficient arrays.

construct_linearoperator_slip(els, BCtype, BCval, mu=1):
    Assemble the linear operator matrix and stacked boundary-condition vector for the antiplane slip problem.
"""


def get_kernels_linforce(x_obs, y_obs, els, connect_matrix, mu=1):
    """
    Compute stress and displacement kernels for an antiplane linear-force basis.
    This function constructs Green's-function-like kernels for a set of observation
    points by assembling element subsets defined in `connect_matrix`, calling the
    bemcs antiplane kernel routine for each subset, and projecting the returned
    tensor kernels onto a two-component linear force basis.
    Parameters
    ----------
    x_obs : array_like, shape (n_obs,)
        x coordinates of observation points where kernels are evaluated.
    y_obs : array_like, shape (n_obs,)
        y coordinates of observation points where kernels are evaluated.
    els : object
        Element container with array attributes x1, y1, x2, y2 describing the
        geometry of all available elements. Each attribute should be indexable by
        integer arrays taken from `connect_matrix`.
    connect_matrix : array_like, shape (n_GFs, n_els_per_GF)
        Integer indices selecting which elements from `els` form each Green's
        function (GF). Each row corresponds to one GF; entries are element indices
        into the arrays in `els`.
    mu : float, optional
        Shear modulus (defaults to 1). Passed to the underlying bemcs kernel
        calculator.
    Returns
    -------
    kernel_sx : ndarray, shape (n_obs, n_GFs)
        Projected kernel for the stress component sigma_x (rows = observation
        points, columns = Green's functions).
    kernel_sy : ndarray, shape (n_obs, n_GFs)
        Projected kernel for the stress component sigma_y (rows = observation
        points, columns = Green's functions).
    kernel_u : ndarray, shape (n_obs, n_GFs)
        Projected out-of-plane displacement kernel (rows = observation points,
        columns = Green's functions).
    Notes
    -----
    - For each GF (each row in `connect_matrix`) a temporary element set `els_mod`
      is created by selecting corresponding entries from `els`. The element
      geometry is standardized with bemcs.standardize_els_geometry(..., reorder=False).
    - The function calls bemcs.get_displacement_stress_kernel_force_antiplane to
      compute raw tensor kernels K_sx, K_sy, K_u for the selected elements, then
      projects these tensors onto the two-component linear force basis
      lincoefs = [[1, 0], [0, 1]] using numpy.tensordot.
    - The input `connect_matrix` must contain valid integer indices into `els`.
    - Returned arrays have dtype float and shape (n_obs, n_GFs).
    Raises
    ------
    IndexError
        If indices in `connect_matrix` are out of range for the arrays in `els`.
    ValueError
        If input array shapes are inconsistent (for example, x_obs and y_obs must
        have the same length).
    Example
    -------
    # Typical usage:
    # kernel_sx, kernel_sy, kernel_u = get_kernels_linforce(x_obs, y_obs, els, connect_matrix, mu=1.0)
    """

    n_obs = len(x_obs)
    n_GFs = len(connect_matrix[:, 0])

    kernel_u = np.zeros((n_obs, n_GFs))
    kernel_sx = np.zeros((n_obs, n_GFs))
    kernel_sy = np.zeros((n_obs, n_GFs))

    # provide coefficients for forces
    lincoefs = np.array([[1.0, 0.0], [0.0, 1.0]])

    for i in range(0, n_GFs):
        # define new els for GF calculation
        els_mod = bemcs.initialize_els()
        els_mod.x1 = els.x1[connect_matrix[i, :].astype(int)]
        els_mod.y1 = els.y1[connect_matrix[i, :].astype(int)]
        els_mod.x2 = els.x2[connect_matrix[i, :].astype(int)]
        els_mod.y2 = els.y2[connect_matrix[i, :].astype(int)]
        bemcs.standardize_els_geometry(els_mod, reorder=False)

        K_sx, K_sy, K_u = bemcs.get_displacement_stress_kernel_force_antiplane(
            x_obs, y_obs, els_mod, mu
        )

        # compute displacements and stress components
        kernel_u[:, i] = np.tensordot(K_u, lincoefs, axes=([2, 1], [0, 1]))
        kernel_sx[:, i] = np.tensordot(K_sx, lincoefs, axes=([2, 1], [0, 1]))
        kernel_sy[:, i] = np.tensordot(K_sy, lincoefs, axes=([2, 1], [0, 1]))

    return kernel_sx, kernel_sy, kernel_u


def get_kernels_trapezoidalforce(x_obs, y_obs, els, connect_matrix, mu=1):
    """
    Compute antiplane (mode-III) displacement and stress kernels for trapezoidal
    force basis functions formed by linking groups of three linear elements.

    This function builds singularity-free trapezoidal basis functions by
    combining three adjacent linear (linear shape function) mesh elements into a
    single trapezoidal force element. For each group defined in connect_matrix
    the corresponding 3 elements are assembled into a temporary element set,
    standardized, and the antiplane Green's-function kernels (displacement and
    stress due to unit forces) are evaluated and collapsed using trapezoidal
    coefficients to produce the final kernels.

    Parameters
    ----------
    x_obs : array_like, shape (n_obs,)
        x-coordinates of observation points where kernels are evaluated.
    y_obs : array_like, shape (n_obs,)
        y-coordinates of observation points where kernels are evaluated.
    els : object
        Element data structure providing element endpoint arrays:
        els.x1, els.y1, els.x2, els.y2 (indexable by integer indices).
    connect_matrix : ndarray, shape (n_groups, 3) or (n_groups, m)
        Integer index matrix linking mesh elements to trapezoidal basis
        functions. Each row selects the element indices (typically three)
        that are combined to form one trapezoidal basis function. Indices are
        used to slice into els.x1, els.y1, els.x2, els.y2.
    mu : float, optional
        Shear modulus used by the antiplane Green's functions. Default is 1.

    Returns
    -------
    kernel_sx : ndarray, shape (n_obs, n_groups)
        Kernel for the shear stress component in the x-direction (σ_xz)
        evaluated at observation points for each trapezoidal basis function.
    kernel_sy : ndarray, shape (n_obs, n_groups)
        Kernel for the shear stress component in the y-direction (σ_yz)
        evaluated at observation points for each trapezoidal basis function.
    kernel_u : ndarray, shape (n_obs, n_groups)
        Kernel for the out-of-plane displacement (u_z) evaluated at observation
        points for each trapezoidal basis function.

    Notes
    -----
    - This routine uses antiplane Green's functions with linear basis
      functions and constructs
      trapezoidal basis functions that are free of singularities by linking
      three mesh elements per basis function.
    - Internally a trapezoidal coefficient matrix
      trapcoefs = [[0., 1.], [1., 1.], [1., 0.]] is applied to the per-element
      kernels returned by bemcs.get_displacement_stress_kernel_force_antiplane
      to collapse the per-element, per-node kernels into one kernel per
      trapezoidal basis function.
    - connect_matrix rows are expected to contain valid integer indices for
      slicing into the els arrays. The function creates a temporary element
      object for each group, calls bemcs.standardize_els_geometry on it, and
      then evaluates the Green's-function kernels.
    - Returned arrays have one column per trapezoidal basis function (one
      row per observation point).

    Raises
    ------
    IndexError
        If indices in connect_matrix are out of bounds for the provided els.
    TypeError
        If input arrays are not indexable or numeric as required.

    Examples
    --------
    # Typical usage (conceptual):
    # kernel_sx, kernel_sy, kernel_u = get_kernels_trapezoidalforce(
    #     x_obs, y_obs, els, connect_matrix, mu=1.0
    # )
    """

    n_obs = len(x_obs)
    n_GFs = len(connect_matrix[:, 0])

    kernel_u = np.zeros((n_obs, n_GFs))
    kernel_sx = np.zeros((n_obs, n_GFs))
    kernel_sy = np.zeros((n_obs, n_GFs))

    # provide coefficients for forces
    trapcoefs = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])

    for i in range(0, n_GFs):
        # define new els for GF calculation
        els_mod = bemcs.initialize_els()
        els_mod.x1 = els.x1[connect_matrix[i, :].astype(int)]
        els_mod.y1 = els.y1[connect_matrix[i, :].astype(int)]
        els_mod.x2 = els.x2[connect_matrix[i, :].astype(int)]
        els_mod.y2 = els.y2[connect_matrix[i, :].astype(int)]
        bemcs.standardize_els_geometry(els_mod, reorder=False)

        K_sx, K_sy, K_u = bemcs.get_displacement_stress_kernel_force_antiplane(
            x_obs, y_obs, els_mod, mu
        )

        # compute displacements and stress components
        kernel_u[:, i] = np.tensordot(K_u, trapcoefs, axes=([2, 1], [0, 1]))
        kernel_sx[:, i] = np.tensordot(K_sx, trapcoefs, axes=([2, 1], [0, 1]))
        kernel_sy[:, i] = np.tensordot(K_sy, trapcoefs, axes=([2, 1], [0, 1]))

    return kernel_sx, kernel_sy, kernel_u


def coeffs_from_GFcoeffs(els, connect_matrix, f):
    n_GFs = len(connect_matrix[:, 0])
    coeffs = np.zeros((len(els.x1), 2))
    for i in range(0, n_GFs):
        coeffs[connect_matrix[i, 0].astype(int), :] += np.array([1, 0]) * f[i]
        coeffs[connect_matrix[i, 1].astype(int), :] += np.array([0, 1]) * f[i]

    return coeffs


def construct_linearoperator_slip(els, BCtype, BCval, mu=1):
    """
    Construct the global linear operator and right-hand side vector for the
    antiplane boundary-integral problem using quadratic slip basis functions.

    Parameters
    ----------
    els : object
        Mesh / element container providing geometry and connectivity required by
        the bemcs helper routines. Must contain at least x_centers, y_centers
        and per-element node arrays (e.g. x1) compatible with the bemcs helpers.
    BCtype : array-like, shape (n_els,)
        Boundary condition labels for each element/central node: 's' for slip
        (use slip-basis operator) or 't' for traction (use traction kernel).
    BCval : array-like, shape (n_els,) or (n_els, 1)
        Values of the prescribed boundary condition at central nodes. Only the
        central-node entries produce non-zero entries in the returned BC vector.
    mu : float, optional
        Shear modulus used when assembling kernels (default 1.0).

    Returns
    -------
    matrix_system_c : ndarray, shape (N_c, Nunknowns)
        Central-node linear operator block. N_c = n_els (one central equation per element),
        Nunknowns = 3 * n_els (antiplane stride = 3 DOFs per element). Each row is either
        a slip-basis row (if BCtype == 's') or a displacement/traction-kernel row (if BCtype == 't' or 'u').
    matrix_nodes : ndarray, shape (N_o + N_i + N_t, Nunknowns)
        Stacked smoothness operator for open nodes, overlapping nodes and triple
        junctions. Row ordering matches the stacked BCnodes ordering below.
    BC_c : ndarray, shape (N_c, 1)
        Central-node prescribed BC values (only these may be non-zero).
    BCnodes : ndarray, shape (Nequations, 1)
        Full stacked BC vector [BC_c; BC_o; BC_i; BC_t] where BC_o, BC_i, BC_t are
        homogeneous zero blocks for open, overlapping and triple-junction nodes.
    """

    stride = 3  # antiplane
    n_els = len(els.x1)

    index_open, index_overlap, index_triple = bemcs.label_nodes(els)
    N_c = n_els  # central node equations
    N_o = len(index_open)  # open node equations
    N_i = 2 * len(index_overlap)  # overlapping node equations
    N_t = 3 * len(index_triple)  # triple junction equations

    Nequations = N_c + N_o + N_i + N_t
    Nunknowns = stride * n_els
    # We will stack this with
    # equations for the element centers
    # equations at open nodes (RHS = 0)
    # equations at overlapping nodes (RHS = 0)
    # equations at triple junctions (RHS = 0)
    BC_c = BCval.reshape(-1, 1)  # these are the only non-zero entries
    BC_o = np.zeros((N_o, 1))
    BC_i = np.zeros((N_i, 1))
    BC_t = np.zeros((N_t, 1))

    # stack all the BCs into 1 big vector
    BCnodes = np.vstack((BC_o, BC_i, BC_t))
    matrix_system_o, matrix_system_i, matrix_system_t = (
        bemcs.construct_smoothoperator_antiplane(
            els, index_open, index_overlap, index_triple
        )
    )
    slip_mat, slip_gradient_mat = bemcs.get_matrices_slip_slip_gradient_antiplane(
        els, flag="node"
    )
    kernels = bemcs.get_displacement_stress_kernel_slip_antiplane(
        els.x_centers, els.y_centers, els, mu
    )
    t_kernels = bemcs.get_traction_kernels_antiplane(els, kernels)
    # displacement kernels
    u_kernels = kernels[2]
    # Linear Operators for the appropriate boundary conditions
    matrix_system_c = np.zeros((N_c, Nunknowns))

    # populate matrix_system for central nodes (check for bc_label)
    for i in range(n_els):
        if BCtype[i] == "s":
            matrix_system_c[i, :] = slip_mat[stride * i + 1, :]
        elif BCtype[i] == "t":
            matrix_system_c[i, :] = t_kernels[i, :]
        elif BCtype[i] == "u":
            matrix_system_c[i, :] = u_kernels[i, :]
        else:
            ValueError("boundary condition label not recognized")

    # stack the matrices and create only the smoothness linear operator
    matrix_nodes = np.vstack((matrix_system_o, matrix_system_i, matrix_system_t))

    return matrix_system_c, matrix_nodes, BC_c, BCnodes
