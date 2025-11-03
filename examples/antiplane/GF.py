import numpy as np
import bemcs
import pandas as pd

try:
    import numba

    print("Numba is installed. Version:", numba.__version__)
    numba_installed = True

except ImportError:
    print("Numba is not installed.")
    numba_installed = False

from numba import njit, prange

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
    trapcoefs = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    # testing other ordering of force (most likely wrong)
    # trapcoefs = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 1.0]])

    C = np.zeros((len(els.x_centers), 2, n_GFs))
    for k in range(n_GFs):
        for m in range(3):
            idx = connect_matrix[k, m]
            C[idx, :, k] = trapcoefs[m, :]

    # compute kernels with all mesh elements
    K_sx, K_sy, K_u = get_displacement_stress_kernel_force_antiplane(
        x_obs.flatten(),
        y_obs.flatten(),
        els.x_centers,
        els.y_centers,
        els.half_lengths,
        els.rot_mats,
        els.rot_mats_inv,
        mu=1.0,
    )
    # multiply kernels with C (forces) to get new trapezoidal kernels
    kernel_sx = np.tensordot(K_sx, C, axes=([1, 2], [1, 0]))  # result: (Nobs, Nnew)
    kernel_sy = np.tensordot(K_sy, C, axes=([1, 2], [1, 0]))
    kernel_u = np.tensordot(K_u, C, axes=([1, 2], [1, 0]))

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


def solveAntiplaneBEM(fileinput, connectivityfile=None, mu=1):
    """
    Solve a 2D antiplane BEM problem given a mesh input file.

    Parameters
    ----------
    fileinput : str
        Path to CSV mesh file containing columns [x1, z1, x2, z2, BC_type, value].
    connectivityfile : str
        Path to CSV file defining connectivity matrix.
    mu : float, optional
        Shear modulus (default=1).

    Returns
    -------
    els : object
        Full mesh element object.
    els_s : object
        Slip-element-only mesh object.
    quadcoefs : ndarray
        Quadratic coefficients (slip field).
    forcecoefs : ndarray
        Force coefficients (if any 'h' type elements exist, otherwise zeros).

    Notes
    -----
    - The function reads the mesh and connectivity files, constructs the necessary
        element objects, builds the BIE system matrix and BC vector, and solves
        for the slip and force coefficients.
    - The mesh file must contain columns: x1, z1, x2, z2, BC_type, value.
    - The connectivity file must define the element connectivity for force elements.
    - The function handles both slip ('s', 't') and heterogeneous ('h') boundary
        conditions.
    """

    # --- Read mesh file ---
    datain = pd.read_csv(fileinput)
    x1 = datain["x1"].values
    x2 = datain["x2"].values
    y1 = datain["z1"].values
    y2 = datain["z2"].values
    BCtype = datain["BC_type"].values
    BCval = datain["value"].values

    if connectivityfile is not None:
        connmatrix = pd.read_csv(connectivityfile, header=None).values
    else:
        print("No connectivity file provided; assuming homogenous medium.")

    # --- Construct mesh objects ---
    stride = 3
    els = bemcs.initialize_els()
    els.x1, els.y1, els.x2, els.y2 = x1, y1, x2, y2

    # Identify slip elements ('s', 't') vs heterogeneous ('h')
    bc_mask = BCtype != "h"
    els_s = bemcs.initialize_els()
    els_s.x1, els_s.y1 = els.x1[bc_mask], els.y1[bc_mask]
    els_s.x2, els_s.y2 = els.x2[bc_mask], els.y2[bc_mask]

    # Standardize geometry
    bemcs.standardize_els_geometry(els, reorder=False)
    bemcs.standardize_els_geometry(els_s, reorder=False)
    # plot mesh
    bemcs.plot_els_geometry(els)
    # ---
    # Define function evaluation points
    # need to shift observation points by an infinitesimal amount to sample discontinuity either in u or du/dn
    # IMPORTANT NOTE: for force elements, observation points are shifted in the element normal direction
    # while for slip elements, observation points are shifted towards the 'interior' of the domain

    dr = -1e-9
    if np.any(BCtype == "h"):
        xo = np.hstack(
            (
                els_s.x_centers,
                els.x_centers[connmatrix[:, 1].astype(int)]
                - dr * els.x_normals[connmatrix[:, 1].astype(int)],
            )
        ).flatten()
        yo = np.hstack(
            (
                els_s.y_centers,
                els.y_centers[connmatrix[:, 1].astype(int)]
                - dr * els.y_normals[connmatrix[:, 1].astype(int)],
            )
        ).flatten()

        # --- Construct kernels ---
        K_x, K_y, _ = get_kernels_trapezoidalforce(xo, yo, els, connmatrix)
        nxvec = np.hstack(
            (els_s.x_normals, els.x_normals[connmatrix[:, 1].astype(int)])
        )
        nyvec = np.hstack(
            (els_s.y_normals, els.y_normals[connmatrix[:, 1].astype(int)])
        )
        Kforce_n = K_x * nxvec[:, None] + K_y * nyvec[:, None]

        # Stress kernels for slip elements
        K_x, K_y, _ = bemcs.get_displacement_stress_kernel_slip_antiplane(
            xo, yo, els_s, mu
        )
        Kslip_n = K_x * nxvec[:, None] + K_y * nyvec[:, None]
    else:
        xo = els_s.x_centers
        yo = els_s.y_centers
        nxvec = els_s.x_normals
        nyvec = els_s.y_normals

    # Slip operator
    matrix_slip_c, matrix_slip_nodes, BC_slip_c, BC_slip_nodes = (
        construct_linearoperator_slip(els_s, BCtype[bc_mask], BCval[bc_mask])
    )

    # --- Assemble and solve system ---
    if np.any(BCtype == "h"):
        # Full system with heterogeneous ('h') BC elements
        N_equations = len(els_s.x1) + len(connmatrix[:, 0])
        N_unknowns = stride * len(els_s.x1) + len(connmatrix[:, 0])
        matrix_system = np.zeros((N_equations, N_unknowns))

        # Populate for slip BCs
        for i in range(len(els_s.x1)):
            if BCtype[i] == "s":
                matrix_system[i, 0 : stride * len(els_s.x1)] = matrix_slip_c[i, :]
            elif BCtype[i] == "t":
                matrix_system[i, :] = np.hstack((matrix_slip_c[i, :], Kforce_n[i, :]))
            else:
                raise ValueError("Unrecognized BC label for slip-elements")

        # Bottom part for heterogeneous ('h') layers
        n_bottom_rows = len(connmatrix[:, 0])
        start_row = len(els_s.x1)
        alpha = np.diag(BCval[connmatrix[:, 1].astype(int)])
        matrix_system[start_row:, : stride * len(els_s.x1)] = (
            alpha @ Kslip_n[len(els_s.x1) :, :]
        )
        matrix_system[start_row:, stride * len(els_s.x1) :] = (
            np.eye(n_bottom_rows) + alpha @ Kforce_n[len(els_s.x1) :, :]
        )

        # Append node continuity constraints
        matrix_system = np.vstack(
            (
                matrix_system,
                np.hstack(
                    (
                        matrix_slip_nodes,
                        np.zeros((matrix_slip_nodes.shape[0], len(connmatrix[:, 0]))),
                    )
                ),
            )
        )
        BC_vector = np.vstack(
            (BC_slip_c, np.zeros((len(connmatrix[:, 0]), 1)), BC_slip_nodes)
        )

        # Solve linear system
        solution_vector = np.linalg.solve(matrix_system, BC_vector)
        quadcoefs = solution_vector[: stride * len(els_s.x1)]
        forcecoefs = solution_vector[stride * len(els_s.x1) :]

    else:
        # Only slip elements → simpler system
        BC_vector = np.vstack((BC_slip_c, BC_slip_nodes))
        matrix_system = np.vstack((matrix_slip_c, matrix_slip_nodes))
        quadcoefs = np.linalg.solve(matrix_system, BC_vector)
        forcecoefs = np.zeros((0, 0))  # empty placeholder

    return els, els_s, quadcoefs, forcecoefs


@njit(fastmath=True, parallel=True)
def get_displacement_stress_kernel_slip_antiplane(
    x_obs,
    y_obs,
    x_centers,
    y_centers,
    half_lengths,
    rot_mats,
    rot_mats_inv,
    mu=1.0,
):
    """
    Compute antiplane-slip Green's-function kernels for a set of rectangular line elements.
    This function evaluates local antiplane displacement and shear-stress kernels for a
    collection of 2D line elements (finite-length cracks/strike-slip elements in the
    z-direction) at a set of observation points. The implementation expects inputs
    already prepared for a local-element coordinate system (rotation matrices and their
    inverses). The outputs are dense kernel arrays where each element contributes three
    columns (three local basis contributions u1,u2,u3 for displacement and corresponding
    stress contributions).
    Intended usage:
    - Call this function inside a numba-jitted context (e.g., with numba.njit(parallel=True))
        because the implementation uses prange for parallel loops and constructs arrays in a
        manner suited for JIT compilation.
    - Provide numpy arrays of appropriate shapes and types (float64 preferred).
    - Ensure observation points are not placed exactly on element singular lines (see
        Warnings) or regularize those cases.
    Parameters
    ----------
    x_obs : array_like, shape (n_obs,)
            x-coordinates of observation points in global coordinates.
    y_obs : array_like, shape (n_obs,)
            y-coordinates of observation points in global coordinates.
    x_centers : array_like, shape (n_els,)
            x-coordinate of the element center (global coordinates).
    y_centers : array_like, shape (n_els,)
            y-coordinate of the element center (global coordinates).
    half_lengths : array_like, shape (n_els,)
            Half-length w of each element (positive). The code assumes the element extends from
            -w to +w in the element local x coordinate.
    rot_mats : array_like, shape (n_els, 2, 2)
            Rotation matrices that map local coordinates -> global coordinates for each element.
            Each rot_mats[j] should be a 2x2 orthonormal rotation matrix.
    rot_mats_inv : array_like, shape (n_els, 2, 2)
            Inverse rotation matrices (global -> local) for each element. Typically the transpose
            of rot_mats if rot_mats are pure rotations.
    mu : float, optional (default 1.0)
            Shear modulus used to convert displacements to stresses. Units must be consistent with
            geometry and slip units.
    Returns
    -------
    kernel_sxz : ndarray, shape (n_obs, 3 * n_els)
            Shear stress kernel component sigma_xz at each observation for each element basis.
            Columns are grouped per element as [basis1, basis2, basis3] for element j occupying
            columns 3*j : 3*(j+1).
    kernel_syz : ndarray, shape (n_obs, 3 * n_els)
            Shear stress kernel component sigma_yz at each observation for each element basis.
    kernel_u : ndarray, shape (n_obs, 3 * n_els)
            Antiplane displacement kernel (u_z) at each observation for each element basis.
    Notes
    -----
    - Column ordering: For element j, columns 3*j, 3*j+1, 3*j+2 correspond to the three local
        basis contributions (u1,u2,u3) used in the analytic antiplane solution. The same
        ordering applies to kernel_sxz and kernel_syz.
    - Coordinate transforms: Observations are translated by element center (dx,dy) and then
        transformed to local coordinates using rot_mats_inv[j]. Stress vectors are rotated
        back to global coordinates using rot_mats[j].
    - This routine evaluates closed-form expressions for Green's functions in the element
        local frame and then rotates stress vectors to global frame.
    - The function is implemented to be friendly to numba JIT compilation. If you call it
        without numba, it will still run in pure Python/numpy but will be slower.
    Warnings
    --------
    - Singular / near-singular evaluation: expressions include terms like log((w +/- x)^2 + y^2),
        arctan((w +/- x)/y), and divisions by ((w +/- x)^2 + y^2) and y. If an observation
        point falls exactly on y == 0 or on the lines x = +/- w with y == 0 the expressions will
        be singular or ill-conditioned. Regularize by shifting observation points slightly
        off the singular line (add small epsilon) or use an analytic singular limit if needed.
    - Numerical stability: for extremely small denominators or very large/small w/x/y ratios,
        numerical cancellation can occur. Use double precision and consider adding checks or
        clamping denominators if necessary.
    - Units: Ensure consistent units across geometry and shear modulus.
    Example
    -------
    # Typical pattern (outside of this docstring):
    # Prepare arrays (x_obs, y_obs, element geometry, rotations, etc.)
    # Optionally compile with numba for performance:
    # get_kernels = njit(parallel=True)(get_kernels_slip_antiplane_numba)
    # kernel_sxz, kernel_syz, kernel_u = get_kernels(x_obs, y_obs,
    #                                               x_centers, y_centers, half_lengths,
    #                                               rot_mats, rot_mats_inv, mu=1.0)
    """

    n_obs = len(x_obs)
    n_els = len(x_centers)

    # Preallocate outputs
    kernel_sxz = np.zeros((n_obs, 3 * n_els))
    kernel_syz = np.zeros((n_obs, 3 * n_els))
    kernel_u = np.zeros((n_obs, 3 * n_els))

    pi = np.pi

    for i in prange(n_obs):
        for j in range(n_els):
            # Translate and rotate observation
            dx = x_obs[i] - x_centers[j]
            dy = y_obs[i] - y_centers[j]
            Rinv = rot_mats_inv[j]
            x = Rinv[0, 0] * dx + Rinv[0, 1] * dy
            y = Rinv[1, 0] * dx + Rinv[1, 1] * dy
            w = half_lengths[j]

            # local Green’s function computations
            log1 = np.log((w - x) ** 2 + y**2)
            log2 = np.log((w + x) ** 2 + y**2)
            atan1 = np.arctan((w - x) / y)
            atan2 = np.arctan((w + x) / y)

            w2 = w**2
            w3 = w**3
            u1 = (3 / (16 * w2 * pi)) * (
                6 * w * y
                + ((-2) * w * x + 3 * (x - y) * (x + y)) * (atan1 + atan2)
                + (w - 3 * x) * y * (-log1 + log2)
            )

            u2 = (1 / (8 * w2 * pi)) * (
                (-18) * w * y
                + (4 * w2 + 9 * y**2) * atan1
                + 9 * x**2 * np.arctan((-w + x) / y)
                + (4 * w2 - 9 * x**2 + 9 * y**2) * atan2
                + 9 * x * y * (-log1 + log2)
            )

            u3 = (3 / (16 * w2 * pi)) * (
                6 * w * y
                + (2 * w * x + 3 * (x - y) * (x + y)) * (atan1 + atan2)
                + (w + 3 * x) * y * (log1 - log2)
            )

            # --- stress kernels (sxz, syz) ---
            # Following your ux1, uy1... formulae but vectorized
            ux1 = (3 / (16 * w2 * pi)) * (
                (-2) * (w - 3 * x) * (atan1 + atan2)
                + y
                * (
                    w2 * ((-1) / ((w - x) ** 2 + y**2) + 5 / ((w + x) ** 2 + y**2))
                    + 3 * log1
                    - 3 * log2
                )
            )

            uy1 = (3 / (16 * w2 * pi)) * (
                w
                * (
                    12
                    + w * (-w + x) / ((w - x) ** 2 + y**2)
                    - 5 * w * (w + x) / ((w + x) ** 2 + y**2)
                )
                - 6 * y * (atan1 + atan2)
                - (w - 3 * x) * (log1 - log2)
            )

            ux2 = (1 / (8 * w2 * pi)) * (
                (-18) * x * (atan1 + atan2)
                + y
                * (
                    20
                    * w3
                    * x
                    / ((w**4 + 2 * w2 * ((-1) * x**2 + y**2) + (x**2 + y**2) ** 2))
                    - 9 * log1
                    + 9 * log2
                )
            )

            uy2 = (-1 / (8 * w2 * pi)) * (
                w
                * (
                    36
                    + 5 * w * (-w + x) / ((w - x) ** 2 + y**2)
                    - 5 * w * (w + x) / ((w + x) ** 2 + y**2)
                )
                - 18 * y * (atan1 + atan2)
                + 9 * x * (log1 - log2)
            )

            ux3 = (3 / (16 * w2 * pi)) * (
                2 * (w + 3 * x) * (atan1 + atan2)
                + y
                * (
                    w2 * ((-5) / ((w - x) ** 2 + y**2) + 1 / ((w + x) ** 2 + y**2))
                    + 3 * log1
                    - 3 * log2
                )
            )

            uy3 = (3 / (16 * w2 * pi)) * (
                w
                * (
                    12
                    + 5 * w * (-w + x) / ((w - x) ** 2 + y**2)
                    - w * (w + x) / ((w + x) ** 2 + y**2)
                )
                - 6 * y * (atan1 + atan2)
                + (w + 3 * x) * (log1 - log2)
            )

            # --- Local stresses ---
            sxz_local = mu * np.array([ux1, ux2, ux3])
            syz_local = mu * np.array([uy1, uy2, uy3])

            # --- Rotate to global coordinates ---
            R = rot_mats[j]
            sxz_global = R[0, 0] * sxz_local + R[0, 1] * syz_local
            syz_global = R[1, 0] * sxz_local + R[1, 1] * syz_local

            # --- Fill output arrays ---
            kernel_u[i, 3 * j : 3 * (j + 1)] = np.array([u1, u2, u3])
            kernel_sxz[i, 3 * j : 3 * (j + 1)] = sxz_global
            kernel_syz[i, 3 * j : 3 * (j + 1)] = syz_global

    return kernel_sxz, kernel_syz, kernel_u


@njit(parallel=True, cache=True, fastmath=True)
def get_displacement_stress_kernel_force_antiplane(
    x_obs,
    y_obs,
    x_centers,
    y_centers,
    half_lengths,
    rot_mats,
    rot_mats_inv,
    mu=1.0,
):
    """
    Compute displacement and stress kernels due to a line force source
    in antiplane geometry, using Numba acceleration.

    Returns 3D arrays:
      kernel_sxz, kernel_syz, kernel_u  [Nobs x 2 basis functions x Nsources]
    """

    n_obs = len(x_obs)
    n_els = len(x_centers)

    kernel_u = np.zeros((n_obs, 2, n_els))
    kernel_sxz = np.zeros((n_obs, 2, n_els))
    kernel_syz = np.zeros((n_obs, 2, n_els))

    pi = np.pi

    for i in prange(n_els):
        # --- Transform observation coordinates into local element frame ---
        x_trans = x_obs - x_centers[i]
        y_trans = y_obs - y_centers[i]

        # Rotate to local coordinates (element is horizontal)
        x = rot_mats_inv[i, 0, 0] * x_trans + rot_mats_inv[i, 0, 1] * y_trans
        y = rot_mats_inv[i, 1, 0] * x_trans + rot_mats_inv[i, 1, 1] * y_trans

        # Local vars
        w = half_lengths[i]

        # --- Compute displacement and stress kernels directly (inlined) ---

        # Displacement kernels (u1, u2)
        u_1 = (
            (1 / 16)
            * np.pi ** (-1)
            * w ** (-1)
            * (
                (-4) * w * (2 * w + x)
                + 4 * (w + x) * y * np.arctan((w - x) / y)
                + 4 * (w + x) * y * np.arctan((w + x) / y)
                + ((w + (-1) * x) * (3 * w + x) + y**2)
                * np.log((w + (-1) * x) ** 2 + y**2)
                + (w + x + (-1) * y) * (w + x + y) * np.log((w + x) ** 2 + y**2)
            )
        )

        u_2 = (
            (1 / 16)
            * np.pi ** (-1)
            * w ** (-1)
            * (
                4 * w * ((-2) * w + x)
                + 4 * (w + (-1) * x) * y * np.arctan((w - x) / y)
                + 4 * (w + (-1) * x) * y * np.arctan((w + x) / y)
                + (w + (-1) * x + (-1) * y)
                * (w + (-1) * x + y)
                * np.log((w + (-1) * x) ** 2 + y**2)
                + ((3 * w + (-1) * x) * (w + x) + y**2) * np.log((w + x) ** 2 + y**2)
            )
        )

        ux_1 = (
            (1 / 8)
            * np.pi ** (-1)
            * w ** (-1)
            * (
                (-4) * w
                + 2 * y * (np.arctan((w - x) / y) + np.arctan((w + x) / y))
                + (-1) * (w + x) * np.log((w + (-1) * x) ** 2 + y**2)
                + (w + x) * np.log((w + x) ** 2 + y**2)
            )
        )

        ux_2 = (
            (-1 / 8)
            * np.pi ** (-1)
            * w ** (-1)
            * (
                (-4) * w
                + 2 * y * (np.arctan((w - x) / y) + np.arctan((w + x) / y))
                + (w + (-1) * x)
                * (
                    np.log((w + (-1) * x) ** 2 + y**2)
                    + (-1) * np.log((w + x) ** 2 + y**2)
                )
            )
        )

        uy_1 = (
            (1 / 8)
            * np.pi ** (-1)
            * w ** (-1)
            * (
                2 * (w + x) * (np.arctan((w - x) / y) + np.arctan((w + x) / y))
                + y
                * (
                    np.log((w + (-1) * x) ** 2 + y**2)
                    + (-1) * np.log((w + x) ** 2 + y**2)
                )
            )
        )

        uy_2 = (
            (1 / 8)
            * np.pi ** (-1)
            * w ** (-1)
            * (
                2 * (w + (-1) * x) * (np.arctan((w - x) / y) + np.arctan((w + x) / y))
                + y
                * (
                    (-1) * np.log((w + (-1) * x) ** 2 + y**2)
                    + np.log((w + x) ** 2 + y**2)
                )
            )
        )

        # Store displacement [Nobs x 2]
        disp = np.empty((n_obs, 2))
        disp[:, 0] = u_1 / mu
        disp[:, 1] = u_2 / mu

        # Local stress [Nobs x 2 x 2]
        stress_x_local = np.empty((n_obs, 2))
        stress_y_local = np.empty((n_obs, 2))
        stress_x_local[:, 0] = ux_1
        stress_x_local[:, 1] = ux_2
        stress_y_local[:, 0] = uy_1
        stress_y_local[:, 1] = uy_2

        # --- Rotate stress from local -> global coordinates ---
        rot = rot_mats[i]
        stress_xz = rot[0, 0] * stress_x_local + rot[0, 1] * stress_y_local
        stress_yz = rot[1, 0] * stress_x_local + rot[1, 1] * stress_y_local

        # --- Store results ---
        kernel_u[:, :, i] = disp
        kernel_sxz[:, :, i] = stress_xz
        kernel_syz[:, :, i] = stress_yz

    return kernel_sxz, kernel_syz, kernel_u
