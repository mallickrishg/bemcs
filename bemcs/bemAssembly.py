from dataclasses import dataclass, field

import numpy as np
import bemcs
from . import bemcs as _core
import pandas as pd
import matplotlib.pyplot as plt

try:
    import numba

    print("Numba is installed. Version:", numba.__version__)
    numba_installed = True

except ImportError:
    print("Numba is not installed.")
    numba_installed = False

from numba import njit, prange


def get_kernels_trapezoidalforce_antiplane(x_obs, y_obs, els, connect_matrix, mu=1):
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


def construct_linearoperator_slip_antiplane(els, BCtype, BCval, mu=1):
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
        Central-node linear operator block. N_c = n_els (one central equation per element;
        callers drop the rows of tip elements, see `tip_elements`, since open nodes are
        clamped by the rows in matrix_nodes),
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
    geometry = "antiplane"
    mesh = Mesh(els, "slip")

    # continuity rows at open, overlapping and triple-junction nodes (RHS = 0)
    matrix_nodes, _ = mesh.continuity_operator(geometry=geometry)
    BC_c = np.asarray(BCval, dtype=float).reshape(-1, 1)  # the only non-zero entries
    BCnodes = np.zeros((len(matrix_nodes), 1))

    slip_mat, _ = slip_node_matrices(els, geometry=geometry)
    U, S, _ = get_slip_kernels(
        els.x_centers, els.y_centers, els, geometry=geometry, mu=mu
    )
    t_kernels = traction_kernels(S, els.x_normals, els.y_normals)[:, :, 0]
    u_kernels = U[:, :, 0]

    # Linear operators for the boundary conditions at the element centres
    matrix_system_c = np.zeros((n_els, stride * n_els))
    for i in range(n_els):
        if BCtype[i] == "s":
            matrix_system_c[i, :] = slip_mat[stride * i + 1, :]
        elif BCtype[i] == "t":
            matrix_system_c[i, :] = t_kernels[i, :]
        elif BCtype[i] == "u":
            matrix_system_c[i, :] = u_kernels[i, :]
        else:
            raise ValueError(f"boundary condition label {BCtype[i]!r} not recognized")

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
        conn = connmatrix.astype(int)
        _, S_f, _ = get_force_kernels(
            xo, yo, els, geometry="antiplane", mu=mu, basis="trapezoid", connectivity=conn
        )
        K_x, K_y = S_f[:, :, 0], S_f[:, :, 1]
        nxvec = np.hstack(
            (els_s.x_normals, els.x_normals[connmatrix[:, 1].astype(int)])
        )
        nyvec = np.hstack(
            (els_s.y_normals, els.y_normals[connmatrix[:, 1].astype(int)])
        )
        Kforce_n = K_x * nxvec[:, None] + K_y * nyvec[:, None]

        # Stress kernels for slip elements
        _, S_s, _ = get_slip_kernels(xo, yo, els_s, geometry="antiplane", mu=mu)
        K_x, K_y = S_s[:, :, 0], S_s[:, :, 1]
        Kslip_n = K_x * nxvec[:, None] + K_y * nyvec[:, None]
    else:
        xo = els_s.x_centers
        yo = els_s.y_centers
        nxvec = els_s.x_normals
        nyvec = els_s.y_normals

    # Slip operator
    matrix_slip_c, matrix_slip_nodes, BC_slip_c, BC_slip_nodes = (
        construct_linearoperator_slip_antiplane(
            els_s, BCtype[bc_mask], BCval[bc_mask], mu=mu
        )
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

        # tip slip elements (open end node) are clamped tapers: drop their BC rows
        tips = tip_elements(els_s)
        matrix_system = np.delete(matrix_system, tips, axis=0)
        BC_vector = np.delete(BC_vector, tips, axis=0)

        # Solve linear system
        solution_vector = np.linalg.solve(matrix_system, BC_vector)
        quadcoefs = solution_vector[: stride * len(els_s.x1)]
        forcecoefs = solution_vector[stride * len(els_s.x1) :]

    else:
        # Only slip elements → simpler system
        # tip elements (open end node) are clamped tapers: drop their BC rows
        tips = tip_elements(els_s)
        BC_vector = np.vstack((np.delete(BC_slip_c, tips, axis=0), BC_slip_nodes))
        matrix_system = np.vstack((np.delete(matrix_slip_c, tips, axis=0), matrix_slip_nodes))
        quadcoefs = np.linalg.solve(matrix_system, BC_vector)
        forcecoefs = np.zeros((0, 0))  # empty placeholder

    return els, els_s, quadcoefs, forcecoefs


@njit(fastmath=True, parallel=True, cache=True)
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


@njit(parallel=True, fastmath=True, cache=True)
def get_displacement_stress_kernel_force_planestrain(
    x_obs,
    y_obs,
    x_centers,
    y_centers,
    half_lengths,
    rot_mats,
    rot_mats_inv,
    mu=1.0,
    nu=0.25,
):
    """
    Compute plane-strain displacement and stress kernels due to force sources
    for multiple horizontal fault elements in one call (Numba-accelerated).

    Parameters
    ----------
    x_obs, y_obs : (Nobs,)
        Observation coordinates.
    x_centers, y_centers : (Nels,)
        Element center coordinates.
    half_lengths : (Nels,)
        Half-lengths of elements.
    rot_mats, rot_mats_inv : (Nels,2,2)
        Rotation matrices for each element.
    mu, nu : float
        Elastic parameters.

    Returns
    -------
    kernel_ux, kernel_uy : (Nobs, 2, 2, Nels)
        Displacement kernels.
    kernel_sxx, kernel_syy, kernel_sxy : (Nobs, 2, 2, Nels)
        Stress kernels.
    """

    # ---- Define local inline functions ----
    # @njit(inline="always")
    def ux_1(fx, fy, w, xo, yo, mu=1, nu=0.25):
        term1 = (
            (1 / 8)
            * fx
            * w ** (-1)
            * (w - xo)
            * (3 * w + xo)
            * mu ** (-1)
            * nu
            * (np.pi - np.pi * nu) ** (-1)
            * np.log((w - xo) ** 2 + yo**2)
        )
        term2 = (
            (1 / 32)
            * fx
            * np.pi ** (-1)
            * w ** (-1)
            * mu ** (-1)
            * (nu - 1) ** (-1)
            * (
                4 * w * (8 * w * (nu - 1) + xo * (-3 + 4 * nu))
                - 16 * (w + xo) * yo * (nu - 1) * np.arctan((w - xo) / yo)
                - 16 * (w + xo) * yo * (nu - 1) * np.arctan((w + xo) / yo)
                + (3 * (w - xo) * (3 * w + xo) + yo**2 * (5 - 4 * nu))
                * np.log((w - xo) ** 2 + yo**2)
                + (
                    3 * (w + xo) ** 2
                    - 5 * yo**2
                    - 4 * (w + xo - yo) * (w + xo + yo) * nu
                )
                * np.log((w + xo) ** 2 + yo**2)
            )
        )
        term3 = (
            (1 / 16)
            * fy
            * np.pi ** (-1)
            * w ** (-1)
            * yo
            * mu ** (-1)
            * (nu - 1) ** (-1)
            * (
                4 * w
                - 2 * yo * (np.arctan((w - xo) / yo) + np.arctan((w + xo) / yo))
                + (w + xo)
                * (np.log((w - xo) ** 2 + yo**2) - np.log((w + xo) ** 2 + yo**2))
            )
        )
        return (term1 + term2 + term3) / 2

    # @njit(inline="always")
    def ux_2(fx, fy, w, xo, yo, mu=1, nu=0.25):
        term1 = (
            fx
            * (w - xo) ** 2
            * (3 - 4 * nu)
            * np.log((w - xo) ** 2 + yo**2)
            / (64 * np.pi * w * mu * (-1 + nu))
        )

        term2 = (
            (w - xo)
            * (
                -8
                * fx
                * yo
                * (-1 + nu)
                * (np.arctan((w - xo) / yo) + np.arctan((w + xo) / yo))
                + fy * yo * np.log((w - xo) ** 2 + yo**2)
            )
            / (32 * np.pi * w * mu * (-1 + nu))
        )

        term3 = (
            1
            / (64 * np.pi * w * mu * (-1 + nu))
            * (
                4 * w * (-2 * fy * yo + fx * xo * (3 - 4 * nu) + 8 * fx * w * (-1 + nu))
                + yo**2
                * (
                    4 * fy * (np.arctan((w - xo) / yo) + np.arctan((w + xo) / yo))
                    + fx * (-5 + 4 * nu) * np.log((w - xo) ** 2 + yo**2)
                )
                + (
                    2 * fy * (-w + xo) * yo
                    + fx
                    * (
                        3 * (3 * w - xo) * (w + xo)
                        + 5 * yo**2
                        - 4 * ((3 * w - xo) * (w + xo) + yo**2) * nu
                    )
                )
                * np.log((w + xo) ** 2 + yo**2)
            )
        )
        return term1 + term2 + term3

    # @njit(inline="always")
    def uy_1(fx, fy, w, xo, yo, mu=1, nu=0.25):
        term1 = (
            (1 / 32)
            * fx
            * np.pi ** (-1)
            * w ** (-1)
            * yo
            * mu ** (-1)
            * (nu - 1) ** (-1)
            * (
                4 * w
                - 2 * yo * (np.arctan((w - xo) / yo) + np.arctan((w + xo) / yo))
                + (w + xo)
                * (np.log((w - xo) ** 2 + yo**2) - np.log((w + xo) ** 2 + yo**2))
            )
        )
        term2 = (
            (1 / 64)
            * fy
            * np.pi ** (-1)
            * w ** (-1)
            * mu ** (-1)
            * (nu - 1) ** (-1)
            * (
                4 * w * (2 * w + xo) * (-3 + 4 * nu)
                + 8 * (w + xo) * yo * (nu * 2 - 1) * np.arctan((-w + xo) / yo)
                - 8 * (w + xo) * yo * (nu * 2 - 1) * np.arctan((w + xo) / yo)
                + (
                    w**2 * (9 - 12 * nu)
                    + yo**2 * (1 - 4 * nu)
                    + 2 * w * xo * (-3 + 4 * nu)
                    + xo**2 * (-3 + 4 * nu)
                )
                * np.log((w - xo) ** 2 + yo**2)
                - (
                    (
                        -3 * (w + xo) ** 2
                        + yo**2
                        + 4 * (w + xo - yo) * (w + xo + yo) * nu
                    )
                )
                * np.log((w + xo) ** 2 + yo**2)
            )
        )
        return term1 + term2

    # @njit(inline="always")
    def uy_2(fx, fy, w, xo, yo, mu=1, nu=0.25):
        term_fx = fx * (
            -(
                yo
                * (
                    4 * w
                    - 2 * yo * (np.arctan((w - xo) / yo) + np.arctan((w + xo) / yo))
                    + (-w + xo) * np.log((w - xo) ** 2 + yo**2)
                )
                / (32 * np.pi * w * mu * (-1 + nu))
            )
            + ((w - xo) * yo * np.log(w**2 + 2 * w * xo + xo**2 + yo**2))
            / (32 * np.pi * w * mu * (1 - nu))  # simplified from denominator structure
        )

        term_fy = fy * (
            -(
                (w - xo)
                * yo
                * (-1 + 2 * nu)
                * (np.arctan((w - xo) / yo) + np.arctan((w + xo) / yo))
                / (8 * np.pi * w * mu * (-1 + nu))
            )
            + ((w - xo) ** 2 * (3 - 4 * nu) * np.log((w - xo) ** 2 + yo**2))
            / (64 * np.pi * w * mu * (-1 + nu))
            + (
                4 * w * (2 * w - xo) * (-3 + 4 * nu)
                + yo**2 * (-1 + 4 * nu) * np.log((w - xo) ** 2 + yo**2)
                + (
                    3 * (3 * w - xo) * (w + xo)
                    + yo**2
                    - 4 * ((3 * w - xo) * (w + xo) + yo**2) * nu
                )
                * np.log((w + xo) ** 2 + yo**2)
            )
            / (64 * np.pi * w * mu * (-1 + nu))
        )

        return term_fx + term_fy

    # Stress kernels
    # @njit(inline="always")
    def sxy_1(fx, fy, w, xo, yo, mu=1, nu=0.25):
        term1 = (
            (-1 / 8)
            * fx
            * np.pi ** (-1)
            * w ** (-1)
            * ((-1) + nu) ** (-1)
            * (
                2 * w * (w - xo) * yo * ((w - xo) ** 2 + yo**2) ** (-1)
                + 2 * (w + xo) * ((-1) + nu) * np.arctan((w - xo) / yo)
                + 2 * (w + xo) * ((-1) + nu) * np.arctan((w + xo) / yo)
                + (1 / 2)
                * yo
                * ((-3) + 2 * nu)
                * (np.log((w - xo) ** 2 + yo**2) - np.log((w + xo) ** 2 + yo**2))
            )
        )

        term2 = (
            (1 / 8)
            * fy
            * np.pi ** (-1)
            * w ** (-1)
            * ((-1) + nu) ** (-1)
            * (
                (-2) * w * (w - xo) ** 2 * ((w - xo) ** 2 + yo**2) ** (-1)
                + 4 * w * nu
                + 2 * yo * nu * np.arctan((-w + xo) / yo)
                + (-2) * yo * nu * np.arctan((w + xo) / yo)
                + (1 / 2)
                * (w + xo)
                * ((-1) + 2 * nu)
                * (np.log((w - xo) ** 2 + yo**2) - np.log((w + xo) ** 2 + yo**2))
            )
        )

        return term1 + term2

    # @njit(inline="always")
    def sxy_2(fx, fy, w, xo, yo, mu=1, nu=0.25):
        term1 = (
            (-1 / 16)
            * fx
            * np.pi ** (-1)
            * w ** (-1)
            * ((-1 + nu) ** (-1))
            * (
                4 * w * (w + xo) * yo * ((w + xo) ** 2 + yo**2) ** (-1)
                + 4 * (w - xo) * (-1 + nu) * np.arctan((w - xo) / yo)
                + 4 * (w - xo) * (-1 + nu) * np.arctan((w + xo) / yo)
                + (-1)
                * yo
                * ((-3) + 2 * nu)
                * (np.log((w - xo) ** 2 + yo**2) - np.log((w + xo) ** 2 + yo**2))
            )
        )

        term2 = (
            (-1 / 16)
            * fy
            * np.pi ** (-1)
            * w ** (-1)
            * ((-1 + nu) ** (-1))
            * (
                (-4) * w * (w + xo) ** 2 * ((w + xo) ** 2 + yo**2) ** (-1)
                + 8 * w * nu
                + (-4) * yo * nu * np.arctan((w - xo) / yo)
                + (-4) * yo * nu * np.arctan((w + xo) / yo)
                + (-1)
                * (w - xo)
                * ((-1) + 2 * nu)
                * (np.log((w - xo) ** 2 + yo**2) - np.log((w + xo) ** 2 + yo**2))
            )
        )

        return term1 + term2

    # @njit(inline="always")
    def sxx_1(fx, fy, w, xo, yo, mu=1, nu=0.25):
        term1_fx = (
            (1 / 8)
            * fx
            * np.pi ** (-1)
            * w ** (-1)
            * (-1 + nu) ** (-1)
            * (
                ((w - xo) ** 2 + yo**2) ** (-1)
                * ((-6) * w * (w - xo) ** 2 - 8 * w * yo**2)
                + 4 * w * nu
                - 2 * yo * (-2 + nu) * np.arctan((w - xo) / yo)
                - 2 * yo * (-2 + nu) * np.arctan((w + xo) / yo)
                + (1 / 2)
                * (w + xo)
                * (-3 + 2 * nu)
                * (np.log((w - xo) ** 2 + yo**2) - np.log((w + xo) ** 2 + yo**2))
            )
        )

        term2_fy = (
            (1 / 8)
            * fy
            * np.pi ** (-1)
            * w ** (-1)
            * (-1 + nu) ** (-1)
            * (
                2 * w * ((-w) + xo) * yo * ((w - xo) ** 2 + yo**2) ** (-1)
                + 2 * (w + xo) * nu * np.arctan((w - xo) / yo)
                + 2 * (w + xo) * nu * np.arctan((w + xo) / yo)
                + (1 / 2)
                * yo
                * (1 + 2 * nu)
                * (np.log((w - xo) ** 2 + yo**2) - np.log((w + xo) ** 2 + yo**2))
            )
        )

        return term1_fx + term2_fy

    # @njit(inline="always")
    def sxx_2(fx, fy, w, xo, yo, mu=1, nu=0.25):
        term1_fx = (
            (1 / 16)
            * fx
            * np.pi ** (-1)
            * w ** (-1)
            * (-1 + nu) ** (-1)
            * (
                4
                * w
                * ((w + xo) ** 2 + yo**2) ** (-1)
                * (3 * (w + xo) ** 2 + 4 * yo**2 - 2 * ((w + xo) ** 2 + yo**2) * nu)
                + 4 * yo * (-2 + nu) * np.arctan((w - xo) / yo)
                + 4 * yo * (-2 + nu) * np.arctan((w + xo) / yo)
                + (w - xo)
                * (-3 + 2 * nu)
                * (np.log((w - xo) ** 2 + yo**2) - np.log((w + xo) ** 2 + yo**2))
            )
        )

        term2_fy = (
            (1 / 16)
            * fy
            * np.pi ** (-1)
            * w ** (-1)
            * (-1 + nu) ** (-1)
            * (
                (-4) * w * (w + xo) * yo * ((w + xo) ** 2 + yo**2) ** (-1)
                + 4 * (w - xo) * nu * np.arctan((w - xo) / yo)
                + 4 * (w - xo) * nu * np.arctan((w + xo) / yo)
                + (-1)
                * yo
                * (1 + 2 * nu)
                * (np.log((w - xo) ** 2 + yo**2) - np.log((w + xo) ** 2 + yo**2))
            )
        )

        return term1_fx + term2_fy

    # @njit(inline="always")
    def syy_1(fx, fy, w, xo, yo, mu=1, nu=0.25):
        term1_fx = (
            (-1 / 16)
            * fx
            * np.pi ** (-1)
            * w ** (-1)
            * (-1 + nu) ** (-1)
            * (
                ((w - xo) ** 2 + yo**2) ** (-1)
                * ((-4) * w * (w - xo) ** 2 - 8 * w * yo**2)
                + 8 * w * nu
                - 4 * yo * (-1 + nu) * np.arctan((w - xo) / yo)
                - 4 * yo * (-1 + nu) * np.arctan((w + xo) / yo)
                + (w + xo)
                * (-1 + 2 * nu)
                * (np.log((w - xo) ** 2 + yo**2) - np.log((w + xo) ** 2 + yo**2))
            )
        )

        term2_fy = (
            (-1 / 8)
            * fy
            * np.pi ** (-1)
            * w ** (-1)
            * (-1 + nu) ** (-1)
            * (
                2 * w * ((-w) + xo) * yo * ((w - xo) ** 2 + yo**2) ** (-1)
                + 2 * (w + xo) * (-1 + nu) * np.arctan((w - xo) / yo)
                + 2 * (w + xo) * (-1 + nu) * np.arctan((w + xo) / yo)
                + (1 / 2)
                * yo
                * (-1 + 2 * nu)
                * (np.log((w - xo) ** 2 + yo**2) - np.log((w + xo) ** 2 + yo**2))
            )
        )

        return term1_fx + term2_fy

    # @njit(inline="always")
    def syy_2(fx, fy, w, xo, yo, mu=1, nu=0.25):
        term1_fx = (
            (-1 / 8)
            * fx
            * np.pi ** (-1)
            * w ** (-1)
            * (-1 + nu) ** (-1)
            * (
                2
                * w
                * ((w + xo) ** 2 + yo**2) ** (-1)
                * ((w + xo) ** 2 + 2 * yo**2 - 2 * ((w + xo) ** 2 + yo**2) * nu)
                + 2 * yo * (-1 + nu) * np.arctan((w - xo) / yo)
                + 2 * yo * (-1 + nu) * np.arctan((w + xo) / yo)
                + (1 / 2)
                * (w - xo)
                * (-1 + 2 * nu)
                * (np.log((w - xo) ** 2 + yo**2) - np.log((w + xo) ** 2 + yo**2))
            )
        )

        term2_fy = (
            (-1 / 8)
            * fy
            * np.pi ** (-1)
            * w ** (-1)
            * (-1 + nu) ** (-1)
            * (
                (-2) * w * (w + xo) * yo * ((w + xo) ** 2 + yo**2) ** (-1)
                + 2 * (w - xo) * (-1 + nu) * np.arctan((w - xo) / yo)
                + 2 * (w - xo) * (-1 + nu) * np.arctan((w + xo) / yo)
                + (-1 / 2)
                * yo
                * (-1 + 2 * nu)
                * (np.log((w - xo) ** 2 + yo**2) - np.log((w + xo) ** 2 + yo**2))
            )
        )

        return term1_fx + term2_fy

    Nobs = len(x_obs)
    Nels = len(x_centers)

    # kernel shape: [Nobs x (fx,fy) x 2 basis functions x Nels]
    kernel_ux = np.zeros((Nobs, 2, 2, Nels))
    kernel_uy = np.zeros((Nobs, 2, 2, Nels))
    kernel_sxx = np.zeros((Nobs, 2, 2, Nels))
    kernel_sxy = np.zeros((Nobs, 2, 2, Nels))
    kernel_syy = np.zeros((Nobs, 2, 2, Nels))

    for i in prange(Nels):
        x_trans = x_obs - x_centers[i]
        y_trans = y_obs - y_centers[i]

        # Rotate coordinates into element-local frame
        x_rot = rot_mats_inv[i, 0, 0] * x_trans + rot_mats_inv[i, 0, 1] * y_trans
        y_rot = rot_mats_inv[i, 1, 0] * x_trans + rot_mats_inv[i, 1, 1] * y_trans

        w = half_lengths[i]

        # ----- Compute displacements and stresses for unit forces along x and y -----
        for basis in range(0, 2):  # 2 linear basis functions
            for fi in range(0, 2):  # force along x/y
                # Compute your analytical kernels here for local coordinates
                # kernel shape: [Nobs x (fx,fy) x 2 basis functions x Nels]
                # define unit force direction
                if fi == 0:
                    fx, fy = 1.0, 0.0
                else:
                    fx, fy = 0.0, 1.0

                # choose appropriate analytical kernel
                if basis == 0:
                    ux_local = ux_1(fx, fy, w, x_rot, y_rot, mu, nu)
                    uy_local = uy_1(fx, fy, w, x_rot, y_rot, mu, nu)
                    sxx_local = sxx_1(fx, fy, w, x_rot, y_rot, mu, nu)
                    syy_local = syy_1(fx, fy, w, x_rot, y_rot, mu, nu)
                    sxy_local = sxy_1(fx, fy, w, x_rot, y_rot, mu, nu)
                else:
                    ux_local = ux_2(fx, fy, w, x_rot, y_rot, mu, nu)
                    uy_local = uy_2(fx, fy, w, x_rot, y_rot, mu, nu)
                    sxx_local = sxx_2(fx, fy, w, x_rot, y_rot, mu, nu)
                    syy_local = syy_2(fx, fy, w, x_rot, y_rot, mu, nu)
                    sxy_local = sxy_2(fx, fy, w, x_rot, y_rot, mu, nu)

                # ----- Rotate kernels from local -> global -----
                rot = rot_mats[i, :, :]
                # Displacements
                ux_global = rot[0, 0] * ux_local + rot[0, 1] * uy_local
                uy_global = rot[1, 0] * ux_local + rot[1, 1] * uy_local
                kernel_ux[:, fi, basis, i] = ux_global
                kernel_uy[:, fi, basis, i] = uy_global

                # stresses (2 rotations) S' = R . S . R'
                s00r = rot[0, 0] * sxx_local + rot[0, 1] * sxy_local
                s01r = rot[0, 0] * sxy_local + rot[0, 1] * syy_local
                s10r = rot[1, 0] * sxx_local + rot[1, 1] * sxy_local
                s11r = rot[1, 0] * sxy_local + rot[1, 1] * syy_local
                sxx_global = rot[0, 0] * s00r + rot[0, 1] * s01r
                syy_global = rot[1, 0] * s10r + rot[1, 1] * s11r
                sxy_global = rot[0, 0] * s10r + rot[0, 1] * s11r
                # sxy_global = rot[1, 0] * s00r + rot[1, 1] * s01r

                kernel_sxx[:, fi, basis, i] = sxx_global
                kernel_syy[:, fi, basis, i] = syy_global
                kernel_sxy[:, fi, basis, i] = sxy_global

    return kernel_ux, kernel_uy, kernel_sxx, kernel_syy, kernel_sxy


def plot_BEM_field(
    toplot, els, xo, yo, xlimits, ylimits, maxval, n_levels=10, cmap="coolwarm"
):
    """
    Plot BEM scalar field with element geometry and automatically computed contours.

    Parameters
    ----------
    toplot : ndarray
        Scalar field values at observation points (1D array of size nx_obs*ny_obs)
    els : object
        Mesh object with attributes x1, x2, y1, y2
    xo, yo : ndarray
        Observation point coordinates (1D arrays)
    xlimits, ylimits : tuple
        (xmin, xmax) and (ymin, ymax)
    maxval : float
        Maximum absolute value for color scaling
    n_levels : int, optional
        Number of contour levels (default 10)
    cmap : str, optional
        Colormap (default "coolwarm")
    """

    # Determine grid shape
    nx_obs = len(np.unique(xo))
    ny_obs = len(np.unique(yo))

    X = xo.reshape(ny_obs, nx_obs)
    Y = yo.reshape(ny_obs, nx_obs)
    Z = toplot.reshape(ny_obs, nx_obs)

    # color limits
    vmin, vmax = -maxval, maxval

    # contour levels
    levels = np.linspace(vmin, vmax, n_levels)

    # plot field using colors
    plt.pcolor(X, Y, Z, cmap=cmap, vmin=vmin, vmax=vmax)

    # overlay mesh elements
    n_els = len(els.x1)
    for i in range(n_els):
        plt.plot(
            [els.x1[i], els.x2[i]],
            [els.y1[i], els.y2[i]],
            "k.-",
            linewidth=0.2,
            markersize=1,
        )

    # colorbar
    plt.colorbar()

    # contour lines
    plt.contour(X, Y, Z, colors="k", levels=levels, linewidths=0.5)

    # limits and aspect
    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.gca().set_aspect("equal", adjustable="box")


def slip_functions(x, a, basis="quadratic"):
    """Get pre-multiplier (L) to quadratic coefficients (x) to compute slip (Lx = slip) at any point on the fault patch"""
    if basis == "quadratic":
        design_matrix = np.zeros((len(x), 3))
        f1 = (x / a) * (9 * (x / a) / 8 - 3 / 4)
        f2 = (1 - 3 * (x / a) / 2) * (1 + 3 * (x / a) / 2)
        f3 = (x / a) * (9 * (x / a) / 8 + 3 / 4)
        design_matrix[:, 0] = f1
        design_matrix[:, 1] = f2
        design_matrix[:, 2] = f3
    elif basis == "linear":
        design_matrix = np.zeros((len(x), 2))
        f1 = 0.5 * (1 + x / a)
        f2 = 0.5 * (1 - x / a)
        design_matrix[:, 0] = f1
        design_matrix[:, 1] = f2
    else:
        raise ValueError("Invalid basis functions. Use either 'linear' or 'quadratic'")

    return design_matrix


# Slip gradient functions
def slipgradient_functions(x, a, basis="quadratic"):
    """Get pre-multiplier (L) to quadratic coefficients (x) to compute slip-gradient (Lx = dslip/dx) at any point on the fault patch.

    Note that the slip gradient is only along the fault."""
    if basis == "quadratic":
        design_matrix = np.zeros((len(x), 3))
        df_1_dx = (9 * x) / (4 * a**2) - 3 / (4 * a)
        df_2_dx = -(9 * x) / (2 * a**2)
        df_3_dx = (9 * x) / (4 * a**2) + 3 / (4 * a)
        design_matrix[:, 0] = df_1_dx
        design_matrix[:, 1] = df_2_dx
        design_matrix[:, 2] = df_3_dx
    elif basis == "linear":
        design_matrix = np.zeros((len(x), 2))
        df_1_dx = 0.5 / a
        df_2_dx = -0.5 / a
        design_matrix[:, 0] = df_1_dx
        design_matrix[:, 1] = df_2_dx
    else:
        raise ValueError("Invalid basis functions. Use either 'linear' or 'quadratic'")
    return design_matrix


def get_matrices_slip_slip_gradient(
    els, flag="node", reference="global", basis="quadratic"
):
    """
    Assemble the slip and slip-gradient matrices for fault elements in 2D (plane-strain or antiplane).

    Each fault element has two slip components:
        - s (shear, along strike)
        - n (normal, perpendicular to strike)

    These matrices map nodal slip values to displacement and displacement gradients
    at element nodes, forming the linear system to compute basis coefficients.

    Parameters
    ----------
    els : object
        Fault element geometry, including x1, y1, x2, y2 endpoints, half-lengths,
        and unit vectors (x_shears, y_shears, x_normals, y_normals) for each element.
    flag : str, optional
        "node"  -> slip is applied at each node of the element (default)
        "mean"  -> slip applied as element-averaged (not supported here)
    reference : str, optional
        "global" -> output matrices are rotated into global (x,y) coordinates
        "local"  -> matrices remain in local (s,n) coordinates
    basis : str, optional
        "linear" or "quadratic" nodal basis functions (default "quadratic")

    Returns
    -------
    mat_slip : ndarray, shape (2*n_nodes*Nels, 2*n_nodes*Nels)
        Design matrix mapping nodal slip to displacements.

        Ordering convention:
        --------------------
        * Columns (input slip degrees of freedom):
              For each element k:
                  [ s1, s2, ..., sn,  n1, n2, ..., nn ]
              i.e., all shear-slip nodes first, followed by all normal-slip nodes.

        * Rows (output displacement components):
              For each element k:
                  [ s1, n1,  s2, n2, ..., sn, nn ]
              i.e., displacements are interleaved by component
              — even rows correspond to shear displacement,
                odd rows correspond to normal displacement.

        If `reference="global"`, these local (s,n) displacements are rotated
        into (x,y) coordinates using the element unit vectors.

    mat_slip_gradient : ndarray, shape same as mat_slip
        Design matrix mapping nodal slip to slip gradients (derivatives of displacement),
        with the same interleaving and ordering as mat_slip.

    Notes
    -----
    - The "interweaving" ensures that before rotation, shear slip only affects shear
      displacement and normal slip only affects normal displacement at each node.
    - For `quadratic` elements, each element contributes 3 nodes (n_nodes=3); for
      `linear` elements, n_nodes=2.
    - `unit_vec_mat` rotates the local s/n directions to global x/y coordinates when
      `reference="global"`.
    """

    n_els = len(els.x1)

    if basis == "quadratic":
        n_nodes = 3
    elif basis == "linear":
        n_nodes = 2
    else:
        raise ValueError("Invalid basis")

    stride = 2 * n_nodes
    mat_slip = np.zeros((stride * n_els, stride * n_els))
    mat_slip_gradient = np.zeros_like(mat_slip)

    for i in range(n_els):
        slip_mat_stack = np.zeros((stride, stride))
        slip_gradient_mat_stack = np.zeros_like(slip_mat_stack)

        unit_vec_mat = np.array(
            [[els.x_shears[i], els.x_normals[i]], [els.y_shears[i], els.y_normals[i]]]
        )
        unit_vec_mat_stack = np.kron(np.eye(n_nodes), unit_vec_mat)

        if flag != "node":
            raise ValueError("Only 'node' flag is supported for now.")

        # evaluation points
        if basis == "quadratic":
            x_obs = np.array([-els.half_lengths[i], 0.0, els.half_lengths[i]])
        else:
            x_obs = np.array([-els.half_lengths[i], els.half_lengths[i]])

        slip_mat = slip_functions(x_obs, els.half_lengths[i], basis)
        slip_gradient_mat = slipgradient_functions(x_obs, els.half_lengths[i], basis)

        # --- Interweave shear & normal components ---
        # The matrix slip_mat_stack has 2*n_nodes rows:
        #   even rows (0,2,4,...) correspond to shear-displacement directions
        #   odd  rows (1,3,5,...) correspond to normal-displacement directions
        # Columns 0:n_nodes represent nodal shear slip coefficients
        # Columns n_nodes:2*n_nodes represent nodal normal slip coefficients
        #
        # This pattern ensures that shear slip affects only shear displacement
        # and normal slip affects only normal displacement before rotation
        # into the global (x,y) reference frame.
        col_shear = slice(0, n_nodes)
        col_normal = slice(n_nodes, 2 * n_nodes)
        slip_mat_stack[0::2, col_shear] = slip_mat
        slip_mat_stack[1::2, col_normal] = slip_mat
        slip_gradient_mat_stack[0::2, col_shear] = slip_gradient_mat
        slip_gradient_mat_stack[1::2, col_normal] = slip_gradient_mat

        if reference == "global":
            mat_slip[stride * i : stride * (i + 1), stride * i : stride * (i + 1)] = (
                unit_vec_mat_stack @ slip_mat_stack
            )
            mat_slip_gradient[
                stride * i : stride * (i + 1), stride * i : stride * (i + 1)
            ] = (unit_vec_mat_stack @ slip_gradient_mat_stack)
        elif reference == "local":
            mat_slip[stride * i : stride * (i + 1), stride * i : stride * (i + 1)] = (
                slip_mat_stack
            )
            mat_slip_gradient[
                stride * i : stride * (i + 1), stride * i : stride * (i + 1)
            ] = slip_gradient_mat_stack
        else:
            raise ValueError("Invalid reference frame. Use either 'global' or 'local'")

    return mat_slip, mat_slip_gradient


def compute_coefs_trapezoidalforce_planestrain(els, connect_matrix):
    """
    Compute trapezoidal distributed force coefficients for plane strain BEM elements.

    This routine constructs local "trapezoidal" force basis functions (shear and normal)
    over three connected boundary elements. Each patch enforces continuity and boundary
    conditions so that distributed tractions vary linearly (trapezoidally) across elements.

    Parameters
    ----------
    els : object
        BEM element structure containing arrays of element endpoints (x1, y1, x2, y2).
        Typically created using `bemcs.initialize_els()`.

    connect_matrix : (n_trapz, 3) int ndarray
        Element connectivity matrix. Each row defines a 3-element trapezoidal patch,
        e.g. [0, 1, 2], used to build local force taper functions.

    Returns
    -------
    fcoefs_s : (Nels, 2, 2, Ntrapz) ndarray
        Shear force coefficients for all trapezoidal patches.
        Dimensions:
            - axis 0 → element index
            - axis 1 → force component (0 = shear, 1 = normal)
            - axis 2 → basis function index
            - axis 3 → trapezium index

    fcoefs_n : (Nels, 2, 2, Ntrapz) ndarray
        Normal force coefficients for all trapezoidal patches, same layout as fcoefs_s.

    Notes
    -----
    Each 3-element patch satisfies:
        1. Unit traction (shear or normal) on the central element.
        2. Continuity of tractions on overlapping nodes.
        3. Zero traction on open boundary nodes.

    The system matrix is built using linear basis functions from
    `get_matrices_slip_slip_gradient()` in both local and global frames.
    The resulting coefficients define distributed force tapers suitable
    for plane strain BEM simulations.
    """
    n_els = len(els.x_centers)
    n_trapz = len(connect_matrix[:, 0])
    fcoefs_s = np.zeros((n_els, 2, 2, n_trapz))
    fcoefs_n = np.zeros((n_els, 2, 2, n_trapz))

    for i in range(n_trapz):
        x1 = els.x1[connect_matrix[i, :]]
        y1 = els.y1[connect_matrix[i, :]]
        x2 = els.x2[connect_matrix[i, :]]
        y2 = els.y2[connect_matrix[i, :]]
        # create new els() from the 3 elements
        els_subset = bemcs.initialize_els()
        els_subset.x1, els_subset.y1, els_subset.x2, els_subset.y2 = x1, y1, x2, y2
        bemcs.standardize_els_geometry(els_subset, reorder=False)

        # find open and overlapping nodes
        index_open, index_overlap, _ = bemcs.label_nodes(els_subset, verbose=False)

        # compute force coefficients at mesh end-points
        mat_force_local, _ = get_matrices_slip_slip_gradient(
            els_subset, reference="local", basis="linear"
        )
        mat_force_global, _ = get_matrices_slip_slip_gradient(
            els_subset, reference="global", basis="linear"
        )

        # constraint equations and build matrix system
        # impose the following: for element 2 (fs = [1,1], fn = [0,0]), taper to 0 for element 1 & 3
        # equations are
        # 1,2,3,4: fs,fn = 1.0,0 at central element
        # 5,6,7,8: continuity at overlap elements
        # 9,10,11,12: fx,fy = 0 at open nodes
        mat_force_open = np.zeros((len(index_open) * 2, len(mat_force_local[0, :])))
        mat_force_overlap = np.zeros(
            (len(index_overlap) * 2, len(mat_force_global[0, :]))
        )
        # construct matrix system for local nodes at the central element
        mat_force_central = mat_force_local[4:8, :]

        # Linear operator for open nodes
        for iter in range(len(index_open)):
            elem_idx = index_open[iter] // 3  # integer division to get element index
            local_node = index_open[iter] % 3  # node within element

            if local_node == 0:
                linear_node = 2 * elem_idx[0]
            elif local_node == 2:
                linear_node = 2 * elem_idx[0] + 1
            else:
                raise ValueError(
                    f"Quadratic mid-node {index_open[iter]} has no equivalent in linear mesh"
                )
            mat_force_open[2 * iter : 2 * iter + 2, :] = mat_force_local[
                2 * linear_node : 2 * linear_node + 2, :
            ]
        # Linear operator for overlapping nodes
        for iter in range(len(index_overlap)):
            elem_idx = index_overlap[iter] // 3  # integer division to get element index
            local_node = index_overlap[iter] % 3  # node within element
            idvals = 2 * elem_idx + local_node  # node number
            # continuity condition
            if (idvals[0] != 0) & (idvals[1] != 0):
                sign1 = np.sign(idvals[0])
                sign2 = np.sign(idvals[1])
            elif (idvals[0] == 0) & (idvals[1] != 0):
                sign1 = 1
                sign2 = -1
            else:
                sign1 = -1
                sign2 = 1

            mat_force_overlap[2 * iter, :] = (
                sign1 * mat_force_global[2 * np.abs(idvals[0]), :]
                + sign2 * mat_force_global[2 * np.abs(idvals[1]), :]
            )  # x
            mat_force_overlap[2 * iter + 1, :] = (
                sign1 * mat_force_global[2 * np.abs(idvals[0]) + 1, :]
                + sign2 * mat_force_global[2 * np.abs(idvals[1]) + 1, :]
            )  # y
        # stack matrices
        mat_system = np.vstack((mat_force_central, mat_force_open, mat_force_overlap))

        # RHS for shear component
        rhs = np.zeros((len(mat_force_global[0, :]), 1))
        rhs[[0, 2], 0] = 1.0
        coefs_shear = np.linalg.solve(mat_system, rhs).flatten()

        # RHS for normal component
        rhs = np.zeros((len(mat_force_global[0, :]), 1))
        rhs[[1, 3], 0] = 1.0
        coefs_normal = np.linalg.solve(mat_system, rhs).flatten()

        for local_id, elem_id in enumerate(connect_matrix[i, :]):
            base = local_id * 4  # 4 coefficients per element (s1,s2,n1,n2)
            # for overall shear source trapezoid
            # shear components
            fcoefs_s[elem_id, 0, :, i] = coefs_shear[base : base + 2]
            # normal components
            fcoefs_s[elem_id, 1, :, i] = coefs_shear[base + 2 : base + 4]

            # for overall normal source trapezoid
            # shear components
            fcoefs_n[elem_id, 0, :, i] = coefs_normal[base : base + 2]
            # normal components
            fcoefs_n[elem_id, 1, :, i] = coefs_normal[base + 2 : base + 4]

    return fcoefs_s, fcoefs_n


def get_kernels_trapezoidalforce_planestrain(
    x_obs, y_obs, els, connect_matrix, mu=1, nu=0.25
):
    """
    Construct trapezoidal basis kernels for plane-strain BEM using 3-element patches.

    Each trapezoidal patch defines a set of distributed force basis functions
    (shear and normal components) over three connected elements. The function
    computes the displacement and stress kernels for all observation points
    and interleaves shear and normal contributions for each trapezoidal patch.

    Parameters
    ----------
    x_obs, y_obs : (Nobs,)
        Observation coordinates.
    els : object
        Element geometry structure with attributes `x_centers`, `y_centers`,
        `half_lengths`, `rot_mats`, `rot_mats_inv`.
    connect_matrix : (Ntrapz, 3)
        Connectivity of trapezoidal patches; each row gives indices of 3 connected elements.
    mu : float, optional
        Shear modulus (default 1.0).
    nu : float, optional
        Poisson’s ratio (default 0.25).

    Returns
    -------
    kernel_ux, kernel_uy, kernel_sxx, kernel_syy, kernel_sxy : (Nobs, 2*Ntrapz)
        Trapezoidal basis kernels for displacement and stress components.
        Columns are interleaved as [shear_patch1, normal_patch1, shear_patch2, normal_patch2, ...].
    """

    n_obs = len(x_obs)
    n_GFs = connect_matrix.shape[0]

    # Allocate final kernels with interleaved force components
    kernel_ux = np.zeros((n_obs, 2 * n_GFs))
    kernel_uy = np.zeros((n_obs, 2 * n_GFs))
    kernel_sxx = np.zeros((n_obs, 2 * n_GFs))
    kernel_syy = np.zeros((n_obs, 2 * n_GFs))
    kernel_sxy = np.zeros((n_obs, 2 * n_GFs))

    # compute kernels with all mesh elements [Nobs, (shear/normal), 2 basis functions, Nels]
    K_ux, K_uy, K_sxx, K_syy, K_sxy = get_displacement_stress_kernel_force_planestrain(
        x_obs.flatten(),
        y_obs.flatten(),
        els.x_centers,
        els.y_centers,
        els.half_lengths,
        els.rot_mats,
        els.rot_mats_inv,
        mu,
        nu,
    )
    # compute coefficients for trapezoidal basis functions
    # each fcoefs matrix : (Nels, 2, 2, Ntrapz) ndarray
    # force coefficients for all trapezoidal patches.
    # Dimensions:
    #     - axis 0 → element index
    #     - axis 1 → force component (0 = shear, 1 = normal)
    #     - axis 2 → basis function index
    #     - axis 3 → trapezoid index
    fcoefs_s, fcoefs_n = compute_coefs_trapezoidalforce_planestrain(els, connect_matrix)

    # Trapezoidal stress kernels (shear) shape: (Nobs, 2, Ntrapz)
    kernel_sxx_s = np.tensordot(K_sxx, fcoefs_s, axes=([1, 2, 3], [1, 2, 0]))
    kernel_syy_s = np.tensordot(K_syy, fcoefs_s, axes=([1, 2, 3], [1, 2, 0]))
    kernel_sxy_s = np.tensordot(K_sxy, fcoefs_s, axes=([1, 2, 3], [1, 2, 0]))
    # normal shape: (Nobs, 2, Ntrapz)
    kernel_sxx_n = np.tensordot(K_sxx, fcoefs_n, axes=([1, 2, 3], [1, 2, 0]))
    kernel_syy_n = np.tensordot(K_syy, fcoefs_n, axes=([1, 2, 3], [1, 2, 0]))
    kernel_sxy_n = np.tensordot(K_sxy, fcoefs_n, axes=([1, 2, 3], [1, 2, 0]))

    # Trapezoidal displacement kernels
    kernel_ux_s = np.tensordot(K_ux, fcoefs_s, axes=([1, 2, 3], [1, 2, 0]))
    kernel_uy_s = np.tensordot(K_uy, fcoefs_s, axes=([1, 2, 3], [1, 2, 0]))
    kernel_ux_n = np.tensordot(K_ux, fcoefs_n, axes=([1, 2, 3], [1, 2, 0]))
    kernel_uy_n = np.tensordot(K_uy, fcoefs_n, axes=([1, 2, 3], [1, 2, 0]))

    # Interleave shear (even columns) and normal (odd columns)
    kernel_ux[:, 0::2] = kernel_ux_s
    kernel_ux[:, 1::2] = kernel_ux_n

    kernel_uy[:, 0::2] = kernel_uy_s
    kernel_uy[:, 1::2] = kernel_uy_n

    kernel_sxx[:, 0::2] = kernel_sxx_s
    kernel_sxx[:, 1::2] = kernel_sxx_n

    kernel_syy[:, 0::2] = kernel_syy_s
    kernel_syy[:, 1::2] = kernel_syy_n

    kernel_sxy[:, 0::2] = kernel_sxy_s
    kernel_sxy[:, 1::2] = kernel_sxy_n

    return kernel_ux, kernel_uy, kernel_sxx, kernel_syy, kernel_sxy


def compute_coefs_quadratichatslip_planestrain(els, connect_matrix):
    """
    Compute quadratic hat slip coefficients for plane strain BEM elements.

    This routine constructs local "quadratic hat" slip basis functions
    over three connected boundary elements. Each patch enforces continuity and boundary
    conditions so that slip varies quadratically across elements.

    Parameters
    ----------
    els : object
        BEM element structure containing arrays of element endpoints (x1, y1, x2, y2).
        Typically created using `bemcs.initialize_els()`.

    connect_matrix : (n_patches, 3) int ndarray
        Element connectivity matrix. Each row defines a 3-element quadratic hat patch,
        e.g. [0, 1, 2], used to build local slip taper functions.

    Returns
    -------
    coefs_s : (Nels, 6, Npatches) ndarray
        Shear slip coefficients for all quadratic hat patches.
        Dimensions:
            - axis 0 → element index
            - axis 1 → nodal coefficients (3 nodes × 2 components = 6)
            - axis 2 → patch index

    coefs_n : (Nels, 6, Npatches) ndarray
        Normal slip coefficients for all quadratic hat patches, same layout as coefs_s.

    Notes
    -----
    Each 3-element patch satisfies:
        1. Unit slip (shear or normal) at the center of the central element.
        2. Continuity of slip and slip-gradient at overlapping nodes.
        3. Zero slip and slip-gradient at open boundary nodes.

    The system matrix is built using quadratic basis functions from
    `get_matrices_slip_slip_gradient()` in local reference frame.
    The resulting coefficients define distributed slip tapers suitable
    for plane strain BEM simulations.
    """
    n_els = len(els.x_centers)
    n_patches = len(connect_matrix[:, 0])
    coefs_s = np.zeros((n_els, 6, n_patches))
    coefs_n = np.zeros((n_els, 6, n_patches))

    for i in range(n_patches):
        print("patch number: ", i)
        x1 = els.x1[connect_matrix[i, :]]
        y1 = els.y1[connect_matrix[i, :]]
        x2 = els.x2[connect_matrix[i, :]]
        y2 = els.y2[connect_matrix[i, :]]
        # create new els() from the 3 elements
        els_subset = bemcs.initialize_els()
        els_subset.x1, els_subset.y1, els_subset.x2, els_subset.y2 = x1, y1, x2, y2
        bemcs.standardize_els_geometry(els_subset, reorder=False)

        # find open and overlapping nodes
        index_open, index_overlap, _ = bemcs.label_nodes(els_subset)

        # operator matrix for slip and slip gradient in global reference frame for all the open and 2-overlap nodes
        mat_slip, mat_slip_gradient = bemcs.get_matrices_slip_slip_gradient(
            els_subset, reference="global"
        )
        # operator matrix ONLY for slip at the central node of the central element in local reference frame (where the quadratic hat is defined)
        mat_slip_local, _ = bemcs.get_matrices_slip_slip_gradient(
            els_subset, reference="local"
        )

        # constraint equations and build matrix system
        # For quadratic, 3 elements × 3 nodes × 2 comp = 18 unknowns
        # Constraints: 2 for central, 4 per open, 4 per overlap = 18
        Nunknowns = 18  # hardcoded for 3-element quadratic hat patch
        N_o = 4 * len(index_open)  # open node equations
        N_i = 4 * len(index_overlap)  # overlapping node equations

        # define matrix system for open and overlapping nodes
        mat_system_o = np.zeros((N_o, Nunknowns))
        mat_system_i = np.zeros((N_i, Nunknowns))

        # Linear operator for open nodes
        for iter in range(int(N_o / 4)):
            id1 = np.abs(index_open[iter])  # node number
            # slip constraints
            mat_system_o[4 * iter, :] = mat_slip[2 * id1, :]  # x component
            mat_system_o[4 * iter + 1, :] = mat_slip[2 * id1 + 1, :]  # y component
            # slip gradient constraints
            mat_system_o[4 * iter + 2, :] = mat_slip_gradient[2 * id1, :]  # x component
            mat_system_o[4 * iter + 3, :] = mat_slip_gradient[
                2 * id1 + 1, :
            ]  # y component

        # Linear operator for overlapping nodes
        for iter in range(len(index_overlap)):
            idvals = index_overlap[iter]  # node number
            # continuity condition
            if (idvals[0] != 0) & (idvals[1] != 0):
                sign1 = np.sign(idvals[0])
                sign2 = np.sign(idvals[1])
            elif (idvals[0] == 0) & (idvals[1] != 0):
                sign1 = 1
                sign2 = -1
            else:
                sign1 = -1
                sign2 = 1

            mat_system_i[4 * iter, :] = (
                sign1 * mat_slip[2 * np.abs(idvals[0]), :]
                + sign2 * mat_slip[2 * np.abs(idvals[1]), :]
            )  # x
            mat_system_i[4 * iter + 1, :] = (
                sign1 * mat_slip[2 * np.abs(idvals[0]) + 1, :]
                + sign2 * mat_slip[2 * np.abs(idvals[1]) + 1, :]
            )  # y
            # smoothing constraints
            mat_system_i[4 * iter + 2, :] = (
                sign1 * mat_slip_gradient[2 * np.abs(idvals[0]), :]
                + sign2 * mat_slip_gradient[2 * np.abs(idvals[1]), :]
            )  # x
            mat_system_i[4 * iter + 3, :] = (
                sign1 * mat_slip_gradient[2 * np.abs(idvals[0]) + 1, :]
                + sign2 * mat_slip_gradient[2 * np.abs(idvals[1]) + 1, :]
            )  # y

        # Linear operator for central node of central element (in local reference frame)
        id1 = (
            4  # hard-coded node [0,1,2],[3,4,5],[5,6,7] → node 4 is center of element 1
        )
        mat_system_c = np.zeros((2, Nunknowns))
        mat_system_c[0, :] = mat_slip_local[2 * id1, :]  # shear component
        mat_system_c[1, :] = mat_slip_local[2 * id1 + 1, :]  # normal component

        # Combine all constraints into a single system
        mat_system = np.vstack((mat_system_c, mat_system_o, mat_system_i))
        # Right-hand side: unit slip at central node for shear and normal cases
        rhs_shear = np.zeros((N_o + N_i + 2, 1))
        rhs_shear[0, 0] = 1  # unit shear slip at central node
        rhs_normal = np.zeros((N_o + N_i + 2, 1))
        rhs_normal[1, 0] = 1  # unit normal slip at central node

        # Solve for coefficients
        coefs_shear = np.linalg.solve(mat_system, rhs_shear).flatten()
        coefs_normal = np.linalg.solve(mat_system, rhs_normal).flatten()

        # Assign to global arrays
        for local_id, elem_id in enumerate(connect_matrix[i, :]):
            base = local_id * 6  # 6 coefficients per element (3 nodes × 2 comp)
            coefs_s[elem_id, :, i] = coefs_shear[base : base + 6]
            coefs_n[elem_id, :, i] = coefs_normal[base : base + 6]

    return coefs_s, coefs_n


def get_kernels_quadratichatslip_planestrain(
    x_obs, y_obs, els, connect_matrix, mu=1, nu=0.25
):
    """
    Compute quadratic-hat slip kernels for plane strain BEM elements.

    This routine constructs displacement and stress kernels for quadratic hat slip basis functions
    over three connected boundary elements. Each patch enforces continuity and boundary
    conditions so that slip varies quadratically across elements.

    Parameters
    ----------
    x_obs, y_obs : array_like, shape (n_obs,)
        Observation coordinates where kernels are evaluated.
    els : object
        BEM element structure containing arrays of element endpoints (x1, y1, x2, y2).
        Typically created using `bemcs.initialize_els()`.
    connect_matrix : (n_patches, 3) int ndarray
        Element connectivity matrix. Each row defines a 3-element quadratic hat patch,
        e.g. [0, 1, 2], used to build local slip taper functions.
    mu : float, optional
        Shear modulus (default 1.0).
    nu : float, optional
        Poisson's ratio (default 0.25).

    Returns
    -------
    kernel_ux, kernel_uy, kernel_sxx, kernel_syy, kernel_sxy : (n_obs, 2*n_patches) ndarray
        Quadratic-hat slip kernels for displacement and stress components.
        Columns are interleaved as [shear_patch1, normal_patch1, shear_patch2, normal_patch2, ...].
        Dimensions:
            - axis 0 → observation point index
            - axis 1 → patch index (interleaved shear/normal)

    Notes
    -----
    Each 3-element patch satisfies:
        1. Unit slip (shear or normal) at the center of the central element.
        2. Continuity of slip and slip-gradient at overlapping nodes.
        3. Zero slip and slip-gradient at open boundary nodes.

    The kernels are computed using quadratic basis functions from
    `bemcs.get_displacement_stress_kernel()` and coefficient distributions from
    `compute_coefs_quadratichatslip_planestrain()`.
    """

    n_obs = len(x_obs)
    n_patches = connect_matrix.shape[0]

    kernel_ux = np.zeros((n_obs, 2 * n_patches))
    kernel_uy = np.zeros((n_obs, 2 * n_patches))
    kernel_sxx = np.zeros((n_obs, 2 * n_patches))
    kernel_syy = np.zeros((n_obs, 2 * n_patches))
    kernel_sxy = np.zeros((n_obs, 2 * n_patches))

    # Quadratic hat slip coefficients should be provided by a coefficient builder.
    coefs_s, coefs_n = compute_coefs_quadratichatslip_planestrain(els, connect_matrix)

    kernels_s = bemcs.get_displacement_stress_kernel(
        x_obs.flatten(), y_obs.flatten(), els, mu, nu, "shear"
    )
    kernels_n = bemcs.get_displacement_stress_kernel(
        x_obs.flatten(), y_obs.flatten(), els, mu, nu, "normal"
    )

    n_els = len(els.x1)
    for k in range(n_patches):
        quadratic_coefs_s = np.zeros(3 * n_els)
        quadratic_coefs_n = np.zeros(3 * n_els)
        for i in range(n_els):
            quadratic_coefs_s[3 * i : 3 * (i + 1)] = coefs_s[i, :3, k]
            quadratic_coefs_n[3 * i : 3 * (i + 1)] = coefs_s[i, 3:, k]

        ux_s, uy_s, sxx_s, syy_s, sxy_s = bemcs.coeffs_to_disp_stress(
            kernels_s, kernels_n, quadratic_coefs_s, quadratic_coefs_n
        )

        quadratic_coefs_s = np.zeros(3 * n_els)
        quadratic_coefs_n = np.zeros(3 * n_els)
        for i in range(n_els):
            quadratic_coefs_s[3 * i : 3 * (i + 1)] = coefs_n[i, :3, k]
            quadratic_coefs_n[3 * i : 3 * (i + 1)] = coefs_n[i, 3:, k]

        ux_n, uy_n, sxx_n, syy_n, sxy_n = bemcs.coeffs_to_disp_stress(
            kernels_s, kernels_n, quadratic_coefs_s, quadratic_coefs_n
        )

        kernel_ux[:, 2 * k] = ux_s
        kernel_ux[:, 2 * k + 1] = ux_n
        kernel_uy[:, 2 * k] = uy_s
        kernel_uy[:, 2 * k + 1] = uy_n
        kernel_sxx[:, 2 * k] = sxx_s
        kernel_sxx[:, 2 * k + 1] = sxx_n
        kernel_syy[:, 2 * k] = syy_s
        kernel_syy[:, 2 * k + 1] = syy_n
        kernel_sxy[:, 2 * k] = sxy_s
        kernel_sxy[:, 2 * k + 1] = sxy_n

    return kernel_ux, kernel_uy, kernel_sxx, kernel_syy, kernel_sxy


# =============================================================================
# Plane-strain element kernels: scalar formulas shared by numpy and numba drivers
#
# _quadratic_slip_f and _linear_force_g were generated with sympy from the
# original closed forms (bemcs.quadratic_kernel_farfield and the inner closures
# of get_displacement_stress_kernel_force_planestrain) by common-subexpression
# elimination, with every (a -/+ x)^2 + y^2 evaluated in that factored form
# (inside logs and denominators) rather than as the expanded polynomial.
# _quadratic_slip_f is in addition expanded about the element endpoint nearest
# the observation point (x = s a + xi), so the small quantities near a node
# (xi, y) are never formed as differences of large numbers: it is accurate to
# ~1e-15 relative down to at least 1e-8 a from a node (tasks.md issue 16).
# The force formulas were already well conditioned near endpoints and are kept
# in their original variables. Both work on scalars (numba drivers) and on
# numpy arrays (numpy drivers, through .py_func).
# =============================================================================

@njit(cache=True, error_model="numpy")
def _quadratic_slip_f(xi, y, a, nu, s, f):
    """Fill f (7, 3) with f0..f6 for the 3 quadratic shape functions.

        The observation point is given relative to the element endpoint x = s a (s = +1 or -1):
        local x = s a + xi. Every polynomial is expanded about that endpoint, so evaluate with the
        endpoint nearest the point. Generated with sympy from bemcs.quadratic_kernel_farfield."""
    am = (1 - s) * a - xi  # a - x, exact near the endpoint x = s a
    ap = (1 + s) * a + xi  # a + x
    Am = np.pi / 2 * np.sign(y / am) - np.arctan(y / am)
    Ap = np.pi / 2 * np.sign(y / ap) - np.arctan(y / ap)
    Rm = am**2 + y**2
    Rp = ap**2 + y**2
    Lm = np.log(Rm)
    Lp = np.log(Rp)
    x0 = 0.1875*xi
    x1 = 0.1875*s
    x2 = a*x1
    x3 = 1/a
    x4 = 0.046875*x3
    x5 = xi**2
    x6 = Lm*x5
    x7 = x4*x6
    x8 = y**2
    x9 = x4*x8
    x10 = Lp*x9
    x11 = s*xi
    x12 = 0.09375*x11
    x13 = Lm*x12
    x14 = Lm*x9
    x15 = Lp*x5
    x16 = x15*x4
    x17 = Lp*x12
    x18 = Am*x1
    x19 = x18*y
    x20 = Ap*x1
    x21 = x20*y
    x22 = x3*y
    x23 = x0*x22
    x24 = Am*x23
    x25 = Ap*x23
    x26 = 0.046875*a
    x27 = Lm*x26
    x28 = Lp*x26
    x29 = 0.1875*x3
    x30 = x29*x8
    x31 = 0.140625*Lp
    x32 = x31*xi
    x33 = 0.28125*Am
    x34 = x33*y
    x35 = 0.28125*Ap
    x36 = x35*y
    x37 = 0.375*x11
    x38 = 0.140625*Lm
    x39 = x38*xi
    x40 = xi**3
    x41 = a**2
    x42 = 1/x41
    x43 = x40*x42
    x44 = 0.046875*x43
    x45 = y**3
    x46 = x42*x45
    x47 = 0.09375*x46
    x48 = s*x8
    x49 = x3*x38
    x50 = x42*x8
    x51 = s*x5
    x52 = x3*x31
    x53 = x42*x5
    x54 = 0.5625*s
    x55 = x22*x54
    x56 = Am*x55
    x57 = Ap*x55
    x58 = -Am*x47 - Ap*x47 - Lm*x44 + Lp*x44 - 0.0625*a - s*x27 + s*x28 + x27 + x28 - x29*x5 + x30 - x32*x50 + x32 + x34*x53 + x34 + x36*x53 + x36 - x37 + x39*x50 - x39 + x48*x49 - x48*x52 - x49*x51 + x51*x52 + x56*xi + x57*xi
    x59 = 1/(np.pi*(nu - 1))
    x60 = 0.375*x3
    x61 = 0.15625*Lp
    x62 = 0.3125*Am
    x63 = 0.3125*Ap
    x64 = 0.28125*x3
    x65 = x8*xi
    x66 = 0.28125*x42
    x67 = Lm*x66
    x68 = x5*y
    x69 = 0.5625*x42
    x70 = Am*x69
    x71 = Ap*x69
    x72 = 1.125*s
    x73 = x22*x72
    x74 = Am*x73
    x75 = Ap*x73
    x76 = 0.09375*x22
    x77 = Lp*x76
    x78 = Lm*x76
    x79 = x0*x3
    x80 = Am*x79
    x81 = Ap*x79
    x82 = Lm*x22
    x83 = 0.28125*s
    x84 = xi*y
    x85 = x3*xi
    x86 = x54*x85
    x87 = Lp*x22
    x88 = Lp*x66
    x89 = Am*x86 + Ap*x86 + 0.5625*x22 - x33*x50 + x33*x53 + x33 - x35*x50 + x35*x53 + x35 + x67*x84 + x82*x83 - x83*x87 - x84*x88
    x90 = 0.5625*x53
    x91 = Lm*x69
    x92 = x72*x85
    x93 = 0.09375*s
    x94 = Lm*x93
    x95 = Lp*x93
    x96 = 0.09375*x85
    x97 = Lm*x96
    x98 = Lp*x96
    x99 = x29*y
    x100 = Am*x99
    x101 = Ap*x99
    x102 = x11*x64
    x103 = -Lm*x102 + Lp*x102 - x31*x50 + x31*x53 + x31 + x38*x50 - x38*x53 - x38 - x54 + x56 + x57 + x70*x84 + x71*x84 - 0.5625*x85
    x104 = 1.125*y
    x105 = x104*x42
    x106 = x105*xi
    x107 = 0.375*x45
    x108 = 0.375*x68
    x109 = 0.75*x41
    x110 = x109*y
    x111 = xi**4
    x112 = Am*x29
    x113 = x111*x112
    x114 = y**4
    x115 = x112*x114
    x116 = Ap*x29
    x117 = x111*x116
    x118 = x114*x116
    x119 = a*x5
    x120 = 0.75*Am
    x121 = x119*x120
    x122 = a*x8
    x123 = x120*x122
    x124 = s*x40
    x125 = x120*x124
    x126 = 0.75*Ap
    x127 = x119*x126
    x128 = x122*x126
    x129 = x124*x126
    x130 = 0.75*x11
    x131 = a*y
    x132 = x130*x131
    x133 = x5*x60
    x134 = Am*x8
    x135 = x133*x134
    x136 = Ap*x8
    x137 = x133*x136
    x138 = x130*x134
    x139 = x130*x136
    x140 = 1.125*x45
    x141 = Lm*x140
    x142 = 4.5*x40
    x143 = Am*x142
    x144 = Ap*x142
    x145 = y**5
    x146 = x145*x66
    x147 = xi**5
    x148 = Am*x147
    x149 = Ap*x147
    x150 = 1.125*x68
    x151 = 4.5*x65
    x152 = a*xi
    x153 = x41*y
    x154 = Lm*x111
    x155 = x66*y
    x156 = Am*x114
    x157 = x3*x54
    x158 = x114*xi
    x159 = Ap*x114
    x160 = 0.5625*x46
    x161 = 1.125*x43
    x162 = 2.25*s
    x163 = Am*x162
    x164 = Ap*x162
    x165 = Am*x111
    x166 = s*x3
    x167 = 2.8125*x166
    x168 = Ap*x111
    x169 = Lp*x111
    x170 = x45*x92
    x171 = x40*x72
    x172 = 3.375*x3
    x173 = x5*x8
    x174 = s*x173
    x175 = x172*x174
    x176 = Am*x151 + Am*x175 + Ap*x151 + Ap*x175 + Lm*x146 + Lm*x150 + Lm*x170 - Lp*x140 - Lp*x146 - Lp*x150 - Lp*x170 - x104*x152 + x119*x163 + x119*x164 + x122*x163 + x122*x164 + x134*x161 + x136*x161 + x141 + x143 + x144 + x148*x69 + x149*x69 - x15*x160 - x153*x72 + x154*x155 - x155*x169 + x156*x157 + x157*x159 + x158*x70 + x158*x71 + x160*x6 + x165*x167 + x167*x168 + x171*x82 - x171*x87
    x177 = x59/(Rm*Rp)
    x178 = 2.25*x45
    x179 = Lm*x178
    x180 = 9.0*x40
    x181 = x145*x69
    x182 = 1.125*x42
    x183 = 2.25*x68
    x184 = 9.0*x65
    x185 = x3*x72
    x186 = Am*x182
    x187 = Ap*x182
    x188 = 2.25*x40
    x189 = x188*x50
    x190 = 4.5*s
    x191 = Am*x190
    x192 = Ap*x190
    x193 = 5.625*x166
    x194 = x11*x3
    x195 = Lm*x188
    x196 = s*x22
    x197 = 6.75*x3
    x198 = x174*x197
    x199 = 0.375*x40
    x200 = 0.375*x65
    x201 = x109*xi
    x202 = 0.09375*x3
    x203 = x154*x202
    x204 = x114*x202
    x205 = Lm*x204
    x206 = 0.375*Lm
    x207 = x119*x206
    x208 = x122*x206
    x209 = s*x199
    x210 = Lm*x209
    x211 = 0.375*x122
    x212 = s*x211
    x213 = x119*x72
    x214 = x169*x202
    x215 = Lp*x204
    x216 = Lp*x119
    x217 = 0.375*x216
    x218 = Lp*x211
    x219 = Lp*x209
    x220 = x30*x6
    x221 = x37*x8
    x222 = Lm*x221
    x223 = x15*x30
    x224 = Lp*x221
    x225 = Am*x178
    x226 = Ap*x178
    x227 = 1.125*x3
    x228 = x147*x66
    x229 = 2.25*x65
    x230 = x41*xi
    x231 = 2.25*x3
    x232 = x11*x8
    x233 = s*x114
    x234 = x233*x64
    x235 = x111*y
    x236 = Lp*x8
    x237 = 0.5625*x43
    x238 = x140*x53
    x239 = x122*x72
    x240 = 1.40625*x166
    x241 = x188*x196
    x242 = x15*x48
    x243 = 1.6875*x3
    x244 = Am*x181 + Am*x183 + Am*x238 + Am*x241 + Ap*x181 + Ap*x183 + Ap*x238 + Ap*x241 - Lm*x213 - Lm*x228 - Lm*x229 - Lm*x234 - Lm*x237*x8 - Lm*x239 + Lp*x188 + Lp*x213 + Lp*x228 + Lp*x229 + Lp*x234 + Lp*x239 - s*x142 - x111*x227 - x114*x227 - 5.0625*x119 - 3.9375*x122 - x154*x240 - x158*x67 + x158*x88 + x169*x240 - x173*x231 + x194*x225 + x194*x226 - x195 + x225 + x226 - x230*x72 - 4.5*x232 + x235*x70 + x235*x71 + x236*x237 + x242*x243 - x243*x48*x6
    x245 = 4.5*x45
    x246 = Am*x245
    x247 = Ap*x245
    x248 = x145*x182
    x249 = 4.5*x68
    x250 = Lp*x122
    x251 = x109*x111
    x252 = x109*x114
    x253 = a**4
    x254 = 3.0*x253
    x255 = x254*x5
    x256 = x254*x8
    x257 = a**3
    x258 = x257*x40
    x259 = 3.0*s
    x260 = x258*x259
    x261 = 4.5*x173*x41
    x262 = 9.0*x232*x257
    x263 = y**6
    x264 = xi**7
    x265 = 2.25*x263
    x266 = Lm*x265
    x267 = xi**6
    x268 = 6.75*x267
    x269 = s*x267
    x270 = a*x147
    x271 = xi**8
    x272 = y**8
    x273 = 4.5*x41
    x274 = x114*x273
    x275 = 4.5*x253
    x276 = x257*x65
    x277 = x114*x40
    x278 = x147*x8
    x279 = 11.25*x114
    x280 = x233*x41
    x281 = 15.75*x8
    x282 = x233*x5
    x283 = x114*x152
    x284 = 16.875*x111
    x285 = s*x41
    x286 = x122*x40
    x287 = x182*x263
    x288 = Lm*x50
    x289 = 1.125*x267
    x290 = x231*x264
    x291 = Lm*s
    x292 = 9.0*x270
    x293 = 9.0*x41
    x294 = x293*x8
    x295 = x114*x42
    x296 = 1.6875*x295
    x297 = x173*x285
    x298 = Lp*s
    x299 = Lm*x11
    x300 = x231*x263
    x301 = Lm*x114
    x302 = 9.0*x11
    x303 = a*x302
    x304 = 18.0*x124
    x305 = Lm*x122
    x306 = x124*x197
    x307 = x147*x48
    x308 = x197*x307
    x309 = Lp*x114
    x310 = Lm*x268 + Lm*x274 + Lm*x308 - Lp*x11*x300 - Lp*x265 - Lp*x268 - Lp*x274 - Lp*x289*x50 - Lp*x308 - x15*x279 - x15*x287 - x15*x294 + x154*x273 + x154*x281 + x154*x296 - x169*x273 - x169*x281 - x169*x296 + x172*x277 + x172*x278 + x227*x264 - x250*x304 + x263*x72 + 1.125*x263*x85 + x266 + 7.875*x269 + 19.125*x270 + x271*x67 - x271*x88 + x272*x67 - x272*x88 + x275*x48 - x275*x51 + 18.0*x276 + x279*x6 + 7.875*x280 + 10.125*x282 + 14.625*x283 + x284*x285 + x284*x48 + 33.75*x286 + x287*x6 + x288*x289 + x290*x291 - x290*x298 + x291*x292 - x292*x298 + x294*x6 + 33.75*x297 + x299*x300 + x301*x303 + x301*x306 - x303*x309 + x304*x305 - x306*x309
    x311 = x59/(Rm**2*Rp**2)
    x312 = 4.5*x263
    x313 = Lm*x312
    x314 = 13.5*x267
    x315 = x114*x293
    x316 = 22.5*x114
    x317 = 31.5*x8
    x318 = 2.25*x267
    x319 = x166*x264
    x320 = 4.5*x319
    x321 = 18.0*x270
    x322 = 18.0*x41
    x323 = 3.375*x295
    x324 = 18.0*a
    x325 = 13.5*x3
    x326 = x124*x325
    x327 = x257*x45
    x328 = x259*x327
    x329 = 3.0*x230*x45
    x330 = 3.0*x153*x40
    x331 = 6.0*x253*x84
    x332 = x257*x68
    x333 = 9.0*x332
    x334 = s*x333
    x335 = Am*x312
    x336 = Ap*x312
    x337 = y**7
    x338 = x111*x293
    x339 = x316*x5
    x340 = x111*x317
    x341 = x265*x53
    x342 = x318*x50
    x343 = x253*y
    x344 = s*x321
    x345 = x173*x322
    x346 = x11*x324
    x347 = 36.0*Am
    x348 = x122*x124
    x349 = 36.0*Ap
    x350 = x325*x48
    x351 = Am*x314 + Am*x315 + Am*x320 + Am*x338 + Am*x339 + Am*x340 + Am*x341 + Am*x342 + Am*x344 + Am*x345 + Ap*x314 + Ap*x315 + Ap*x320 + Ap*x338 + Ap*x339 + Ap*x340 + Ap*x341 + Ap*x342 + Ap*x344 + Ap*x345 - 12.375*a*x111*y - 7.875*a*x145 - 20.25*a*x45*x5 - 6.75*s*x145*xi - 6.75*s*x147*y - 4.5*s*x40*x41*y - 13.5*s*x40*x45 - 13.5*s*x41*x45*xi - 3.375*x111*x3*x45 - 3.375*x145*x3*x5 + x148*x350 + x149*x350 + x156*x326 + x156*x346 + x159*x326 + x159*x346 + x165*x323 + x168*x323 + x194*x335 + x194*x336 - 9.0*x257*x45 - 1.125*x267*x3*y + x271*x70 + x271*x71 + x272*x70 + x272*x71 - 1.125*x3*x337 + x302*x343 + x333 + x335 + x336 + x347*x348 + x348*x349
    x352 = 9.0*x263
    x353 = Am*x352
    x354 = Ap*x352
    x355 = 27.0*x267
    x356 = x114*x322
    x357 = 45.0*x5
    x358 = 63.0*x8
    x359 = 4.5*x267*x50
    x360 = 9.0*x319
    x361 = s*x270
    x362 = x173*x41
    x363 = 6.75*x295
    x364 = 36.0*a*x11
    x365 = 72.0*x348
    x366 = 27.0*x3
    x367 = x124*x366
    x368 = x366*x48
    f[0, 0] = x59*(x0 + x10 + x13 - x14 - x16 - x17 - x19 + x2 - x21 - x24 - x25 + x58 + x7)
    f[0, 1] = x59*(0.1875*Am*x42*x45 + 0.1875*Ap*x42*x45 - 0.03125*Lm*a*s + 0.03125*Lm*a + 0.28125*Lm*s*x3*x5 + 0.09375*Lm*x40*x42 - Lm*x48*x64 + 0.15625*Lm*xi + 0.03125*Lp*a*s + 0.03125*Lp*a + 0.28125*Lp*s*x3*x8 + 0.28125*Lp*x42*x8*xi - 0.09375*Lp*x43 + 0.125*a - s*x15*x64 + 0.75*s*xi + 0.375*x3*x5 - x60*x8 - x61*xi - x62*y - x63*y - x65*x67 - x68*x70 - x68*x71 - x74*xi - x75*xi)
    f[0, 2] = x59*(-x0 - x10 - x13 + x14 + x16 + x17 + x19 - x2 + x21 + x24 + x25 + x58 - x7)
    f[1, 0] = x59*(-x18 - x20 + x77 - x78 - x80 - x81 + x89)
    f[1, 1] = x59*(0.5625*Am*x42*x8 - Am*x90 - Am*x92 + 0.5625*Ap*x42*x8 - Ap*x90 - Ap*x92 + 0.5625*Lp*s*x3*y + 0.5625*Lp*x42*xi*y - 1.125*x22 - x54*x82 - x62 - x63 - x84*x91)
    f[1, 2] = x59*(x18 + x20 - x77 + x78 + x80 + x81 + x89)
    f[2, 0] = x59*(-x100 - x101 + x103 + x94 - x95 + x97 - x98 + 0.375)
    f[2, 1] = x59*(-Am*x106 - Ap*x106 + 0.5625*Lm*s*x3*xi + 0.28125*Lm*x42*x5 - 0.28125*Lm*x50 + 0.15625*Lm + 0.28125*Lp*x42*x8 - Lp*x86 + 1.125*s - x15*x66 + 1.125*x3*xi - x61 - x74 - x75)
    f[2, 2] = x59*(x100 + x101 + x103 - x94 + x95 - x97 + x98 - 0.375)
    f[3, 0] = x177*(x107 + x108 + x110 - x113 - x115 - x117 - x118 - x121 - x123 - x125 - x127 - x128 - x129 + x132 - x135 - x137 - x138 - x139 + x176)
    f[3, 1] = x177*(-Am*x180 - Am*x184 - Am*x189 - Am*x198 - Ap*x180 - Ap*x184 - Ap*x189 - Ap*x198 - Lm*x181 - Lm*x183 + 2.25*Lp*s*x3*x40*y + 2.25*Lp*s*x3*x45*xi + 0.5625*Lp*x111*x42*y + 0.5625*Lp*x145*x42 + 1.125*Lp*x42*x45*x5 + 2.25*Lp*x45 + 2.25*Lp*x5*y + 1.25*a*xi*y + 1.25*s*x41*y - x119*x191 - x119*x192 - x122*x191 - x122*x192 - x141*x53 - x148*x182 - x149*x182 - x154*x69*y - x156*x185 - x158*x186 - x158*x187 - x159*x185 - x165*x193 - x168*x193 - x179*x194 - x179 - x195*x196)
    f[3, 2] = x177*(-x107 - x108 - x110 + x113 + x115 + x117 + x118 + x121 + x123 + x125 + x127 + x128 + x129 - x132 + x135 + x137 + x138 + x139 + x176)
    f[4, 0] = x177*(x199 + x200 + x201 + x203 + x205 + x207 + x208 + x210 + x212 + x213 - x214 - x215 - x217 - x218 - x219 + x220 + x222 - x223 - x224 + x244)
    f[4, 1] = x177*(-Am*x248 - Am*x249 - Ap*x248 - Ap*x249 + 2.25*Lm*a*s*x5 + 2.25*Lm*a*s*x8 + 2.8125*Lm*s*x111*x3 + 0.5625*Lm*s*x114*x3 + 3.375*Lm*s*x3*x5*x8 + 0.5625*Lm*x114*x42*xi + 0.5625*Lm*x147*x42 + 1.125*Lm*x40*x42*x8 + 4.5*Lm*x40 + 4.5*Lm*x8*xi - Lp*x114*x157 - Lp*x142 - Lp*x147*x69 - Lp*x151 - Lp*x158*x69 + 9.625*a*x5 + 8.375*a*x8 + 9.0*s*x40 + 1.25*s*x41*xi + 9.0*s*x8*xi - x105*x165 - x105*x168 + 2.25*x111*x3 + 2.25*x114*x3 - x143*x196 - x144*x196 - x161*x236 - x162*x216 - x162*x250 - x167*x169 - x172*x242 - x194*x246 - x194*x247 - x225*x53 - x226*x53 - x246 - x247 + 4.5*x3*x5*x8)
    f[4, 2] = x177*(-x199 - x200 - x201 - x203 - x205 - x207 - x208 - x210 - x212 - x213 + x214 + x215 + x217 + x218 + x219 - x220 - x222 + x223 + x224 + x244)
    f[5, 0] = x311*(x251 + x252 + x255 - x256 + x260 - x261 - x262 + x310)
    f[5, 1] = x311*(-Lm*x307*x325 - Lm*x314 - Lm*x315 - Lm*x320 + 18.0*Lp*a*s*x114*xi + 18.0*Lp*a*s*x147 + 36.0*Lp*a*s*x40*x8 + 13.5*Lp*s*x114*x3*x40 + 13.5*Lp*s*x147*x3*x8 + 4.5*Lp*s*x263*x3*xi + 4.5*Lp*s*x264*x3 + 3.375*Lp*x111*x114*x42 + 9.0*Lp*x111*x41 + 31.5*Lp*x111*x8 + 9.0*Lp*x114*x41 + 22.5*Lp*x114*x5 + 2.25*Lp*x263*x42*x5 + 4.5*Lp*x263 + 2.25*Lp*x267*x42*x8 + 13.5*Lp*x267 + 0.5625*Lp*x271*x42 + 0.5625*Lp*x272*x42 + 18.0*Lp*x41*x5*x8 + 5.0*s*x253*x5 - s*x265 - 38.75*x111*x285 - 33.75*x111*x48 - x114*x299*x324 - 36.0*x124*x305 - x154*x293 - x154*x317 - x154*x323 - x194*x313 - x197*x277 - x197*x278 - 5.0*x253*x48 - 8.0*x258 - x265*x85 - x266*x53 - 15.75*x269 - 39.25*x270 - x271*x91 - x272*x91 - 28.0*x276 - 12.75*x280 - 20.25*x282 - 26.25*x283 - 65.5*x286 - x288*x318 - x290 - x291*x321 - 61.5*x297 - x301*x326 - x313 - x316*x6 - x322*x6*x8)
    f[5, 2] = x311*(-x251 - x252 - x255 + x256 - x260 + x261 + x262 + x310)
    f[6, 0] = x311*(-x328 - x329 + x330 + x331 + x334 - x351)
    f[6, 1] = x311*(Am*x355 + Am*x356 + Am*x359 + Am*x360 + Am*x365 + Ap*x355 + Ap*x356 + Ap*x359 + Ap*x360 + Ap*x365 - 14.75*a*x145 - 13.5*s*x147*y - 13.5*x11*x145 + 10.0*x11*x343 - 31.0*x11*x41*x45 - 27.75*x111*x131 - x111*x197*x45 - 42.5*x119*x45 - 21.0*x124*x153 - 27.0*x124*x45 - x145*x197*x5 + x148*x368 + x149*x368 + x156*x357 + x156*x364 + x156*x367 + x159*x357 + x159*x364 + x159*x367 + x165*x322 + x165*x358 + x165*x363 + x168*x322 + x168*x358 + x168*x363 + x186*x271 + x186*x272 + x187*x271 + x187*x272 + x194*x353 + x194*x354 - x22*x318 - x231*x337 - 18.0*x327 + 2.0*x332 + x335*x53 + x336*x53 + x347*x361 + x347*x362 + x349*x361 + x349*x362 + x353 + x354)
    f[6, 2] = x311*(x328 + x329 - x330 - x331 - x334 - x351)


@njit(cache=True, error_model="numpy")
def _linear_force_g(x, y, w, mu, nu, g):
    """Fill g (2 ramps, 2 force directions, 5 = [ux, uy, sxx, syy, sxy]) in the local
        frame at local (x, y). Generated from the closures of the old
        get_displacement_stress_kernel_force_planestrain (common subexpressions only)."""
    RFp = (w + x) ** 2 + y**2
    RFm = (w - x) ** 2 + y**2
    LFp = np.log(RFp)
    LFm = np.log(RFm)
    Tp = np.arctan((w + x) / y)
    Tm = np.arctan((w - x) / y)
    x0 = y**2
    x1 = 5*x0
    x2 = w + x
    x3 = x2**2
    x4 = 3*x3
    x5 = 4*nu
    x6 = -x4 + x5*(x2 - y)*(x2 + y)
    x7 = x0*(x5 - 5)
    x8 = -x
    x9 = w + x8
    x10 = 3*w
    x11 = x + x10
    x12 = x5 - 3
    x13 = x*x12
    x14 = nu - 1
    x15 = 8*w
    x16 = x14*x15
    x17 = LFm*x9
    x18 = Tm*x14
    x19 = x2*y
    x20 = 0.25*x19
    x21 = Tp*x14
    x22 = 1/mu
    x23 = 1/x14
    x24 = 1/np.pi
    x25 = 1/w
    x26 = x23*x24*x25
    x27 = x22*x26
    x28 = 4*w
    x29 = LFm - LFp
    x30 = x2*x29
    x31 = Tm + Tp
    x32 = 2*y
    x33 = x31*x32
    x34 = x27*y
    x35 = 0.03125*x34*(x28 + x30 - x33)
    x36 = -w*x5
    x37 = nu - 2
    x38 = x32*x37
    x39 = x9**2
    x40 = 4*x0
    x41 = 1/RFm
    x42 = 2*w
    x43 = x41*x42
    x44 = 2*nu
    x45 = x44 - 3
    x46 = 0.125*x26
    x47 = -nu*x15
    x48 = x44 - 1
    x49 = x30*x48
    x50 = 4*y
    x51 = 2*x0
    x52 = 0.0625*x26
    x53 = x29*x45
    x54 = x53*y
    x55 = x9*y
    x56 = x43*x55
    x57 = 2*x2
    x58 = x18*x57 + x21*x57
    x59 = x0*(x5 - 1)
    x60 = x12*x28
    x61 = 8*x19*x48
    x62 = x2*x44
    x63 = x29*y
    x64 = x63*(x44 + 1)
    x65 = -x56
    x66 = 0.5*x48*x63
    x67 = x44*y
    x68 = x2*(x10 + x8)
    x69 = -x5*(x0 + x68) + 3*x68
    x70 = LFm*x12*x39
    x71 = -LFp*x9 - 4*w + x17 + x33
    x72 = (1/32)*x34
    x73 = x37*x50
    x74 = -x44*(x0 + x3)
    x75 = 1/RFp
    x76 = x28*x75
    x77 = x42*x75
    x78 = x48*x9
    x79 = x29*x78
    x80 = 4*x9
    x81 = x19*x76
    x82 = x5*x9
    x83 = 2*x9
    x84 = x5*y
    g[0, 0, 0] = x27*(0.015625*LFm*(3*x11*x9 - x7) - 0.015625*LFp*(x1 + x6) - 0.0625*nu*x11*x17 + 0.0625*w*(x13 + x16) - x18*x20 - x20*x21)
    g[0, 0, 1] = x35
    g[0, 0, 2] = -x46*(Tm*x38 + Tp*x38 - 0.5*x30*x45 + x36 + x43*(3*x39 + x40))
    g[0, 0, 3] = x52*(x18*x50 + x21*x50 + x28*x41*(x39 + x51) + x47 - x49)
    g[0, 0, 4] = -x46*(0.5*x54 + x56 + x58)
    g[0, 1, 0] = x35
    g[0, 1, 1] = -0.015625*x27*(-LFm*(-3*w**2*x12 + 2*w*x*x12 + x**2*x12 - x59) + LFp*(x0 + x6) + Tm*x61 + Tp*x61 - x60*(x + x42))
    g[0, 1, 2] = x46*(Tm*x62 + Tp*x62 + 0.5*x64 + x65)
    g[0, 1, 3] = -x46*(x58 + x65 + x66)
    g[0, 1, 4] = -x46*(Tm*x67 + Tp*x67 + x36 + x39*x43 - 0.5*x49)
    g[1, 0, 0] = (1/64)*x22*x24*x25*(-x23*x70 + x23*(LFm*x7 + LFp*(x1 + x69) + x28*(-x13 + x16)) - 16*x31*x55)
    g[1, 0, 1] = x71*x72
    g[1, 0, 2] = x52*(Tm*x73 + Tp*x73 + x53*x9 + x76*(x4 + x40 + x74))
    g[1, 0, 3] = -x46*(x18*x32 + x21*x32 + x77*(x3 + x51 + x74) + 0.5*x79)
    g[1, 0, 4] = -x52*(x18*x80 + x21*x80 - x54 + x81)
    g[1, 1, 0] = x71*x72
    g[1, 1, 1] = (1/64)*x27*(LFm*x59 + LFp*(x0 + x69) - 8*x31*x78*y + x60*(x42 + x8) - x70)
    g[1, 1, 2] = x52*(Tm*x82 + Tp*x82 - x64 - x81)
    g[1, 1, 3] = -x46*(x18*x83 - x19*x77 + x21*x83 - x66)
    g[1, 1, 4] = x52*(Tm*x84 + Tp*x84 + x3*x76 + x47 + x79)


@njit(cache=True, error_model="numpy")
def _slip_local(f, node, y, mu, nu, cx, cy):
    """Local displacement and stress of quadratic node `node` for slip (cx, cy)
    along (t, n); generalization of Crouch & Starfield (sign as in
    bemcs.f_slip_to_displacement_stress)."""
    cx = -cx
    cy = -cy
    f1 = f[1, node]
    f2 = f[2, node]
    f3 = f[3, node]
    f4 = f[4, node]
    f5 = f[5, node]
    f6 = f[6, node]
    ux = cx * (2 * (1 - nu) * f1 - y * f4) + cy * (-(1 - 2 * nu) * f2 - y * f3)
    uy = cx * ((1 - 2 * nu) * f2 - y * f3) + cy * (2 * (1 - nu) * f1 + y * f4)
    sxx = 2 * mu * (cx * (2 * f3 + y * f5) + cy * (-f4 + y * f6))
    syy = 2 * mu * (cx * (-y * f5) + cy * (-f4 - y * f6))
    sxy = 2 * mu * (cx * (-f4 + y * f6) + cy * (-y * f5))
    return ux, uy, sxx, syy, sxy


@njit(cache=True, error_model="numpy")
def _to_global(R, ux, uy, sxx, syy, sxy):
    """Rotate a local displacement vector and stress tensor to global: u = R u', S = R S' R^T."""
    gx = R[0, 0] * ux + R[0, 1] * uy
    gy = R[1, 0] * ux + R[1, 1] * uy
    gxx = R[0, 0] ** 2 * sxx + 2 * R[0, 0] * R[0, 1] * sxy + R[0, 1] ** 2 * syy
    gyy = R[1, 0] ** 2 * sxx + 2 * R[1, 0] * R[1, 1] * sxy + R[1, 1] ** 2 * syy
    gxy = (
        R[0, 0] * R[1, 0] * sxx
        + (R[0, 0] * R[1, 1] + R[0, 1] * R[1, 0]) * sxy
        + R[0, 1] * R[1, 1] * syy
    )
    return gx, gy, gxx, gyy, gxy


# Points with |y_local| <= _ON_ELEMENT * (half_length + |xi|) are on the element (to
# rounding): the kernels return the average of the two sides there. The slip
# formula does this for y == 0 exactly; the force formula is evaluated at
# y = +-_TINY and averaged.
_ON_ELEMENT = 1e-12
_TINY = 1e-300


@njit(parallel=True, cache=True, error_model="numpy")
def _slip_planestrain_numba(
    x_obs, y_obs, x_centers, y_centers, x1, y1, x2, y2, half_lengths, rot_mats, mu, nu
):
    """Quadratic slip, plane strain: U (n_obs, 3 n_els, 2, 2), S (n_obs, 3 n_els, 3, 2)."""
    n_obs = x_obs.shape[0]
    n_els = x_centers.shape[0]
    U = np.zeros((n_obs, 3 * n_els, 2, 2))
    S = np.zeros((n_obs, 3 * n_els, 3, 2))
    for i in prange(n_obs):
        f = np.empty((7, 3))
        for e in range(n_els):
            R = rot_mats[e]
            # pick the nearest endpoint (x = s a) and work relative to it
            xc = R[0, 0] * (x_obs[i] - x_centers[e]) + R[1, 0] * (y_obs[i] - y_centers[e])
            s = 1.0 if xc >= 0 else -1.0
            dx = x_obs[i] - (x2[e] if s > 0 else x1[e])
            dy = y_obs[i] - (y2[e] if s > 0 else y1[e])
            xi = R[0, 0] * dx + R[1, 0] * dy  # local = R^T (global - endpoint)
            yl = R[0, 1] * dx + R[1, 1] * dy
            if abs(yl) <= _ON_ELEMENT * (half_lengths[e] + abs(xi)):
                yl = 0.0  # on the element: two-sided average
            _quadratic_slip_f(xi, yl, half_lengths[e], nu, s, f)
            for node in range(3):
                for l in range(2):
                    loc = _slip_local(f, node, yl, mu, nu, 1.0 - l, float(l))
                    g = _to_global(R, loc[0], loc[1], loc[2], loc[3], loc[4])
                    j = 3 * e + node
                    U[i, j, 0, l] = g[0]
                    U[i, j, 1, l] = g[1]
                    S[i, j, 0, l] = g[2]
                    S[i, j, 1, l] = g[3]
                    S[i, j, 2, l] = g[4]
    return U, S


def _slip_planestrain_numpy(
    x_obs, y_obs, x_centers, y_centers, x1, y1, x2, y2, half_lengths, rot_mats, mu, nu
):
    """Same as _slip_planestrain_numba, vectorized over observation points."""
    n_obs, n_els = x_obs.shape[0], x_centers.shape[0]
    U = np.zeros((n_obs, 3 * n_els, 2, 2))
    S = np.zeros((n_obs, 3 * n_els, 3, 2))
    f = np.empty((7, 3, n_obs))
    with np.errstate(divide="ignore", invalid="ignore"):
        for e in range(n_els):
            R = rot_mats[e]
            xc = R[0, 0] * (x_obs - x_centers[e]) + R[1, 0] * (y_obs - y_centers[e])
            s = np.where(xc >= 0, 1.0, -1.0)
            dx = x_obs - np.where(s > 0, x2[e], x1[e])
            dy = y_obs - np.where(s > 0, y2[e], y1[e])
            xi = R[0, 0] * dx + R[1, 0] * dy
            yl = R[0, 1] * dx + R[1, 1] * dy
            yl = np.where(np.abs(yl) <= _ON_ELEMENT * (half_lengths[e] + np.abs(xi)), 0.0, yl)
            _quadratic_slip_f.py_func(xi, yl, half_lengths[e], nu, s, f)
            for node in range(3):
                for l in range(2):
                    loc = _slip_local.py_func(f, node, yl, mu, nu, 1.0 - l, float(l))
                    g = _to_global.py_func(R, *loc)
                    j = 3 * e + node
                    U[:, j, 0, l], U[:, j, 1, l] = g[0], g[1]
                    S[:, j, 0, l], S[:, j, 1, l], S[:, j, 2, l] = g[2], g[3], g[4]
    return U, S


@njit(parallel=True, cache=True, error_model="numpy")
def _force_planestrain_numba(x_obs, y_obs, x_centers, y_centers, half_lengths, rot_mats, mu, nu):
    """Linear force, plane strain: U (n_obs, 2 n_els, 2, 2), S (n_obs, 2 n_els, 3, 2)."""
    n_obs = x_obs.shape[0]
    n_els = x_centers.shape[0]
    U = np.zeros((n_obs, 2 * n_els, 2, 2))
    S = np.zeros((n_obs, 2 * n_els, 3, 2))
    for i in prange(n_obs):
        g = np.empty((2, 2, 5))
        g2 = np.empty((2, 2, 5))
        for e in range(n_els):
            R = rot_mats[e]
            dx = x_obs[i] - x_centers[e]
            dy = y_obs[i] - y_centers[e]
            xl = R[0, 0] * dx + R[1, 0] * dy
            yl = R[0, 1] * dx + R[1, 1] * dy
            if abs(yl) <= _ON_ELEMENT * (half_lengths[e] + abs(xl)):
                # on the element: average of the two sides
                _linear_force_g(xl, _TINY, half_lengths[e], mu, nu, g)
                _linear_force_g(xl, -_TINY, half_lengths[e], mu, nu, g2)
                for r in range(2):
                    for l in range(2):
                        for c in range(5):
                            g[r, l, c] = 0.5 * (g[r, l, c] + g2[r, l, c])
            else:
                _linear_force_g(xl, yl, half_lengths[e], mu, nu, g)
            for r in range(2):
                for l in range(2):
                    o = _to_global(R, g[r, l, 0], g[r, l, 1], g[r, l, 2], g[r, l, 3], g[r, l, 4])
                    j = 2 * e + r
                    U[i, j, 0, l] = o[0]
                    U[i, j, 1, l] = o[1]
                    S[i, j, 0, l] = o[2]
                    S[i, j, 1, l] = o[3]
                    S[i, j, 2, l] = o[4]
    return U, S


def _force_planestrain_numpy(x_obs, y_obs, x_centers, y_centers, half_lengths, rot_mats, mu, nu):
    """Same as _force_planestrain_numba, vectorized over observation points."""
    n_obs, n_els = x_obs.shape[0], x_centers.shape[0]
    U = np.zeros((n_obs, 2 * n_els, 2, 2))
    S = np.zeros((n_obs, 2 * n_els, 3, 2))
    g = np.empty((2, 2, 5, n_obs))
    with np.errstate(divide="ignore", invalid="ignore"):
        for e in range(n_els):
            R = rot_mats[e]
            dx, dy = x_obs - x_centers[e], y_obs - y_centers[e]
            xl = R[0, 0] * dx + R[1, 0] * dy
            yl = R[0, 1] * dx + R[1, 1] * dy
            on = np.abs(yl) <= _ON_ELEMENT * (half_lengths[e] + np.abs(xl))
            _linear_force_g.py_func(xl, np.where(on, _TINY, yl), half_lengths[e], mu, nu, g)
            if on.any():  # on the element: average of the two sides
                g2 = np.empty_like(g)
                _linear_force_g.py_func(xl, np.where(on, -_TINY, yl), half_lengths[e], mu, nu, g2)
                g[..., on] = 0.5 * (g[..., on] + g2[..., on])
            for r in range(2):
                for l in range(2):
                    o = _to_global.py_func(R, *g[r, l])
                    j = 2 * e + r
                    U[:, j, 0, l], U[:, j, 1, l] = o[0], o[1]
                    S[:, j, 0, l], S[:, j, 1, l], S[:, j, 2, l] = o[2], o[3], o[4]
    return U, S


@njit(cache=True, error_model="numpy")
def _antiplane_slip_point(x, y, w):
    """Quadratic antiplane slip at local (x, y) of an element with half-length w:
    displacement (u1, u2, u3) and displacement gradients (ux1..3, uy1..3) of the
    3 nodal shape functions (stress = mu * gradient). Scalars or arrays."""
    pi = np.pi
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

    return u1, u2, u3, ux1, ux2, ux3, uy1, uy2, uy3


@njit(parallel=True, cache=True, error_model="numpy")
def _slip_antiplane_numba(x_obs, y_obs, x_centers, y_centers, half_lengths, rot_mats, mu):
    """Quadratic slip, antiplane: U (n_obs, 3 n_els, 1), S (n_obs, 3 n_els, 2)."""
    n_obs = x_obs.shape[0]
    n_els = x_centers.shape[0]
    U = np.zeros((n_obs, 3 * n_els, 1))
    S = np.zeros((n_obs, 3 * n_els, 2))
    for i in prange(n_obs):
        for e in range(n_els):
            R = rot_mats[e]
            dx = x_obs[i] - x_centers[e]
            dy = y_obs[i] - y_centers[e]
            x = R[0, 0] * dx + R[1, 0] * dy
            y = R[0, 1] * dx + R[1, 1] * dy
            v = _antiplane_slip_point(x, y, half_lengths[e])
            for k in range(3):
                gx = mu * v[3 + k]
                gy = mu * v[6 + k]
                U[i, 3 * e + k, 0] = v[k]
                S[i, 3 * e + k, 0] = R[0, 0] * gx + R[0, 1] * gy
                S[i, 3 * e + k, 1] = R[1, 0] * gx + R[1, 1] * gy
    return U, S


def _slip_antiplane_numpy(x_obs, y_obs, x_centers, y_centers, half_lengths, rot_mats, mu):
    """Same as _slip_antiplane_numba, vectorized over observation points."""
    n_obs, n_els = x_obs.shape[0], x_centers.shape[0]
    U = np.zeros((n_obs, 3 * n_els, 1))
    S = np.zeros((n_obs, 3 * n_els, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        for e in range(n_els):
            R = rot_mats[e]
            dx, dy = x_obs - x_centers[e], y_obs - y_centers[e]
            x = R[0, 0] * dx + R[1, 0] * dy
            y = R[0, 1] * dx + R[1, 1] * dy
            v = _antiplane_slip_point.py_func(x, y, half_lengths[e])
            for k in range(3):
                gx, gy = mu * v[3 + k], mu * v[6 + k]
                U[:, 3 * e + k, 0] = v[k]
                S[:, 3 * e + k, 0] = R[0, 0] * gx + R[0, 1] * gy
                S[:, 3 * e + k, 1] = R[1, 0] * gx + R[1, 1] * gy
    return U, S


# =============================================================================
# Kernel API in three levels (see tasks.md section 7)
#   level 1  Mesh + get_kernels              (what users work with)
#   level 2  get_slip_kernels, get_force_kernels
#   level 3  element_kernels_slip/force x basis_transform
# =============================================================================

_GEOMETRIES = ("antiplane", "planestrain")
_KINDS = ("slip", "force")  # block order along the source axis at level 1
_ELEMENT_BASIS = {"slip": "quadratic", "force": "linear"}
_NODES = {"slip": 3, "force": 2}  # nodes per element of the element basis
_BASES = {
    "slip": ("quadratic", "constant", "hat"),
    "force": ("linear", "constant", "trapezoid"),
}
_PATCH_BASES = ("hat", "trapezoid")
_HAS_NUMBA_KERNEL = {
    ("antiplane", "slip"): True,
    ("antiplane", "force"): True,
    ("planestrain", "slip"): True,
    ("planestrain", "force"): True,
}
_COMPONENTS = {
    "antiplane": (("uz",), ("sxz", "syz"), ()),
    "planestrain": (("ux", "uy"), ("sxx", "syy", "sxy"), ("t", "n")),
}


@dataclass
class KernelInfo:
    """Describes the arrays returned by the kernel functions.

    Axes of U and S: (observation, source coefficient, component[, direction]).
    The direction axis exists only in plane strain: "t" is along the element
    (shear slip or tangential force) and "n" is the element normal, both in the
    element-local frame. Components are in the global x-y frame.

    Per source coefficient j (axis 1): `src_label[j]` is "s" (slip) or "f"
    (force); `src_index[j]` is the global element index, or the connectivity
    row for patch bases; `src_node[j]` is the node of the element basis, or -1
    for constant and patch bases. At the mesh level, slip comes before force.
    """

    geometry: str
    basis: dict
    mu: float
    nu: object
    backend: str
    n_obs: int
    displacement_components: tuple
    stress_components: tuple
    directions: tuple
    src_label: np.ndarray = field(repr=False)
    src_index: np.ndarray = field(repr=False)
    src_node: np.ndarray = field(repr=False)

    @property
    def n_src(self):
        return len(self.src_label)

    def _shape(self, components):
        extra = (len(self.directions),) if self.directions else ()
        return (self.n_obs, self.n_src, len(components)) + extra

    @property
    def u_shape(self):
        return self._shape(self.displacement_components)

    @property
    def s_shape(self):
        return self._shape(self.stress_components)

    def __str__(self):
        dirs = f", direction{list(self.directions)}" if self.directions else ""
        nu = f", nu={self.nu}" if self.nu is not None else ""
        lines = [
            f"KernelInfo: {self.geometry}, basis={self.basis}, backend={self.backend}, "
            f"mu={self.mu}{nu}",
            f"  U {self.u_shape}: (obs, src, component{list(self.displacement_components)}{dirs})",
            f"  S {self.s_shape}: (obs, src, component{list(self.stress_components)}{dirs})",
        ]
        for kind in _KINDS:
            sel = self.src_label == kind[0]
            if not sel.any():
                continue
            first = int(np.flatnonzero(sel)[0])
            unit = "patches" if self.basis[kind] in _PATCH_BASES else "elements"
            lines.append(
                f"  src {first}:{first + int(sel.sum())} -> {kind}, {self.basis[kind]} basis "
                f"on {len(np.unique(self.src_index[sel]))} {unit}"
            )
        if self.directions:
            lines.append(
                "  directions are element-local (t = along element, n = normal); a system "
                "block S[:, :, k, :].reshape(n_obs, -1) has columns [t0, n0, t1, n1, ...]"
            )
        return "\n".join(lines)


# ----------------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------------


def _backend(fn, use_numba):
    """The numba-jitted kernel, or its pure-Python source (same code, no JIT)."""
    return fn if use_numba else fn.py_func


def _els_arrays(els):
    return tuple(
        np.ascontiguousarray(getattr(els, name), dtype=float)
        for name in ("x_centers", "y_centers", "half_lengths", "rot_mats", "rot_mats_inv")
    )


def _obs(x_obs, y_obs):
    xo = np.ascontiguousarray(x_obs, dtype=float).ravel()
    yo = np.ascontiguousarray(y_obs, dtype=float).ravel()
    if xo.shape != yo.shape:
        raise ValueError(f"x_obs and y_obs differ in size: {xo.size} vs {yo.size}")
    return xo, yo


def _check_material(geometry, mu, nu):
    if geometry not in _GEOMETRIES:
        raise ValueError(f"geometry must be one of {_GEOMETRIES}, got {geometry!r}")
    if not np.isscalar(mu) or not mu > 0:
        raise ValueError(f"mu must be a positive scalar (reference shear modulus), got {mu!r}")
    if geometry == "planestrain":
        if nu is None:
            raise TypeError("nu is required for planestrain")
        if not np.isscalar(nu) or not -1 < nu < 0.5:
            raise ValueError(f"nu must be a scalar in (-1, 0.5), got {nu!r}")


def _check_connectivity(conn, n_els):
    c = np.asarray(conn)
    if c.ndim != 2 or c.shape[1] != 3 or not np.issubdtype(c.dtype, np.integer):
        raise ValueError(
            f"connectivity must be an (n_patches, 3) int array, got shape {c.shape} {c.dtype}"
        )
    if c.size and (c.min() < 0 or c.max() >= n_els):
        raise ValueError("connectivity has element indices out of range")
    return c


def _subset_els(els, idx):
    sub = _core.initialize_els()
    for name in ("x1", "y1", "x2", "y2"):
        setattr(sub, name, np.asarray(getattr(els, name), dtype=float)[idx].copy())
    _core.standardize_els_geometry(sub, reorder=False)
    return sub


# ----------------------------------------------------------------------------
# level 3: element kernels and basis transforms
# ----------------------------------------------------------------------------


def element_kernels_slip(x_obs, y_obs, els, *, geometry, mu, nu=None, use_numba=False):
    """Quadratic-slip kernels, 3 nodes per element: U, S in the get_kernels layout.

    Stresses are singular at element end points; the singular parts cancel
    between neighbouring elements only for co-planar elements with continuous
    slip and slip gradient (see `get_kernels`, Notes).
    """
    _check_material(geometry, mu, nu)
    xo, yo = _obs(x_obs, y_obs)
    if geometry == "antiplane":
        driver = _slip_antiplane_numba if use_numba else _slip_antiplane_numpy
        xc, yc, hl, rot, _ = _els_arrays(els)
        return driver(xo, yo, xc, yc, hl, rot, float(mu))
    driver = _slip_planestrain_numba if use_numba else _slip_planestrain_numpy
    xc, yc, hl, rot, _ = _els_arrays(els)
    x1, y1, x2, y2 = (np.ascontiguousarray(getattr(els, k), dtype=float) for k in ("x1", "y1", "x2", "y2"))
    return driver(xo, yo, xc, yc, x1, y1, x2, y2, hl, rot, float(mu), float(nu))


def element_kernels_force(x_obs, y_obs, els, *, geometry, mu, nu=None, use_numba=False):
    """Linear-force kernels, 2 ramps per element: U, S in the get_kernels layout.

    Ramp 0 is (w+s)/2w and ramp 1 is (w-s)/2w, with s in [-w, w] along the element.
    """
    _check_material(geometry, mu, nu)
    xo, yo = _obs(x_obs, y_obs)
    n_obs = len(xo)
    if geometry == "antiplane":
        f = _backend(get_displacement_stress_kernel_force_antiplane, use_numba)
        sxz, syz, u = f(xo, yo, *_els_arrays(els), float(mu))  # (n_obs, ramp, n_els)

        def r(K):
            return K.transpose(0, 2, 1).reshape(n_obs, -1)

        return r(u)[:, :, None], np.stack((r(sxz), r(syz)), axis=2)
    driver = _force_planestrain_numba if use_numba else _force_planestrain_numpy
    xc, yc, hl, rot, _ = _els_arrays(els)
    return driver(xo, yo, xc, yc, hl, rot, float(mu), float(nu))


def _per_element_transform(n_els, n_nodes, weights, geometry):
    """Block-diagonal transform: per element, basis = weights . element nodes."""
    w = np.asarray(weights, dtype=float).reshape(n_nodes, -1)  # (node, n_basis_per_element)
    T = np.kron(np.eye(n_els), w)  # (n_els * n_nodes, n_els * n_basis)
    if geometry == "antiplane":
        return T
    return np.einsum("sb,lm->slbm", T, np.eye(2))


def hat_coefficients(els, connectivity, *, geometry):
    """Transform from quadratic-slip element DOFs to hat patches (one per row of connectivity).

    Each hat has unit slip (t or n in plane strain, element-local frame) at the
    central node of its middle element, zero slip and slip gradient at the open
    ends, and continuous slip and slip gradient (global frame) at the two
    internal nodes. Returns (3 n_els, n_patches) in antiplane and
    (3 n_els, 2, n_patches, 2) [dof, direction_in, patch, direction_out] in plane strain.
    """
    if geometry not in _GEOMETRIES:
        raise ValueError(f"geometry must be one of {_GEOMETRIES}, got {geometry!r}")
    conn = _check_connectivity(connectivity, len(els.x_centers))
    n_els, n_p = len(els.x_centers), len(conn)
    pe = 1 if geometry == "antiplane" else 2
    n_unknowns = 9 * pe
    T = np.zeros((n_els * 3, pe, n_p, pe))
    for p in range(n_p):
        sub = _subset_els(els, conn[p])
        index_open, index_overlap, _ = _core.label_nodes(sub, verbose=False)
        m_slip, m_grad = slip_node_matrices(sub, geometry=geometry, reference="global")
        m_local, _ = slip_node_matrices(sub, geometry=geometry, reference="local")
        rows = [m_local[pe * 4 + c] for c in range(pe)]  # centre of the middle element
        for node in index_open:
            n = int(np.abs(node).item())
            rows += [M[pe * n + c] for M in (m_slip, m_grad) for c in range(pe)]
        for a, b in index_overlap:
            if a != 0 and b != 0:
                sa, sb = np.sign(a), np.sign(b)
            elif a == 0:
                sa, sb = 1, -1
            else:
                sa, sb = -1, 1
            rows += [
                sa * M[pe * abs(a) + c] + sb * M[pe * abs(b) + c]
                for M in (m_slip, m_grad)
                for c in range(pe)
            ]
        A = np.array(rows)
        if A.shape != (n_unknowns, n_unknowns):
            raise ValueError(
                f"patch {p} ({conn[p].tolist()}) is not a chain of 3 connected elements"
            )
        rhs = np.zeros((n_unknowns, pe))
        rhs[:pe, :] = np.eye(pe)  # unit slip along each local direction
        coefs = np.linalg.solve(A, rhs).reshape(3, 3, pe, pe)  # (element, node, l_in, m_out)
        for local, e in enumerate(conn[p]):
            T[3 * e : 3 * e + 3, :, p, :] = coefs[local]
    return T[:, 0, :, 0] if geometry == "antiplane" else T


def trapezoid_coefficients(els, connectivity, *, geometry):
    """Transform from linear-force element DOFs to trapezoid patches.

    Returns (2 n_els, n_patches) in antiplane and (2 n_els, 2, n_patches, 2) in plane strain.
    """
    conn = _check_connectivity(connectivity, len(els.x_centers))
    n_els, n_p = len(els.x_centers), len(conn)
    if geometry == "planestrain":
        fs, fn = compute_coefs_trapezoidalforce_planestrain(els, conn)  # (e, l, ramp, p)
        c = np.stack((fs, fn), axis=-1)
        return c.transpose(0, 2, 1, 3, 4).reshape(n_els * 2, 2, n_p, 2)
    if geometry != "antiplane":
        raise ValueError(f"geometry must be one of {_GEOMETRIES}, got {geometry!r}")
    trapcoefs = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    C = np.zeros((n_els, 2, n_p))
    for p in range(n_p):
        for m in range(3):
            C[conn[p, m], :, p] = trapcoefs[m, :]
    return C.reshape(n_els * 2, n_p)


def basis_transform(els, *, kind, basis, geometry, connectivity=None):
    """Transform T from element-basis DOFs to the requested basis (see tasks.md 7.2)."""
    n_els = len(els.x_centers)
    n_nodes = _NODES[kind]
    if basis == _ELEMENT_BASIS[kind]:
        return _per_element_transform(n_els, n_nodes, np.eye(n_nodes), geometry)
    if basis == "constant":
        return _per_element_transform(n_els, n_nodes, np.ones(n_nodes), geometry)
    if basis == "hat":
        return hat_coefficients(els, connectivity, geometry=geometry)
    return trapezoid_coefficients(els, connectivity, geometry=geometry)


def _apply_transform(U, S, T, geometry):
    if geometry == "antiplane":
        return np.einsum("osc,sb->obc", U, T), np.einsum("osc,sb->obc", S, T)
    return np.einsum("oscl,slbm->obcm", U, T), np.einsum("oscl,slbm->obcm", S, T)


# ----------------------------------------------------------------------------
# level 3: node operators, continuity rows, tractions, values along elements
#
# Column convention (same as the kernels): element-basis coefficient j =
# n_nodes * element + node; in plane strain each j has a direction l in (t, n)
# and the flattened column is 2 j + l (interleaved [t0, n0, t1, n1, ...]).
# ----------------------------------------------------------------------------

_FORCE_AT_NODES = np.array([[0.0, 1.0], [1.0, 0.0]])  # rows: start, end node; cols: ramp 0, 1


def _unit_vectors(els, e):
    """[[x_shear, x_normal], [y_shear, y_normal]]: local (t, n) -> global (x, y)."""
    return np.array(
        [[els.x_shears[e], els.x_normals[e]], [els.y_shears[e], els.y_normals[e]]]
    )


def _node_matrix(els, per_element, geometry, reference):
    """Assemble block-diagonal node matrices from per-element (rows=nodes, cols=basis) blocks."""
    n_els = len(els.x_centers)
    blocks = [per_element(e) for e in range(n_els)]
    n_rows, n_cols = blocks[0].shape
    if geometry == "antiplane":
        M = np.zeros((n_els * n_rows, n_els * n_cols))
        for e, B in enumerate(blocks):
            M[e * n_rows : (e + 1) * n_rows, e * n_cols : (e + 1) * n_cols] = B
        return M
    if reference not in ("global", "local"):
        raise ValueError(f"reference must be 'global' or 'local', got {reference!r}")
    M = np.zeros((2 * n_els * n_rows, 2 * n_els * n_cols))
    for e, B in enumerate(blocks):
        Uv = _unit_vectors(els, e) if reference == "global" else np.eye(2)
        M[2 * e * n_rows : 2 * (e + 1) * n_rows, 2 * e * n_cols : 2 * (e + 1) * n_cols] = np.kron(B, Uv)
    return M


def slip_node_matrices(els, *, geometry, reference="global"):
    """Slip and slip gradient (d/ds along each element) at the 3 nodes of every element.

    Rows: node 3 e + k (k = 0, 1, 2 at s = -w, 0, w); in plane strain each row has a
    component c (x, y if reference="global"; t, n if "local") at row 2 (3 e + k) + c.
    Columns: quadratic-slip coefficients in the kernel convention (interleaved in
    plane strain).
    """
    def nodes(e):
        w = els.half_lengths[e]
        return np.array([-w, 0.0, w]), w

    def slip(e):
        x, w = nodes(e)
        return _core.slip_functions(x, w)

    def grad(e):
        x, w = nodes(e)
        return _core.slipgradient_functions(x, w)

    return (
        _node_matrix(els, slip, geometry, reference),
        _node_matrix(els, grad, geometry, reference),
    )


def force_node_matrix(els, *, geometry, reference="global"):
    """Force at the 2 end nodes of every element (row 2 e + k, k = 0 start, 1 end).

    Columns: linear-force coefficients (ramp 0 = (w+s)/2w, ramp 1 = (w-s)/2w),
    interleaved in plane strain; rows as in `slip_node_matrices`.
    """
    return _node_matrix(els, lambda e: _FORCE_AT_NODES, geometry, reference)


def continuity_rows(els, *, kind, geometry):
    """Continuity rows A (A c = 0) for element-basis coefficients of one source type.

    kind="slip" (quadratic): open nodes (fault tips) are clamped: slip = 0 and
    slip gradient = 0; overlapping nodes: slip and slip-gradient continuity;
    triple junctions: slip kinematics and slip-gradient (Jacobian) conditions.
    An element with an open node is a taper (tip) element and should get no
    boundary-condition row (see `tip_elements`). kind="force" (linear): open nodes:
    force = 0; overlapping nodes: force continuity; triple junctions are an
    error. Rows act on global-frame quantities. Returns (A, info) where info
    holds per-row arrays: node_type, quantity, node (label_nodes index), x, y.
    """
    index_open, index_overlap, index_triple = _core.label_nodes(els, verbose=False)
    pe = 1 if geometry == "antiplane" else 2  # components per node
    if kind == "slip":
        M_val, M_grad = slip_node_matrices(els, geometry=geometry)

        def row(M, node3, c):  # node3 = 3 e + k in label_nodes numbering
            return M[pe * node3 + c]

        quantities = (("slip", M_val), ("gradient", M_grad))
    elif kind == "force":
        if index_triple:
            raise ValueError("force elements cannot form triple junctions")
        M_val = force_node_matrix(els, geometry=geometry)

        def row(M, node3, c):  # label_nodes uses 3 e (start) and 3 e + 2 (end)
            e, k = divmod(int(node3), 3)
            return M[pe * (2 * e + (k // 2)) + c]

        quantities = (("force", M_val),)
    else:
        raise ValueError(f"kind must be 'slip' or 'force', got {kind!r}")

    rows, node_type, quantity, node_id = [], [], [], []

    def add(r, t, q, n):
        rows.append(r)
        node_type.append(t)
        quantity.append(q)
        node_id.append(int(n))

    for node in index_open:  # slip: clamped (value and gradient); force: value
        n = int(np.abs(node).item())
        for q, M in quantities:
            for c in range(pe):
                add(row(M, n, c), "open", q, n)

    for a, b in index_overlap:
        if a != 0 and b != 0:
            sa, sb = np.sign(a), np.sign(b)
        elif a == 0:
            sa, sb = 1, -1
        else:
            sa, sb = -1, 1
        for q, M in quantities:
            for c in range(pe):
                add(sa * row(M, abs(a), c) + sb * row(M, abs(b), c), "overlap", q, abs(a))

    for ids in index_triple:  # slip only
        ids = np.asarray(ids)
        signs = np.where(ids < 0, -1.0, 1.0)
        nodes = np.abs(ids)
        for c in range(pe):
            add(sum(sg * row(M_val, n, c) for sg, n in zip(signs, nodes)), "triple", "slip", nodes[0])
        # slip gradient along each element, projected on the x and y axes (Jacobian form)
        shear = np.array([[els.x_shears[n // 3], els.y_shears[n // 3]] for n in nodes])
        for c in range(pe):
            for axis in range(2):
                add(
                    sum(sg * row(M_grad, n, c) * shear[i, axis] for i, (sg, n) in enumerate(zip(signs, nodes))),
                    "triple", "gradient", nodes[0],
                )

    n_cols = M_val.shape[1]
    A = np.array(rows) if rows else np.zeros((0, n_cols))
    node_id = np.array(node_id, dtype=int)
    e, k = node_id // 3, node_id % 3
    s_node = np.where(k == 0, -1.0, np.where(k == 2, 1.0, 0.0))
    R = np.asarray(els.rot_mats)
    hl = np.asarray(els.half_lengths)
    x = np.asarray(els.x_centers)[e] + R[e, 0, 0] * s_node * hl[e] if len(e) else np.zeros(0)
    y = np.asarray(els.y_centers)[e] + R[e, 1, 0] * s_node * hl[e] if len(e) else np.zeros(0)
    info = {
        "node_type": np.array(node_type, dtype=object),
        "quantity": np.array(quantity, dtype=object),
        "node": node_id,
        "x": x,
        "y": y,
    }
    return A, info


def tip_elements(els):
    """Indices of elements with an open end node (fault tips).

    With clamped tips (slip and slip gradient = 0 at open nodes, tasks.md 7.4) a
    tip element is a taper element: it gets no boundary-condition row. An element
    with both end nodes open cannot be represented and raises ValueError.
    """
    index_open, _, _ = _core.label_nodes(els, verbose=False)
    elements = np.array([int(np.abs(n).item()) // 3 for n in index_open], dtype=int)
    tips, counts = np.unique(elements, return_counts=True)
    both = tips[counts > 1]
    if both.size:
        raise ValueError(
            f"element(s) {both.tolist()} have both end nodes open; a clamped-tip slip "
            "element needs at least one connected neighbour"
        )
    return tips


def traction_kernels(S, nx, ny, sx=None, sy=None):
    """Traction kernels t = sigma . n from stress kernels S (get_kernels layout).

    nx, ny: unit normal at each observation point (length n_obs). Returns
    planestrain (n_obs, n_src, 2, 2) = [tx, ty] x [t, n] and antiplane
    (n_obs, n_src, 1). If the unit tangent (sx, sy) is given (plane strain),
    the traction is returned in that local frame: [t . s, t . n].
    """
    nx = np.asarray(nx, dtype=float).ravel()[:, None]
    ny = np.asarray(ny, dtype=float).ravel()[:, None]
    if S.ndim == 3:  # antiplane: [sxz, syz]
        return (S[:, :, 0] * nx + S[:, :, 1] * ny)[:, :, None]
    nx, ny = nx[..., None], ny[..., None]
    tx = S[:, :, 0] * nx + S[:, :, 2] * ny
    ty = S[:, :, 2] * nx + S[:, :, 1] * ny
    if sx is None:
        return np.stack((tx, ty), axis=2)
    sx = np.asarray(sx, dtype=float).ravel()[:, None, None]
    sy = np.asarray(sy, dtype=float).ravel()[:, None, None]
    return np.stack((tx * sx + ty * sy, tx * nx + ty * ny), axis=2)


def kernel_matrix(K):
    """Flatten a kernel array into a system-matrix block.

    planestrain (n_obs, n_src, n_comp, 2) -> (n_obs * n_comp, 2 * n_src): rows are
    observation-major then component; columns interleaved [t0, n0, t1, n1, ...].
    antiplane (n_obs, n_src, n_comp) -> (n_obs * n_comp, n_src).
    """
    K = np.asarray(K)
    if K.ndim == 3:
        return K.transpose(0, 2, 1).reshape(K.shape[0] * K.shape[2], K.shape[1])
    if K.ndim == 4:
        n_obs, n_src, n_comp, n_dir = K.shape
        return K.transpose(0, 2, 1, 3).reshape(n_obs * n_comp, n_src * n_dir)
    raise ValueError(f"expected a 3-D or 4-D kernel array, got shape {K.shape}")


def values_on_elements(els, coefs, *, kind, geometry, n_eval=10, frame="global"):
    """Slip or force along every element from element-basis coefficients.

    coefs: flattened coefficient vector in the kernel convention (quadratic
    slip or linear force). Evaluates at n_eval points per element (midpoints of
    n_eval equal sub-intervals). Returns x, y (n_els * n_eval,) and values:
    antiplane (n,) ; plane strain (n, 2) in global (x, y) or local (t, n) frame.
    """
    n_els = len(els.x_centers)
    n_nodes = _NODES[kind]
    pe = 1 if geometry == "antiplane" else 2
    c = np.asarray(coefs, dtype=float).ravel().reshape(n_els, n_nodes, pe)
    t = (np.arange(n_eval) + 0.5) / n_eval * 2 - 1  # in (-1, 1), as s / w
    if kind == "slip":
        P = _core.slip_functions(t, 1.0)  # (n_eval, 3)
    else:
        P = np.stack(((1 + t) / 2, (1 - t) / 2), axis=1)  # ramps 0, 1
    vals = np.einsum("pk,ekc->epc", P, c)  # (n_els, n_eval, pe)
    R = np.asarray(els.rot_mats)
    x = (np.asarray(els.x_centers)[:, None] + R[:, 0, 0][:, None] * t * np.asarray(els.half_lengths)[:, None]).ravel()
    y = (np.asarray(els.y_centers)[:, None] + R[:, 1, 0][:, None] * t * np.asarray(els.half_lengths)[:, None]).ravel()
    if geometry == "antiplane":
        return x, y, vals.reshape(-1)
    if frame == "global":
        Uv = np.stack([_unit_vectors(els, e) for e in range(n_els)])  # (e, 2, 2)
        vals = np.einsum("ecl,epl->epc", Uv, vals)
    elif frame != "local":
        raise ValueError(f"frame must be 'global' or 'local', got {frame!r}")
    return x, y, vals.reshape(-1, 2)


# ----------------------------------------------------------------------------
# level 2: one source type
# ----------------------------------------------------------------------------


def _source_kernels(kind, x_obs, y_obs, els, geometry, mu, nu, basis, connectivity, use_numba):
    _check_material(geometry, mu, nu)
    if basis not in _BASES[kind]:
        raise ValueError(f"basis {basis!r} is not available for {kind}; choose from {_BASES[kind]}")
    n_els = len(els.x_centers)
    if basis in _PATCH_BASES:
        if connectivity is None:
            raise ValueError(f"basis {basis!r} needs connectivity (n_patches, 3)")
        connectivity = _check_connectivity(connectivity, n_els)
    elif connectivity is not None:
        raise ValueError(f"connectivity is only used with patch bases, not {basis!r}")
    if use_numba and not _HAS_NUMBA_KERNEL[(geometry, kind)]:
        raise NotImplementedError(f"no numba kernel for {geometry} {kind}; use use_numba=False")

    element_kernels = element_kernels_slip if kind == "slip" else element_kernels_force
    U, S = element_kernels(
        x_obs, y_obs, els, geometry=geometry, mu=mu, nu=nu, use_numba=use_numba
    )
    if basis != _ELEMENT_BASIS[kind]:
        T = basis_transform(
            els, kind=kind, basis=basis, geometry=geometry, connectivity=connectivity
        )
        U, S = _apply_transform(U, S, T, geometry)

    n_nodes = _NODES[kind]
    if basis == _ELEMENT_BASIS[kind]:
        index, node = np.repeat(np.arange(n_els), n_nodes), np.tile(np.arange(n_nodes), n_els)
    elif basis == "constant":
        index, node = np.arange(n_els), np.full(n_els, -1)
    else:
        index, node = np.arange(len(connectivity)), np.full(len(connectivity), -1)

    disp_c, stress_c, dirs = _COMPONENTS[geometry]
    info = KernelInfo(
        geometry=geometry,
        basis={kind: basis},
        mu=mu,
        nu=nu if geometry == "planestrain" else None,
        backend="numba" if use_numba else "numpy",
        n_obs=U.shape[0],
        displacement_components=disp_c,
        stress_components=stress_c,
        directions=dirs,
        src_label=np.full(U.shape[1], kind[0]),
        src_index=index,
        src_node=node,
    )
    return U, S, info


def get_slip_kernels(
    x_obs, y_obs, els, *, geometry, mu, nu=None, basis="quadratic", connectivity=None,
    use_numba=False,
):
    """Displacement and stress kernels of slip sources on all elements of `els`.

    Most users should build a `Mesh` and call `get_kernels` instead.

    Parameters
    ----------
    x_obs, y_obs : array_like
        Observation coordinates (any shape; flattened).
    els : addict.Dict
        Standardized element geometry, all of them slip elements.
    geometry : {"antiplane", "planestrain"}
    mu : float
        Reference shear modulus.
    nu : float
        Poisson's ratio, uniform; required for plane strain, ignored in antiplane.
    basis : {"quadratic", "constant", "hat"}
        "quadratic": 3 nodes per element; "constant": 1 per element (derived from
        the quadratic one); "hat": 1 per 3-element patch in `connectivity`.
    connectivity : (n_patches, 3) int array
        Element indices of each patch; only for basis="hat".
    use_numba : bool
        Use the numba-jitted kernel (antiplane only). The first call compiles.

    Returns
    -------
    U, S, info : see `get_kernels`.

    Notes
    -----
    Singularity cancellation. Each element's stress is singular at its end
    points; with slip and slip gradient continuous across a node, the singular
    parts of neighbouring elements cancel exactly only if the elements are
    co-planar. At a kink or triple junction the mesh geometry is only C0, so a
    weak (logarithmic) stress singularity remains, with a strength set by the
    angle between the neighbouring elements (the discrete curvature). The
    kernels are evaluated relative to the nearest element end point, so in
    float64 the cancellation holds to about 1e-16 L/d at a distance d from a
    node (L = element length).

    Points on an element. Stresses (and, for slip, displacements) jump across
    an element. For an observation point on an element (local |y| within
    rounding, <= 1e-12 of the element scale) the plane-strain kernels return the
    average of the two sides. To evaluate one side, offset the point along the
    element normal explicitly.
    """
    return _source_kernels(
        "slip", x_obs, y_obs, els, geometry, mu, nu, basis, connectivity, use_numba
    )


def get_force_kernels(
    x_obs, y_obs, els, *, geometry, mu, nu=None, basis="linear", connectivity=None,
    use_numba=False,
):
    """Displacement and stress kernels of line-force sources on all elements of `els`.

    Most users should build a `Mesh` and call `get_kernels` instead.

    basis : {"linear", "constant", "trapezoid"}
        "linear": 2 ramps per element, (w+s)/2w and (w-s)/2w; "constant": 1 per
        element (sum of the ramps); "trapezoid": 1 per 3-element patch in
        `connectivity`.

    Other parameters and returns are as in `get_slip_kernels`; `use_numba`
    works in both geometries.
    """
    return _source_kernels(
        "force", x_obs, y_obs, els, geometry, mu, nu, basis, connectivity, use_numba
    )


# ----------------------------------------------------------------------------
# level 1: mesh
# ----------------------------------------------------------------------------


class Mesh:
    """A boundary mesh: elements, their source type, and the basis used for each type.

    Parameters
    ----------
    els : addict.Dict
        Standardized element geometry.
    labels : "slip" | "force" | sequence of "s"/"f"
        Source type of every element.
    basis : dict, optional
        {"slip": ..., "force": ...}. Defaults: "quadratic" slip, "linear" force
        (the general element-DOF formulation).
    connectivity : dict, optional
        {"slip": (n, 3) int, "force": (m, 3) int} with *global* element
        indices; required for the patch bases "hat" and "trapezoid".
    bc_type, bc_value, alpha : optional
        Per-element boundary-condition data (used by the solvers; T1/T2).
    """

    def __init__(self, els, labels, basis=None, connectivity=None, bc_type=None,
                 bc_value=None, alpha=None):
        n_els = len(els.x_centers)
        if isinstance(labels, str):
            if labels not in _KINDS:
                raise ValueError(f"labels must be 'slip', 'force' or per-element 's'/'f', got {labels!r}")
            labels = np.full(n_els, labels[0])
        labels = np.asarray(labels).astype(str)
        if labels.shape != (n_els,):
            raise ValueError(f"labels must have shape ({n_els},), got {labels.shape}")
        bad = sorted(set(labels.tolist()) - {"s", "f"})
        if bad:
            raise ValueError(f"labels must be 's' or 'f', got {bad}")

        basis = dict(basis or {})
        connectivity = dict(connectivity or {})
        for name, d in (("basis", basis), ("connectivity", connectivity)):
            unknown = set(d) - set(_KINDS)
            if unknown:
                raise ValueError(f"{name} keys must be 'slip'/'force', got {sorted(unknown)}")

        self.els = els
        self.labels = labels
        self.kinds = [k for k in _KINDS if (labels == k[0]).any()]
        self.basis = {}
        self.connectivity = {}
        for k in self.kinds:
            b = basis.get(k, _ELEMENT_BASIS[k])
            if b not in _BASES[k]:
                raise ValueError(f"basis {b!r} is not available for {k}; choose from {_BASES[k]}")
            c = connectivity.get(k)
            if b in _PATCH_BASES:
                if c is None:
                    raise ValueError(f"{k} basis {b!r} needs connectivity[{k!r}]")
                c = _check_connectivity(c, n_els)
                if not (labels[c] == k[0]).all():
                    raise ValueError(f"{k} connectivity refers to elements not labelled {k[0]!r}")
            elif c is not None:
                raise ValueError(f"connectivity[{k!r}] is only used with patch bases, not {b!r}")
            self.basis[k], self.connectivity[k] = b, c
        for k in set(basis) | set(connectivity):
            if k not in self.kinds:
                raise ValueError(f"basis/connectivity given for {k!r} but no element is labelled {k[0]!r}")
        self.bc_type, self.bc_value, self.alpha = bc_type, bc_value, alpha

    @property
    def n_els(self):
        return len(self.labels)

    def elements(self, kind):
        """Global indices of the elements of one source type."""
        return np.flatnonzero(self.labels == kind[0])

    def subset(self, kind):
        """(els, global indices, connectivity in subset indices) for one source type."""
        idx = self.elements(kind)
        sub = self.els if len(idx) == self.n_els else _subset_els(self.els, idx)
        conn = self.connectivity[kind]
        if conn is not None:
            g2l = np.full(self.n_els, -1)
            g2l[idx] = np.arange(len(idx))
            conn = g2l[conn]
        return sub, idx, conn

    def __repr__(self):
        parts = [f"{k}: {len(self.elements(k))} elements, {self.basis[k]} basis" for k in self.kinds]
        return f"Mesh({self.n_els} elements; " + "; ".join(parts) + ")"

    # --- layout of the flattened coefficient vector x = [slip; force] -------------

    def n_src(self, kind):
        """Number of source coefficients of one type (axis 1 of the kernels)."""
        b = self.basis[kind]
        n = len(self.elements(kind))
        if b in _PATCH_BASES:
            return len(self.connectivity[kind])
        return n if b == "constant" else n * _NODES[kind]

    def columns(self, kind, *, geometry):
        """Slice of one source type in the flattened coefficient vector."""
        pe = 1 if geometry == "antiplane" else 2
        start = 0
        for k in self.kinds:
            n = pe * self.n_src(k)
            if k == kind:
                return slice(start, start + n)
            start += n
        raise ValueError(f"no element is labelled {kind[0]!r}")

    def n_columns(self, *, geometry):
        pe = 1 if geometry == "antiplane" else 2
        return pe * sum(self.n_src(k) for k in self.kinds)

    # --- continuity (general, element-DOF formulation) ------------------------------

    def tip_elements(self, kind="slip"):
        """Global indices of the tip (taper) elements of one source type: elements
        with an open end node. They get no boundary-condition row (tasks.md 7.4)."""
        sub, idx, _ = self.subset(kind)
        return idx[tip_elements(sub)]

    def bc_elements(self, kind="slip"):
        """Global indices of the elements of one source type that carry a
        boundary-condition row: all except the tip elements."""
        return np.setdiff1d(self.elements(kind), self.tip_elements(kind))

    def continuity_operator(self, *, geometry):
        """Continuity rows A (A x = 0, or = prescribed values at open nodes) for the mesh.

        Rows are produced for source types with an element basis ("quadratic" slip,
        "linear" force); patch bases satisfy continuity by construction and the
        constant bases have none. Each type is labelled on its own elements, so a
        node shared by a slip and a force element is an open node of both
        (tasks.md 7.4). Returns (A, info), A of shape (n_rows, n_columns); info
        holds per-row arrays: kind, node_type ("open"/"overlap"/"triple"),
        quantity ("slip"/"gradient"/"force"), element, x, y.
        """
        if geometry not in _GEOMETRIES:
            raise ValueError(f"geometry must be one of {_GEOMETRIES}, got {geometry!r}")
        blocks, info = [], {k: [] for k in ("kind", "node_type", "quantity", "element", "x", "y")}
        n_cols = self.n_columns(geometry=geometry)
        for k in self.kinds:
            if self.basis[k] != _ELEMENT_BASIS[k]:
                continue
            sub, idx, _ = self.subset(k)
            A_k, inf = continuity_rows(sub, kind=k, geometry=geometry)
            A = np.zeros((len(A_k), n_cols))
            A[:, self.columns(k, geometry=geometry)] = A_k
            blocks.append(A)
            info["kind"].append(np.full(len(A_k), k[0]))
            info["node_type"].append(inf["node_type"])
            info["quantity"].append(inf["quantity"])
            info["element"].append(idx[inf["node"] // 3])
            info["x"].append(inf["x"])
            info["y"].append(inf["y"])
        if not blocks:
            return np.zeros((0, n_cols)), {k: np.zeros(0) for k in info}
        return np.vstack(blocks), {k: np.concatenate(v) for k, v in info.items()}

    # --- from solution coefficients to element-basis coefficients and values -----

    def element_coefficients(self, x, kind, *, geometry):
        """Element-basis (quadratic slip / linear force) coefficients of one source type,
        flattened in the kernel convention, from the mesh solution vector x."""
        x = np.asarray(x, dtype=float).ravel()
        if x.size != self.n_columns(geometry=geometry):
            raise ValueError(f"x has {x.size} entries, the mesh has {self.n_columns(geometry=geometry)}")
        a = x[self.columns(kind, geometry=geometry)]
        if self.basis[kind] == _ELEMENT_BASIS[kind]:
            return a
        sub, _, conn = self.subset(kind)
        T = basis_transform(sub, kind=kind, basis=self.basis[kind], geometry=geometry, connectivity=conn)
        if geometry == "antiplane":
            return T @ a
        return np.einsum("slbm,bm->sl", T, a.reshape(-1, 2)).ravel()

    def values_on_elements(self, x, kind, *, geometry, n_eval=10, frame="global"):
        """Slip or force along the elements of one source type (see `values_on_elements`)."""
        sub, _, _ = self.subset(kind)
        c = self.element_coefficients(x, kind, geometry=geometry)
        return values_on_elements(sub, c, kind=kind, geometry=geometry, n_eval=n_eval, frame=frame)


def get_kernels(x_obs, y_obs, mesh, *, geometry, mu, nu=None, use_numba=False):
    """Displacement and stress kernels of every source in a mesh.

    Parameters
    ----------
    x_obs, y_obs : array_like
        Observation coordinates (any shape; flattened).
    mesh : Mesh
        Elements, their source type (slip/force) and the basis for each type.
    geometry : {"antiplane", "planestrain"}
        Required.
    mu : float
        Required reference shear modulus. Heterogeneity is mesh data (alpha =
        d ln(mu)/dn on interface elements), not a kernel argument.
    nu : float
        Uniform Poisson's ratio; required for plane strain.
    use_numba : bool
        Use numba-jitted kernels where they exist (not plane-strain slip). The
        first call compiles; default False runs the same code as plain Python.

    Returns
    -------
    U : ndarray
        planestrain: (n_obs, n_src, 2, 2) = [ux, uy] x [t, n];
        antiplane: (n_obs, n_src, 1) = [uz].
    S : ndarray
        planestrain: (n_obs, n_src, 3, 2) = [sxx, syy, sxy] x [t, n];
        antiplane: (n_obs, n_src, 2) = [sxz, syz].
    info : KernelInfo
        Meaning of every axis and source coefficient; `print(info)` summarizes.
        All slip coefficients come first, then all force coefficients.

    Notes
    -----
    Singularity cancellation. Each element's stress is singular at its end
    points; with slip and slip gradient continuous across a node, the singular
    parts of neighbouring elements cancel exactly only if the elements are
    co-planar. At a kink or triple junction the mesh geometry is only C0, so a
    weak (logarithmic) stress singularity remains, with a strength set by the
    angle between the neighbouring elements (the discrete curvature). The
    kernels are evaluated relative to the nearest element end point, so in
    float64 the cancellation holds to about 1e-16 L/d at a distance d from a
    node (L = element length).

    Points on an element. Stresses (and, for slip, displacements) jump across
    an element. For an observation point on an element (local |y| within
    rounding, <= 1e-12 of the element scale) the plane-strain kernels return the
    average of the two sides. To evaluate one side, offset the point along the
    element normal explicitly.
    """
    if not isinstance(mesh, Mesh):
        raise TypeError(
            "get_kernels takes a bemcs.Mesh; for one source type on bare els use "
            "get_slip_kernels / get_force_kernels"
        )
    _check_material(geometry, mu, nu)
    for k in mesh.kinds:  # check every block before computing any
        if use_numba and not _HAS_NUMBA_KERNEL[(geometry, k)]:
            raise NotImplementedError(f"no numba kernel for {geometry} {k}; use use_numba=False")

    U_blocks, S_blocks, infos = [], [], []
    for k in mesh.kinds:
        sub, idx, conn = mesh.subset(k)
        U, S, info = _source_kernels(
            k, x_obs, y_obs, sub, geometry, mu, nu, mesh.basis[k], conn, use_numba
        )
        if mesh.basis[k] not in _PATCH_BASES:
            info.src_index = idx[info.src_index]  # subset -> global element index
        U_blocks.append(U)
        S_blocks.append(S)
        infos.append(info)

    info = infos[0]
    info.basis = dict(mesh.basis)
    info.src_label = np.concatenate([i.src_label for i in infos])
    info.src_index = np.concatenate([i.src_index for i in infos])
    info.src_node = np.concatenate([i.src_node for i in infos])
    return np.concatenate(U_blocks, axis=1), np.concatenate(S_blocks, axis=1), info
