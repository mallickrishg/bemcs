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
        K_x, K_y, _ = get_kernels_trapezoidalforce_antiplane(xo, yo, els, connmatrix)
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
        construct_linearoperator_slip_antiplane(els_s, BCtype[bc_mask], BCval[bc_mask])
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
        print("trapezoid number: ", i)
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
    ("planestrain", "slip"): False,
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
    """Quadratic-slip kernels, 3 nodes per element: U, S in the get_kernels layout."""
    _check_material(geometry, mu, nu)
    if use_numba and not _HAS_NUMBA_KERNEL[(geometry, "slip")]:
        raise NotImplementedError(f"no numba kernel for {geometry} slip; use use_numba=False")
    xo, yo = _obs(x_obs, y_obs)
    if geometry == "antiplane":
        f = _backend(get_displacement_stress_kernel_slip_antiplane, use_numba)
        sxz, syz, u = f(xo, yo, *_els_arrays(els), float(mu))
        return u[:, :, None], np.stack((sxz, syz), axis=2)
    kt = _core.get_displacement_stress_kernel(xo, yo, els, mu, nu, "shear")
    kn = _core.get_displacement_stress_kernel(xo, yo, els, mu, nu, "normal")
    comp = [np.stack((kt[c], kn[c]), axis=-1) for c in range(5)]  # sxx syy sxy ux uy
    return np.stack(comp[3:5], axis=2), np.stack(comp[0:3], axis=2)


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
    f = _backend(get_displacement_stress_kernel_force_planestrain, use_numba)
    ux, uy, sxx, syy, sxy = f(xo, yo, *_els_arrays(els), float(mu), float(nu))

    def r(K):  # (n_obs, direction, ramp, n_els) -> (n_obs, n_els * ramp, direction)
        return K.transpose(0, 3, 2, 1).reshape(n_obs, -1, 2)

    return np.stack((r(ux), r(uy)), axis=2), np.stack((r(sxx), r(syy), r(sxy)), axis=2)


def _per_element_transform(n_els, n_nodes, weights, geometry):
    """Block-diagonal transform: per element, basis = weights . element nodes."""
    w = np.asarray(weights, dtype=float).reshape(n_nodes, -1)  # (node, n_basis_per_element)
    T = np.kron(np.eye(n_els), w)  # (n_els * n_nodes, n_els * n_basis)
    if geometry == "antiplane":
        return T
    return np.einsum("sb,lm->slbm", T, np.eye(2))


def hat_coefficients(els, connectivity, *, geometry):
    """Transform from quadratic-slip element DOFs to hat patches (one per row of connectivity).

    Each hat has unit slip at the central node of its middle element, zero slip and
    slip gradient at the open ends, and continuous slip and slip gradient at the two
    internal nodes. Returns (3 n_els, n_patches) in antiplane and
    (3 n_els, 2, n_patches, 2) [dof, direction_in, patch, direction_out] in plane strain.
    """
    conn = _check_connectivity(connectivity, len(els.x_centers))
    n_els, n_p = len(els.x_centers), len(conn)
    if geometry == "planestrain":
        cs, cn = compute_coefs_quadratichatslip_planestrain(els, conn)  # (n_els, 6, n_p)
        c = np.stack((cs, cn), axis=-1).reshape(n_els, 2, 3, n_p, 2)
        return c.transpose(0, 2, 1, 3, 4).reshape(n_els * 3, 2, n_p, 2)
    if geometry != "antiplane":
        raise ValueError(f"geometry must be one of {_GEOMETRIES}, got {geometry!r}")
    T = np.zeros((n_els * 3, n_p))
    for p in range(n_p):
        sub = _subset_els(els, conn[p])
        index_open, index_overlap, _ = _core.label_nodes(sub)
        m_slip, m_grad = _core.get_matrices_slip_slip_gradient_antiplane(sub)
        rows = [m_slip[4]]  # central node of the middle element
        for node in index_open:
            n = int(np.abs(node).item())
            rows += [m_slip[n], m_grad[n]]
        for a, b in index_overlap:
            if a != 0 and b != 0:
                sa, sb = np.sign(a), np.sign(b)
            elif a == 0:
                sa, sb = 1, -1
            else:
                sa, sb = -1, 1
            rows += [
                sa * m_slip[abs(a)] + sb * m_slip[abs(b)],
                sa * m_grad[abs(a)] + sb * m_grad[abs(b)],
            ]
        A = np.array(rows)
        if A.shape != (9, 9):
            raise ValueError(
                f"patch {p} ({conn[p].tolist()}) is not a chain of 3 connected elements"
            )
        rhs = np.zeros(9)
        rhs[0] = 1.0
        coefs = np.linalg.solve(A, rhs)
        for local, e in enumerate(conn[p]):
            T[3 * e : 3 * e + 3, p] = coefs[3 * local : 3 * local + 3]
    return T


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
