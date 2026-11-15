import numpy as np
import bemcs


def logistic(x, L=1, k=1, x0=0):
    """
    Logistic function.

    Parameters:
        x (float or array-like): Input value(s)
        L (float): Maximum value (asymptote)
        k (float): Growth rate (steepness)
        x0 (float): Midpoint (inflection point)

    Returns:
        float or array: Value(s) of logistic function at x
    """
    return L / (1 + np.exp(-k * (x - x0)))


def central_bc_rows(els, bctype_x, bctype_y, U, T, matrix_slip_local=None):
    """
    Rows of the element-centre boundary conditions, in the kernel column convention.

    For element j and component c (x or y) the BC type is one of
        "u_global": displacement,  "t_global": traction (global x, y),
        "s_local":  slip at the element centre (local t, n), slip sources only.

    Parameters
    ----------
    els : element structure
    bctype_x, bctype_y : array_like of str, length n_els
    U, T : (n_els, n_src, 2, 2) displacement and global traction kernels at the
        element centres (from bemcs.get_kernels and bemcs.traction_kernels)
    matrix_slip_local : slip node matrix in the local frame (needed for "s_local";
        quadratic slip only)

    Returns
    -------
    rows : (2 n_els, 2 n_src) ndarray, rows ordered [x0, y0, x1, y1, ...]
    """
    n_els = len(els.x1)
    rows = np.zeros((2 * n_els, 2 * U.shape[1]))
    for j in range(n_els):
        for c, bctype in enumerate((bctype_x[j], bctype_y[j])):
            if bctype == "u_global":
                rows[2 * j + c] = U[j, :, c, :].ravel()
            elif bctype == "t_global":
                rows[2 * j + c] = T[j, :, c, :].ravel()
            elif bctype == "s_local" and matrix_slip_local is not None:
                rows[2 * j + c] = matrix_slip_local[2 * (3 * j + 1) + c]
            else:
                raise ValueError(f"unrecognized boundary condition type {bctype!r}")
    return rows


def solve_bem_system_slip(els, bc_x, bc_y, BCtype_x, BCtype_y, mu=1, nu=0.25):
    """
    Solve a plane-strain BEM problem with quadratic slip on every element.

    Parameters
    ----------
    els : element structure
    bc_x, bc_y : array_like, length n_els
        Boundary condition values at the element centres.
    BCtype_x, BCtype_y : array_like of str
        "u_global", "t_global" or "s_local" for each element and component.
    mu, nu : float
        Shear modulus and Poisson's ratio.

    Returns
    -------
    quadratic_coefs : (6 n_els,) ndarray
        Slip coefficients in the kernel convention: element-major, then node,
        then direction (t, n): [t0, n0, t1, n1, ...].
    mesh : bemcs.Mesh
        The mesh used (all elements quadratic slip), for evaluating fields.
    """
    geometry = "planestrain"
    mesh = bemcs.Mesh(els, "slip")

    # kernels at the element centres (small offset to the -n side)
    dr = -1e-6
    x_obs = els.x_centers + dr * els.x_normals
    y_obs = els.y_centers + dr * els.y_normals
    U, S, _ = bemcs.get_kernels(x_obs, y_obs, mesh, geometry=geometry, mu=mu, nu=nu)
    T = bemcs.traction_kernels(S, els.x_normals, els.y_normals)
    matrix_slip_local, _ = bemcs.bemAssembly.slip_node_matrices(
        els, geometry=geometry, reference="local"
    )

    matrix_system_c = central_bc_rows(els, BCtype_x, BCtype_y, U, T, matrix_slip_local)
    matrix_continuity, _ = mesh.continuity_operator(geometry=geometry)

    # tip elements (with an open end node) are clamped tapers: they get no BC row
    keep = mesh.bc_elements("slip")
    n_els = len(els.x1)
    matrix_system_c = matrix_system_c.reshape(n_els, 2, -1)[keep].reshape(2 * len(keep), -1)
    matrix_system = np.vstack((matrix_system_c, matrix_continuity))

    BC_c = np.c_[np.ravel(bc_x), np.ravel(bc_y)][keep].ravel()
    BCvector = np.concatenate((BC_c, np.zeros(len(matrix_continuity))))

    quadratic_coefs = np.linalg.solve(matrix_system, BCvector)
    print("Linear Operator Condition Number:", np.linalg.cond(matrix_system))
    return quadratic_coefs, mesh


def solve_bem_system_force(
    els, connect_matrix, bc_x, bc_y, BCtype_x, BCtype_y, mu=1, nu=0.25
):
    """
    Solve a plane-strain BEM problem with a trapezoidal force basis.

    Collocation points are the centres of the middle element of each patch.

    Parameters
    ----------
    els : element structure
    connect_matrix : (n_patches, 3) int ndarray
        The 3 elements of each trapezoidal patch.
    bc_x, bc_y : ndarray, length n_patches
        Prescribed values at the collocation points.
    BCtype_x, BCtype_y : array_like of str
        "u_global" or "t_global" for each patch and component.
    mu, nu : float
        Shear modulus and Poisson's ratio.

    Returns
    -------
    fcoefs : (2 n_patches,) ndarray
        Force coefficients, interleaved [t0, n0, t1, n1, ...] (local frame of the
        middle element of each patch).
    """
    mid = connect_matrix[:, 1]
    dr = -1e-9  # small offset to avoid discontinuity
    x_obs = (els.x_centers + els.x_normals * dr)[mid]
    y_obs = (els.y_centers + els.y_normals * dr)[mid]
    U, S, _ = bemcs.get_force_kernels(
        x_obs, y_obs, els, geometry="planestrain", mu=mu, nu=nu,
        basis="trapezoid", connectivity=connect_matrix,
    )
    T = bemcs.traction_kernels(S, els.x_normals[mid], els.y_normals[mid])

    # patch-collocation version of central_bc_rows
    n_p = len(mid)
    matrix_system = np.zeros((2 * n_p, 2 * n_p))
    for j in range(n_p):
        for c, bctype in enumerate((BCtype_x[j], BCtype_y[j])):
            K = {"u_global": U, "t_global": T}.get(bctype)
            if K is None:
                raise ValueError(f"unknown BC type {bctype!r}")
            matrix_system[2 * j + c] = K[j, :, c, :].ravel()
    BCvector = np.c_[np.ravel(bc_x), np.ravel(bc_y)].ravel()
    print("Force Linear Operator Condition Number:", np.linalg.cond(matrix_system))
    return np.linalg.solve(matrix_system, BCvector)
