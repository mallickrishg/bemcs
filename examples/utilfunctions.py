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


def kernel_constructor(
    els,
    bctype_x,
    bctype_y,
    kernels_s,
    kernels_n,
    traction_kernels_s,
    traction_kernels_n,
):

    n_els = len(els.x1)
    Nunknowns = 6 * n_els

    matrix_slip, _ = bemcs.get_matrices_slip_slip_gradient(els, reference="local")

    # Linear operator for central node BCs
    kerneleval_x = np.zeros((n_els, Nunknowns))
    kerneleval_y = np.zeros((n_els, Nunknowns))

    # x,y-kernels

    for j in range(0, n_els):
        if bctype_x[j] == "u_global":
            for k in range(0, 3):
                kerneleval_x[j, k::6] = kernels_s[3][j, k::3]
                kerneleval_x[j, k + 3 :: 6] = kernels_n[3][j, k::3]
        elif bctype_x[j] == "t_global":
            for k in range(0, 3):
                kerneleval_x[j, k::6] = traction_kernels_s[0][j, k::3]
                kerneleval_x[j, k + 3 :: 6] = traction_kernels_n[0][j, k::3]
        elif bctype_x[j] == "s_local":
            kerneleval_x[j, :] = matrix_slip[2::6, :][j, :]
        else:
            raise ValueError("unrecognized boundary condition type")

        if bctype_y[j] == "u_global":
            for k in range(0, 3):
                kerneleval_y[j, k::6] = kernels_s[4][j, k::3]
                kerneleval_y[j, k + 3 :: 6] = kernels_n[4][j, k::3]
        elif bctype_y[j] == "t_global":
            for k in range(0, 3):
                kerneleval_y[j, k::6] = traction_kernels_s[1][j, k::3]
                kerneleval_y[j, k + 3 :: 6] = traction_kernels_n[1][j, k::3]
        elif bctype_y[j] == "s_local":
            kerneleval_y[j, :] = matrix_slip[3::6, :][j, :]
        else:
            raise ValueError("unrecognized boundary condition type")

    return kerneleval_x, kerneleval_y


def solve_bem_system_slip(els, bc_x, bc_y, BCtype, stride=6, mu=1, nu=0.25):
    """
    Solve the boundary element system and return quadratic node coefficients.

    Parameters
    ----------
    els : bemcs.ElementClass
        Element structure containing geometry, normals, centers, etc.
    bc_x, bc_y : array_like
        Boundary condition values evaluated at element centers (length = n_els)
    mu, nu : float
        Shear modulus and Poisson ratio.
    BCtype : str
        Boundary condition type ("slip", "traction", etc.) passed to UF.kernel_constructor.
    stride : int
        Number of unknowns per element (default = 6 for quadratic slip/traction basis).

    Returns
    -------
    quadratic_coefs_s : ndarray (3*n_els, 1)
    quadratic_coefs_n : ndarray (3*n_els, 1)
    """

    n_els = len(els.x1)

    # ---------------------------------------------------------
    # 1. Construct Boundary Condition Vectors
    # ---------------------------------------------------------
    index_open, index_overlap, index_triple = bemcs.label_nodes(els)

    N_c = 2 * n_els
    N_o = 2 * len(index_open)
    N_i = 4 * len(index_overlap)
    N_t = 6 * len(index_triple)

    Neq = N_c + N_o + N_i + N_t
    Nunknowns = stride * n_els

    # central nodes
    BC_c = np.zeros((N_c, 1))
    BC_c[0::2, 0] = bc_x
    BC_c[1::2, 0] = bc_y

    # homogeneous nodes (open, overlap, triple)
    BC_o = np.zeros((N_o, 1))
    BC_i = np.zeros((N_i, 1))
    BC_t = np.zeros((N_t, 1))

    # Stack entire RHS
    BCvector = np.vstack((BC_c, BC_o, BC_i, BC_t))

    # ---------------------------------------------------------
    # 2. Build Kernels
    # ---------------------------------------------------------
    kernels_s = bemcs.get_displacement_stress_kernel(
        els.x_centers, els.y_centers, els, mu, nu, "shear"
    )
    kernels_n = bemcs.get_displacement_stress_kernel(
        els.x_centers, els.y_centers, els, mu, nu, "normal"
    )

    traction_kernels_s = bemcs.get_traction_kernels(els, kernels_s, flag="global")
    traction_kernels_n = bemcs.get_traction_kernels(els, kernels_n, flag="global")

    # ---------------------------------------------------------
    # 3. Build Central-node Linear Operator
    # ---------------------------------------------------------
    matrix_system_c = np.zeros((N_c, Nunknowns))

    kerneleval_x, kerneleval_y = kernel_constructor(
        els,
        BCtype,
        BCtype,
        kernels_s,
        kernels_n,
        traction_kernels_s,
        traction_kernels_n,
    )

    matrix_system_c[0::2, :] = kerneleval_x
    matrix_system_c[1::2, :] = kerneleval_y

    # ---------------------------------------------------------
    # 4. Build Open/Overlap/Triple Node Operators
    # ---------------------------------------------------------
    matrix_system_o, matrix_system_i, matrix_system_t = bemcs.construct_smoothoperator(
        els, index_open, index_overlap, index_triple
    )

    matrix_system = np.vstack(
        (matrix_system_c, matrix_system_o, matrix_system_i, matrix_system_t)
    )

    # ---------------------------------------------------------
    # 5. Solve the Linear System
    # ---------------------------------------------------------
    quadratic_coefs = np.linalg.solve(matrix_system, BCvector)

    print("Linear Operator Condition Number:", np.linalg.cond(matrix_system))

    # ---------------------------------------------------------
    # 6. Extract s and n direction coefficients
    # ---------------------------------------------------------
    quadratic_coefs_s = np.zeros((3 * n_els, 1))
    quadratic_coefs_n = np.zeros((3 * n_els, 1))

    for i in range(n_els):
        quadratic_coefs_s[3 * i : 3 * (i + 1)] = quadratic_coefs[6 * i : 6 * i + 3]
        quadratic_coefs_n[3 * i : 3 * (i + 1)] = quadratic_coefs[
            6 * i + 3 : 6 * (i + 1)
        ]

    return quadratic_coefs_s, quadratic_coefs_n


def solve_bem_system_force(els, connect_matrix, bc_x, bc_y, mu=1, nu=0.25):

    dr = -1e-9  # small offset to avoid discontinuity
    x_obs = els.x_centers + els.x_normals * dr
    y_obs = els.y_centers + els.y_normals * dr
    nxvec = els.x_normals
    nyvec = els.y_normals

    # _, _, K_sxx, K_syy, K_sxy = (
    #     bemcs.bemAssembly.get_kernels_trapezoidalforce_planestrain(
    #         x_obs, y_obs, els, connect_matrix
    #     )
    # )
    _, _, K_sxx0, K_syy0, K_sxy0 = (
        bemcs.bemAssembly.get_displacement_stress_kernel_force_planestrain(
            x_obs.flatten(),
            y_obs.flatten(),
            els.x_centers,
            els.y_centers,
            els.half_lengths,
            els.rot_mats,
            els.rot_mats_inv,
            mu=1,
            nu=0.25,
        )
    )
    # sum over basis functions (using constant basis for now)
    K_sxx0 = K_sxx0[:, :, 0, :] + K_sxx0[:, :, 1, :]
    K_syy0 = K_syy0[:, :, 0, :] + K_syy0[:, :, 1, :]
    K_sxy0 = K_sxy0[:, :, 0, :] + K_sxy0[:, :, 1, :]
    K_sxx = np.empty((K_sxx0.shape[0], 2 * K_sxx0.shape[2]))
    K_sxx[:, 0::2] = K_sxx0[:, 0, :]
    K_sxx[:, 1::2] = K_sxx0[:, 1, :]
    K_syy = np.empty((K_syy0.shape[0], 2 * K_syy0.shape[2]))
    K_syy[:, 0::2] = K_syy0[:, 0, :]
    K_syy[:, 1::2] = K_syy0[:, 1, :]
    K_sxy = np.empty((K_sxy0.shape[0], 2 * K_sxy0.shape[2]))
    K_sxy[:, 0::2] = K_sxy0[:, 0, :]
    K_sxy[:, 1::2] = K_sxy0[:, 1, :]
    K_tx = K_sxx * nxvec[:, None] + K_sxy * nyvec[:, None]
    K_ty = K_sxy * nxvec[:, None] + K_syy * nyvec[:, None]

    matrix_system = np.vstack((K_tx, K_ty))
    BCvector = np.hstack((bc_x, bc_y)).reshape(-1, 1)
    print("Force Linear Operator Condition Number:", np.linalg.cond(matrix_system))
    fcoefs = np.linalg.solve(matrix_system, BCvector)

    fcoefs_s = fcoefs[0::2]
    fcoefs_n = fcoefs[1::2]

    return fcoefs_s, fcoefs_n
