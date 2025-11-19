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


def kernel_constructor_slip(els, bctype_x, bctype_y, kernels_s, kernels_n):
    """
    Assemble the linear operator for a 2D BEM problem using a quadratic slip basis.

    Each element has:
        - 3 shear DOF + 3 normal DOF → 6 DOF per element.

    The function returns the kernel matrices that map the unknown slip coefficients
    to the x- and y-components of boundary equations, taking into account the
    specified boundary conditions.

    Parameters
    ----------
    els : object
        Mesh element object containing coordinates, element centers, and normals.
    bctype_x, bctype_y : array_like
        Boundary condition types for each element in the x- and y-directions.
        Allowed types: "u_global", "t_global", "s_local".
    kernels_s, kernels_n : list of arrays
        Precomputed displacement & stress kernels for shear and normal directions.
        [0]: σ_xx, [1]: σ_yy, [2]: σ_xy, [3]: u_x, [4]: u_y

    Returns
    -------
    kerneleval_x, kerneleval_y : ndarray
        Linear operators for x- and y-equations of shape (n_elements, 6*n_elements),
        which can be used to assemble the global BEM system.
    """

    n_els = len(els.x1)
    Nunknowns = 6 * n_els

    matrix_slip, _ = bemcs.get_matrices_slip_slip_gradient(els, reference="local")
    traction_kernels_s = bemcs.get_traction_kernels(els, kernels_s, flag="global")
    traction_kernels_n = bemcs.get_traction_kernels(els, kernels_n, flag="global")

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


def kernel_constructor_force(
    els,
    bctype_x,
    bctype_y,
    K_ux,
    K_uy,
    K_sxx,
    K_syy,
    K_sxy,
    connect_matrix,
):
    """
    Assemble the linear operator for a 2D BEM problem using a trapezoidal force basis.

    Each element has:
        - 1 shear DOF + 1 normal DOF → 2 DOF per element.

    The function computes the kernel matrices that map the unknown force coefficients
    to the x- and y-components of the boundary equations, including conversion of
    stress kernels into traction kernels using element normals.

    Parameters
    ----------
    els : object
        Mesh element object containing coordinates, element centers, and normals.
    bctype_x, bctype_y : array_like
        Boundary condition types for each element in the x- and y-directions.
        Allowed types: "u_global", "t_global".
    K_ux, K_uy : ndarray
        Displacement kernels in x- and y-directions for shear and normal DOFs.
    K_sxx, K_syy, K_sxy : ndarray
        Stress kernels used to compute traction kernels.
    connect_matrix : ndarray
        Element connectivity for trapezoidal basis (used for mid-node normals).

    Returns
    -------
    ker_x, ker_y : ndarray
        Linear operators for x- and y-equations of shape (n_elements, 2*n_elements),
        suitable for assembling the global BEM system.
    """

    n_els = len(connect_matrix[:, 1])
    Nunknowns = 2 * n_els

    # --- element normals (use mid-node connectivity) -----------------------
    nx = els.x_normals[connect_matrix[:, 1]]
    ny = els.y_normals[connect_matrix[:, 1]]

    # --- convert stresses into traction kernels ----------------------------
    # K_tx = sxx * nx + sxy * ny
    # K_ty = sxy * nx + syy * ny
    K_tx = K_sxx * nx[:, None] + K_sxy * ny[:, None]
    K_ty = K_sxy * nx[:, None] + K_syy * ny[:, None]

    # allocate
    ker_x = np.zeros((n_els, Nunknowns))
    ker_y = np.zeros((n_els, Nunknowns))

    for j in range(n_els):

        # --- X equation ----------------------------------------------------
        if bctype_x[j] == "u_global":
            ker_x[j, :] = K_ux[j, :]

        elif bctype_x[j] == "t_global":
            ker_x[j, :] = K_tx[j, :]

        else:
            raise ValueError("Unknown BC type for x")

        # --- Y equation ----------------------------------------------------
        if bctype_y[j] == "u_global":
            ker_y[j, :] = K_uy[j, :]

        elif bctype_y[j] == "t_global":
            ker_y[j, :] = K_ty[j, :]

        else:
            raise ValueError("Unknown BC type for y")

    return ker_x, ker_y


def solve_bem_system_slip(els, bc_x, bc_y, BCtype_x, BCtype_y, mu=1, nu=0.25):
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
    BCtype_x, BCtype_y : str
        Boundary condition types for each element in x- and y-directions.
        Allowed types: ("s_local", "t_global", "u_global")

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
    Nunknowns = 6 * n_els  # 6 dof per mesh element

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
    dr = -1e-6  # small offset to avoid discontinuity
    x_obs = els.x_centers + dr * els.x_normals
    y_obs = els.y_centers + dr * els.y_normals
    kernels_s = bemcs.get_displacement_stress_kernel(x_obs, y_obs, els, mu, nu, "shear")
    kernels_n = bemcs.get_displacement_stress_kernel(
        x_obs, y_obs, els, mu, nu, "normal"
    )

    # ---------------------------------------------------------
    # 3. Build Central-node Linear Operator
    # ---------------------------------------------------------
    matrix_system_c = np.zeros((N_c, Nunknowns))

    kerneleval_x, kerneleval_y = kernel_constructor_slip(
        els,
        BCtype_x,
        BCtype_y,
        kernels_s,
        kernels_n,
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


def solve_bem_system_force(
    els, connect_matrix, bc_x, bc_y, BCtype_x, BCtype_y, mu=1, nu=0.25
):
    """
    Solve a 2D BEM system for a trapezoidal force basis in plane strain.

    This function assembles the linear operator from displacement and traction kernels,
    applies boundary conditions, solves the resulting linear system, and returns the
    shear and normal force coefficients for each element.

    Parameters
    ----------
    els : object
        Mesh element object containing coordinates, element centers, and normals.
    connect_matrix : ndarray
        Element connectivity for trapezoidal basis (used for mid-node normals).
    bc_x, bc_y : ndarray
        Prescribed boundary values in the x- and y-directions.
    BCtype_x, BCtype_y : array_like
        Boundary condition types for each element in x- and y-directions.
        Allowed types: "u_global", "t_global".
    mu : float, optional
        Shear modulus (default: 1).
    nu : float, optional
        Poisson's ratio (default: 0.25).

    Returns
    -------
    fcoefs_s, fcoefs_n : ndarray
        Shear and normal force coefficients for each element, corresponding to
        the trapezoidal force basis.
    """

    dr = -1e-9  # small offset to avoid discontinuity
    x_obs = (els.x_centers + els.x_normals * dr)[connect_matrix[:, 1]]
    y_obs = (els.y_centers + els.y_normals * dr)[connect_matrix[:, 1]]

    K_ux, K_uy, K_sxx, K_syy, K_sxy = (
        bemcs.bemAssembly.get_kernels_trapezoidalforce_planestrain(
            x_obs, y_obs, els, connect_matrix, mu, nu
        )
    )

    kernels_x, kernels_y = kernel_constructor_force(
        els,
        BCtype_x,
        BCtype_y,
        K_ux,  # disp-x due to shear/normal
        K_uy,  # disp-y due to shear/normal
        K_sxx,
        K_syy,
        K_sxy,
        connect_matrix,
    )

    matrix_system = np.vstack((kernels_x, kernels_y))
    BCvector = np.hstack((bc_x, bc_y)).reshape(-1, 1)
    print("Force Linear Operator Condition Number:", np.linalg.cond(matrix_system))
    fcoefs = np.linalg.solve(matrix_system, BCvector)
    #  ---------------------------------------------------------
    # Extract s and n force coefficients
    fcoefs_s = fcoefs[0::2]
    fcoefs_n = fcoefs[1::2]

    return fcoefs_s, fcoefs_n
