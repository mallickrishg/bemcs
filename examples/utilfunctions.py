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


def solve_planestrain_heterogeneous(
    els,
    connect_slip,
    BCtype_slip,
    BCval_slip,
    connect_force,
    alpha_force,
    mu=1.0,
    nu=0.25,
):
    """
    Solve a plane strain heterogeneous boundary element problem using
    quadratic hat slip elements and trapezoidal force elements.

    Parameters
    ----------
    els : object
        BEM element structure containing arrays of element endpoints (x1, y1, x2, y2).
    connect_slip : (N_s, 3) int ndarray
        Element connectivity matrix for slip patches.
    BCtype_slip : list or array of strings, length N_s
        Boundary condition type for each slip patch. 's' for slip, 't' for projected strain.
    BCval_slip : (N_s, 2) float ndarray
        Prescribed values for each slip patch.
        If 's', it's local (shear, normal) slip.
        If 't', it's global (x, y) projected strain.
    connect_force : (N_f, 3) int ndarray
        Element connectivity matrix for force patches.
    alpha_force : (N_f,) float ndarray
        Normalized gradient of shear modulus alpha = (1/mu) * d_mu/d_n.
    mu : float, optional
        Shear modulus (default 1.0).
    nu : float, optional
        Poisson's ratio (default 0.25).

    Returns
    -------
    coefs_slip : (2*N_s,) float ndarray
        Slip coefficients interleaved [shear1, normal1, shear2, normal2, ...]
    coefs_force : (2*N_f,) float ndarray
        Force coefficients interleaved [fx1, fy1, fx2, fy2, ...]
    """
    import bemcs.bemAssembly as GF

    # Allow empty connectivity arrays
    if connect_slip is not None and len(connect_slip) > 0:
        N_s = len(connect_slip[:, 0])
        colloc_slip_idx = connect_slip[:, 1].astype(int)
    else:
        N_s = 0
        colloc_slip_idx = np.array([], dtype=int)
        connect_slip = np.zeros((0, 3), dtype=int)

    if connect_force is not None and len(connect_force) > 0:
        N_f = len(connect_force[:, 0])
        colloc_force_idx = connect_force[:, 1].astype(int)
    else:
        N_f = 0
        colloc_force_idx = np.array([], dtype=int)
        connect_force = np.zeros((0, 3), dtype=int)

    N_obs = N_s + N_f
    if N_obs == 0:
        return np.zeros(0), np.zeros(0)

    dr = -1e-9

    # Extract properties of collocation elements (the central element of each patch)
    colloc_idx = np.hstack((colloc_slip_idx, colloc_force_idx))

    x_obs = els.x_centers[colloc_idx] + dr * els.x_normals[colloc_idx]
    y_obs = els.y_centers[colloc_idx] + dr * els.y_normals[colloc_idx]
    nx_obs = els.x_normals[colloc_idx]
    ny_obs = els.y_normals[colloc_idx]

    # Compute kernels
    if N_s > 0:
        # kernels_slip will have shape (N_obs, 2 * N_s)
        K_ux_s, K_uy_s, K_sxx_s, K_syy_s, K_sxy_s = (
            GF.get_kernels_quadratichatslip_planestrain(
                x_obs, y_obs, els, connect_slip, mu, nu
            )
        )
    else:
        K_sxx_s = np.zeros((N_obs, 0))
        K_syy_s = np.zeros((N_obs, 0))
        K_sxy_s = np.zeros((N_obs, 0))

    if N_f > 0:
        # kernels_force will have shape (N_obs, 2 * N_f)
        K_ux_f, K_uy_f, K_sxx_f, K_syy_f, K_sxy_f = (
            GF.get_kernels_trapezoidalforce_planestrain(
                x_obs, y_obs, els, connect_force, mu, nu
            )
        )
    else:
        K_sxx_f = np.zeros((N_obs, 0))
        K_syy_f = np.zeros((N_obs, 0))
        K_sxy_f = np.zeros((N_obs, 0))

    # Helper to convert stresses to projected strains and augmented strains
    def compute_strains(sxx, syy, sxy, nx, ny):
        if sxx.shape[1] == 0:
            return sxx.copy(), sxx.copy(), sxx.copy(), sxx.copy()

        nu_prime = nu / (1 + nu)
        exx = (1 / (2 * mu)) * ((1 - nu_prime) * sxx - nu_prime * syy)
        eyy = (1 / (2 * mu)) * (-nu_prime * sxx + (1 - nu_prime) * syy)
        exy = (1 / (2 * mu)) * sxy

        Lambda = exx + eyy

        Psi_x = exx * nx[:, None] + exy * ny[:, None]
        Psi_y = exy * nx[:, None] + eyy * ny[:, None]

        factor = nu / (1 - 2 * nu)
        Psi_tilde_x = Psi_x + factor * nx[:, None] * Lambda
        Psi_tilde_y = Psi_y + factor * ny[:, None] * Lambda

        return Psi_x, Psi_y, Psi_tilde_x, Psi_tilde_y

    Psi_x_s, Psi_y_s, Psi_t_x_s, Psi_t_y_s = compute_strains(
        K_sxx_s, K_syy_s, K_sxy_s, nx_obs, ny_obs
    )
    Psi_x_f, Psi_y_f, Psi_t_x_f, Psi_t_y_f = compute_strains(
        K_sxx_f, K_syy_f, K_sxy_f, nx_obs, ny_obs
    )

    # Evaluate slip kernels at slip element centers for enforcing slip BCs
    if N_s > 0:
        dr_slip = 1e-9
        x_slip = els.x_centers[colloc_slip_idx]
        y_slip = els.y_centers[colloc_slip_idx]
        nx_slip = els.x_normals[colloc_slip_idx]
        ny_slip = els.y_normals[colloc_slip_idx]
        
        K_ux_s_p, K_uy_s_p, _, _, _ = GF.get_kernels_quadratichatslip_planestrain(
            x_slip + dr_slip * nx_slip,
            y_slip + dr_slip * ny_slip,
            els, connect_slip, mu, nu
        )
        K_ux_s_m, K_uy_s_m, _, _, _ = GF.get_kernels_quadratichatslip_planestrain(
            x_slip - dr_slip * nx_slip,
            y_slip - dr_slip * ny_slip,
            els, connect_slip, mu, nu
        )
        K_sx_global_s = K_ux_s_p - K_ux_s_m
        K_sy_global_s = K_uy_s_p - K_uy_s_m
        
        if N_f > 0:
            K_ux_f_p, K_uy_f_p, _, _, _ = GF.get_kernels_trapezoidalforce_planestrain(
                x_slip + dr_slip * nx_slip,
                y_slip + dr_slip * ny_slip,
                els, connect_force, mu, nu
            )
            K_ux_f_m, K_uy_f_m, _, _, _ = GF.get_kernels_trapezoidalforce_planestrain(
                x_slip - dr_slip * nx_slip,
                y_slip - dr_slip * ny_slip,
                els, connect_force, mu, nu
            )
            K_sx_global_f = K_ux_f_p - K_ux_f_m
            K_sy_global_f = K_uy_f_p - K_uy_f_m
        else:
            K_sx_global_f = np.zeros((N_s, 0))
            K_sy_global_f = np.zeros((N_s, 0))
            
        K_slip_shear_s = np.zeros_like(K_sx_global_s)
        K_slip_normal_s = np.zeros_like(K_sx_global_s)
        K_slip_shear_f = np.zeros_like(K_sx_global_f)
        K_slip_normal_f = np.zeros_like(K_sx_global_f)
        
        for i in range(N_s):
            Rinv = els.rot_mats_inv[colloc_slip_idx[i]]
            K_slip_shear_s[i, :] = Rinv[0, 0] * K_sx_global_s[i, :] + Rinv[0, 1] * K_sy_global_s[i, :]
            K_slip_normal_s[i, :] = Rinv[1, 0] * K_sx_global_s[i, :] + Rinv[1, 1] * K_sy_global_s[i, :]
            if N_f > 0:
                K_slip_shear_f[i, :] = Rinv[0, 0] * K_sx_global_f[i, :] + Rinv[0, 1] * K_sy_global_f[i, :]
                K_slip_normal_f[i, :] = Rinv[1, 0] * K_sx_global_f[i, :] + Rinv[1, 1] * K_sy_global_f[i, :]

    N_eq = 2 * N_obs
    A = np.zeros((N_eq, N_eq))
    b = np.zeros((N_eq, 1))

    # Assembly
    # Part 1: Equations for slip/traction boundaries (first 2 * N_s rows)
    for i in range(N_s):
        row_x = 2 * i
        row_y = 2 * i + 1

        if BCtype_slip[i] == "s":
            # Prescribed slip
            A[row_x, :2*N_s] = K_slip_shear_s[i, :]
            A[row_y, :2*N_s] = K_slip_normal_s[i, :]
            if N_f > 0:
                A[row_x, 2*N_s:] = K_slip_shear_f[i, :]
                A[row_y, 2*N_s:] = K_slip_normal_f[i, :]
                
            b[row_x] = BCval_slip[i, 0]
            b[row_y] = BCval_slip[i, 1]

        elif BCtype_slip[i] == "t":
            # Prescribed projected strain
            A[row_x, : 2 * N_s] = Psi_x_s[i, :]
            if N_f > 0:
                A[row_x, 2 * N_s :] = Psi_x_f[i, :]

            A[row_y, : 2 * N_s] = Psi_y_s[i, :]
            if N_f > 0:
                A[row_y, 2 * N_s :] = Psi_y_f[i, :]

            b[row_x] = BCval_slip[i, 0]
            b[row_y] = BCval_slip[i, 1]
        else:
            raise ValueError(f"Unknown BCtype_slip: {BCtype_slip[i]}")

    # Part 2: Equations for heterogeneous materials ('h' rows)
    for i in range(N_f):
        row_x = 2 * N_s + 2 * i
        row_y = 2 * N_s + 2 * i + 1
        colloc_idx_f = N_s + i

        alpha = alpha_force[i]

        # alpha * Psi_tilde^(s) * s + (I + alpha * Psi_tilde^(f)) * f = 0
        if N_s > 0:
            A[row_x, : 2 * N_s] = alpha * Psi_t_x_s[colloc_idx_f, :]
            A[row_y, : 2 * N_s] = alpha * Psi_t_y_s[colloc_idx_f, :]

        A[row_x, 2 * N_s :] = alpha * Psi_t_x_f[colloc_idx_f, :]
        A[row_x, 2 * N_s + 2 * i] += 1.0  # The I part for f_x

        A[row_y, 2 * N_s :] = alpha * Psi_t_y_f[colloc_idx_f, :]
        A[row_y, 2 * N_s + 2 * i + 1] += 1.0  # The I part for f_y

        b[row_x] = 0.0
        b[row_y] = 0.0

    print("Heterogeneous BEM Linear Operator Condition Number:", np.linalg.cond(A))
    sol = np.linalg.solve(A, b)

    coefs_slip = sol[: 2 * N_s].flatten()
    coefs_force = sol[2 * N_s :].flatten()

    return coefs_slip, coefs_force
