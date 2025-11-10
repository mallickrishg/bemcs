# %% import libraries
import numpy as np
import matplotlib.pyplot as plt
import bemcs
import pandas as pd
import warnings
import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "antiplane/")))
import GF

warnings.filterwarnings("ignore", category=RuntimeWarning)


# %% define functions
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


def get_kernels_trapezoidalforce_planestrain(x_obs, y_obs, els, connect_matrix, mu=1):
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
    K_ux, K_uy, K_sxx, K_syy, K_sxy = (
        GF.get_displacement_stress_kernel_force_planestrain(
            x_obs.flatten(),
            y_obs.flatten(),
            els.x_centers,
            els.y_centers,
            els.half_lengths,
            els.rot_mats,
            els.rot_mats_inv,
            mu=1.0,
        )
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


# %% input file and test kernels
# fileinput = "antiplane/HeterogeneousDomainMesh.csv"
fileinput = "dummy_mesh.csv"
# connectvitiyfile = "HeterogeneousDomainMeshConnectivity.csv"
# connmatrix = pd.read_csv(connectvitiyfile, header=None).values
datain = pd.read_csv(fileinput)
x1 = datain["x1"].values
x2 = datain["x2"].values
y1 = datain["z1"].values
y2 = datain["z2"].values
# BCtype = datain["BC_type"].values
# BCval = datain["value"].values

els = bemcs.initialize_els()
els.x1, els.y1, els.x2, els.y2 = x1, y1, x2, y2
bemcs.standardize_els_geometry(els, reorder=False)
bemcs.plot_els_geometry(els)

# provide connections to compute trapeoidal basis functions
connect_matrix = np.array([[2, 1, 0], [1, 0, 3]])
fcoefs_s, fcoefs_n = compute_coefs_trapezoidalforce_planestrain(els, connect_matrix)

# %% compute kernels and plot displacements
nx = 100
ny = 100
x_vec = np.linspace(-3.0, 2.0, nx)
y_vec = np.linspace(-2.0, 2.0, ny)
x_obs, y_obs = np.meshgrid(x_vec, y_vec)
x_obs = x_obs.flatten()
y_obs = y_obs.flatten()
xlimits = [-3, 2]
ylimits = [-2, 2]
# # compute kernels [Nobs x 2*N_trapz]
K_ux, K_uy, K_sxx, K_syy, K_sxy = get_kernels_trapezoidalforce_planestrain(
    x_obs, y_obs, els, connect_matrix
)

# compute original 4-D kernels
Ko_ux, Ko_uy, Ko_sxx, Ko_syy, Ko_sxy = (
    GF.get_displacement_stress_kernel_force_planestrain(
        x_obs.flatten(),
        y_obs.flatten(),
        els.x_centers,
        els.y_centers,
        els.half_lengths,
        els.rot_mats,
        els.rot_mats_inv,
        mu=1.0,
    )
)
sourcecoefs = np.array((0, 1, 0, 0))
coefs_s = np.array((0.25, 1, 0.25, 0))
plt.figure(figsize=(7, 6))
toplot = K_syy @ sourcecoefs
# toplot = (Ko_syy[:, 1, 0, :] + Ko_syy[:, 1, 1, :]) @ coefs_s
maxval = 0.5
plot_BEM_field(
    toplot, els, x_obs, y_obs, xlimits, ylimits, maxval, n_levels=10, cmap="coolwarm"
)
# %%


# %%
