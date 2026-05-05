#%%
import numpy as np
import bemcs
import bemcs.bemAssembly as GF
import matplotlib.pyplot as plt
import pandas as pd
import warnings

#%%
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
            els_subset, reference="global")
        # operator matrix ONLY for slip at the central node of the central element in local reference frame (where the quadratic hat is defined)
        mat_slip_local, _ = bemcs.get_matrices_slip_slip_gradient(
            els_subset, reference="local")

        # constraint equations and build matrix system
        # For quadratic, 3 elements × 3 nodes × 2 comp = 18 unknowns
        # Constraints: 2 for central, 4 per open, 4 per overlap = 18
        Nunknowns = 18 # hardcoded for 3-element quadratic hat patch
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
            mat_system_o[4 * iter + 3, :] = mat_slip_gradient[2 * id1 + 1, :]  # y component

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
        id1 = 4  # hard-coded node [0,1,2],[3,4,5],[5,6,7] → node 4 is center of element 1
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


# Test code
if __name__ == "__main__":
    # Load mesh
    fileinput = "dummy_mesh.csv"
    datain = pd.read_csv(fileinput)
    x1 = datain.x1.values
    y1 = datain.z1.values  # z is y
    x2 = datain.x2.values
    y2 = datain.z2.values

    # Initialize elements
    els = bemcs.initialize_els()
    els.x1, els.y1, els.x2, els.y2 = x1, y1, x2, y2
    bemcs.standardize_els_geometry(els, reorder=False)
    bemcs.plot_els_geometry(els)

    # Define connectivity for quadratic hat patches
    connect_matrix = np.array([[2, 1, 0], [1, 0, 3], [0, 3, 4]])

    coefs_s, coefs_n = compute_coefs_quadratichatslip_planestrain(els, connect_matrix)

    print("Shape of coefs_s:", coefs_s.shape)
    print("Shape of coefs_n:", coefs_n.shape)

    # provide number of points per fault element
    n_eval = 9

    # Compute coefficients from coefs_s[:,:,k] or coefs_n[:,:,k] to a format that get_slip_vector_on_fault() can use i.e., [3x shear; 3x normal] for each mesh element
    coefs_combined = np.zeros((len(els.x1)*6, coefs_s.shape[2]))  # 6 = 3 nodes × 2 comp
    for k in range(coefs_s.shape[2]):
        for i in range(len(els.x1)):
            coefs_combined[i*6:(i+1)*6, k] = coefs_s[i, :, k]
    
    # Evaluate slip using get_slip_vector_on_fault()
    x_obs, y_obs, fault_slip_x, fault_slip_y = bemcs.get_slipvector_on_fault(
        els, coefs_combined[:, 2], n_eval=n_eval
    )

    # plot as quiver in map/cross-sectional view
    plt.figure(figsize=(10, 5))
    for i in range(len(els.x1)):
        plt.plot([els.x1[i], els.x2[i]], [els.y1[i], els.y2[i]], "k.-", linewidth=0.1)
    # plt.plot(els.x_centers, els.y_centers, "ro")
    plt.quiver(
        x_obs,
        y_obs,
        fault_slip_x,
        fault_slip_y,
        np.sqrt(fault_slip_x**2 + fault_slip_y**2),
        cmap="cool",
    )
    plt.scatter(x_obs, y_obs,c=np.sqrt(fault_slip_x**2 + fault_slip_y**2), cmap="cool", s=50)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.ylim(-1,1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label="|slip|")
    plt.show()

# %%
