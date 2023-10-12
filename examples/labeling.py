import numpy as np
import bemcs


# Function to label open nodes, overlapping interior nodes and triple junctions automatically
def label_nodes(els):
    """provide a dictionary of line segments using bemcs.standardize_els_geometry(els),
    to return labelled indices for open nodes, overlapping nodes and triple junctions.

    the indices are provided as a number 3(i-1)<= id <=(3i-1),
    where 'i' corresponds to the mesh element
    """
    n_els = len(els.x1)
    # first find all unique points
    points = np.zeros((2 * n_els, 2))
    x1y1 = np.vstack((els.x1, els.y1)).T
    x2y2 = np.vstack((els.x2, els.y2)).T
    points[0::2, :] = x1y1
    points[1::2, :] = x2y2
    unique_points, id_unique = np.unique(points, axis=0, return_index=True)

    # Find number of open, 2-overlap & triple junction nodes
    index_matrix1 = []  # open
    index_matrix2 = []  # 2-overlap
    index_matrix3 = []  # triple junction
    for i in range(len(unique_points)):
        pts = unique_points[i, :].reshape(1, -1)

        # Which element(s) contains this point
        id1 = np.where(np.all(pts == x1y1, axis=1))
        id2 = np.where(np.all(pts == x2y2, axis=1))

        # The negative signs are for the triple junction equations
        # s_1 + s_2 + s_3 = 0 with the negative sign going to any 2 elements that are both id1 or id2
        if (np.size(id1) == 2) & (np.size(id2) == 1):
            id_combo = np.hstack((-id1[0] * 3, id2[0] * 3 + 2))
        elif (np.size(id2) == 2) & (np.size(id1) == 1):
            id_combo = np.hstack((id1[0] * 3, -(id2[0] * 3 + 2)))
        elif (np.size(id2) == 1) & (np.size(id1) == 1):
            id_combo = np.hstack((id1[0] * 3, -(id2[0] * 3 + 2)))
        elif (np.size(id2) == 2) & (np.size(id1) == 0):
            id_combo = np.hstack(((id2[0][0] * 3 + 2), -(id2[0][1] * 3 + 2)))
        elif (np.size(id1) == 2) & (np.size(id2) == 0):
            id_combo = np.hstack(((id1[0][0] * 3), -(id1[0][1] * 3)))
        else:
            id_combo = np.hstack((id1[0] * 3, (id2[0] * 3 + 2)))

        if np.size(id_combo) == 1:
            index_matrix1.append(id_combo)
        elif np.size(id_combo) == 2:
            index_matrix2.append(id_combo)
        elif np.size(id_combo) == 3:
            index_matrix3.append(id_combo)
        else:
            print(id_combo)
            raise ValueError("Cannot deal with more than 3 lines at a node")

    print("Number of open nodes =", len(index_matrix1))
    print(":", index_matrix1)
    print("Number of 2-overlap nodes =", len(index_matrix2))
    print(":", index_matrix2)
    print("Number of triple junctions =", len(index_matrix3))
    print(":", index_matrix3)

    return index_matrix1, index_matrix2, index_matrix3


def construct_smoothoperator(els, index_open, index_overlap, index_triple):
    """function to construct linear operator that enforces
    continuity and smoothness conditions at non-central nodes
    """

    n_els = len(els.x1)
    Nunknowns = 6 * n_els
    # Design matrices (in x,y coordinates) for slip and slip gradients at each 3qn
    matrix_slip, matrix_slip_gradient = bemcs.get_matrices_slip_slip_gradient(els)

    N_o = 2 * len(index_open)  # open node equations
    N_i = 4 * len(index_overlap)  # overlapping node equations
    N_t = 6 * len(index_triple)  # triple junction equations

    matrix_system_o = np.zeros((N_o, Nunknowns))
    matrix_system_i = np.zeros((N_i, Nunknowns))
    matrix_system_t = np.zeros((N_t, Nunknowns))

    # Linear operator for open nodes
    for i in range(int(N_o / 2)):
        id1 = np.abs(index_open[i])  # node number
        matrix_system_o[2 * i, :] = matrix_slip[2 * id1, :]  # x component
        matrix_system_o[2 * i + 1, :] = matrix_slip[2 * id1 + 1, :]  # y component

    # Linear operator for overlapping nodes
    for i in range(int(N_i / 4)):
        idvals = index_overlap[i]  # node number
        # continuity condition
        sign1 = np.sign(idvals[0])
        sign2 = np.sign(idvals[1])
        matrix_system_i[4 * i, :] = (
            sign1 * matrix_slip[2 * np.abs(idvals[0]), :]
            + sign2 * matrix_slip[2 * np.abs(idvals[1]), :]
        )  # x
        matrix_system_i[4 * i + 1, :] = (
            sign1 * matrix_slip[2 * np.abs(idvals[0]) + 1, :]
            + sign2 * matrix_slip[2 * np.abs(idvals[1]) + 1, :]
        )  # y
        # smoothing constraints
        matrix_system_i[4 * i + 2, :] = (
            sign1 * matrix_slip_gradient[2 * np.abs(idvals[0]), :]
            + sign2 * matrix_slip_gradient[2 * np.abs(idvals[1]), :]
        )  # x
        matrix_system_i[4 * i + 3, :] = (
            sign1 * matrix_slip_gradient[2 * np.abs(idvals[0]) + 1, :]
            + sign2 * matrix_slip_gradient[2 * np.abs(idvals[1]) + 1, :]
        )  # y

    # Linear operator for triple junction nodes
    for k in range(int(N_t / 6)):
        id1 = index_triple[k]
        idvalst = np.abs(id1)

        # node number that need to be subtracted in TJ kinematics
        id_neg = idvalst[id1 < 0]
        # node numbers that need to be added
        id_pos = idvalst[id1 >= 0]
        # triple junction kinematics equations
        if len(id_neg) == 2:
            matrix_system_t[6 * k, :] = (
                matrix_slip[2 * id_pos, :]
                - matrix_slip[2 * id_neg[0], :]
                - matrix_slip[2 * id_neg[1], :]
            )  # x component
            matrix_system_t[6 * k + 1, :] = (
                matrix_slip[2 * id_pos + 1, :]
                - matrix_slip[2 * id_neg[0] + 1, :]
                - matrix_slip[2 * id_neg[1] + 1, :]
            )  # y component
        else:
            matrix_system_t[6 * k, :] = (
                matrix_slip[2 * id_pos[0], :]
                + matrix_slip[2 * id_pos[1], :]
                - matrix_slip[2 * id_neg, :]
            )  # x component
            matrix_system_t[6 * k + 1, :] = (
                matrix_slip[2 * id_pos[0] + 1, :]
                + matrix_slip[2 * id_pos[1] + 1, :]
                - matrix_slip[2 * id_neg + 1, :]
            )  # y component

        # smoothing constraints (2 nodes at a time)
        matrix_system_t[6 * k + 2, :] = (
            matrix_slip_gradient[2 * idvalst[0], :]
            - matrix_slip_gradient[2 * idvalst[1], :]
        )  # x
        matrix_system_t[6 * k + 3, :] = (
            matrix_slip_gradient[2 * idvalst[0] + 1, :]
            - matrix_slip_gradient[2 * idvalst[1] + 1, :]
        )  # y
        matrix_system_t[6 * k + 4, :] = (
            matrix_slip_gradient[2 * idvalst[0], :]
            - matrix_slip_gradient[2 * idvalst[2], :]
        )  # x
        matrix_system_t[6 * k + 5, :] = (
            matrix_slip_gradient[2 * idvalst[0] + 1, :]
            - matrix_slip_gradient[2 * idvalst[2] + 1, :]
        )  # y

    return matrix_system_o, matrix_system_i, matrix_system_t
