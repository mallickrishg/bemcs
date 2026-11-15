"""Frozen pre-revamp continuity operators, used only as test references.

Copied from bemcs/bemcs.py at commit 74a77ec (deleted in stage 3). They use the
old block column order [s0 s1 s2 n0 n1 n2] per element and impose slip only at
open nodes; tests/test_operators.py permutes the columns and adds the clamped-tip
gradient rows before comparing with the new interleaved operators.
"""

import numpy as np

import bemcs.bemcs as core


def get_matrices_slip_slip_gradient(els, flag="node", reference="global"):
    """Assemble design matrix in (x,y) coordinate system for 2 slip components (s,n) for a
    linear system of equations to calculate quadratic coefficients from applied boundary conditions for an ordered list of fault elements.

    flag = "node" : slip is applied at each node of a fault element
    flag = "mean" : slip is applied as a mean value over the entire fault element, not just at nodes

    reference = "global" : slip is applied in a global (x,y) coordinate system
    reference = "local"  : slip is applied in a local (s,n) coordinate system

    Unit vectors for each patch are used to premultiply the input matrices for the global reference frame
    [dx nx] [f1 f2 f3 0  0  0]
    [dy ny] [0  0  0  f1 f2 f3]"""

    stride = 6
    n_els = len(els.x1)
    mat_slip = np.zeros((stride * n_els, stride * n_els))
    mat_slip_gradient = np.zeros_like(mat_slip)

    for i in range(n_els):
        slip_mat_stack = np.zeros((stride, stride))
        slip_gradient_mat_stack = np.zeros_like(slip_mat_stack)
        unit_vec_mat = np.array(
            [
                [els.x_shears[i], els.x_normals[i]],
                [els.y_shears[i], els.y_normals[i]],
            ]
        )
        unit_vec_mat_stack = np.kron(np.eye(3), unit_vec_mat)
        x_obs = np.array([-els.half_lengths[i], 0.0, els.half_lengths[i]])

        if flag == "node":
            slip_mat = core.slip_functions(x_obs, els.half_lengths[i])
        elif flag == "mean":
            slip_mat = core.slip_functions_mean(x_obs)
        else:
            raise ValueError("Invalid flag. Use either 'node' or 'mean'.")

        slip_gradient_mat = core.slipgradient_functions(x_obs, els.half_lengths[i])
        slip_mat_stack[0::2, 0:3] = slip_mat
        slip_mat_stack[1::2, 3:] = slip_mat
        slip_gradient_mat_stack[0::2, 0:3] = slip_gradient_mat
        slip_gradient_mat_stack[1::2, 3:] = slip_gradient_mat

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


def get_matrices_slip_slip_gradient_antiplane(els, flag="node"):
    """Assemble design matrix in (x,y) coordinate system for antiplane slip for a
    linear system of equations to calculate quadratic coefficients from applied boundary conditions for an ordered list of fault elements.

    flag = "node" : slip is applied at each node of a fault element
    flag = "mean" : slip is applied as a mean value over the entire fault element, not just at nodes

    Unit vectors for each patch are used to premultiply the input matrices for the global reference frame
    [s   =  [f1   f2   f3 ].[φ1
    ds/dζ]  [f1'  f2'  f3']  φ2
                             φ3]"""

    stride = 3
    n_els = len(els.x1)
    mat_slip = np.zeros((stride * n_els, stride * n_els))
    mat_slip_gradient = np.zeros_like(mat_slip)

    for i in range(n_els):

        x_obs = np.array([-els.half_lengths[i], 0.0, els.half_lengths[i]])

        if flag == "node":
            slip_mat = core.slip_functions(x_obs, els.half_lengths[i])
        elif flag == "mean":
            slip_mat = core.slip_functions_mean(x_obs)
        else:
            raise ValueError("Invalid flag. Use either 'node' or 'mean'.")

        slip_gradient_mat = core.slipgradient_functions(x_obs, els.half_lengths[i])

        mat_slip[stride * i : stride * (i + 1), stride * i : stride * (i + 1)] = (
            slip_mat
        )
        mat_slip_gradient[
            stride * i : stride * (i + 1), stride * i : stride * (i + 1)
        ] = slip_gradient_mat

    return mat_slip, mat_slip_gradient


def construct_smoothoperator(els, index_open, index_overlap, index_triple):
    """function to construct linear operator that enforces
    continuity and smoothness conditions at non-central nodes

    returns 3 matrices: matrix_system_o, matrix_system_i, matrix_system_t for open, overlapping and triple junctions
    """

    n_els = len(els.x1)
    Nunknowns = 6 * n_els
    # Design matrices (in x,y coordinates) for slip and slip gradients at each 3qn
    matrix_slip, matrix_slip_gradient = get_matrices_slip_slip_gradient(els)

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
        if (idvals[0] != 0) & (idvals[1] != 0):
            sign1 = np.sign(idvals[0])
            sign2 = np.sign(idvals[1])
        elif (idvals[0] == 0) & (idvals[1] != 0):
            sign1 = 1
            sign2 = -1
        else:
            sign1 = -1
            sign2 = 1

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

            # Jacobian form of smoothing equations
            matrix_system_t[6 * k + 2, :] = (
                matrix_slip_gradient[2 * id_pos, :]
                * els.x_shears[int(np.floor(id_pos / 3))]
                - matrix_slip_gradient[2 * id_neg[0], :]
                * els.x_shears[int(np.floor(id_neg[0] / 3))]
                - matrix_slip_gradient[2 * id_neg[1], :]
                * els.x_shears[int(np.floor(id_neg[1] / 3))]
            )
            matrix_system_t[6 * k + 3, :] = (
                matrix_slip_gradient[2 * id_pos, :]
                * els.y_shears[int(np.floor(id_pos / 3))]
                - matrix_slip_gradient[2 * id_neg[0], :]
                * els.y_shears[int(np.floor(id_neg[0] / 3))]
                - matrix_slip_gradient[2 * id_neg[1], :]
                * els.y_shears[int(np.floor(id_neg[1] / 3))]
            )
            matrix_system_t[6 * k + 4, :] = (
                matrix_slip_gradient[2 * id_pos + 1, :]
                * els.x_shears[int(np.floor(id_pos / 3))]
                - matrix_slip_gradient[2 * id_neg[0] + 1, :]
                * els.x_shears[int(np.floor(id_neg[0] / 3))]
                - matrix_slip_gradient[2 * id_neg[1] + 1, :]
                * els.x_shears[int(np.floor(id_neg[1] / 3))]
            )
            matrix_system_t[6 * k + 5, :] = (
                matrix_slip_gradient[2 * id_pos + 1, :]
                * els.y_shears[int(np.floor(id_pos / 3))]
                - matrix_slip_gradient[2 * id_neg[0] + 1, :]
                * els.y_shears[int(np.floor(id_neg[0] / 3))]
                - matrix_slip_gradient[2 * id_neg[1] + 1, :]
                * els.y_shears[int(np.floor(id_neg[1] / 3))]
            )
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

            # Jacobian form of smoothing equations
            matrix_system_t[6 * k + 2, :] = (
                matrix_slip_gradient[2 * id_neg, :]
                * els.x_shears[int(np.floor(id_neg / 3))]
                - matrix_slip_gradient[2 * id_pos[0], :]
                * els.x_shears[int(np.floor(id_pos[0] / 3))]
                - matrix_slip_gradient[2 * id_pos[1], :]
                * els.x_shears[int(np.floor(id_pos[1] / 3))]
            )
            matrix_system_t[6 * k + 3, :] = (
                matrix_slip_gradient[2 * id_neg, :]
                * els.y_shears[int(np.floor(id_neg / 3))]
                - matrix_slip_gradient[2 * id_pos[0], :]
                * els.y_shears[int(np.floor(id_pos[0] / 3))]
                - matrix_slip_gradient[2 * id_pos[1], :]
                * els.y_shears[int(np.floor(id_pos[1] / 3))]
            )
            matrix_system_t[6 * k + 4, :] = (
                matrix_slip_gradient[2 * id_neg + 1, :]
                * els.x_shears[int(np.floor(id_neg / 3))]
                - matrix_slip_gradient[2 * id_pos[0] + 1, :]
                * els.x_shears[int(np.floor(id_pos[0] / 3))]
                - matrix_slip_gradient[2 * id_pos[1] + 1, :]
                * els.x_shears[int(np.floor(id_pos[1] / 3))]
            )
            matrix_system_t[6 * k + 5, :] = (
                matrix_slip_gradient[2 * id_neg + 1, :]
                * els.y_shears[int(np.floor(id_neg / 3))]
                - matrix_slip_gradient[2 * id_pos[0] + 1, :]
                * els.y_shears[int(np.floor(id_pos[0] / 3))]
                - matrix_slip_gradient[2 * id_pos[1] + 1, :]
                * els.y_shears[int(np.floor(id_pos[1] / 3))]
            )

        """# smoothing constraints (2 nodes at a time)
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
        )  # y"""

    return matrix_system_o, matrix_system_i, matrix_system_t


def construct_smoothoperator_antiplane(els, index_open, index_overlap, index_triple):
    """function to construct linear operator that enforces
    continuity and smoothness conditions at non-central nodes for antiplane geometry

    returns 3 matrices: matrix_system_o, matrix_system_i, matrix_system_t for open, overlapping and triple junctions
    """

    n_els = len(els.x1)
    Nunknowns = 3 * n_els
    # Design matrices (in x,y coordinates) for slip and slip gradients at each 3qn
    matrix_slip, matrix_slip_gradient = get_matrices_slip_slip_gradient_antiplane(els)

    N_o = len(index_open)  # open node equations
    N_i = 2 * len(index_overlap)  # overlapping node equations
    N_t = 3 * len(index_triple)  # triple junction equations

    matrix_system_o = np.zeros((N_o, Nunknowns))
    matrix_system_i = np.zeros((N_i, Nunknowns))
    matrix_system_t = np.zeros((N_t, Nunknowns))

    # Linear operator for open nodes
    for i in range(N_o):
        id1 = np.abs(index_open[i])  # node number
        matrix_system_o[i, :] = matrix_slip[id1, :]  # x component

    # Linear operator for overlapping nodes
    for i in range(int(N_i / 2)):
        idvals = index_overlap[i]  # node number

        if (idvals[0] != 0) & (idvals[1] != 0):
            sign1 = np.sign(idvals[0])
            sign2 = np.sign(idvals[1])
        elif (idvals[0] == 0) & (idvals[1] != 0):
            sign1 = 1
            sign2 = -1
        else:
            sign1 = -1
            sign2 = 1

        # continuity condition
        matrix_system_i[2 * i, :] = (
            sign1 * matrix_slip[np.abs(idvals[0]), :]
            + sign2 * matrix_slip[np.abs(idvals[1]), :]
        )

        # smoothing constraints
        matrix_system_i[2 * i + 1, :] = (
            sign1 * matrix_slip_gradient[np.abs(idvals[0]), :]
            + sign2 * matrix_slip_gradient[np.abs(idvals[1]), :]
        )

    # Linear operator for triple junction nodes
    for k in range(int(N_t / 3)):
        id1 = index_triple[k]
        idvalst = np.abs(id1)

        # node number that need to be subtracted in TJ kinematics
        id_neg = idvalst[id1 < 0]
        # node numbers that need to be added
        id_pos = idvalst[id1 >= 0]
        # triple junction kinematics equations
        if len(id_neg) == 2:
            matrix_system_t[3 * k, :] = (
                matrix_slip[id_pos, :]
                - matrix_slip[id_neg[0], :]
                - matrix_slip[id_neg[1], :]
            )
            # smoothing constraints
            matrix_system_t[3 * k + 1, :] = (
                matrix_slip_gradient[id_pos, :]
                * els.x_shears[int(np.floor(id_pos / 3))]
                - matrix_slip_gradient[id_neg[0], :]
                * els.x_shears[int(np.floor(id_neg[0] / 3))]
                - matrix_slip_gradient[id_neg[1], :]
                * els.x_shears[int(np.floor(id_neg[1] / 3))]
            )

            matrix_system_t[3 * k + 2, :] = (
                matrix_slip_gradient[id_pos, :]
                * els.y_shears[int(np.floor(id_pos / 3))]
                - matrix_slip_gradient[id_neg[0], :]
                * els.y_shears[int(np.floor(id_neg[0] / 3))]
                - matrix_slip_gradient[id_neg[1], :]
                * els.y_shears[int(np.floor(id_neg[1] / 3))]
            )

        else:
            matrix_system_t[3 * k, :] = (
                matrix_slip[id_pos[0], :]
                + matrix_slip[id_pos[1], :]
                - matrix_slip[id_neg, :]
            )
            # smoothing constraints
            matrix_system_t[3 * k + 1, :] = (
                matrix_slip_gradient[id_pos[0], :]
                * els.x_shears[int(np.floor(id_pos[0] / 3))]
                + matrix_slip_gradient[id_pos[1], :]
                * els.x_shears[int(np.floor(id_pos[1] / 3))]
                - matrix_slip_gradient[id_neg, :]
                * els.x_shears[int(np.floor(id_neg / 3))]
            )

            matrix_system_t[3 * k + 2, :] = (
                matrix_slip_gradient[id_pos[0], :]
                * els.y_shears[int(np.floor(id_pos[0] / 3))]
                + matrix_slip_gradient[id_pos[1], :]
                * els.y_shears[int(np.floor(id_pos[1] / 3))]
                - matrix_slip_gradient[id_neg, :]
                * els.y_shears[int(np.floor(id_neg / 3))]
            )

        """# smoothing constraints (2 nodes at a time)
        matrix_system_t[3 * k + 1, :] = (
            matrix_slip_gradient[idvalst[0], :] - matrix_slip_gradient[idvalst[1], :]
        )

        matrix_system_t[3 * k + 2, :] = (
            matrix_slip_gradient[idvalst[0], :] - matrix_slip_gradient[idvalst[2], :]
        )"""

    return matrix_system_o, matrix_system_i, matrix_system_t

