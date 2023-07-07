import numpy as np
import bemcs


def get_individualdesignmatrix_3qn(elements):
    """Compute design matrix for a linear system of equations to calculate quadratic coefficients from applied boundary conditions for an ordered list of fault elements.
    The resulting matrix is only for 1 component of slip or slip-gradient. Use get_designmatrix_3qn() for the full matrix.

    This function provides 2 design matrices - (1) for slip at every node, (2) for slip gradients at every node"""

    designmatrix_slip = np.zeros((3 * len(elements), 3 * len(elements)))
    designmatrix_slipgradient = np.zeros((3 * len(elements), 3 * len(elements)))

    for i in range(len(elements)):
        slip_matrix = np.zeros((3, 3))
        slipgradient_matrix = np.zeros((3, 3))

        # set x_obs to be oriented along the fault
        x_obs = np.array((-elements[i]["half_length"], 0.0, elements[i]["half_length"]))

        slip_matrix = bemcs.slip_functions(x_obs, elements[i]["half_length"])
        designmatrix_slip[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = slip_matrix

        slipgradient_matrix = bemcs.slipgradient_functions(
            x_obs, elements[i]["half_length"]
        )
        designmatrix_slipgradient[
            3 * i : 3 * i + 3, 3 * i : 3 * i + 3
        ] = slipgradient_matrix

    return designmatrix_slip, designmatrix_slipgradient


def get_designmatrix_xy_3qn(elements,flag="node"):
    """Assemble design matrix in (x,y) coordinate system for 2 slip components (s,n) for a
    linear system of equations to calculate quadratic coefficients from applied boundary conditions for an ordered list of fault elements.

    flag = "node" : slip is applied at each node of a fault element
    flag = "mean" : slip is applied as a mean value over the entire fault element, not just at nodes
    
    Unit vectors for each patch are used to premultiply the input matrices
    [dx nx] [f1 f2 f3 0  0  0]
    [dy ny] [0  0  0  f1 f2 f3]"""

    designmatrix_slip = np.zeros((6 * len(elements), 6 * len(elements)))
    designmatrix_slipgradient = np.zeros((6 * len(elements), 6 * len(elements)))

    for i in range(len(elements)):
        slip_matrixstack = np.zeros((6, 6))
        slipgradient_matrixstack = np.zeros((6, 6))

        unitvec_matrix = np.array(
            [
                [elements[i]["x_shear"], elements[i]["x_normal"]],
                [elements[i]["y_shear"], elements[i]["y_normal"]],
            ]
        )
        unitvec_matrixstack = np.kron(np.eye(3), unitvec_matrix)

        # set x_obs to be oriented along the fault
        x_obs = np.array((-elements[i]["half_length"], 0.0, elements[i]["half_length"]))

        if flag == "node":
            slip_matrix = bemcs.slip_functions(x_obs, elements[i]["half_length"])
        elif flag == "mean":
            slip_matrix = bemcs.slip_functions_mean(x_obs)
        else:
            raise ValueError("Invalid flag. Use either 'node' or 'mean'.")
        
        slipgradient_matrix = bemcs.slipgradient_functions(
            x_obs, elements[i]["half_length"]
        )

        slip_matrixstack[0::2, 0:3] = slip_matrix
        slip_matrixstack[1::2, 3:] = slip_matrix
        slipgradient_matrixstack[0::2, 0:3] = slipgradient_matrix
        slipgradient_matrixstack[1::2, 3:] = slipgradient_matrix

        designmatrix_slip[6 * i : 6 * (i + 1), 6 * i : 6 * (i + 1)] = (
            unitvec_matrixstack @ slip_matrixstack
        )
        designmatrix_slipgradient[6 * i : 6 * (i + 1), 6 * i : 6 * (i + 1)] = (
            unitvec_matrixstack @ slipgradient_matrixstack
        )

    return designmatrix_slip, designmatrix_slipgradient


def get_designmatrix_xy_3qn_mean(elements,flag="node"):
    """Assemble design matrix in (x,y) coordinate system for 2 slip components (s,n) for a
    linear system of equations to calculate quadratic coefficients from applied boundary conditions for an ordered list of fault elements.

    Unit vectors for each patch are used to premultiply the input matrices
    [dx nx] [f1 f2 f3 0  0  0]
    [dy ny] [0  0  0  f1 f2 f3]"""

    designmatrix_slip = np.zeros((6 * len(elements), 6 * len(elements)))
    designmatrix_slipgradient = np.zeros((6 * len(elements), 6 * len(elements)))

    for i in range(len(elements)):
        slip_matrixstack = np.zeros((6, 6))
        slipgradient_matrixstack = np.zeros((6, 6))

        unitvec_matrix = np.array(
            [
                [elements[i]["x_shear"], elements[i]["x_normal"]],
                [elements[i]["y_shear"], elements[i]["y_normal"]],
            ]
        )
        unitvec_matrixstack = np.kron(np.eye(3), unitvec_matrix)

        # set x_obs to be oriented along the fault
        x_obs = np.array((-elements[i]["half_length"], 0.0, elements[i]["half_length"]))        
        
        slipgradient_matrix = bemcs.slipgradient_functions(
            x_obs, elements[i]["half_length"]
        )

        slip_matrixstack[0::2, 0:3] = slip_matrix
        slip_matrixstack[1::2, 3:] = slip_matrix
        slipgradient_matrixstack[0::2, 0:3] = slipgradient_matrix
        slipgradient_matrixstack[1::2, 3:] = slipgradient_matrix

        designmatrix_slip[6 * i : 6 * (i + 1), 6 * i : 6 * (i + 1)] = (
            unitvec_matrixstack @ slip_matrixstack
        )
        designmatrix_slipgradient[6 * i : 6 * (i + 1), 6 * i : 6 * (i + 1)] = (
            unitvec_matrixstack @ slipgradient_matrixstack
        )

    return designmatrix_slip, designmatrix_slipgradient


def rotate_displacement_stress(displacement, stress, inverse_rotation_matrix):
    """Rotate displacements stresses from local to global reference frame"""
    displacement = np.matmul(displacement.T, inverse_rotation_matrix).T
    for i in range(0, stress.shape[1]):
        stress_tensor = np.array(
            [[stress[0, i], stress[2, i]], [stress[2, i], stress[1, i]]]
        )
        stress_tensor_global = (
            inverse_rotation_matrix.T @ stress_tensor @ inverse_rotation_matrix
        )
        stress[0, i] = stress_tensor_global[0, 0]
        stress[1, i] = stress_tensor_global[1, 1]
        stress[2, i] = stress_tensor_global[0, 1]
    return displacement, stress


def get_quadratic_displacement_stress_kernel(x_obs, y_obs, elements, mu, nu, flag):
    """INPUTS

    x_obs,y_obs - locations to compute kernels
    elements - provide list of elements (geometry & rotation matrices)
    mu, nu - Elastic parameters (Shear Modulus, Poisson ratio)
    flag - 1 for shear, 0 for tensile kernels

    OUTPUTS

    Each stress kernel is a matrix of dimensions

    Kxx = Nobs x 3xNpatches
    Kyy = Nobs x 3xNpatches
    Kxy = Nobs x 3xNpatches

    Each displacement kernel is a matrix of dimensions

    Gx = Nobs x 3xNpatches
    Gy = Nobs x 3xNpatches"""
    Kxx = np.zeros((len(x_obs), 3 * len(elements)))
    Kyy = np.zeros((len(x_obs), 3 * len(elements)))
    Kxy = np.zeros((len(x_obs), 3 * len(elements)))
    Gx = np.zeros((len(x_obs), 3 * len(elements)))
    Gy = np.zeros((len(x_obs), 3 * len(elements)))

    # check for which slip component kernels the user wants
    if flag == 1:
        flag_strikeslip = 1.0
        flag_tensileslip = 0.0
    elif flag == 0:
        flag_strikeslip = 0.0
        flag_tensileslip = 1.0
    else:
        raise ValueError("shear/tensile flag must be 1/0, no other values allowed")

    for i in range(len(elements)):
        # center observation locations (no translation needed)
        x_trans = x_obs - elements[i]["x_center"]
        y_trans = y_obs - elements[i]["y_center"]
        # rotate observations such that fault element is horizontal
        rotated_coordinates = elements[i]["inverse_rotation_matrix"] @ np.vstack(
            (x_trans.T, y_trans.T)
        )
        x_rot = rotated_coordinates[0, :].T + elements[i]["x_center"]
        y_rot = rotated_coordinates[1, :].T + elements[i]["y_center"]

        # go through each of the 3 components for a given patch
        # component 1
        slip_vector = np.array([1.0, 0.0, 0.0])
        strike_slip = slip_vector * flag_strikeslip
        tensile_slip = slip_vector * flag_tensileslip
        # Calculate displacements and stresses for current element
        (
            displacement_local,
            stress_local,
        ) = bemcs.displacements_stresses_quadratic_no_rotation(
            x_rot,
            y_rot,
            elements[i]["half_length"],
            mu,
            nu,
            strike_slip,
            tensile_slip,
            elements[i]["x_center"],
            elements[i]["y_center"],
        )
        displacement_eval, stress_eval = rotate_displacement_stress(
            displacement_local, stress_local, elements[i]["inverse_rotation_matrix"]
        )
        # displacement_eval,stress_eval = displacement_local,stress_local
        index = 3 * i
        Kxx[:, index] = stress_eval[0, :]
        Kyy[:, index] = stress_eval[1, :]
        Kxy[:, index] = stress_eval[2, :]
        Gx[:, index] = displacement_eval[0, :]
        Gy[:, index] = displacement_eval[1, :]

        # component 2
        slip_vector = np.array([0.0, 1.0, 0.0])
        strike_slip = slip_vector * flag_strikeslip
        tensile_slip = slip_vector * flag_tensileslip
        # Calculate displacements and stresses for current element
        (
            displacement_local,
            stress_local,
        ) = bemcs.displacements_stresses_quadratic_no_rotation(
            x_rot,
            y_rot,
            elements[i]["half_length"],
            mu,
            nu,
            strike_slip,
            tensile_slip,
            elements[i]["x_center"],
            elements[i]["y_center"],
        )
        displacement_eval, stress_eval = rotate_displacement_stress(
            displacement_local, stress_local, elements[i]["inverse_rotation_matrix"]
        )
        index = 3 * i + 1
        Kxx[:, index] = stress_eval[0, :]
        Kyy[:, index] = stress_eval[1, :]
        Kxy[:, index] = stress_eval[2, :]
        Gx[:, index] = displacement_eval[0, :]
        Gy[:, index] = displacement_eval[1, :]

        # component 3
        slip_vector = np.array([0.0, 0.0, 1.0])
        strike_slip = slip_vector * flag_strikeslip
        tensile_slip = slip_vector * flag_tensileslip
        # Calculate displacements and stresses for current element
        (
            displacement_local,
            stress_local,
        ) = bemcs.displacements_stresses_quadratic_no_rotation(
            x_rot,
            y_rot,
            elements[i]["half_length"],
            mu,
            nu,
            strike_slip,
            tensile_slip,
            elements[i]["x_center"],
            elements[i]["y_center"],
        )
        displacement_eval, stress_eval = rotate_displacement_stress(
            displacement_local, stress_local, elements[i]["inverse_rotation_matrix"]
        )
        index = 3 * i + 2
        Kxx[:, index] = stress_eval[0, :]
        Kyy[:, index] = stress_eval[1, :]
        Kxy[:, index] = stress_eval[2, :]
        Gx[:, index] = displacement_eval[0, :]
        Gy[:, index] = displacement_eval[1, :]

    return Kxx, Kyy, Kxy, Gx, Gy


def compute_tractionkernels(elements, kernels):
    """Function to calculate kernels of traction vector from a set of stress kernels and unit vectors.

    Provide elements as a list with ["x_normal"] & ["y_normal"] for the unit normal vector.

    kernels must be provided as kernels[0] = Kxx, kernels[1] = Kyy, kernels[2] = Kxy
    """
    Kxx = kernels[0]
    Kyy = kernels[1]
    Kxy = kernels[2]
    nrows = np.shape(Kxx)[0]
    # nrows = len(elements)
    # ncols = np.shape(Kxx)[1]

    tx = np.zeros_like(Kxx)
    ty = np.zeros_like(Kxx)
    # unit vector in normal direction
    nvec = np.zeros((nrows, 2))

    for i in range(nrows):
        nvec[i, :] = np.array((elements[i]["x_normal"], elements[i]["y_normal"]))
    nx_matrix = np.zeros_like(Kxx)
    ny_matrix = np.zeros_like(Kxx)

    nx_matrix[:, 0::3] = nvec[:, 0]
    nx_matrix[:, 1::3] = nvec[:, 0]
    nx_matrix[:, 2::3] = nvec[:, 0]
    ny_matrix[:, 0::3] = nvec[:, 1]
    ny_matrix[:, 1::3] = nvec[:, 1]
    ny_matrix[:, 2::3] = nvec[:, 1]

    # traction vector t = n.σ
    tx = Kxx * nx_matrix + Kxy * ny_matrix
    ty = Kxy * nx_matrix + Kyy * ny_matrix

    return tx, ty
