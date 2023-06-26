import numpy as np
import bemcs

def get_designmatrix_3qn(elements):
    """ Compute design matrix for a linear system of equations to calculate quadratic coefficients from applied boundary conditions for an ordered list of fault elements. 
    
    This function provides 2 design matrices - (1) for slip at every node, (2) for slip gradients at every node
        
    Currently designed to handle the following:
        
    (1) slip at patch centers, (2) slip-gradient at boundary nodes, (3) slip continuity at all internal overlapping nodes and (4) slip smoothness at all internal overlapping nodes"""
    designmatrix_slip = np.zeros((3*len(elements),3*len(elements)))
    designmatrix_slipgradient = np.zeros((3*len(elements),3*len(elements)))

    for i in range(len(elements)):
        slip_matrix = np.zeros((3,3))
        slipgradient_matrix = np.zeros((3,3))
        
        # set x_obs to be oriented along the fault
        x_obs = np.array((-elements[i]["half_length"], 0., elements[i]["half_length"]))

        slip_matrix = bemcs.slip_functions(x_obs, elements[i]["half_length"])        
        designmatrix_slip[3*i:3*i+3,3*i:3*i+3] = slip_matrix

        slipgradient_matrix = bemcs.slipgradient_functions(x_obs, elements[i]["half_length"])        
        designmatrix_slipgradient[3*i:3*i+3,3*i:3*i+3] = slipgradient_matrix

    return designmatrix_slip, designmatrix_slipgradient

def rotate_displacement_stress(displacement, stress, inverse_rotation_matrix):
    """ Rotate displacements stresses from local to global reference frame """
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

def get_quadratic_displacement_stress_kernel(x_obs,y_obs,elements,mu,nu,flag=1):
    """ INPUTS

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
    Kxx = np.zeros((len(x_obs),3*len(elements)))
    Kyy = np.zeros((len(x_obs),3*len(elements)))
    Kxy = np.zeros((len(x_obs),3*len(elements)))
    Gx = np.zeros((len(x_obs),3*len(elements)))
    Gy = np.zeros((len(x_obs),3*len(elements)))    

    # check for which slip component kernels the user wants
    if flag==1:
        flag_strikeslip=1.
        flag_tensileslip=0.
    elif flag==0:
        flag_strikeslip=0.
        flag_tensileslip=1.
    else:
        raise ValueError("shear/tensile flag must be 0/1, no other values allowed")
    
    for i in range(len(elements)):
        # center observation locations (no translation needed)
        x_trans = x_obs - elements[i]["x_center"]
        y_trans = y_obs - elements[i]["y_center"]
        # rotate observations such that fault element is horizontal
        rotated_coordinates = elements[i]["inverse_rotation_matrix"]@np.vstack((x_trans.T,y_trans.T))
        x_rot = rotated_coordinates[0,:].T + elements[i]["x_center"]
        y_rot = rotated_coordinates[1,:].T + elements[i]["y_center"]

        # go through each of the 3 components for a given patch
        # component 1
        slip_vector = np.array([1.0, 0.0, 0.0])
        strike_slip = slip_vector*flag_strikeslip
        tensile_slip = slip_vector*flag_tensileslip    
        # Calculate displacements and stresses for current element
        displacement_local,stress_local = bemcs.displacements_stresses_quadratic_no_rotation(
            x_rot,y_rot,elements[i]["half_length"],mu,nu,strike_slip,tensile_slip,
            elements[i]["x_center"],elements[i]["y_center"])
        displacement_eval,stress_eval = rotate_displacement_stress(displacement_local,stress_local, elements[i]["rotation_matrix"])
        # displacement_eval,stress_eval = displacement_local,stress_local
        index = 3*i
        Kxx[:,index] = stress_eval[0,:]
        Kyy[:,index] = stress_eval[1,:]
        Kxy[:,index] = stress_eval[2,:]
        Gx[:,index] = displacement_eval[0,:]
        Gy[:,index] = displacement_eval[1,:]

        # component 2
        slip_vector = np.array([0.0, 1.0, 0.0])
        strike_slip = slip_vector*flag_strikeslip    
        tensile_slip = slip_vector*flag_tensileslip   
        # Calculate displacements and stresses for current element
        displacement_local,stress_local = bemcs.displacements_stresses_quadratic_no_rotation(
            x_rot,y_rot,elements[i]["half_length"],mu,nu,strike_slip,tensile_slip,
            elements[i]["x_center"],elements[i]["y_center"])
        displacement_eval,stress_eval = rotate_displacement_stress(displacement_local,stress_local, elements[i]["rotation_matrix"])
        index = 3*i + 1
        Kxx[:,index] = stress_eval[0,:]
        Kyy[:,index] = stress_eval[1,:]
        Kxy[:,index] = stress_eval[2,:]
        Gx[:,index] = displacement_eval[0,:]
        Gy[:,index] = displacement_eval[1,:]

        # component 3
        slip_vector = np.array([0.0, 0.0, 1.0])
        strike_slip = slip_vector*flag_strikeslip    
        tensile_slip = slip_vector*flag_tensileslip
        # Calculate displacements and stresses for current element
        displacement_local,stress_local = bemcs.displacements_stresses_quadratic_no_rotation(
            x_rot,y_rot,elements[i]["half_length"],mu,nu,strike_slip,tensile_slip,
            elements[i]["x_center"],elements[i]["y_center"])
        displacement_eval,stress_eval = rotate_displacement_stress(displacement_local,stress_local, elements[i]["rotation_matrix"])
        index = 3*i + 2
        Kxx[:,index] = stress_eval[0,:]
        Kyy[:,index] = stress_eval[1,:]
        Kxy[:,index] = stress_eval[2,:]
        Gx[:,index] = displacement_eval[0,:]
        Gy[:,index] = displacement_eval[1,:]

    return Kxx, Kyy, Kxy, Gx, Gy