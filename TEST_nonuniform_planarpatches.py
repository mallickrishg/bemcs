import numpy as np
import matplotlib.pyplot as plt
import bemcs

def quadratic_stresskernel(x_obs,y_obs,elements):
    """Each stress kernel is a matrix of dimensions 
        Kxx = Nobs x 3xNpatches 
        Kyy = Nobs x 3xNpatches
        Kxy = Nobs x 3xNpatches"""
    Kxx = np.zeros((len(x_obs),3*len(elements)))
    Kyy = np.zeros((len(x_obs),3*len(elements)))
    Kxy = np.zeros((len(x_obs),3*len(elements)))

    tensile_slip = [0.0, 0.0, 0.0]

    for i in range(len(elements)):    
        # go through each of the 3 components for a given patch
        # component 1
        strike_slip = [1.0, 0.0, 0.0]        
        # Calculate displacements and stresses for current element
        displacement_eval,stress_eval= bemcs.displacements_stresses_quadratic_no_rotation(
            x_obs,
            y_obs,
            elements[i]["half_length"],
            mu,
            nu,
            strike_slip,
            tensile_slip,
            elements[i]["x_center"],
            elements[i]["y_center"],
        )
        index = 3*i
        Kxx[:,index] = stress_eval[0,:]
        Kyy[:,index] = stress_eval[1,:]
        Kxy[:,index] = stress_eval[2,:]

        # component 2
        strike_slip = [0.0, 1.0, 0.0]        
        # Calculate displacements and stresses for current element
        displacement_eval,stress_eval= bemcs.displacements_stresses_quadratic_no_rotation(
            x_obs,
            y_obs,
            elements[i]["half_length"],
            mu,
            nu,
            strike_slip,
            tensile_slip,
            elements[i]["x_center"],
            elements[i]["y_center"],
        )
        index = 3*i + 1
        Kxx[:,index] = stress_eval[0,:]
        Kyy[:,index] = stress_eval[1,:]
        Kxy[:,index] = stress_eval[2,:]

        # component 3
        strike_slip = [0.0, 0.0, 1.0]        
        # Calculate displacements and stresses for current element
        displacement_eval,stress_eval= bemcs.displacements_stresses_quadratic_no_rotation(
            x_obs,
            y_obs,
            elements[i]["half_length"],
            mu,
            nu,
            strike_slip,
            tensile_slip,
            elements[i]["x_center"],
            elements[i]["y_center"],
        )
        index = 3*i + 2
        Kxx[:,index] = stress_eval[0,:]
        Kyy[:,index] = stress_eval[1,:]
        Kxy[:,index] = stress_eval[2,:]

    return Kxx, Kyy, Kxy

def smooth_3qncoef_from_slip(elements,slip):
    for i in range(len(elements)):
        slip, slip_gradient = bemcs.get_slip_slipgradient(x, a, phi)

    return quadratic_coefs

# define a mesh
n_elements = 4
mu = np.array([1])
nu = np.array([0.25])
elements = []
element = {}
L = 1

# x1, y1, x2, y2 = bemcs.discretized_line(-L, 0, L, 0, n_elements)
x1 = np.array((-1,-0.5,0.,0.3))
x2 = np.array((-0.5,0.,0.3,1.))
y1 = np.zeros_like(x1)
y2 = np.zeros_like(x2)

for i in range(n_elements):
    element["x1"] = x1[i]
    element["y1"] = y1[i]
    element["x2"] = x2[i]
    element["y2"] = y2[i]
    elements.append(element.copy())

elements = bemcs.standardize_elements(elements)

# plot geometry of mesh
plt.figure()
bemcs.plot_element_geometry(elements)

# observation coordinates for displacement,slip and stress
n_obs = 1001
x_obs = np.linspace(-1.5, 1.5, n_obs)
y_obs = 1e-9 * np.ones_like(x_obs)

# Stress kernels
Kxx,Kyy,Kxy = quadratic_stresskernel(x_obs,y_obs,elements)
