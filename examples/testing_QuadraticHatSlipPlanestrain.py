#%%
import numpy as np
import bemcs
import bemcs.bemAssembly as GF
import matplotlib.pyplot as plt
import pandas as pd
import warnings

#%%
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

coefs_s, coefs_n = GF.compute_coefs_quadratichatslip_planestrain(els, connect_matrix)
    
print("Shape of coefs_s:", coefs_s.shape)
print("Shape of coefs_n:", coefs_n.shape)

# Visualize a single quadratic-hat patch on a regular observation grid
n_obs = 100
width = 3.0
x_obs = np.linspace(-width, width, n_obs)
y_obs = np.linspace(-width/2, width/2, n_obs)
x_obs, y_obs = np.meshgrid(x_obs, y_obs)
x_obs = x_obs.flatten()
y_obs = y_obs.flatten()

mu = 1.0
nu = 0.25

patch_index = 2
quadratic_coefs_s = np.zeros(3 * len(els.x1))
quadratic_coefs_n = np.zeros(3 * len(els.x1))
for i in range(len(els.x1)):
    quadratic_coefs_s[3 * i : 3 * (i + 1)] = coefs_s[i, :3, patch_index]
    quadratic_coefs_n[3 * i : 3 * (i + 1)] = coefs_s[i, 3:, patch_index]

kernels_s = bemcs.get_displacement_stress_kernel(
    x_obs, y_obs, els, mu, nu, "shear"
)
kernels_n = bemcs.get_displacement_stress_kernel(
    x_obs, y_obs, els, mu, nu, "normal"
)

ux, uy, sxx, syy, sxy = bemcs.coeffs_to_disp_stress(
    kernels_s, kernels_n, quadratic_coefs_s, quadratic_coefs_n
)

bemcs.plot_displacements_stresses_els(
    els,
    n_obs,
    ux,
    uy,
    sxx,
    syy,
    sxy,
    x_obs,
    y_obs,
    n_skip_plot=31,
)

# Test with the combined kernels for all patches
K_ux, K_uy, K_sxx, K_syy, K_sxy = GF.get_kernels_quadratichatslip_planestrain(
    x_obs, y_obs, els, connect_matrix, mu, nu)
slip = np.zeros((len(connect_matrix[:, 0])*2, 1))  # zero slip for all elements
slip[4,0] = 1
slip[0,0] = 1
# compute ux,uy,sxx,syy,sxy for this slip distribution
ux = K_ux @ slip
uy = K_uy @ slip
sxx = K_sxx @ slip
syy = K_syy @ slip
sxy = K_sxy @ slip

bemcs.plot_displacements_stresses_els(
    els,
    n_obs,
    ux,
    uy,
    sxx,
    syy,
    sxy,
    x_obs,
    y_obs,
    n_skip_plot=31,
)
