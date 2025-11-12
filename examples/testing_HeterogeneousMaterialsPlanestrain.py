# %% import libraries
import numpy as np
import matplotlib.pyplot as plt
import bemcs
import pandas as pd
import warnings
import bemcs.bemAssembly as GF

warnings.filterwarnings("ignore", category=RuntimeWarning)


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
fcoefs_s, fcoefs_n = GF.compute_coefs_trapezoidalforce_planestrain(els, connect_matrix)

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
K_ux, K_uy, K_sxx, K_syy, K_sxy = GF.get_kernels_trapezoidalforce_planestrain(
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
sourcecoefs = np.array((0, 0, 1, 0))
coefs_s = np.array((0.5, 1, 0.5, 0))
plt.figure(figsize=(7, 6))
toplot = K_syy @ sourcecoefs
# toplot = (Ko_sxy[:, 0, 0, :] + Ko_sxy[:, 0, 1, :]) @ coefs_s
maxval = 0.5
GF.plot_BEM_field(
    toplot, els, x_obs, y_obs, xlimits, ylimits, maxval, n_levels=10, cmap="coolwarm"
)
# %%


# %%
