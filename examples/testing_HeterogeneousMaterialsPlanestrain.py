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
connect_matrix = np.array([[2, 1, 0], [1, 0, 3], [0, 3, 4]])
fcoefs_s, fcoefs_n = GF.compute_coefs_trapezoidalforce_planestrain(els, connect_matrix)

# %% compute kernels and plot displacements
nx = 100
ny = 100
x_vec = np.linspace(-3.0, 3.0, nx)
y_vec = np.linspace(-2.0, 2.0, ny)
x_obs, y_obs = np.meshgrid(x_vec, y_vec)
x_obs = x_obs.flatten()
y_obs = y_obs.flatten()
xlimits = [-3, 3]
ylimits = [-2, 2]
# compute kernels [Nobs x 2*N_trapz]
K_ux, K_uy, K_sxx, K_syy, K_sxy = GF.get_kernels_trapezoidalforce_planestrain(
    x_obs, y_obs, els, connect_matrix
)
K_ux0, K_uy0, K_sxx0, K_syy0, K_sxy0 = (
    GF.get_displacement_stress_kernel_force_planestrain(
        x_obs.flatten(),
        y_obs.flatten(),
        els.x_centers,
        els.y_centers,
        els.half_lengths,
        els.rot_mats,
        els.rot_mats_inv,
        mu=1,
        nu=0.25,
    )
)
# compute kernels by summing over both basis functions
K_sxx0 = K_sxx0[:, :, 0, :] + K_sxx0[:, :, 1, :]
K_syy0 = K_syy0[:, :, 0, :] + K_syy0[:, :, 1, :]
K_sxy0 = K_sxy0[:, :, 0, :] + K_sxy0[:, :, 1, :]
# impose force coefficients as [fs,fn] pairs for same number of trapezoids as in connect_matrix
sourcecoefs = np.array((1, 0, 0, 0, 1, 0))
# sourcecoefs0 is defined for every mesh element assuming constant basis functions
sourcecoefs0 = np.array([[1.0, 0], [1, 0], [0.5, 0], [1, 0], [0.5, 0]])
# compute stress components
sxx = K_sxx @ sourcecoefs
syy = K_syy @ sourcecoefs
sxy = K_sxy @ sourcecoefs
sxx0 = K_sxx0[:, 0, :] @ sourcecoefs0[:, 0] + K_sxx0[:, 1, :] @ sourcecoefs0[:, 1]
syy0 = K_syy0[:, 0, :] @ sourcecoefs0[:, 0] + K_syy0[:, 1, :] @ sourcecoefs0[:, 1]
sxy0 = K_sxy0[:, 0, :] @ sourcecoefs0[:, 0] + K_sxy0[:, 1, :] @ sourcecoefs0[:, 1]

# plot all 3 stress components as contour plots
plt.figure(figsize=(10, 10))
ux = K_ux @ sourcecoefs
uy = K_uy @ sourcecoefs
nskip = 13
maxval = 0.5
plt.subplot(3, 1, 1)
toplot = sxx0
GF.plot_BEM_field(
    toplot, els, x_obs, y_obs, xlimits, ylimits, maxval, n_levels=11, cmap="Spectral_r"
)
plt.quiver(x_obs[0::nskip], y_obs[0::nskip], ux[0::nskip], 0 * uy[0::nskip])
plt.title("$\\sigma_{xx}$")
plt.subplot(3, 1, 2)
toplot = syy0
GF.plot_BEM_field(
    toplot, els, x_obs, y_obs, xlimits, ylimits, maxval, n_levels=11, cmap="Spectral_r"
)
plt.quiver(x_obs[0::nskip], y_obs[0::nskip], 0 * ux[0::nskip], uy[0::nskip])
plt.title("$\\sigma_{yy}$")
plt.subplot(3, 1, 3)
toplot = sxy0
GF.plot_BEM_field(
    toplot, els, x_obs, y_obs, xlimits, ylimits, maxval, n_levels=11, cmap="Spectral_r"
)
plt.quiver(x_obs[0::nskip], y_obs[0::nskip], ux[0::nskip], uy[0::nskip])
plt.title("$\\sigma_{xy}$")


# %% compute points on the mesh (shifted a little to avoid discontinuitites)

npts_per_elt = 50
dr = -1e-9
xmesh = np.zeros((len(els.x1) * npts_per_elt,))
ymesh = np.zeros((len(els.x1) * npts_per_elt,))
for i in range(len(els.x1)):
    xstart = els.x1[i]
    xend = els.x2[i]
    ystart = els.y1[i]
    yend = els.y2[i]
    xpts = np.linspace(xstart, xend, npts_per_elt + 2)[1:-1]
    ypts = np.linspace(ystart, yend, npts_per_elt + 2)[1:-1]
    xmesh[i * npts_per_elt : (i + 1) * npts_per_elt] = xpts + els.x_normals[i] * dr
    ymesh[i * npts_per_elt : (i + 1) * npts_per_elt] = ypts + els.y_normals[i] * dr

# compute kernels [Nobs x 2*N_trapz]
_, _, K_sxx, K_syy, K_sxy = GF.get_kernels_trapezoidalforce_planestrain(
    xmesh, ymesh, els, connect_matrix
)
# plot all 3 stress components as line plots along the mesh
plt.figure(figsize=(6, 10))
plt.subplot(3, 1, 1)
plt.plot(xmesh, K_sxx @ sourcecoefs, ".")
plt.ylabel("$\\sigma_{xx}$")
plt.subplot(3, 1, 2)
plt.plot(xmesh, K_syy @ sourcecoefs, ".")
plt.ylabel("$\\sigma_{yy}$")
plt.subplot(3, 1, 3)
plt.plot(xmesh, K_sxy @ sourcecoefs, ".")
plt.ylabel("$\\sigma_{xy}$")
plt.show()
# %%
