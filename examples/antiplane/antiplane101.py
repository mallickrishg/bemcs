# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bemcs
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# # BEM Problem Description
# Provide an input file with mesh endpoints and boundary condition labels. Below we consider a standard problem of computing the displacement and stress components of a strike-slip fault segment with a free surface.
#
# The boundary conditions for this problem are as follows:
# - $t$: traction-free boundary conditions ($\hat{n}\cdot \sigma = \tau = 0$) on the top surface (free-surface) 
# - $s$: imposed slip on the fault segments 
#
# The boundary conditions are imposed at the center of each mesh element. In addition to these equations we also enforce continuity and differentiability of the BEM coefficients for neighboring elements - these appear as additional constraint equations. So, for a mesh with $N$ elements we have $3N$ unknowns (because we use quadratic polynomials as our basis functions) and we have $3N$ equations.
#
# ## Input file and mesh geometry

# %%
fileinput = "example_mesh.csv"

datain = pd.read_csv(fileinput)

x1 = datain["x1"].values
x2 = datain["x2"].values
y1 = datain["z1"].values
y2 = datain["z2"].values
BCtype = datain["BC_type"].values
BCval = datain["value"].values

# Elastic parameter
mu = 30  # in GPa

# construct mesh
els = bemcs.initialize_els()
stride = 3
# convert units to km
els.x1 = x1 / 1e3
els.y1 = y1 / 1e3
els.x2 = x2 / 1e3
els.y2 = y2 / 1e3

bemcs.standardize_els_geometry(els, reorder=False)

n_els = len(els.x1)
bemcs.plot_els_geometry(els)

# %% [markdown]
# Everything below this point is handled by various functions within `bemcs` and this notebook provides a template for applying it in antiplane geometry. It is unlikely that anything needs to be changed by the user until they get to the display and visualization section.
#
# ## Boundary Conditions
# Here we implement the various boundary conditions as a big vector that goes in the right hand side of the Integral equations. The convention we follow is:
# - BCs at node centers: `BC_c`
# - BCs at open nodes: `BC_o`
# - BCs at interior 2-overlap nodes: `BC_i`
# - BCs at triple junctions: `BC_t`

# %%
index_open, index_overlap, index_triple = bemcs.label_nodes(els)
N_c = n_els  # central node equations
N_o = len(index_open)  # open node equations
N_i = 2 * len(index_overlap)  # overlapping node equations
N_t = 3 * len(index_triple)  # triple junction equations

Nequations = N_c + N_o + N_i + N_t
Nunknowns = stride * n_els
# We will stack this with
# equations for the element centers
# equations at open nodes (RHS = 0)
# equations at overlapping nodes (RHS = 0)
# equations at triple junctions (RHS = 0)
BC_c = BCval.reshape(-1, 1)  # these are the only non-zero entries
BC_o = np.zeros((N_o, 1))
BC_i = np.zeros((N_i, 1))
BC_t = np.zeros((N_t, 1))

# stack all the BCs into 1 big vector
BCvector = np.vstack((BC_c, BC_o, BC_i, BC_t))

# %% [markdown]
# ## Construct linear operator $L\cdot x = b$
# Here we construct a discretized version of the linear operator $L$ for the BEM solve. $L$ contains equations that solve:
# - Mechanical equilibrium with appropriate BC at the center of each element
# - Imposed slip $s = 0$ at open nodes
# - Slip continuity and differentiability and overlapping nodes
# - Sllip continuity and differentiability at triple junctions

# %%
matrix_system_o, matrix_system_i, matrix_system_t = (
    bemcs.construct_smoothoperator_antiplane(
        els, index_open, index_overlap, index_triple
    )
)
slip_mat, slip_gradient_mat = bemcs.get_matrices_slip_slip_gradient_antiplane(
    els, flag="node"
)
kernels = bemcs.get_displacement_stress_kernel_slip_antiplane(
    els.x_centers, els.y_centers, els, mu
)
t_kernels = bemcs.get_traction_kernels_antiplane(els, kernels)

# Linear Operators for the appropriate boundary conditions
matrix_system_c = np.zeros((N_c, Nunknowns))

# populate matrix_system for central nodes (check for bc_label)
for i in range(n_els):
    if BCtype[i] == "s":
        matrix_system_c[i, :] = slip_mat[stride * i + 1, :]
    elif BCtype[i] == "t":
        matrix_system_c[i, :] = t_kernels[i, :]
    else:
        ValueError("boundary condition label not recognized")

# stack the matrices and create the full linear operator
matrix_system = np.vstack(
    (matrix_system_c, matrix_system_o, matrix_system_i, matrix_system_t)
)

# %% [markdown]
# # Solve linear system of equations and get solution coeffcients (on the boundary)

# %%
# Solve for quadratic node coefficients
quadcoefs = np.linalg.inv(matrix_system) @ BCvector

# %% [markdown]
# # Visualization of results
# Below I show how one can plot a number of useful quantities
# - Extract the slip distribution and plot it
# - Internal displacement and stress components
#
# ## Fault slip distribution
# Use the variable `BCtype` to find which elements had $s$-type BCs, and plot their slip coefficients.

# %%
# extract slip at fault-slip nodes
index = BCtype == "s"
nfaultnodes = np.sum(index)

yf = np.zeros(3 * nfaultnodes)
slipnodes = np.zeros((3 * nfaultnodes, 1))
yf[0::3] = els.y1[index]
yf[1::3] = els.y_centers[index]
yf[2::3] = els.y2[index]

slipnodes[0::3] = (slip_mat @ quadcoefs)[0::3][index]
slipnodes[1::3] = (slip_mat @ quadcoefs)[1::3][index]
slipnodes[2::3] = (slip_mat @ quadcoefs)[2::3][index]

# plot slip distribution on the fault as a function of depth
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(slipnodes, yf, "o-")
plt.xlabel("slip (m)")
plt.ylabel("depth (km)")
plt.show()

# %% [markdown]
# ## Internal displacement and stress field
# To compute displacement and stress components at arbitrary points in the medium we first need to construct an array of observation locations.

# %%
xlimits = [-20, 20]
nx_obs = 200
ny_obs = nx_obs
x_obs = np.linspace(-20, 20, nx_obs)
y_obs = np.linspace(-20, 4.41, ny_obs)
x_obs, y_obs = np.meshgrid(x_obs, y_obs)
xo = x_obs.flatten().reshape(-1, 1)
yo = y_obs.flatten().reshape(-1, 1)

# %% [markdown]
# Once we have the observation points `xo,yo`, we need to construct the relevant kernels that will give us the relationship between unit slip on every element in `els` and its response at `(xo,yo)`.

# %%
K_sx, K_sy, K_u = bemcs.get_displacement_stress_kernel_slip_antiplane(xo, yo, els, mu)

# compute displacement and stress components
u = K_u @ quadcoefs
sx = K_sx @ quadcoefs
sy = K_sy @ quadcoefs

# %% [markdown]
# ## Plot displacements and stress components

# %%
plt.figure(figsize=(15, 15))
plt.subplot(3, 1, 1)
toplot = u.reshape(ny_obs, nx_obs)
maxval = 1
minval = -maxval
levels = np.linspace(minval, maxval, 21)
plt.contourf(
    xo.reshape(ny_obs, nx_obs),
    yo.reshape(ny_obs, nx_obs),
    toplot,
    cmap="coolwarm",
    vmin=minval,
    vmax=maxval,
    levels=levels,
)
for i in range(n_els):
    plt.plot([els.x1[i], els.x2[i]], [els.y1[i], els.y2[i]], "k.-")
plt.colorbar(label="$u$ (m)")
plt.contour(
    xo.reshape(ny_obs, nx_obs),
    yo.reshape(ny_obs, nx_obs),
    toplot,
    colors="k",
    levels=levels,
    linewidths=0.5,
)
plt.xlim(xlimits)
plt.gca().set_aspect("equal", adjustable="box")

plt.subplot(3, 1, 2)
toplot = sx.reshape(ny_obs, nx_obs)
maxval = 10
minval = -maxval
levels = np.linspace(minval, maxval, 21)
plt.contourf(
    xo.reshape(ny_obs, nx_obs),
    yo.reshape(ny_obs, nx_obs),
    toplot,
    cmap="RdYlBu_r",
    vmin=minval,
    vmax=maxval,
    levels=levels,
)
for i in range(n_els):
    plt.plot([els.x1[i], els.x2[i]], [els.y1[i], els.y2[i]], "k-")
plt.colorbar(label="$\sigma_x$ (MPa)")
plt.contour(
    xo.reshape(ny_obs, nx_obs),
    yo.reshape(ny_obs, nx_obs),
    toplot,
    colors="k",
    levels=levels,
    linewidths=0.5,
)
plt.xlim(xlimits)
plt.gca().set_aspect("equal", adjustable="box")

plt.subplot(3, 1, 3)
toplot = sy.reshape(ny_obs, nx_obs)
plt.contourf(
    xo.reshape(ny_obs, nx_obs),
    yo.reshape(ny_obs, nx_obs),
    toplot,
    cmap="RdYlBu_r",
    vmin=minval,
    vmax=maxval,
    levels=levels,
)
plt.colorbar(label="$\sigma_y$ (MPa)")
plt.contour(
    xo.reshape(ny_obs, nx_obs),
    yo.reshape(ny_obs, nx_obs),
    toplot,
    colors="k",
    levels=levels,
    linewidths=0.5,
)
for i in range(n_els):
    plt.plot([els.x1[i], els.x2[i]], [els.y1[i], els.y2[i]], "k-")
plt.xlim(xlimits)
plt.gca().set_aspect("equal", adjustable="box")
plt.show()
