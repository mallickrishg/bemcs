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
# ## Mesh and boundary conditions
# The right-hand side holds the boundary conditions at the element centres, followed by zeros
# for the continuity rows at open nodes, interior 2-overlap nodes and triple junctions.

# %%
# every element carries quadratic slip (3 coefficients per element)
mesh = bemcs.Mesh(els, "slip")
geometry = "antiplane"

# %% [markdown]
# ## Construct linear operator $L\cdot x = b$
# Here we construct a discretized version of the linear operator $L$ for the BEM solve. $L$ contains equations that solve:
# - Mechanical equilibrium with appropriate BC at the center of each element
# - Imposed slip $s = 0$ at open nodes
# - Slip continuity and differentiability at overlapping nodes
# - Slip continuity and differentiability at triple junctions

# %%
# continuity rows at open, overlapping and triple-junction nodes (RHS = 0)
matrix_continuity, continuity_info = mesh.continuity_operator(geometry=geometry)

# slip and slip gradient at the 3 nodes of every element
slip_mat, slip_gradient_mat = bemcs.bemAssembly.slip_node_matrices(els, geometry=geometry)

# stress kernels at the element centres -> traction kernels (n_els, 3 n_els)
_, S, info = bemcs.get_kernels(els.x_centers, els.y_centers, mesh, geometry=geometry, mu=mu)
t_kernels = bemcs.traction_kernels(S, els.x_normals, els.y_normals)[:, :, 0]

# Linear Operators for the appropriate boundary conditions at element centres
matrix_system_c = np.zeros((n_els, mesh.n_columns(geometry=geometry)))
for i in range(n_els):
    if BCtype[i] == "s":
        matrix_system_c[i, :] = slip_mat[stride * i + 1, :]  # slip at the centre node
    elif BCtype[i] == "t":
        matrix_system_c[i, :] = t_kernels[i, :]
    else:
        raise ValueError("boundary condition label not recognized")

# tip elements (with an open end node) are clamped tapers: they get no BC row
BCval_c = BCval.reshape(-1, 1)
keep = mesh.bc_elements("slip")
matrix_system_c = matrix_system_c.reshape(n_els, 1, -1)[keep].reshape(1 * len(keep), -1)
BCval_c = BCval_c.reshape(n_els, 1, -1)[keep].reshape(-1, 1)

# stack the matrices and the boundary conditions
matrix_system = np.vstack((matrix_system_c, matrix_continuity))
BCvector = np.vstack((BCval_c, np.zeros((len(matrix_continuity), 1))))

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
U, S, _ = bemcs.get_kernels(xo, yo, mesh, geometry=geometry, mu=mu)

# compute displacement and stress components
u = U[:, :, 0] @ quadcoefs
sx = S[:, :, 0] @ quadcoefs
sy = S[:, :, 1] @ quadcoefs

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
plt.colorbar(label=r"$\sigma_x$ (MPa)")
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
plt.colorbar(label=r"$\sigma_y$ (MPa)")
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
