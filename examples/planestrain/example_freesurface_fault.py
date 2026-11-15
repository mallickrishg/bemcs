# %%
import numpy as np
import matplotlib.pyplot as plt
import warnings
import bemcs

warnings.filterwarnings("ignore")


# Linear operator for the element-centre boundary conditions. For each element and
# component (x or y), the BC is one of
#   "u_global": displacement,  "t_global": traction (global x, y),
#   "s_local":  slip at the element centre (local t, n).
def central_bc_rows(els, bctype_x, bctype_y, U, T):
    """Rows [x0, y0, x1, y1, ...] acting on the interleaved slip coefficients."""
    n_els = len(els.x1)
    matrix_slip_local, _ = bemcs.bemAssembly.slip_node_matrices(
        els, geometry="planestrain", reference="local"
    )
    rows = np.zeros((2 * n_els, matrix_slip_local.shape[1]))
    for j in range(n_els):
        for c, bctype in enumerate((bctype_x[j], bctype_y[j])):
            if bctype == "u_global":
                rows[2 * j + c] = U[j, :, c, :].ravel()
            elif bctype == "t_global":
                rows[2 * j + c] = T[j, :, c, :].ravel()
            elif bctype == "s_local":  # centre node (node 1) of element j
                rows[2 * j + c] = matrix_slip_local[2 * (3 * j + 1) + c]
            else:
                raise ValueError("unrecognized boundary condition type")
    return rows

# %% [markdown]
# ## Create line mesh (free surface/topography and fault geometry)

# %%
# Elastic parameters
mu = 1
nu = 0.25

# create a mesh of topography
topo_n = 50  # only even number
topo_L = 50  # topo extent in x from [-L/2,L/2]
topo_amp = -0.5  # amplitude of topography as a Gaussian
topo_sigma = 1  # spread of Gaussian topography

xlogvals = np.logspace(-1.0, np.log10(topo_L / 2), int(topo_n / 2))
xmesh = np.concatenate((-xlogvals[::-1], [0.0], xlogvals))
xt1 = xmesh[0:-1]
xt2 = xmesh[1:]
yt1 = topo_amp * np.exp(-(xt1**2) / 2 / (topo_sigma**2))
yt2 = topo_amp * np.exp(-(xt2**2) / 2 / (topo_sigma**2))

# create a mesh of a dipping fault intersecting the free surface
fault_n = 6
slipvalues_x = -np.array([1, 1, 1.0, 0.9, 0.4, 0.1])
slipvalues_y = np.zeros(fault_n)

fault_L = 2.0
fault_dip = 60 * np.pi / 180
xlocs = np.linspace(0, fault_L * np.cos(fault_dip), fault_n + 1)
ylocs = (
    -(np.linspace(0, fault_L * np.sin(fault_dip), fault_n + 1))
    + yt1[np.abs(yt1) == np.max(np.abs(yt1))]
)

# fault geometry
xf1 = xlocs[0:-1]
xf2 = xlocs[1:]
yf1 = ylocs[0:-1]
yf2 = ylocs[1:]

# create bemcs data structure
els = bemcs.initialize_els()
els.x1 = np.concatenate([xt1, xf1])
els.y1 = np.concatenate([yt1, yf1])
els.x2 = np.concatenate([xt2, xf2])
els.y2 = np.concatenate([yt2, yf2])
bemcs.standardize_els_geometry(els, reorder=False)
n_els = len(els.x1)
bemcs.plot_els_geometry(els)

# label Boundary Conditions for each element
bctype_x = np.concatenate(
    (np.repeat("t_global", len(xt1)), np.repeat("s_local", len(xf1)))
)
bctype_y = np.concatenate(
    (np.repeat("t_global", len(xt1)), np.repeat("s_local", len(xf1)))
)

# %% [markdown]
# ## Mesh: quadratic slip on every element (general, element-DOF formulation)
# Open nodes, overlapping interior nodes and triple junctions are identified automatically
# by the continuity operator.

# %%
mesh = bemcs.Mesh(els, "slip")
geometry = "planestrain"

# Boundary conditions are
# topography - traction-free (tx,ty) = (0,0)
# fault mesh - specified slip values
bc_x = np.zeros((n_els, 1))
bc_y = np.zeros((n_els, 1))
# logical index for topography mesh in 'els'
topo_index = np.zeros(n_els, dtype=bool)
topo_index[0:topo_n] = True

bc_x[topo_index, 0] = 0.0
bc_y[topo_index, 0] = 0.0
bc_x[~topo_index, 0] = slipvalues_x
bc_y[~topo_index, 0] = slipvalues_y

BC_c = np.c_[bc_x, bc_y].reshape(-1, 1)  # [x0, y0, x1, y1, ...]

# %% [markdown]
# ## Design Matrix System 
# - At element centers, we implement either displacement or traction boundary conditions
# - At open nodes, we force slip to be 0
# - At 2-overlap (interior) nodes we have continuity & smoothness of slip
# - At triple junctions, we implement 2 equations for conservation of slip and 4 equations for smoothness
#
# We first deal with the matrix system for the central nodes

# %%
# Kernels evaluated ONLY at the element centres: U (n_els, 3 n_els, 2, 2), S (n_els, 3 n_els, 3, 2)
U, S, info = bemcs.get_kernels(
    els.x_centers, els.y_centers, mesh, geometry=geometry, mu=mu, nu=nu
)
# traction kernels in global (x, y) coordinates
T = bemcs.traction_kernels(S, els.x_normals, els.y_normals)
matrix_system_c = central_bc_rows(els, bctype_x, bctype_y, U, T)

# %% [markdown]
# ### Open nodes, overlapping nodes & triple junctions

# %%
matrix_continuity, continuity_info = mesh.continuity_operator(geometry=geometry)

# tip elements (with an open end node) are clamped tapers: they get no BC row
keep = mesh.bc_elements("slip")
matrix_system_c = matrix_system_c.reshape(n_els, 2, -1)[keep].reshape(2 * len(keep), -1)
BC_c = BC_c.reshape(n_els, 2, -1)[keep].reshape(-1, 1)

# stack the matrices and create the full linear operator
matrix_system = np.vstack((matrix_system_c, matrix_continuity))
BCvector = np.vstack((BC_c, np.zeros((len(matrix_continuity), 1))))

# plot design matrix
plt.figure(figsize=(10, 6))
plt.imshow(matrix_system, cmap="coolwarm")
plt.colorbar()
plt.clim(-1, 1)
plt.show()

# %% [markdown]
# # Solve system

# %%
# compute quadratic node coefficients (slip in local (t, n) coordinates, interleaved)
quadratic_coefs = np.linalg.inv(matrix_system) @ BCvector
print("Linear Operator Condition Number:", np.linalg.cond(matrix_system))

# %%
# plot slip distribution on fault source (global x, y components)
xnodes, ynodes, slip_xy = mesh.values_on_elements(
    quadratic_coefs, "slip", geometry=geometry, n_eval=10
)
sx, sy = slip_xy[:, 0], slip_xy[:, 1]

# Slip at the 3 nodes of every element, global (x, y) and local (t, n) components
matrix_slip, _ = bemcs.bemAssembly.slip_node_matrices(els, geometry=geometry)
slip_nodes = matrix_slip @ quadratic_coefs
slip_nodes_x = slip_nodes[0::2]
slip_nodes_y = slip_nodes[1::2]
slip_nodes_s = quadratic_coefs[0::2]  # quadratic coefficients are nodal values (local)
slip_nodes_n = quadratic_coefs[1::2]

index = (ynodes < 0) | (xnodes == 0)


plt.figure(figsize=(7, 5))
plt.plot(np.sqrt(sx[index] ** 2 + sy[index] ** 2), ynodes[index], "-", linewidth=3)

ynodes = np.zeros((n_els * 3))
xnodes = np.zeros((n_els * 3))

ynodes[0::3] = els.y1
ynodes[1::3] = els.y_centers
ynodes[2::3] = els.y2
xnodes[0::3] = els.x1
xnodes[1::3] = els.x_centers
xnodes[2::3] = els.x2
index = ynodes < 0  # & (xnodes == 0)
plt.plot(
    np.sqrt(
        slip_nodes_x[index] ** 2 + slip_nodes_y[index] ** 2
    ),
    ynodes[index],
    "o",
)
# plt.ylim([-3, 0])
plt.xlabel("slip", fontsize=14)
plt.ylabel("depth (km)", fontsize=14)
plt.tick_params(labelsize=13)
plt.grid()
plt.show()


plt.figure(figsize=(10, 4))
plt.plot(slip_nodes_x, "b.", label="$s_x$")
plt.plot(slip_nodes_y, "g.", label="$s_y$")
plt.xlabel("node number")
plt.ylabel("slip at nodes")
plt.legend()
plt.grid()
plt.show()

# slip_nodes_s + quadratic_coefs

# %% [markdown]
# ## Plot displacement and stress components in the medium

# %%
# Observation points
n_obs = 100
width = 3
x_obs = np.linspace(-width, width, n_obs)
y_obs = np.linspace(-width, 0, n_obs)
x_obs, y_obs = np.meshgrid(x_obs, y_obs)
x_obs = x_obs.flatten()
y_obs = y_obs.flatten()

# Displacement and stress kernels at the observation points
U, S, _ = bemcs.get_kernels(x_obs, y_obs, mesh, geometry=geometry, mu=mu, nu=nu)

# Compute displacement and stresses (we will plot it later)
c = quadratic_coefs.ravel()
ux, uy = (U[:, :, k, :].reshape(len(x_obs), -1) @ c for k in range(2))
sxx, syy, sxy = (S[:, :, k, :].reshape(len(x_obs), -1) @ c for k in range(3))

bemcs.plot_displacements_stresses_els(
    els, n_obs, ux, uy, sxx, syy, sxy, x_obs, y_obs, n_skip_plot=31
)

# %%
# make a nice plot
toplot = sxy.reshape(n_obs, n_obs)
maxval = 0.5  # np.max(np.abs(toplot))
minval = -maxval
levels = np.linspace(minval, maxval, 21)

plt.figure(figsize=(15, 5))
n_skip_plot = 6
img = plt.pcolor(
    x_obs.reshape(n_obs, n_obs),
    y_obs.reshape(n_obs, n_obs),
    toplot,
    cmap="coolwarm",
)
plt.clim([minval, maxval])
cb = plt.colorbar(img, shrink=0.9, extend="both")
cb.ax.tick_params(labelsize=15)
cb.set_label(r"$\frac{\sigma_{xy}}{\mu}$", fontsize=20)
plt.contour(
    x_obs.reshape(n_obs, n_obs),
    y_obs.reshape(n_obs, n_obs),
    toplot,
    linewidths=0.5,
    colors="k",
    levels=levels,
)
for i in range(n_els):
    plt.plot([els.x1[i], els.x2[i]], [els.y1[i], els.y2[i]], "k-")
plt.quiver(
    x_obs[0::n_skip_plot],
    y_obs[0::n_skip_plot],
    ux[0::n_skip_plot],
    uy[0::n_skip_plot],
    width=0.001,
    scale=10,
)
plt.xlim([-3, 3])
plt.xlabel("x (km)", fontsize=15)
plt.ylabel("depth (km)", fontsize=15)
plt.tick_params(labelsize=15)
plt.gca().set_aspect("equal", adjustable="box")
plt.show()
