# %%
import numpy as np
import matplotlib.pyplot as plt
import warnings
import bemcs

warnings.filterwarnings("ignore")


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

# %% [markdown]
# ## Mesh, boundary conditions and solve
# - topography: traction-free (`"t"`, global tx, ty = 0)
# - fault: prescribed slip (`"s"`, local shear and normal slip at the element centre)
#
# Quadratic slip on every element (general form): continuity and smoothness of slip at interior
# nodes and triple junctions; fault tips are clamped. Homogeneous medium: the same shear modulus
# on both sides of every element.

# %%
geometry = "planestrain"
topo_index = np.zeros(n_els, dtype=bool)
topo_index[0:topo_n] = True
bc_value = np.zeros((n_els, 2))
bc_value[~topo_index, 0] = slipvalues_x  # shear slip
bc_value[~topo_index, 1] = slipvalues_y  # normal slip
mesh = bemcs.Mesh(
    els,
    bc_type=np.where(topo_index, "t", "s"),
    bc_value=bc_value,
    mu_minus=mu,
    mu_plus=mu,
)
plt.figure(figsize=(10, 3))
bemcs.plot_mesh(mesh)
plt.xlim(-3, 3)
plt.show()

quadratic_coefs, (matrix_system, BCvector, rows) = bemcs.solve(
    mesh, geometry=geometry, mu=mu, nu=nu, return_system=True
)
print("Linear Operator Condition Number:", np.linalg.cond(matrix_system))

# plot design matrix
plt.figure(figsize=(10, 6))
plt.imshow(matrix_system, cmap="coolwarm")
plt.colorbar()
plt.clim(-1, 1)
plt.show()

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
matrix_slip_local, _ = bemcs.bemAssembly.slip_node_matrices(els, geometry=geometry, reference="local")
slip_nodes_local = matrix_slip_local @ quadratic_coefs
slip_nodes_s = slip_nodes_local[0::2]
slip_nodes_n = slip_nodes_local[1::2]

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
# Displacement and stress fields on a grid (plane strain: ux, uy, sxx, syy, sxy)
n_obs = 100
width = 3
fig, axes, fields = bemcs.plot_fields(
    mesh, quadratic_coefs, geometry=geometry, mu=mu, nu=nu,
    xlim=(-width, width), ylim=(-width, 0), n=(n_obs, n_obs),
)
plt.show()
x_obs, y_obs = fields["x"].ravel(), fields["y"].ravel()
ux, uy = fields["ux"].ravel(), fields["uy"].ravel()
sxx, syy, sxy = fields["sxx"].ravel(), fields["syy"].ravel(), fields["sxy"].ravel()

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
