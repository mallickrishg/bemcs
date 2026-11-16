# %%
# %load_ext autoreload  # (IPython magic, disabled in .py)
# %autoreload 2  # (IPython magic, disabled in .py)
# %config InlineBackend.figure_format = "retina"  # (IPython magic, disabled in .py)

import numpy as np
import matplotlib.pyplot as plt

import bemcs

# define els_obs as an object containing x_normals and y_normals with same as size as n_pts
class els_normals:
    def __init__(self, els, n_pts):
        x_normals = els.x_normals
        y_normals = els.y_normals
        x_shears = els.x_shears
        y_shears = els.y_shears

        self.x_normals = np.zeros(n_pts * len(x_normals))
        self.y_normals = np.zeros(n_pts * len(x_normals))
        self.x_normals[0:n_pts] = x_normals[0]
        self.y_normals[0:n_pts] = y_normals[0]
        self.x_normals[n_pts : 2 * n_pts] = x_normals[1]
        self.y_normals[n_pts : 2 * n_pts] = y_normals[1]
        self.x_normals[2 * n_pts :] = x_normals[2]
        self.y_normals[2 * n_pts :] = y_normals[2]

        self.x_shears = np.zeros(n_pts * len(x_normals))
        self.y_shears = np.zeros(n_pts * len(x_normals))
        self.x_shears[0:n_pts] = x_shears[0]
        self.y_shears[0:n_pts] = y_shears[0]
        self.x_shears[n_pts : 2 * n_pts] = x_shears[1]
        self.y_shears[n_pts : 2 * n_pts] = y_shears[1]
        self.x_shears[2 * n_pts :] = x_shears[2]
        self.y_shears[2 * n_pts :] = y_shears[2]

# %%
els = bemcs.initialize_els()
# els.x1 = np.array([-1.0, 0.0, 0.0])
# els.y1 = np.array([0.5, 0.5, -0.5])
# els.x2 = np.array([0.0, 0.0, 1.0])
# els.y2 = np.array([0.5, -0.5, -0.5])

els.x1 = np.array([-1.0, -0.3, 0.3])
els.y1 = np.array([0.5, 0.3, -0.25])
els.x2 = np.array([-0.3, 0.3, 1.0])
els.y2 = np.array([0.3, -0.25, -0.5])

# Slip imposed at central nodes of each patch (these are in x,y coordinates)
slip_vector_x = np.array([0.3, 0.8, 0.3])
slip_vector_y = np.array([0.1, 0.0, 0.1])
# slip_vector_x = np.random.rand(3)
# slip_vector_y = np.random.rand(3)

bemcs.standardize_els_geometry(els)
n_els = len(els.x1)
bemcs.plot_els_geometry(els)

# %%
# Quadratic slip on every element; unknowns are interleaved [t0, n0, t1, n1, ...] per node
geometry = "planestrain"
mu = 1
nu = 0.25

# Boundary conditions: slip imposed at the centre of every element. The values above are global
# (x, y); BC type "s" takes local (shear, normal) slip, so rotate them into each element's frame.
# Open nodes are clamped (slip and gradient = 0) and interior nodes continuous and smooth; the end
# elements carry a fault tip and are tapers without a BC row.
slip_local = np.einsum("eji,ej->ei", els.rot_mats, np.c_[slip_vector_x, slip_vector_y])  # R^T s
mesh = bemcs.Mesh(els, bc_type=["s"] * n_els, bc_value=slip_local, mu_minus=mu, mu_plus=mu)

# Slip and slip gradient at the 3 nodes of every element, global (x, y) components
slip_mat, slip_gradient_mat = bemcs.bemAssembly.slip_node_matrices(els, geometry=geometry)

# Solve for quadratic node coefficients (in local (t, n) coordinates)
coeffs = bemcs.solve(mesh, geometry=geometry, mu=mu, nu=nu)

# %% [markdown]
# # Plot Slip along the fault

# %%
# provide number of points per fault element
n_eval = 15

# calculate slip at all nodes
x_nodes = np.zeros((3 * n_els, 1))
x_nodes[0::3, 0] = els.x1
x_nodes[2::3, 0] = els.x2
x_nodes[1::3, 0] = els.x_centers
fault_slip = slip_mat @ coeffs
fault_slip_x_nodes = fault_slip[0::2]
fault_slip_y_nodes = fault_slip[1::2]

x_obs, y_obs, fault_slip = mesh.values_on_elements(coeffs, "slip", geometry=geometry, n_eval=n_eval)
fault_slip_x, fault_slip_y = fault_slip[:, 0], fault_slip[:, 1]

# plot as a line plot
plt.figure(figsize=(9, 7))
plt.subplot(2, 1, 1)
plt.stem(x_nodes, fault_slip_x_nodes, "o", label="x-slip")
plt.plot(x_obs, fault_slip_x, "k-")
plt.grid()
plt.legend()
plt.xlabel("x")
plt.ylabel("slip")
plt.subplot(2, 1, 2)
plt.stem(x_nodes, fault_slip_y_nodes, "o", label="y-slip")
plt.plot(x_obs, fault_slip_y, "k-")
plt.grid()
plt.legend()
plt.xlabel("x")
plt.ylabel("slip")
plt.show()

# plot as quiver in map/cross-sectional view
plt.figure(figsize=(10, 5))
for i in range(n_els):
    plt.plot([els.x1[i], els.x2[i]], [els.y1[i], els.y2[i]], "k.-", linewidth=2)
plt.plot(els.x_centers, els.y_centers, "ro")
plt.quiver(
    x_obs,
    y_obs,
    fault_slip_x,
    fault_slip_y,
    np.sqrt(fault_slip_x**2 + fault_slip_y**2),
    cmap="cool",
)
plt.gca().set_aspect("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(label="|slip|")
plt.title("Continuous fault slip on a non-planar fault")
plt.show()

# %%
# Compute displacement and stress kernels

# Compute displacements along a line
n_pts = 15
dr = 3e-2

els_obs = els_normals(els, n_pts)

x_plot = np.hstack(
    (
        np.linspace(
            els.x1[0] + dr * els.x_shears[0], els.x2[0] - dr * els.x_shears[0], n_pts
        ),
        np.linspace(
            els.x1[1] + dr * els.x_shears[1], els.x2[1] - dr * els.x_shears[1], n_pts
        ),
        np.linspace(
            els.x1[2] + dr * els.x_shears[2], els.x2[2] - dr * els.x_shears[2], n_pts
        ),
    )
).reshape(-1, 1)
y_plot = np.hstack(
    (
        np.linspace(
            els.y1[0] + dr * els.y_shears[0], els.y2[0] - dr * els.y_shears[0], n_pts
        ),
        np.linspace(
            els.y1[1] + dr * els.y_shears[1], els.y2[1] - dr * els.y_shears[1], n_pts
        ),
        np.linspace(
            els.y1[2] + dr * els.y_shears[2], els.y2[2] - dr * els.y_shears[2], n_pts
        ),
    )
).reshape(-1, 1)

# Stress kernels along the line
_, S, _ = bemcs.get_kernels(x_plot, y_plot, mesh, geometry=geometry, mu=mu, nu=nu)
n_plot = len(x_plot)
sxx, syy, sxy = (S[:, :, k, :].reshape(n_plot, -1) @ coeffs for k in range(3))

# tractions on the element planes (global x, y components)
T = bemcs.traction_kernels(S, els_obs.x_normals, els_obs.y_normals)
tx, ty = (T[:, :, k, :].reshape(n_plot, -1) @ coeffs for k in range(2))

plt.figure(figsize=(8, 5))
plt.subplot(2, 1, 1)
for i in range(n_els):
    plt.plot([els.x1[i], els.x2[i]], [0, 0], "k.-")
plt.plot(x_obs, fault_slip_x, ".-", linewidth=2, label="$s_x$")
plt.plot(x_obs, fault_slip_y, ".-", linewidth=2, label="$s_y$")
plt.legend()
plt.xticks(())
plt.ylabel("slip")
plt.subplot(2, 1, 2)
for i in range(n_els):
    plt.plot([els.x1[i], els.x2[i]], [0, 0], "k.-")
plt.plot(x_plot, tx, ".-", linewidth=2, label="$t_x$")
plt.plot(x_plot, ty, ".-", linewidth=2, label="$t_y$")
plt.legend()
plt.xlabel("x")
plt.ylabel("tractions")
plt.ylim(-2, 2)
plt.show()

plt.figure(figsize=(14, 4))
# plt.figure(figsize=(8, 6))

for i in range(n_els):
    plt.plot([els.x1[i], els.x2[i]], [els.y1[i], els.y2[i]], "k-", linewidth=2)
# plt.plot(els.x_centers, els.y_centers, "ro")
q_slip = plt.quiver(
    x_obs,
    y_obs,
    fault_slip_x,
    fault_slip_y,
    np.sqrt(fault_slip_x**2 + fault_slip_y**2),
    cmap="copper",
)
q_traction = plt.quiver(
    x_plot,
    y_plot,
    tx,
    ty,
    np.sqrt(tx**2 + ty**2),
    cmap="cool",
    width=0.005,
)
# plt.gca().set_aspect("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim([-1.25, 1.25])
plt.ylim([-0.6, 0.6])
plt.xticks([-1.25, -0.75, 0, 0.75, 1.25])
plt.yticks([-0.6, -0.3, 0.0, 0.3, 0.6])
plt.colorbar(q_traction, label="$|\\tau|$", pad=-0.05)
plt.colorbar(q_slip, label=r"$|\mathbf{s}|$")
plt.gca().set_aspect("equal", adjustable="box")
# plt.savefig("spatially_smooth_slip_tractions_quiver.pdf")
plt.show()
