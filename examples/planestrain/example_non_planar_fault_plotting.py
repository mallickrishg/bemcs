# %%
# %load_ext autoreload  # (IPython magic, disabled in .py)
# %autoreload 2  # (IPython magic, disabled in .py)
# %config InlineBackend.figure_format = "retina"  # (IPython magic, disabled in .py)

import numpy as np
import matplotlib.pyplot as plt

import bemcs

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
slip_vector_x = np.array([0.5, 0.8, 0.3])
slip_vector_y = np.array([0.3, 0.9, 0.1])
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

# Mesh of quadratic slip elements (used for kernels and plotting only)
mesh = bemcs.Mesh(els, labels="slip")

# Slip and slip gradient at the 3 nodes of every element, global (x, y) components.
# Row 2 * (3 e + k) + c: element e, node k (0 = start, 1 = centre, 2 = end), component c (x, y).
slip_mat, slip_gradient_mat = bemcs.bemAssembly.slip_node_matrices(
    els, geometry=geometry
)


# General system, written out by hand. bemcs.solve clamps fault tips (slip and slip
# gradient = 0) and gives tip elements no BC row, so on this 3-element fault it would ignore
# the slip imposed on both end elements. Instead (6 n_els equations, 6 n_els unknowns):
#   - slip = 0 at the two fault tips (slip gradient left free)
#   - slip imposed (global x, y) at the centre node of every element
#   - slip continuity and smoothness (slip-gradient continuity) at every interior node
def node_row(M, e, k, c):
    return M[2 * (3 * e + k) + c]


rows, rhs = [], []
for c in range(2):
    rows += [node_row(slip_mat, 0, 0, c), node_row(slip_mat, n_els - 1, 2, c)]
    rhs += [0.0, 0.0]
for e, s in enumerate(np.c_[slip_vector_x, slip_vector_y]):
    for c in range(2):
        rows.append(node_row(slip_mat, e, 1, c))
        rhs.append(s[c])
for e in range(n_els - 1):
    for M in (slip_mat, slip_gradient_mat):
        for c in range(2):
            rows.append(node_row(M, e, 2, c) - node_row(M, e + 1, 0, c))
            rhs.append(0.0)
design_mat = np.array(rows)

# Solve for quadratic node coefficients (in local (t, n) coordinates)
coeffs = np.linalg.solve(design_mat, np.array(rhs))

# %% [markdown]
# # Plot Slip along the fault

# %%
# Slip at the 3 nodes of every element and at n_eval points along each element (global x, y)
n_eval = 15
x_nodes = np.c_[els.x1, els.x_centers, els.x2].ravel()
slip_nodes = mesh.node_values(
    coeffs, "slip", geometry=geometry, frame="global"
).reshape(-1, 2)
x_obs, y_obs, fault_slip = mesh.values_on_elements(
    coeffs, "slip", geometry=geometry, n_eval=n_eval
)
fault_slip_x, fault_slip_y = fault_slip[:, 0], fault_slip[:, 1]

# fault trace, as one line per element
fault_x = np.c_[els.x1, els.x2].T
fault_y = np.c_[els.y1, els.y2].T

# plot as a line plot
plt.figure(figsize=(9, 7))
for i, label in enumerate(("x-slip", "y-slip")):
    plt.subplot(2, 1, i + 1)
    plt.stem(x_nodes, slip_nodes[:, i], "o", label=label)
    plt.plot(x_obs, fault_slip[:, i], "k-")
    plt.grid()
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("slip")
plt.show()

# plot as quiver in map/cross-sectional view
plt.figure(figsize=(10, 5))
plt.plot(fault_x, fault_y, "k.-", linewidth=2)
plt.plot(els.x_centers, els.y_centers, "ro")
plt.quiver(
    x_obs,
    y_obs,
    fault_slip_x,
    fault_slip_y,
    np.hypot(fault_slip_x, fault_slip_y),
    cmap="cool",
)
plt.gca().set_aspect("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(label="|slip|")
plt.title("Continuous fault slip on a non-planar fault")
plt.show()

# %%
# Tractions on the fault, at the same points along the elements. Points on an element get the
# two-sided average of the stress; traction (sigma . n) is continuous across the fault.
fields = bemcs.evaluate_fields(
    mesh, coeffs, x_obs, y_obs, geometry=geometry, mu=mu, nu=nu
)
nx = np.repeat(els.x_normals, n_eval)
ny = np.repeat(els.y_normals, n_eval)
tx = fields["sxx"] * nx + fields["sxy"] * ny
ty = fields["sxy"] * nx + fields["syy"] * ny

plt.figure(figsize=(8, 5))
plt.subplot(2, 1, 1)
plt.plot(fault_x, np.zeros_like(fault_x), "k.-")
plt.plot(x_obs, fault_slip_x, ".-", linewidth=2, label="$s_x$")
plt.plot(x_obs, fault_slip_y, ".-", linewidth=2, label="$s_y$")
plt.legend()
plt.xticks(())
plt.ylabel("slip")
plt.subplot(2, 1, 2)
plt.plot(fault_x, np.zeros_like(fault_x), "k.-")
plt.plot(x_obs, tx, ".-", linewidth=2, label="$t_x$")
plt.plot(x_obs, ty, ".-", linewidth=2, label="$t_y$")
plt.legend()
plt.xlabel("x")
plt.ylabel("tractions")
plt.ylim(-2, 2)
plt.show()

plt.figure(figsize=(14, 4))
plt.plot(fault_x, fault_y, "k-", linewidth=2)
q_slip = plt.quiver(
    x_obs,
    y_obs,
    fault_slip_x,
    fault_slip_y,
    np.hypot(fault_slip_x, fault_slip_y),
    cmap="copper",
)
q_traction = plt.quiver(
    x_obs,
    y_obs,
    tx,
    ty,
    np.hypot(tx, ty),
    cmap="cool",
    width=0.005,
)
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
