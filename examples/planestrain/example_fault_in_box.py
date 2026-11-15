# %%
import numpy as np
import matplotlib.pyplot as plt

import bemcs


# define els_obs as an object containing x_normals and y_normals with same as size as n_pts
class els_normals:
    def __init__(self, els, n):
        n_els = len(els.x1)
        self.x = np.zeros(n_els * n)
        self.y = np.zeros(n_els * n)
        self.x_shears = np.zeros(n_els * n)
        self.y_shears = np.zeros(n_els * n)
        self.x_normals = np.zeros(n_els * n)
        self.y_normals = np.zeros(n_els * n)
        self.lengths = np.zeros(n_els * n)

        for i in range(n_els):
            xvec = np.linspace(els.x1[i], els.x2[i], n + 1)
            yvec = np.linspace(els.y1[i], els.y2[i], n + 1)
            self.x[i * n : (i + 1) * n] = 0.5 * (xvec[1:] + xvec[0:-1])
            self.y[i * n : (i + 1) * n] = 0.5 * (yvec[1:] + yvec[0:-1])
            self.x_shears[i * n : (i + 1) * n] = els.x_shears[i]
            self.y_shears[i * n : (i + 1) * n] = els.y_shears[i]
            self.x_normals[i * n : (i + 1) * n] = els.x_normals[i]
            self.y_normals[i * n : (i + 1) * n] = els.y_normals[i]
            self.lengths[i * n : (i + 1) * n] = els.lengths[i] / n


# %% [markdown]
# # Construct a fault inside an arbitrary sized box domain

# %%
# Create a box mesh with a sinusoidal fault in the middle
L = 2.0
H = L * 1.0
ngrid = 4
corners = np.array([[-L, H], [L, H], [L, -H], [-L, -H], [-L, H]])

# Create arrays of the x and y coordinates of the corners
x, z = [], []
for i in range(4):
    lsegment = np.sqrt(
        (corners[i, 0] - corners[(i + 1), 0]) ** 2
        + (corners[i, 1] - corners[(i + 1), 1]) ** 2
    )
    xi = np.linspace(
        corners[i][0],
        corners[(i + 1) % 4][0],
        num=round(lsegment * ngrid / H),
        endpoint=False,
    )
    zi = np.linspace(
        corners[i][1],
        corners[(i + 1) % 4][1],
        num=round(lsegment * ngrid / H),
        endpoint=False,
    )
    x.append(xi)
    z.append(zi)
x.append(corners[0, 0])
z.append(corners[0, 1])
x = np.hstack(x)
z = np.hstack(z)

x1_box = x[0:-1]
x2_box = x[1:]
y1_box = z[0:-1]
y2_box = z[1:]

# Across the middle: Create a sinusoidal fault centered around y=0
Lfault = 1.0
x = np.linspace(-0.5 * Lfault, 0.5 * Lfault, int(4 * ngrid))
z = 0.0 * Lfault * np.sin(4 * np.pi * x / 2 / Lfault)
x1_across = x[0:-1]
x2_across = x[1:]
y1_across = z[0:-1]
y2_across = z[1:]

# Elastic parameters
mu = 1
nu = 0.25

els = bemcs.initialize_els()
els.x1 = np.concatenate([x1_box, x1_across])
els.y1 = np.concatenate([y1_box, y1_across])
els.x2 = np.concatenate([x2_box, x2_across])
els.y2 = np.concatenate([y2_box, y2_across])

bemcs.standardize_els_geometry(els, reorder=False)
n_els = len(els.x1)
bemcs.plot_els_geometry(els)

# %% [markdown]
# # Imposed tractions as BC (specified in terms of a global stress tensor)
# Traction vector, as a spatial variable, is calculated from the global uniform stress tensor as:
#
# $t(\mathbf{x}) = \sigma \cdot \hat{n}(\mathbf{x})$

# %%
# Global stress tensor
sigma_xx = 0
sigma_yy = -sigma_xx
sigma_xy = 1

# Traction imposed at central nodes of each patch (these are in x,y coordinates)
traction_vector_x = np.hstack(
    [
        sigma_xx * els.x_normals[0 : len(x1_box)]
        + sigma_xy * els.y_normals[0 : len(x1_box)],
        np.zeros_like(x1_across),
    ]
)
traction_vector_y = np.hstack(
    [
        sigma_xy * els.x_normals[0 : len(x1_box)]
        + sigma_yy * els.y_normals[0 : len(x1_box)],
        np.zeros_like(x1_across),
    ]
)

plt.figure(figsize=(23, 4))
for i in range(n_els):
    plt.plot([els.x1[i], els.x2[i]], [els.y1[i], els.y2[i]], "k-")
plt.scatter(
    els.x_centers,
    els.y_centers,
    c=np.sqrt(traction_vector_x**2 + traction_vector_y**2),
    marker="o",
    cmap="magma_r",
)
plt.colorbar(label="imposed traction")
plt.quiver(
    els.x_centers, els.y_centers, traction_vector_x, traction_vector_y, width=0.005
)
plt.xlabel("x")
plt.ylabel("y")
plt.gca().set_aspect("equal", adjustable="box")
plt.show()

# %% [markdown]
# # Mesh, kernels and continuity conditions
# All elements carry quadratic slip with 3 nodes per element (the general, element-DOF formulation).
# Unknowns are ordered element-major, then node, then direction (t, n): [t0, n0, t1, n1, ...].

# %%
mesh = bemcs.Mesh(els, "slip")  # default basis for slip: quadratic
geometry = "planestrain"

# stress kernels evaluated ONLY at the element centres: S has shape (n_els, 3 n_els, 3, 2)
_, S, info = bemcs.get_kernels(
    els.x_centers, els.y_centers, mesh, geometry=geometry, mu=mu, nu=nu
)
print(info)

# traction kernels t = sigma . n in global (x, y) coordinates: (n_els, 3 n_els, 2, 2)
T = bemcs.traction_kernels(S, els.x_normals, els.y_normals)

# continuity and smoothness rows at open, overlapping and triple-junction nodes (RHS = 0)
matrix_continuity, rows = mesh.continuity_operator(geometry=geometry)

# %% [markdown]
# # Assemble the design matrix and boundary conditions

# %%
# traction BCs at element centres: rows [tx0, ty0, tx1, ty1, ...]
matrix_system_c = bemcs.kernel_matrix(T)
BC_c = np.c_[traction_vector_x, traction_vector_y].reshape(-1, 1)

# tip elements (with an open end node) are clamped tapers: they get no BC row
keep = mesh.bc_elements("slip")
matrix_system_c = matrix_system_c.reshape(n_els, 2, -1)[keep].reshape(2 * len(keep), -1)
BC_c = BC_c.reshape(n_els, 2, -1)[keep].reshape(-1, 1)

matrix_system = np.vstack((matrix_system_c, matrix_continuity))
BCvector = np.vstack((BC_c, np.zeros((len(matrix_continuity), 1))))
print("unknowns:", mesh.n_columns(geometry=geometry), " equations:", len(matrix_system))

# compute quadratic node coefficients (slip in local (t, n) coordinates)
quadratic_coefs = np.linalg.pinv(matrix_system) @ BCvector
print("Linear Operator Condition Number:", np.linalg.cond(matrix_system))

# evaluate slip (global x, y components) on a finer set of points along the elements
n_eval = 4
els_obs = els_normals(els, n_eval)
_, _, slip_xy = mesh.values_on_elements(quadratic_coefs, "slip", geometry=geometry, n_eval=n_eval)
sx, sy = slip_xy[:, 0], slip_xy[:, 1]

# %% [markdown]
# # Plot displacements and stresses

# %%
faultindex = np.arange((n_els - len(x1_across)) * n_eval, n_els * n_eval)

plt.figure(figsize=(7, 7))
for i in range(n_els):
    plt.plot([els.x1[i], els.x2[i]], [els.y1[i], els.y2[i]], "k.-")
plt.scatter(
    els_obs.x,
    els_obs.y,
    c=np.sqrt(sx**2 + sy**2),
    marker="o",
    cmap="BuGn",
)
plt.colorbar(label="slip (BEM)")
plt.quiver(
    els_obs.x[faultindex],
    els_obs.y[faultindex],
    sx[faultindex],
    sy[faultindex],
    width=0.005,
)
plt.xlabel("x")
plt.ylabel("y")
# plt.gca().set_aspect("equal", adjustable="box")
plt.show()

# plot fault slip
plt.figure(figsize=(7, 4))
plt.plot(els_obs.x[faultindex], sx[faultindex], ".-", label="$s_x$")
plt.plot(els_obs.x[faultindex], sy[faultindex], ".-", label="$s_y$")
plt.xlabel("x")
plt.ylabel("slip")
plt.legend()
plt.show()
