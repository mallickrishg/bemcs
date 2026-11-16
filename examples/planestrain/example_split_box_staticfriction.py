# %%
# %load_ext autoreload  # (IPython magic, disabled in .py)
# %autoreload 2  # (IPython magic, disabled in .py)
# %config InlineBackend.figure_format = "retina"  # (IPython magic, disabled in .py)

import numpy as np
import matplotlib.pyplot as plt
from importlib import reload

import bemcs
plt.rcParams.update({"font.size": 12})

class els_labels:
    def __init__(self, els, H, L):
        xmid = np.copy(els.x_centers)
        ymid = np.copy(els.y_centers)
        self.top = ymid == H  # top
        self.right = xmid == L  # right
        self.bot = ymid == -H  # top
        self.left = xmid == -L  # left
        self.interior = (xmid > -L) & (xmid < L) & (ymid > -H) & (ymid < H)

# %% [markdown]
# # Wavy fault in a box
# Below we consider a toy problem based on laboratory rock deformation experiments. A sinusoidal fault is embedded inside a rectangular uniform elastic rectangular box. 
#
# The boundary conditions for this problem are as follows:
# - $u_x = 1, u_y = 0$ at the top
# - $u_x = -1, u_y = 0$ at the bottom
# - traction-free boundary conditions ($\tau_x,\tau_y = 0$) on the left & right boundaries 
# - mixed: static friction ($\tau_s - f\tau_n= 0$) and no tensile slip ($s_n= 0$) at interior fault elements

# %%
# Elastic parameters
mu = 1
nu = 0.25
friction_coeff = 0.2

# %%
# Create a box mesh with a sinusoidal fault in the middle
L = 4.0
H = 4.0
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

# %%
# Across the middle: Create a sinusoidal fault centered around y=0
ngridf = 10

x = np.linspace(-1.0 * L, 1.0 * L, int(2 * ngridf))
z = 0.1 * np.sin(2 * np.pi * x / 2 / 2)
# Force the first and last z-values to be 0
z[0] = 0
z[-1] = 0
x1_across = x[0:-1]
x2_across = x[1:]
y1_across = z[0:-1]
y2_across = z[1:]

els = bemcs.initialize_els()
els.x1 = np.concatenate([x1_box, x1_across])
els.y1 = np.concatenate([y1_box, y1_across])
els.x2 = np.concatenate([x2_box, x2_across])
els.y2 = np.concatenate([y2_box, y2_across])

bemcs.standardize_els_geometry(els, reorder=False)
n_els = len(els.x1)
bemcs.plot_els_geometry(els)

# add labels for top,right,bot,left,interior
labels = els_labels(els, H, L)

# %% [markdown]
# ## Mesh: quadratic slip on every element (general, element-DOF formulation)
# Open nodes, overlapping interior nodes and triple junctions are identified automatically by
# the continuity operator. Unknowns are interleaved [t0, n0, t1, n1, ...] per node.
#
# The interior fault obeys static friction (t_s - f t_n = 0, no normal slip), a linear combination
# that `bemcs.solve` does not offer as a BC type, so this example assembles its system from the
# building blocks (kernels, traction kernels, slip node matrices, continuity operator).

# %%
mesh = bemcs.Mesh(els, "slip")
geometry = "planestrain"

# %% [markdown]
# # Boundary Conditions
# Here we implement the various boundary conditions as a big vector that goes in the right hand side of the Integral equations. The convention we follow is:
# - BCs at element centres (2 per element)
# - continuity rows at open, 2-overlap and triple-junction nodes (RHS = 0)

# %%
# Boundary conditions are
# top: ux = 1, uy = 0
# right: traction-free
# bot: ux = -1, uy = 0
# left: traction-free
# interior: static friction, no normal slip
bc_x = np.zeros((n_els, 1))
bc_y = np.zeros((n_els, 1))

bc_x[labels.top] = 1.0
bc_y[labels.top] = -0.0
bc_x[labels.bot] = -1.0
bc_y[labels.bot] = -0.0

bc_x[labels.right] = -0.0
bc_y[labels.right] = 0.0
bc_x[labels.left] = 0.0
bc_y[labels.left] = 0.0

# traction BCs at interior fault centers
# set t_s - f*t_n = 0, or t_s - f*t_n = residual_traction
# s_n = 0
bc_x[labels.interior] = 0.0  # residual or pre-stress level
# bc_x[labels.interior] = np.random.normal(
#     loc=0, scale=0.1, size=(len(x1_across), 1)
# )  # random residual stress field
bc_y[labels.interior] = 0.0

BC_c = np.c_[bc_x, bc_y].reshape(-1, 1)  # [x0, y0, x1, y1, ...]

# %% [markdown]
# # Design Matrix System 
# Here we implement various boundary conditions at the different types of nodes we have
# - At element centers, we implement either displacement or traction boundary conditions
# - At open nodes, we force slip to be 0
# - At 2-overlap (interior) nodes we have continuity & smoothness of slip
# - At triple junctions, we are currently implementing 2 equations for conservation of slip and 4 equations for smoothness (these choices are likely to change in the future)

# %%
# Patch center locations (need to be shifted an infinitesimal amount in -ve unit normal direction)
obs_xy = (
    np.vstack((els.x_centers, els.y_centers)).T
    - np.vstack((els.x_normals, els.y_normals)).T * 1e-8
)
x_obs = obs_xy[:, 0]
y_obs = obs_xy[:, 1]

# Kernels evaluated ONLY at the element centres
U, S, info = bemcs.get_kernels(x_obs, y_obs, mesh, geometry=geometry, mu=mu, nu=nu)

# Traction kernels in local (shear, normal) coordinates of each element: [ts, tn]
T = bemcs.traction_kernels(S, els.x_normals, els.y_normals, els.x_shears, els.y_shears)

# slip at element nodes in local (t, n) coordinates
matrix_slip, matrix_slip_gradient = bemcs.bemAssembly.slip_node_matrices(
    els, geometry=geometry, reference="local"
)

# Linear operator for central node BCs: rows [x0, y0, x1, y1, ...]
kerneleval_x = np.zeros((n_els, mesh.n_columns(geometry=geometry)))
kerneleval_y = np.zeros_like(kerneleval_x)


def rows(K, idx, c):
    """Rows of kernel component c at the elements idx, interleaved columns."""
    return K[idx, :, c, :].reshape(int(np.sum(idx)), -1)


for i in range(5):
    if i == 0:  # top: displacement
        idx = labels.top.flatten()
        kerneleval_x[idx], kerneleval_y[idx] = rows(U, idx, 0), rows(U, idx, 1)
    elif i == 1:  # bottom: displacement
        idx = labels.bot.flatten()
        kerneleval_x[idx], kerneleval_y[idx] = rows(U, idx, 0), rows(U, idx, 1)
    elif i == 2:  # left: traction
        idx = labels.left.flatten()
        kerneleval_x[idx], kerneleval_y[idx] = rows(T, idx, 0), rows(T, idx, 1)
    elif i == 3:  # right: traction
        idx = labels.right.flatten()
        kerneleval_x[idx], kerneleval_y[idx] = rows(T, idx, 0), rows(T, idx, 1)
    else:  # interior fault: t_s - f t_n = 0 and no normal slip at the centre node
        idx = labels.interior.flatten()
        kerneleval_x[idx] = rows(T, idx, 0) - friction_coeff * rows(T, idx, 1)
        centre_normal = 2 * (3 * np.flatnonzero(idx) + 1) + 1
        kerneleval_y[idx] = matrix_slip[centre_normal]

# populate matrix_system for central nodes
matrix_system_c = np.zeros((2 * n_els, mesh.n_columns(geometry=geometry)))
matrix_system_c[0::2, :] = kerneleval_x
matrix_system_c[1::2, :] = kerneleval_y

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
plt.figure(figsize=(15, 6))
plt.imshow(matrix_system, cmap="coolwarm")
plt.colorbar()
plt.clim(-0.1, 0.1)
plt.show()

# %% [markdown]
# # Solve system

# %%
# compute quadratic node coefficients (in local (s,n) coordinates)
quadratic_coefs = np.linalg.inv(matrix_system) @ BCvector
print("Linear Operator Condition Number:", np.linalg.cond(matrix_system))

c = quadratic_coefs.ravel()

# compute stress components exactly on the fault centers
sxy_f = S[:, :, 2, :].reshape(n_els, -1) @ c
syy_f = S[:, :, 1, :].reshape(n_els, -1) @ c

# %% [markdown]
# # Plot displacement and stress components in the medium

# %%
# Displacement and stress fields on a grid (plane strain: ux, uy, sxx, syy, sxy)
n_obs = 50
width = 2.001
fig, axes, fields = bemcs.plot_fields(
    mesh, c, geometry=geometry, mu=mu, nu=nu,
    xlim=(-width, width), ylim=(-width, width), n=(n_obs, n_obs),
)
plt.show()
x_obs, y_obs = fields["x"].ravel(), fields["y"].ravel()
ux, uy = fields["ux"].ravel(), fields["uy"].ravel()
sxx, syy, sxy = fields["sxx"].ravel(), fields["syy"].ravel(), fields["sxy"].ravel()

# %% [markdown]
# ## Plot slip on the wavy fault

# %%
# Extract slip at nodes from quadratic coefficients in the required coordinate system (global - x,y or local - s,n)
slip_nodes = matrix_slip @ quadratic_coefs
slip_nodes_x = slip_nodes[0::2]
slip_nodes_y = slip_nodes[1::2]
slip_vector = np.hstack((slip_nodes_x, slip_nodes_y))

# compute traction components
ts = T[:, :, 0, :].reshape(n_els, -1) @ c
tn = T[:, :, 1, :].reshape(n_els, -1) @ c
# ts = kerneleval_x @ quadratic_coefs
# tn = kerneleval_y @ quadratic_coefs

plt.figure(figsize=(12, 6))
xplot = np.zeros([len(els.x_centers[labels.interior]) * 3, 1])
s_s = np.zeros_like(xplot)
s_n = np.zeros_like(xplot)

xplot[0::3, 0] = els.x1[labels.interior]
xplot[1::3, 0] = els.x_centers[labels.interior]
xplot[2::3, 0] = els.x2[labels.interior]
for i in range(0, 3):
    s_s[i::3] = slip_nodes_x[i::3][labels.interior]
    s_n[i::3] = slip_nodes_y[i::3][labels.interior]

plt.plot(xplot, s_s, "b.-", label="$s_s$")
plt.plot(xplot, s_n, "g.-", label="$s_n$")
plt.plot(els.x_centers[labels.interior], ts[labels.interior], "bo-", label="$t_s$")
plt.plot(els.x_centers[labels.interior], tn[labels.interior], "go-", label="$t_n$")
plt.plot(
    els.x_centers[labels.interior], sxy_f[labels.interior], "b--", label="$\sigma_{xy}$"
)
plt.plot(
    els.x_centers[labels.interior], syy_f[labels.interior], "g--", label="$\sigma_{yy}$"
)
plt.xlabel("x")
plt.ylabel("slip or stress")
plt.ylim([-0.5, 2.2])
plt.xlim([-2, 2])
plt.legend()
plt.grid()
plt.show()

# %% [markdown]
# ## Plot shear stress in the medium

# %%
n_skip_plot = 9
plt.figure(figsize=(12, 12))
plt.subplot(2, 1, 1)
toplot = sxy.reshape(n_obs, n_obs)
maxval = 1  # np.max(np.abs(toplot))
minval = -maxval
img = plt.contourf(
    x_obs.reshape(n_obs, n_obs),
    y_obs.reshape(n_obs, n_obs),
    toplot,
    cmap="PiYG",
    vmin=minval,
    vmax=maxval,
    levels=np.linspace(minval, maxval, 21),
)
plt.clim([-1, 1])
plt.contour(
    x_obs.reshape(n_obs, n_obs),
    y_obs.reshape(n_obs, n_obs),
    toplot,
    linewidths=0.5,
    colors="k",
    levels=np.linspace(-maxval, maxval, 21),
)
plt.colorbar(img, label="$\Delta\sigma_{xy}$")
for i in range(n_els):
    plt.plot([els.x1[i], els.x2[i]], [els.y1[i], els.y2[i]], "k-")
plt.xlim([np.min(x_obs), np.max(x_obs)])
plt.ylim([np.min(y_obs), np.max(y_obs)])
plt.gca().set_aspect("equal", adjustable="box")
plt.quiver(
    x_obs[0::n_skip_plot],
    y_obs[0::n_skip_plot],
    ux[0::n_skip_plot],
    uy[0::n_skip_plot],
    width=0.002,
    scale=30,
)
plt.xlabel("x")
plt.ylabel("y")
plt.xlim([-2, 2])
plt.ylim([-1, 1])

plt.subplot(2, 1, 2)
toplot = (sxx + syy).reshape(n_obs, n_obs) / 2
img = plt.contourf(
    x_obs.reshape(n_obs, n_obs),
    y_obs.reshape(n_obs, n_obs),
    toplot,
    cmap="coolwarm",
    vmin=minval,
    vmax=maxval,
    levels=np.linspace(minval, maxval, 21),
)
plt.clim([-1, 1])
plt.contour(
    x_obs.reshape(n_obs, n_obs),
    y_obs.reshape(n_obs, n_obs),
    toplot,
    linewidths=1,
    colors="k",
    levels=np.linspace(-maxval, maxval, 21),
)
plt.colorbar(img, label="$\Delta$pressure")
for i in range(n_els):
    plt.plot([els.x1[i], els.x2[i]], [els.y1[i], els.y2[i]], "k.-")
plt.xlim([-2, 2])
plt.ylim([-1, 1])
plt.gca().set_aspect("equal", adjustable="box")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
