# %% import libraries
import matplotlib.pyplot as plt
import numpy as np
import bemcs
import pandas as pd
import GF
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")
# %% Read source file and mesh of the domain
fileinput = "HeterogeneousDomainMesh.csv"
connectvitiyfile = "HeterogeneousDomainMeshConnectivity.csv"
# read mesh file
datain = pd.read_csv(fileinput)

x1 = datain["x1"].values
x2 = datain["x2"].values
y1 = datain["z1"].values
y2 = datain["z2"].values
BCtype = datain["BC_type"].values
BCval = datain["value"].values
connmatrix = pd.read_csv(connectvitiyfile, header=None).values

# Elastic parameter
mu = 1
# in normalized units (only use 1 since mu variations are accounted for in BCval)

# construct mesh
els = bemcs.initialize_els()
# specify stride for problem (antiplane - 3; planestrain - 6)
stride = 3
# construct mesh from mesh file
els.x1 = x1
els.y1 = y1
els.x2 = x2
els.y2 = y2

# construct separate mesh just for 's', 'u' or 't' BCtype
# 'h' type BC elements are force elements (used for 'heterogeneous' material properties)
bc_mask = BCtype != "h"
els_s = bemcs.initialize_els()
els_s.x1 = els.x1[bc_mask]
els_s.y1 = els.y1[bc_mask]
els_s.x2 = els.x2[bc_mask]
els_s.y2 = els.y2[bc_mask]

# standardize element geometry
bemcs.standardize_els_geometry(els, reorder=False)
bemcs.standardize_els_geometry(els_s, reorder=False)
# plot mesh
n_els = len(els.x1)
bemcs.plot_els_geometry(els)

# Define observation points
# need to shift observation points by an infinitesimal amount to sample discontinuity either in u or du/dn
dr = -1e-6

xo = np.hstack(
    (
        els_s.x_centers + dr * els_s.x_normals,
        els.x_centers[connmatrix[:, 1].astype(int)]
        + dr * els.x_normals[connmatrix[:, 1].astype(int)],
    )
).flatten()
yo = np.hstack(
    (
        els_s.y_centers + dr * els_s.y_normals,
        els.y_centers[connmatrix[:, 1].astype(int)]
        + dr * els.y_normals[connmatrix[:, 1].astype(int)],
    )
).flatten()

# %% construct kernels
# compute force kernels (assuming trapezoidal spatial functions) for "h" BCtype elements [Nobs x Nsources]
K_x, K_y, _ = GF.get_kernels_trapezoidalforce(xo, yo, els, connmatrix)
# compute traction kernel by dotting stress with normal vector
nxvec = np.hstack(
    (
        els_s.x_normals,
        els.x_normals[connmatrix[:, 1].astype(int)],
    )
)
nyvec = np.hstack(
    (
        els_s.y_normals,
        els.y_normals[connmatrix[:, 1].astype(int)],
    )
)

# convert from stress to traction kernels [Nobs x Nsources]
Kforce_n = K_x * nxvec[:, None] + K_y * nyvec[:, None]

# compute slip kernels for "s" & "t" BCtype elements (will be a square matrix of dimension N = stride x N_slip_elements)
matrix_slip_c, matrix_slip_nodes, BC_slip_c, BC_slip_nodes = (
    GF.construct_linearoperator_slip(els_s, BCtype[bc_mask], BCval[bc_mask])
)

# compute stress kernels for "s" & "t" BCtype elements [Nobs x Nsources]
K_x, K_y, _ = bemcs.get_displacement_stress_kernel_slip_antiplane(xo, yo, els_s, mu)
# convert from stress to traction kernels [Nobs x Nsources]
Kslip_n = K_x * nxvec[:, None] + K_y * nyvec[:, None]


# %% Assemble BIE matrices for slip & force kernels and appropriate Boundary Conditions

N_equations = len(els_s.x1) + len(
    connmatrix[:, 0]
)  # number of equations at mesh centers
N_unknowns = stride * len(els_s.x1) + len(connmatrix[:, 0])  # number of unknowns

# linear operator for slip & force elements
matrix_system = np.zeros((N_equations, N_unknowns))

# populate matrix_system for slip elements (check for bc_label)
for i in range(len(els_s.x1)):
    if BCtype[i] == "s":
        matrix_system[i, 0 : stride * len(els_s.x1)] = matrix_slip_c[i, :]
    elif BCtype[i] == "t":
        matrix_system[i, :] = np.hstack((matrix_slip_c[i, :], Kforce_n[i, :]))
    else:
        ValueError("boundary condition label not recognized")

# Add bottom part of matrix_system for 'h' type BC elements
n_bottom_rows = len(connmatrix[:, 0])
start_row = len(els_s.x1)

# Create diagonal matrix of shear modulus gradients
alpha = np.diag(BCval[connmatrix[:, 1].astype(int)])

# Assemble [diag(BCval)*Kslip_n, I + diag(BCval)*Kforce_n]
matrix_system[start_row:, : stride * len(els_s.x1)] = (
    alpha @ Kslip_n[len(els_s.x1) :, :]
)
matrix_system[start_row:, stride * len(els_s.x1) :] = (
    np.eye(n_bottom_rows) + alpha @ Kforce_n[len(els_s.x1) :, :]
)
# append matrix_slip_nodes below matrix_system and maintain correct shape
matrix_system = np.vstack(
    (
        matrix_system,
        np.hstack(
            (
                matrix_slip_nodes,
                np.zeros((matrix_slip_nodes.shape[0], len(connmatrix[:, 0]))),
            )
        ),
    )
)

# assemble BC vector
BC_vector = np.vstack((BC_slip_c, np.zeros((len(connmatrix[:, 0]), 1)), BC_slip_nodes))

# %% solve Linear system
solution_vector = np.linalg.solve(matrix_system, BC_vector)

# extract quadratic coefficients for slip elements
quadcoefs = solution_vector[0 : stride * len(els_s.x1)]
forcecoefs = solution_vector[stride * len(els_s.x1) :]

# %% compute and plot displacement, displacement gradient fields inside the domain
xlimits = [-4, 4]
ylimits = [-2, 0]
nx_obs = 200
ny_obs = nx_obs
x_obs = np.linspace(-5, 5, nx_obs)
y_obs = np.linspace(-6, -1e-3, ny_obs)
x_obs, y_obs = np.meshgrid(x_obs, y_obs)
xo = x_obs.flatten().reshape(-1, 1)
yo = y_obs.flatten().reshape(-1, 1)
# compute kernels at observation points
Kslip_x, Kslip_y, Kslip_u = bemcs.get_displacement_stress_kernel_slip_antiplane(
    xo, yo, els_s, mu
)
Kforce_x, Kforce_y, Kforce_u = GF.get_kernels_trapezoidalforce(xo, yo, els, connmatrix)
# compute displacement and stress components
u = Kslip_u @ quadcoefs + Kforce_u @ forcecoefs
sx = Kslip_x @ quadcoefs + Kforce_x @ forcecoefs
sy = Kslip_y @ quadcoefs + Kforce_y @ forcecoefs

plt.figure(figsize=(8, 10))
plt.subplot(3, 1, 1)
toplot = u.reshape(ny_obs, nx_obs)
maxval = 0.5
minval = -maxval
levels = np.linspace(minval, maxval, 21)
plt.pcolor(
    xo.reshape(ny_obs, nx_obs),
    yo.reshape(ny_obs, nx_obs),
    toplot,
    cmap="coolwarm",
    vmin=minval,
    vmax=maxval,
)
for i in range(n_els):
    plt.plot(
        [els.x1[i], els.x2[i]],
        [els.y1[i], els.y2[i]],
        "k.-",
        linewidth=0.2,
        markersize=1,
    )
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
plt.ylim(ylimits)
plt.gca().set_aspect("equal", adjustable="box")

plt.subplot(3, 1, 2)
toplot = sx.reshape(ny_obs, nx_obs)
maxval = 2
minval = -maxval
levels = np.linspace(minval, maxval, 21)
plt.pcolor(
    xo.reshape(ny_obs, nx_obs),
    yo.reshape(ny_obs, nx_obs),
    toplot,
    cmap="RdYlBu_r",
    vmin=minval,
    vmax=maxval,
)
for i in range(n_els):
    plt.plot(
        [els.x1[i], els.x2[i]],
        [els.y1[i], els.y2[i]],
        "k.-",
        linewidth=0.2,
        markersize=1,
    )
plt.colorbar(label="$u_{,x}$")
plt.contour(
    xo.reshape(ny_obs, nx_obs),
    yo.reshape(ny_obs, nx_obs),
    toplot,
    colors="k",
    levels=levels,
    linewidths=0.5,
)
plt.xlim(xlimits)
plt.ylim(ylimits)
plt.gca().set_aspect("equal", adjustable="box")

plt.subplot(3, 1, 3)
toplot = sy.reshape(ny_obs, nx_obs)
plt.pcolor(
    xo.reshape(ny_obs, nx_obs),
    yo.reshape(ny_obs, nx_obs),
    toplot,
    cmap="RdYlBu_r",
    vmin=minval,
    vmax=maxval,
)
plt.colorbar(label="$u_{,y}$")
plt.contour(
    xo.reshape(ny_obs, nx_obs),
    yo.reshape(ny_obs, nx_obs),
    toplot,
    colors="k",
    levels=levels,
    linewidths=0.5,
)
for i in range(n_els):
    plt.plot(
        [els.x1[i], els.x2[i]],
        [els.y1[i], els.y2[i]],
        "k.-",
        linewidth=0.2,
        markersize=1,
    )
plt.xlim(xlimits)
plt.ylim(ylimits)
plt.gca().set_aspect("equal", adjustable="box")
plt.show()

# plot fields at the surface of the domain
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
index = yo == np.max(yo)
plt.plot(xo[index], u[index], ".-")
plt.xlabel("x ")
plt.ylabel("u")
plt.title("Surface displacement")
plt.xlim(xlimits)
plt.subplot(2, 1, 2)
plt.plot(xo[index], sx[index], ".-", label="$u_{,x}$")
plt.plot(xo[index], sy[index], ".-", label="$u_{,y}$")
plt.xlabel("x ")
plt.ylabel("displacement gradients")
plt.title("Surface displacement gradients")
plt.legend()
plt.xlim(xlimits)
# plt.ylim([-10, 10])
plt.show()

# %% plot slip on 's' mesh elements

xf, yf, slipnodes = bemcs.get_slipvector_on_fault_antiplane(
    els_s, quadcoefs.flatten(), 20
)
plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.plot(xf[yf == 0], slipnodes[yf == 0], "-")
plt.xlabel("x")
plt.ylabel("slip at nodes")
plt.xlim(xlimits)
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(slipnodes[yf < 0], yf[yf < 0], "-")
plt.ylabel("y")
plt.xlabel("slip at nodes")
plt.grid()
plt.show()

# %%
