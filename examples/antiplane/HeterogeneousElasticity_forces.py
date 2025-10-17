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
mu = 1  # in GPa

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
bc_mask = BCtype != "h"
els_s = bemcs.initialize_els()
els_s.x1 = els.x1[bc_mask]
els_s.y1 = els.y1[bc_mask]
els_s.x2 = els.x2[bc_mask]
els_s.y2 = els.y2[bc_mask]
# construct a mesh only for 'h' BCtype
els_h = bemcs.initialize_els()
els_h.x1 = els.x1[~bc_mask]
els_h.y1 = els.y1[~bc_mask]
els_h.x2 = els.x2[~bc_mask]
els_h.y2 = els.y2[~bc_mask]

bemcs.standardize_els_geometry(els, reorder=False)
bemcs.standardize_els_geometry(els_s, reorder=False)
bemcs.standardize_els_geometry(els_h, reorder=False)

n_els = len(els.x1)
bemcs.plot_els_geometry(els)

# Define observation points
# need to shift observation points by an infinitesimal amount to sample discontinuity either in u or du/dn
dr = -1e-9

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
K_n = K_x * nxvec[:, None] + K_y * nyvec[:, None]

# compute slip kernels for "s" & "t" BCtype elements (will be a square matrix of dimension N = stride x N_slip_elements)
matrix_slip_c, matrix_slip_nodes, BC_slip_c, BC_slip_nodes = (
    GF.construct_linearoperator_slip(els_s, BCtype[bc_mask], BCval[bc_mask])
)

plt.figure(figsize=(8, 8))
plt.imshow(K_n, cmap="seismic")
plt.colorbar()
plt.clim([-0.5, 0.5])
plt.show()
plt.figure(figsize=(8, 8))
plt.imshow(np.vstack((matrix_slip_c, matrix_slip_nodes)), cmap="seismic")
plt.colorbar()
plt.clim([-0.2, 0.2])
plt.show()
# %% Assemble BIE matrices for slip & force kernels and appropriate Boundary Conditions
