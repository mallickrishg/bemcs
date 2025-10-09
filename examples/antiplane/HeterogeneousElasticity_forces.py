# %% import libraries
import matplotlib.pyplot as plt
import numpy as np
import bemcs
import pandas as pd

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

# construct separate mesh just for 's' BCtype
els_s = bemcs.initialize_els()
els_s.x1 = els.x1[BCtype == "s"]
els_s.y1 = els.y1[BCtype == "s"]
els_s.x2 = els.x2[BCtype == "s"]
els_s.y2 = els.y2[BCtype == "s"]

bemcs.standardize_els_geometry(els, reorder=False)
bemcs.standardize_els_geometry(els_s, reorder=False)

n_els = len(els.x1)
bemcs.plot_els_geometry(els)
# %%
# Define observation points
# need to shift observation points by an infinitesimal amount to sample discontinuity either in u or du/dn
dr = -1e-6
xo = els.xc + dr * els.x_normals
yo = els.yc + dr * els.y_normals
# %%
