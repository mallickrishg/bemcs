# %% import libraries
import matplotlib.pyplot as plt
import numpy as np
import bemcs
import pandas as pd
import GF
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")
# Elastic parameter (shear modulus)
mu = 1.0
fileinput = "testing_mesh.csv"
datain = pd.read_csv(fileinput)
x1 = datain["x1"].values
x2 = datain["x2"].values
y1 = datain["z1"].values
y2 = datain["z2"].values

# boundary conditions must be 's' ONLY
BCtype = np.full_like(x1, "s", dtype=object)

els = bemcs.initialize_els()
els.x1, els.y1, els.x2, els.y2 = x1, y1, x2, y2
bemcs.standardize_els_geometry(els, reorder=False)
bemcs.plot_els_geometry(els)

# construct distance-weighted eigenmodes
r_mat = np.zeros((len(els.x1), len(els.x1)))
for i in range(len(els.x1)):
    r_mat[:, i] = np.sqrt(
        (els.x_centers[i] - els.x_centers) ** 2
        + (els.y_centers[i] - els.y_centers) ** 2
    )
C_r = np.exp(-(r_mat**1))
# compute K-L expansion of the distance-covariance matrix
KLmodes, _, _ = np.linalg.svd(C_r)

# Slip operator
MaxMode = 5
for i in range(MaxMode):
    index = i  # choose KL mode index
    weights = KLmodes[:, index]
    matrix_slip_c, matrix_slip_nodes, BC_slip_c, BC_slip_nodes = (
        GF.construct_linearoperator_slip(els, BCtype, weights)
    )
    # Only slip elements → simpler system
    BC_vector = np.vstack((BC_slip_c, BC_slip_nodes))
    matrix_system = np.vstack((matrix_slip_c, matrix_slip_nodes))
    # solve system of equations and get quadratic slip coefficients that match the given KLmode
    quadcoefs = np.linalg.solve(matrix_system, BC_vector)

    # compute slip at element nodes
    slip_mat, _ = bemcs.get_matrices_slip_slip_gradient_antiplane(els)
    slipnodecenters = slip_mat[1::3, :] @ quadcoefs

    xf, yf, slipnodes = bemcs.get_slipvector_on_fault_antiplane(
        els, quadcoefs.flatten(), 10
    )

    # plot slip at nodes and element centers
    # xlimits = [-5, 5]
    plt.figure(figsize=(8, 2))
    plt.subplot2grid((1, 3), (0, 0), colspan=2)
    plt.plot(xf[yf == 0], slipnodes[yf == 0], "-")
    plt.plot(
        els.x_centers[els.y_centers == 0],
        slipnodecenters[els.y_centers == 0],
        ".",
        label="slip at element centers",
    )
    plt.xlabel("x")
    plt.ylabel("slip at nodes")
    # plt.xlim(xlimits)
    plt.grid()
    plt.subplot2grid((1, 3), (0, 2))
    plt.plot(slipnodes[yf < 0], yf[yf < 0], "-")
    plt.plot(
        slipnodecenters[els.y_centers < 0],
        els.y_centers[els.y_centers < 0],
        ".",
        label="slip at element centers",
    )
    plt.ylabel("y")
    plt.xlabel("slip at nodes")
    plt.grid()
    plt.show()

# %%
