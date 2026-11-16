# %% import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% topography
# Elastic parameter (shear modulus)
mu = 1.0
Lscale = 10
npts_layer = 101
xvals = np.linspace(-Lscale, Lscale, npts_layer)
xt1 = xvals[0:-1]
xt2 = xvals[1:]
yt1 = np.zeros_like(xt1)
yt2 = np.zeros_like(xt2)

# setup a fault geometry (source) - in this case it is a vertical strike-slip fault segment
xf1 = np.array([-0.0, 0.0, 0.0]) + np.min(xvals[xvals >= 0])
# yf1 = np.array([-1.5, -1.0, -0.5])
yf1 = np.array([0, -0.5, -1.0])
xf2 = np.array([0, 0, 0]) + np.min(xvals[xvals >= 0])
# yf2 = np.array([-1.0, -0.5, 0])
yf2 = np.array([-0.5, -1.0, -1.5])
slip_values = np.array([0.75, 1.0, 0.5])  # slip on fault segments

# %% provide layered structure in terms of number of layers, location of layers (iterface with jump in μ), and μ values
nlayers = 2
zlayer = np.linspace(-2, 0, nlayers + 1)[0:-1]
mulayer = np.linspace(5.0, 1, nlayers + 1)

x1 = []
x2 = []
y1 = []
y2 = []

# shear modulus below (-n side) and above (+n side) each interface element: elements run
# left to right, so their unit normal points up (+n = above)
mu_below = np.zeros(nlayers * (npts_layer - 1))
mu_above = np.zeros(nlayers * (npts_layer - 1))
y1perturb = np.sin(2 * np.pi * 4 * xt1 / Lscale) * 0.005
y2perturb = np.sin(2 * np.pi * 4 * xt2 / Lscale) * 0.005
for i in range(nlayers):
    xvals = np.linspace(-Lscale, Lscale, npts_layer)
    x1 = np.hstack([x1, xvals[0:-1]])
    x2 = np.hstack([x2, xvals[1:]])
    y1 = np.hstack([y1, np.ones(npts_layer - 1) * zlayer[i] + y1perturb])
    y2 = np.hstack([y2, np.ones(npts_layer - 1) * zlayer[i] + y2perturb])
    rows = slice(i * (npts_layer - 1), (i + 1) * (npts_layer - 1))
    mu_below[rows] = mulayer[i]  # layer i lies below interface i
    mu_above[rows] = mulayer[i + 1]
print("interfaces (mu below, mu above):", sorted(set(zip(mu_below, mu_above))))
# No connectivity file is needed: bemcs.solve builds one trapezoidal force patch per non-tip
# interface element from the mesh topology.


def mu_at_depth(z):
    """Shear modulus of the layered structure at depth z (z < 0)."""
    mu_z = np.full(np.shape(z), mulayer[-1], dtype=float)
    for i in range(nlayers):
        mu_z = np.where(z < zlayer[nlayers - 1 - i], mulayer[nlayers - 1 - i], mu_z)
    return mu_z



# plot elastic structure
plt.figure()
for i in range(nlayers):
    if i == 0:
        plt.plot(
            [mulayer[i], mulayer[i], mulayer[i + 1]],
            [zlayer[i] - 1, zlayer[i], zlayer[i]],
            "k.-",
        )
    else:
        plt.plot(
            [mulayer[i], mulayer[i], mulayer[i + 1]],
            [zlayer[i - 1], zlayer[i], zlayer[i]],
            "k.-",
        )
plt.plot(
    [mulayer[i + 1], mulayer[i + 1]],
    [zlayer[i], 0],
    "k.-",
)
plt.xlabel("$\\mu$")
plt.ylabel("depth")
plt.show()
# %% Store all x,y pairs, BC types and values, and the shear modulus on both sides in a csv file
# xf,yf have a BC type 's' with the slip value
# xt,yt have a BC type 't' with a value of 0.0 (traction free); their -n side is the ground
# x,y pairs have a BC type of 'h' (material interface): mu_minus below, mu_plus above

BCtype = np.hstack(
    [
        np.full_like(xf1, "s", dtype=object),
        np.full_like(xt1, "t", dtype=object),
        np.full_like(x1, "h", dtype=object),
    ]
)
BCval = np.hstack([slip_values, np.ones_like(xt1) * 0.0, np.zeros_like(x1)])
# fault: modulus of the layer at each element's centre (same on both sides);
# free surface: the top layer below it (air above: only the -n side is sampled)
mu_fault = mu_at_depth(0.5 * (yf1 + yf2))
mu_minus = np.hstack([mu_fault, np.full_like(xt1, mulayer[-1]), mu_below])
mu_plus = np.hstack([mu_fault, np.full_like(xt1, mulayer[-1]), mu_above])

# export in the format xs,ys,xe,y2e,BCtype,BCval
# where(xs,ys) is start point and (xe,ye) is end point of each segment
dataout = pd.DataFrame(
    {
        "x1": np.hstack([xf1, xt1, x1]),
        "z1": np.hstack([yf1, yt1, y1]),
        "x2": np.hstack([xf2, xt2, x2]),
        "z2": np.hstack([yf2, yt2, y2]),
        "BC_type": BCtype,
        "value": BCval,
        "mu_minus": mu_minus,
        "mu_plus": mu_plus,
    }
)

dataout.to_csv("HeterogeneousDomainMesh.csv", index=False)
print("Mesh file created: HeterogeneousDomainMesh.csv")

# %%
