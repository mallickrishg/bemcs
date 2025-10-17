# %% import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% topography
# Elastic parameter (shear modulus)
mu = 1.0
Lscale = 10
npts_layer = 10
xvals = np.linspace(-Lscale, Lscale, npts_layer)
xt1 = xvals[0:-1]
xt2 = xvals[1:]
yt1 = np.zeros_like(xt1)
yt2 = np.zeros_like(xt1)

# setup a fault geometry (source) - in this case it is a vertical strike-slip fault segment
xf1 = np.array([-0.0]) + np.min(xvals[xvals >= 0])
yf1 = np.array([-0.0])
xf2 = np.array([0]) + np.min(xvals[xvals >= 0])
yf2 = np.array([-1.5])

# %% provide layered structure in terms of number of layers, location of layers (iterface with jump in μ), and μ values
nlayers = 3
zlayer = np.linspace(-6, 0, nlayers + 1)[0:-1]
mulayer = np.linspace(10, 1, nlayers + 1)

x1 = []
x2 = []
y1 = []
y2 = []

# calculation the dμ/dx and dμ/dy terms as α,β
beta = np.zeros(nlayers * (npts_layer - 1))
for i in range(nlayers):
    xvals = np.linspace(-Lscale, Lscale, npts_layer)
    x1 = np.hstack([x1, xvals[0:-1]])
    x2 = np.hstack([x2, xvals[1:]])
    y1 = np.hstack([y1, np.ones(npts_layer - 1) * zlayer[i]])
    y2 = np.hstack([y2, np.ones(npts_layer - 1) * zlayer[i]])
    beta[i * (npts_layer - 1) : (npts_layer - 1) * (i + 1)] = (
        -(mulayer[i + 1] - mulayer[i]) / mulayer[i + 1]
    )
# along with x1,y1 and x2,y2 we also need to provide a connectivity matrix tying three elements together
# this connectivity matrix is for two elements at a time - i want to connect [0,1,2] then [2,3,4] and so on
# connectivity construction
connectivity = []
for i in range(nlayers):
    start_idx = i * (npts_layer - 1) + len(xf1) + len(xt1)
    end_idx = start_idx + (npts_layer - 1)
    local_idx = np.arange(start_idx, end_idx)
    for j in range(0, len(local_idx) - 2, 2):
        connectivity.append(local_idx[j : j + 3])
connectivity = np.array(connectivity)

print(connectivity)
# export connectivity in a csv file
pd.DataFrame(connectivity).to_csv(
    "HeterogeneousDomainMeshConnectivity.csv", index=False, header=False
)
print("connectivity file created: HeterogeneousDomainMeshConnectivity.csv")


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
# %% Store all x,y pairs and BC types and values in a csv file
# xf,yf have a BC type 's' with a value of 1.0 (unit slip)
# xt,yt have a BC type 't' with a value of 0.0 (traction free)
# x,y pairs have a BC type of 'h' with a value of beta (dμ/dy term)

BCtype = np.hstack(
    [
        np.full_like(xf1, "s", dtype=object),
        np.full_like(xt1, "t", dtype=object),
        np.full_like(x1, "h", dtype=object),
    ]
)
BCval = np.hstack([np.ones_like(xf1) * 1.0, np.ones_like(xt1) * 0.0, beta])

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
    }
)

dataout.to_csv("HeterogeneousDomainMesh.csv", index=False)
print("Mesh file created: HeterogeneousDomainMesh.csv")

# %%
