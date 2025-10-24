# %% import libraries
import matplotlib.pyplot as plt
import numpy as np
import bemcs
import pandas as pd
import GF
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


def plot_BEM_field(
    toplot, els, xo, yo, xlimits, ylimits, maxval, n_levels=10, cmap="coolwarm"
):
    """
    Plot BEM scalar field with element geometry and automatically computed contours.

    Parameters
    ----------
    toplot : ndarray
        Scalar field values at observation points (1D array of size nx_obs*ny_obs)
    els : object
        Mesh object with attributes x1, x2, y1, y2
    xo, yo : ndarray
        Observation point coordinates (1D arrays)
    xlimits, ylimits : tuple
        (xmin, xmax) and (ymin, ymax)
    maxval : float
        Maximum absolute value for color scaling
    n_levels : int, optional
        Number of contour levels (default 10)
    cmap : str, optional
        Colormap (default "coolwarm")
    """

    # Determine grid shape
    nx_obs = len(np.unique(xo))
    ny_obs = len(np.unique(yo))

    X = xo.reshape(ny_obs, nx_obs)
    Y = yo.reshape(ny_obs, nx_obs)
    Z = toplot.reshape(ny_obs, nx_obs)

    # color limits
    vmin, vmax = -maxval, maxval

    # contour levels
    levels = np.linspace(vmin, vmax, n_levels)

    # plot field using colors
    plt.pcolor(X, Y, Z, cmap=cmap, vmin=vmin, vmax=vmax)

    # overlay mesh elements
    n_els = len(els.x1)
    for i in range(n_els):
        plt.plot(
            [els.x1[i], els.x2[i]],
            [els.y1[i], els.y2[i]],
            "k.-",
            linewidth=0.2,
            markersize=1,
        )

    # colorbar
    plt.colorbar()

    # contour lines
    plt.contour(X, Y, Z, colors="k", levels=levels, linewidths=0.5)

    # limits and aspect
    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.gca().set_aspect("equal", adjustable="box")


# %% Read source file and mesh of the domain
# fileinput = "testing_mesh.csv"

# need to fix issue with connectivity file reading for non-heterogeneous cases

fileinput = "HeterogeneousDomainMesh.csv"
connectvitiyfile = "HeterogeneousDomainMeshConnectivity.csv"
# connmatrix = pd.read_csv(connectvitiyfile, header=None).values
# %% solve BEM to get quadratic(slip) & force coefficients
els, els_s, quadcoefs, forcecoefs = GF.solveAntiplaneBEM(fileinput, connectvitiyfile)

# %% compute and plot displacement, displacement gradient fields inside the domain
n_els = len(els.x1)
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
    xo, yo, els_s, mu=1
)
# compute displacement and stress components
if "connmatrix" in locals():
    Kforce_x, Kforce_y, Kforce_u = GF.get_kernels_trapezoidalforce(
        xo, yo, els, connmatrix
    )
    u = Kslip_u @ quadcoefs + Kforce_u @ forcecoefs
    sx = Kslip_x @ quadcoefs + Kforce_x @ forcecoefs
    sy = Kslip_y @ quadcoefs + Kforce_y @ forcecoefs
else:
    u = Kslip_u @ quadcoefs
    sx = Kslip_x @ quadcoefs
    sy = Kslip_y @ quadcoefs


# %% plot fields inside the domain
plt.figure(figsize=(10, 10))
plt.subplot(3, 1, 1)
toplot = u.reshape(ny_obs, nx_obs)
maxval = 0.5
plot_BEM_field(
    toplot, els, xo, yo, xlimits, ylimits, maxval, n_levels=11, cmap="coolwarm"
)
plt.title("Displacement field $u$")

plt.subplot(3, 1, 2)
toplot = sx.reshape(ny_obs, nx_obs)
maxval = 1
plot_BEM_field(
    toplot, els, xo, yo, xlimits, ylimits, maxval, n_levels=11, cmap="RdYlBu_r"
)
plt.title("Displacement gradient $u_{,x}$")

plt.subplot(3, 1, 3)
toplot = sy.reshape(ny_obs, nx_obs)
plot_BEM_field(
    toplot, els, xo, yo, xlimits, ylimits, maxval, n_levels=11, cmap="RdYlBu_r"
)
plt.title("Displacement gradient $u_{,y}$")
plt.show()

# %% plot fields at the surface of the domain
plt.figure(figsize=(8, 10))
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
    els_s, quadcoefs.flatten(), 10
)
plt.figure(figsize=(4, 4))
# plt.subplot2grid((1, 3), (0, 0), colspan=2)
# plt.plot(xf[yf == 0], slipnodes[yf == 0], "-")
# plt.xlabel("x")
# plt.ylabel("slip at nodes")
# plt.xlim(xlimits)
# plt.grid()
# plt.subplot2grid((1, 3), (0, 2))
plt.plot(slipnodes[yf < 0], yf[yf < 0], ".-")
plt.ylabel("y")
plt.xlabel("slip at nodes")
plt.grid()
plt.show()
# %% end of file
