# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import bemcs
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")




def logistic(x, L=1, k=1, x0=0):
    """Logistic function L / (1 + exp(-k (x - x0)))."""
    return L / (1 + np.exp(-k * (x - x0)))


# %% Define mesh and boundary conditions
# construct mesh
Lscale = 30
N_top = 50  # always choose odd npts so that 0 can be included symmetrically
hmax = 1
# xvals = np.linspace(-Lscale, Lscale, npts)
# build symmetric log-spaced x array from –Lscale → 0 → +Lscale
epsmin = 5e-3 * Lscale
h_fixed = 2.0  # uniform spacing for left/right/bottom
L_y = -5.0  # <-- USER-DEFINED bottom value

# ------------------------------------------------------------
# 1. Top boundary (logistic spacing)
# ------------------------------------------------------------
half = np.logspace(np.log10(epsmin), np.log10(Lscale), N_top)
x_top = np.concatenate([-half[::-1], [0.0], half])
y_top = logistic(x_top, L=hmax, k=2, x0=0)
N_top_pts = len(x_top)

# ------------------------------------------------------------
# 2. Left boundary (fixed spacing)
# bottom → top
# ------------------------------------------------------------
y_left_vals = np.arange(L_y, y_top[0] - h_fixed, h_fixed)
x_left_vals = np.full_like(y_left_vals, x_top[0])

# ------------------------------------------------------------
# 3. Right boundary (fixed spacing)
# top → bottom
# ------------------------------------------------------------
y_right_vals = np.arange(y_top[-1], L_y, -h_fixed)
x_right_vals = np.full_like(y_right_vals, x_top[-1])

# ------------------------------------------------------------
# 4. Bottom boundary (fixed spacing)
# right → left
# ------------------------------------------------------------
x_bottom_vals = np.arange(Lscale, -Lscale - h_fixed, -h_fixed)
y_bottom_vals = np.full_like(x_bottom_vals, L_y)

# ------------------------------------------------------------
# 5. Concatenate boundaries in reversed circulation:
# left → top → right → bottom
# ------------------------------------------------------------
xvals = np.concatenate([x_left_vals, x_top, x_right_vals[1:], x_bottom_vals[0:]])
yvals = np.concatenate([y_left_vals, y_top, y_right_vals[1:], y_bottom_vals[0:]])

x1 = xvals[0:-1]
x2 = xvals[1:]
y1 = yvals[0:-1]
y2 = yvals[1:]

# construct mesh
els = bemcs.initialize_els()

els.x1 = x1
els.y1 = y1
els.x2 = x2
els.y2 = y2

bemcs.standardize_els_geometry(els, reorder=False)
bemcs.plot_els_geometry(els)

# ------------------------------------------------------------
# 8. Assign BC types
# top = "t" (traction, global), others = "u" (displacement, global)
# ------------------------------------------------------------
BCtype = np.empty_like(x1, dtype=object)
N_left = len(x_left_vals)
N_top_elems = N_top_pts - 1
# left boundary
BCtype[:N_left] = "u"
# top boundary
BCtype[N_left : N_left + N_top_elems] = "t"
# right + bottom boundaries
BCtype[N_left + N_top_elems :] = "u"

# apply a localized Gaussian load at the top boundary
BCval = np.exp(-(els.x_centers**2) / (2 * np.pi * 0.5**2))
# BCval = 1.0 * (els.y_centers - 0.0)
BCval[BCtype == "u"] = 0.0

# provide connections to compute trapeoidal basis functions
# connectivity matrix: each row is a group of 3 consecutive elements
connect_matrix = np.vstack(
    [np.array([i, i + 1, i + 2]) for i in range(0, len(els.x1) - 2, 2)]
)

# Define material properties
# Elastic parameters and lithostatic effects
mu = 1  # normalized units (mostly does not matter for topographic problem)
nu = 0.25  # Poisson's ratio
rho = 1  # rho*g/mu
ydatum = 1  # provide a datum for verticals

# %% Solve the topographic load problem twice: slip basis and force basis
geometry = "planestrain"
bc_value = np.c_[BCval * els.x_normals, BCval * els.y_normals]  # traction = load * n on top
common = dict(bc_type=BCtype, bc_value=bc_value, mu_minus=mu, mu_plus=mu)

# (1) quadratic slip on every element (general form with continuity rows)
mesh_slip = bemcs.Mesh(els, labels="slip", **common)
quadratic_coefs = bemcs.solve(mesh_slip, geometry=geometry, mu=mu, nu=nu)

# (2) trapezoidal force patches; BCs at the centres of their middle elements
mesh_force = bemcs.Mesh(
    els, labels="force", basis={"force": "trapezoid"}, connectivity={"force": connect_matrix}, **common
)
fcoefs = bemcs.solve(mesh_force, geometry=geometry, mu=mu, nu=nu)

# %% Fields: deviatoric stress and pressure, then stress components (slip vs force)
n_obs = 100
xlimits = [-1.5 * np.abs(L_y), 1.5 * np.abs(L_y)]
ylimits = [0.99 * L_y, 1.2 * hmax]


def inside(X, Y):
    """Inside the domain (below the topography)."""
    return bemcs.inpolygon(X.ravel(), Y.ravel(), els.x1, els.y1).reshape(X.shape)


derived = {
    "deviatoric stress": lambda f: np.sqrt(0.25 * (f["sxx"] - f["syy"]) ** 2 + f["sxy"] ** 2),
    "pressure": lambda f: 0.5 * (f["sxx"] + f["syy"]),
}
plot_kw = dict(geometry=geometry, mu=mu, nu=nu, xlim=xlimits, ylim=ylimits, n=(n_obs, n_obs),
               mask=inside, derived=derived)
results = {}
for name, mesh_k, coefs in (("slip", mesh_slip, quadratic_coefs), ("force", mesh_force, fcoefs)):
    fig, axes, f = bemcs.plot_fields(
        mesh_k, coefs, components=["deviatoric stress", "pressure", "sxx", "syy", "sxy"], **plot_kw
    )
    fig.suptitle(f"{name} basis")
    plt.show()
    results[name] = f
pressure, pressure_f = results["slip"]["pressure"], results["force"]["pressure"]
deviatoric_stress = results["slip"]["deviatoric stress"]
deviatoric_stress_f = results["force"]["deviatoric stress"]
ux, uy = results["slip"]["ux"], results["slip"]["uy"]

# %% evaluate stress fields at the surface
npts_per_elt = 3
dr = -1e-9  # small offset below the surface
top = BCtype == "t"
pts = [np.linspace(a, b, npts_per_elt + 2)[1:-1] for a, b in zip(els.x1[top], els.x2[top])]
x_obs = np.concatenate(pts) + np.repeat(els.x_normals[top], npts_per_elt) * dr
pts = [np.linspace(a, b, npts_per_elt + 2)[1:-1] for a, b in zip(els.y1[top], els.y2[top])]
y_obs = np.concatenate(pts) + np.repeat(els.y_normals[top], npts_per_elt) * dr
kw = dict(geometry=geometry, mu=mu, nu=nu)
fs = bemcs.evaluate_fields(mesh_slip, quadratic_coefs, x_obs, y_obs, **kw)
ff = bemcs.evaluate_fields(mesh_force, fcoefs, x_obs, y_obs, **kw)
sxx, syy, sxy = fs["sxx"], fs["syy"], fs["sxy"]
sxx_f, syy_f, sxy_f = ff["sxx"], ff["syy"], ff["sxy"]

plt.figure(figsize=(15, 5))
for count, name in enumerate(("sxx", "syy", "sxy")):
    plt.subplot(1, 3, count + 1)
    plt.plot(x_obs, fs[name], ".", label="slip")
    plt.plot(x_obs, ff[name], "-", label="force")
    plt.xlabel("x", fontsize=10)
    plt.ylabel(name, fontsize=10)
    plt.xlim(xlimits)
    plt.ylim([-1, 1])
    if count == 0:
        plt.legend()
plt.show()

# tractions at the centres of the surface elements
x_obs = (els.x_centers + els.x_normals * dr)[top]
y_obs = (els.y_centers + els.y_normals * dr)[top]
nx, ny = els.x_normals[top], els.y_normals[top]
fs = bemcs.evaluate_fields(mesh_slip, quadratic_coefs, x_obs, y_obs, **kw)
ff = bemcs.evaluate_fields(mesh_force, fcoefs, x_obs, y_obs, **kw)
plt.figure(figsize=(10, 2))
for count, (label, tx) in enumerate((
    (r"$\tau_x$", lambda f: f["sxx"] * nx + f["sxy"] * ny),
    (r"$\tau_y$", lambda f: f["sxy"] * nx + f["syy"] * ny),
)):
    plt.subplot(1, 2, count + 1)
    plt.plot(x_obs, tx(fs), "o-", label="slip")
    plt.plot(x_obs, tx(ff), ".-", label="force")
    plt.xlabel("x", fontsize=10)
    plt.ylabel(label, fontsize=10)
    plt.xlim(xlimits)
    if count == 0:
        plt.legend()
plt.show()

# %%
