# %% [markdown]
# # Topographic loading: slip basis vs force basis
# The same plane-strain boundary-value problem solved twice: once with quadratic slip elements on the
# boundary, and once with trapezoidal force elements, to compare the two source bases.
#
# The domain is a closed polygon: a smooth (logistic) topographic step on top, and vertical sides and a
# flat bottom below it. A Gaussian normal traction acts on the topography; the sides and bottom are fixed
# (u = 0). The last section checks both bases against an exact (Kelvin point-force) solution for several
# Poisson's ratios.

# %% Imports
import numpy as np
import matplotlib.pyplot as plt

import bemcs


def logistic(x, L=1, k=1, x0=0):
    """Logistic function L / (1 + exp(-k (x - x0)))."""
    return L / (1 + np.exp(-k * (x - x0)))


# %% Geometry: one closed polygon, traversed left side -> top -> right side -> bottom
Lscale = 30  # half-width of the domain
L_y = -5.0  # depth of the bottom
hmax = 1  # height of the topographic step
N_top = 50  # log-spaced points on each half of the topography (refined towards x = 0)
h_fixed = 1.0  # element length on the sides and bottom

half = np.logspace(np.log10(5e-3 * Lscale), np.log10(Lscale), N_top)
x_top = np.concatenate([-half[::-1], [0.0], half])
x_top[[0, -1]] = -Lscale, Lscale  # exact corners, so the polygon closes
y_top = logistic(x_top, L=hmax, k=2, x0=0)

y_left = np.arange(L_y, y_top[0], h_fixed)  # bottom -> top
y_right = np.arange(y_top[-1], L_y, -h_fixed)[1:]  # top -> bottom
x_bottom = np.arange(Lscale, -Lscale - h_fixed / 2, -h_fixed)  # right -> left, ends at the start

xvals = np.concatenate([np.full_like(y_left, -Lscale), x_top, np.full_like(y_right, Lscale), x_bottom])
yvals = np.concatenate([y_left, y_top, y_right, np.full_like(x_bottom, L_y)])
on_top = np.concatenate([np.zeros(len(y_left)), np.ones(len(x_top) - 1), np.zeros(len(y_right) + len(x_bottom))]) > 0

els = bemcs.initialize_els()
els.x1, els.y1 = xvals[:-1], yvals[:-1]
els.x2, els.y2 = xvals[1:], yvals[1:]
bemcs.standardize_els_geometry(els, reorder=False)
n_els = len(els.x1)

# %% Boundary conditions and material
geometry = "planestrain"
mu = 1.0  # reference shear modulus (stresses do not depend on it; displacements scale as 1/mu)
nu = 0.25  # Poisson's ratio

# Topography: traction in the local (shear, normal) frame, a Gaussian normal traction (positive =
# along the outward normal) with no shear. Sides and bottom: zero displacement (global x, y).
load = np.exp(-(els.x_centers**2) / (2 * np.pi * 0.5**2))
bc_type = np.where(on_top, "t_local", "u")
bc_value = np.where(on_top[:, None], np.c_[np.zeros(n_els), load], 0.0)

meshes = {
    # (1) quadratic slip on every element (general form with continuity rows)
    "slip": bemcs.Mesh(els, labels="slip", bc_type=bc_type, bc_value=bc_value, mu_minus=mu, mu_plus=mu),
    # (2) trapezoidal force patches (default connectivity: centred on every other element around the
    #     closed loop); BCs at the centres of their middle elements
    "force": bemcs.Mesh(els, labels="force", basis={"force": "trapezoid"}, bc_type=bc_type,
                        bc_value=bc_value, mu_minus=mu, mu_plus=mu),
}
fig, ax = plt.subplots(figsize=(12, 3))
bemcs.plot_mesh(meshes["slip"], ax=ax, show_moduli=False, offset=0.5)
ax.set_title("mesh, outward normals")
plt.show()

# %% Solve with both bases
coefs = {name: bemcs.solve(m, geometry=geometry, mu=mu, nu=nu) for name, m in meshes.items()}

# %% Fields on a grid: deviatoric stress, pressure and stress components
xlimits = [-1.5 * abs(L_y), 1.5 * abs(L_y)]
ylimits = [0.99 * L_y, 1.2 * hmax]


def inside(X, Y):
    """Inside the domain (below the topography)."""
    return bemcs.inpolygon(X.ravel(), Y.ravel(), els.x1, els.y1).reshape(X.shape)


derived = {
    "deviatoric stress": lambda f: np.sqrt(0.25 * (f["sxx"] - f["syy"]) ** 2 + f["sxy"] ** 2),
    "pressure": lambda f: 0.5 * (f["sxx"] + f["syy"]),
}
grid = {}
for name, m in meshes.items():
    fig, axes, grid[name] = bemcs.plot_fields(
        m, coefs[name], geometry=geometry, mu=mu, nu=nu, xlim=xlimits, ylim=ylimits, n=(100, 100),
        mask=inside, derived=derived, components=["deviatoric stress", "pressure", "sxx", "syy", "sxy"],
    )
    fig.suptitle(f"{name} basis")
    plt.show()

# %% Stresses and tractions just below the topography
dr = -1e-9  # offset along the outward normal, i.e. just inside the domain
n_per_el = 3
t = (np.arange(n_per_el) + 0.5) / n_per_el  # fractions along each element
x_surf = (els.x1[on_top, None] + t * (els.x2 - els.x1)[on_top, None] + dr * els.x_normals[on_top, None]).ravel()
y_surf = (els.y1[on_top, None] + t * (els.y2 - els.y1)[on_top, None] + dr * els.y_normals[on_top, None]).ravel()
nx, ny = np.repeat(els.x_normals[on_top], n_per_el), np.repeat(els.y_normals[on_top], n_per_el)
sx, sy = np.repeat(els.x_shears[on_top], n_per_el), np.repeat(els.y_shears[on_top], n_per_el)
surf = {
    name: bemcs.evaluate_fields(m, coefs[name], x_surf, y_surf, geometry=geometry, mu=mu, nu=nu)
    for name, m in meshes.items()
}
for f in surf.values():
    tx, ty = f["sxx"] * nx + f["sxy"] * ny, f["sxy"] * nx + f["syy"] * ny
    f["shear traction"], f["normal traction"] = tx * sx + ty * sy, tx * nx + ty * ny

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, name in zip(axes, ("sxx", "syy", "sxy", "shear traction", "normal traction")):
    ax.plot(x_surf, surf["slip"][name], ".", label="slip")
    ax.plot(x_surf, surf["force"][name], "-", label="force")
    if name == "normal traction":
        ax.plot(els.x_centers[on_top], load[on_top], "k--", lw=1, label="imposed")
    ax.set_xlabel("x")
    ax.set_title(name)
    ax.set_xlim(xlimits)
axes[0].legend()
plt.show()

# %% Effect of Poisson's ratio on the surface displacement
nus = [0.0, 0.1, 0.25, 0.4, 0.49]
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, (name, m) in zip(axes, meshes.items()):
    for nu_k in nus:
        c = bemcs.solve(m, geometry=geometry, mu=mu, nu=nu_k)
        f = bemcs.evaluate_fields(m, c, x_surf, y_surf, geometry=geometry, mu=mu, nu=nu_k)
        ax.plot(x_surf, f["uy"], label=f"nu = {nu_k}")
    ax.set_xlim(xlimits)
    ax.set_xlabel("x")
    ax.set_title(f"{name} basis: surface uy")
axes[0].set_ylabel("uy")
axes[0].legend()
plt.show()

# %% Check against an exact solution for several Poisson's ratios
# Impose on the same boundary the tractions (top) and displacements (sides, bottom) of a Kelvin point
# force placed outside the domain; inside, the BEM solution must reproduce the Kelvin field.
X, Y = np.meshgrid(np.linspace(-6, 6, 25), np.linspace(-4, 0, 9))
keep = inside(X, Y)
X, Y = X[keep], Y[keep]
sources = {"above": (1.0, 3.0, 0.3, -1.0), "below": (0.0, -9.0, 1.0, 0.5)}  # x0, y0, fx, fy


def kelvin(x, y, x0, y0, fx, fy, nu_k):
    ux, uy = bemcs.kelvin_point_source_disp(x, y, x0, y0, fx, fy, mu, nu_k)
    sxx, syy, sxy = bemcs.kelvin_point_source_stress(x, y, x0, y0, fx, fy, mu, nu_k)
    return {k: np.ravel(v) for k, v in zip(("ux", "uy", "sxx", "syy", "sxy"), (ux, uy, sxx, syy, sxy))}


print("relative max error inside the domain (displacement, stress)")
for src_name, src in sources.items():
    for nu_k in nus:
        kb = kelvin(els.x_centers, els.y_centers, *src, nu_k)
        tx = kb["sxx"] * els.x_normals + kb["sxy"] * els.y_normals
        ty = kb["sxy"] * els.x_normals + kb["syy"] * els.y_normals
        exact_bc = np.where(on_top[:, None], np.c_[tx, ty], np.c_[kb["ux"], kb["uy"]])
        ke = kelvin(X, Y, *src, nu_k)
        line = f"  source {src_name}, nu = {nu_k:.2f}:"
        for name, m in meshes.items():
            mk = bemcs.Mesh(els, labels=name, basis=m.basis, bc_type=np.where(on_top, "t", "u"),
                            bc_value=exact_bc, mu_minus=mu, mu_plus=mu)
            f = bemcs.evaluate_fields(mk, bemcs.solve(mk, geometry=geometry, mu=mu, nu=nu_k), X, Y,
                                      geometry=geometry, mu=mu, nu=nu_k)
            err_u = np.hypot(f["ux"] - ke["ux"], f["uy"] - ke["uy"]).max() / np.hypot(ke["ux"], ke["uy"]).max()
            err_s = max(np.abs(f[k] - ke[k]).max() for k in ("sxx", "syy", "sxy"))
            err_s /= max(np.abs(ke[k]).max() for k in ("sxx", "syy", "sxy"))
            line += f"  {name}: {err_u:.1e}, {err_s:.1e}"
        print(line)
