# %% [markdown]
# # A strike-slip fault on a vertical material contact (antiplane)
# A half-space (free surface y = 0) made of two quarter-spaces welded along a vertical contact at x = 0:
# shear modulus `MU1` for x < 0 and `MU2` = 4 `MU1` for x > 0. A vertical strike-slip fault lies on the
# contact and breaks the surface, with slip 1 at the surface tapering smoothly to 0 at depth D: a
# **bimaterial fault**.
#
# **The mesh** has three lines:
# - the fault, x = 0 from the surface to depth D: slip elements with prescribed slip (`'s'`);
# - the free surface, y = 0: slip elements, traction-free (`'t'`);
# - the contact, x = 0 from the surface down: force elements (`'h'`). Along the fault it lies on the same
#   nodes as the fault elements, so the fault is meshed twice: as a slip line and as a material contact.
#
# Every element lists the shear modulus on its -n and +n side. The fault and the contact run downward, so
# their unit normal points to +x: `mu_minus` = `MU1`, `mu_plus` = `MU2`. The free surface and the contact
# extend far (|x|, depth up to L), with elements growing away from the fault.
#
# **Exact solution** (Rybicki 1971). With S the field of the same fault in a homogeneous half-space, the
# displacement is 2 `MU2` / (`MU1` + `MU2`) S on the `MU1` side and 2 `MU1` / (`MU1` + `MU2`) S on the `MU2`
# side: the compliant side moves more.

# %% imports
import numpy as np
import matplotlib.pyplot as plt
import bemcs

MU1, MU2 = 1.0, 4.0
D = 1.0  # fault depth
L = 1000.0  # extent of the free surface (|x| < L) and of the contact (depth < L)
H_EL = 0.01  # element length near the fault (|x| < 2, depth < 2)
geometry = "antiplane"


def slip_profile(depth):
    return np.where(depth < D, np.cos(0.5 * np.pi * depth / D) ** 2, 0.0)


# %% the exact solution


def S_halfspace(x, y=0.0, n_quad=400):
    """Homogeneous half-space displacement of the tapered fault at x = 0. Uniform slip from the
    surface to depth d gives (atan((d - z)/x) + atan((d + z)/x)) / 2 pi at depth z (fault plus its
    free-surface image); the tapered profile is a superposition with weights -s'(d)."""
    g, w = np.polynomial.legendre.leggauss(n_quad)
    d, wd = 0.5 * D * (g + 1), 0.5 * D * w
    neg_ds = 0.5 * np.pi / D * np.sin(np.pi * d / D)
    z = -np.asarray(y, float)
    return sum(
        wi * si * (np.arctan((di - z) / x) + np.arctan((di + z) / x)) / (2 * np.pi)
        for di, wi, si in zip(d, wd, neg_ds)
    )


def exact(x, y=0.0):
    return np.where(x < 0, 2 * MU2, 2 * MU1) / (MU1 + MU2) * S_halfspace(x, y)


# %% the mesh


def graded(h, L, flat=2.0, ratio=1.08, h_max=5.0):
    """0, h, 2h, ... up to `flat`, then log-spaced (each element 8% longer) up to h_max, out to L."""
    r, step = list(np.arange(0.0, flat + 1e-12, h)), h
    while r[-1] < L:
        r.append(r[-1] + step)
        step = min(step * ratio, h_max)
    return np.array(r)


r = graded(H_EL, L)
xs = np.r_[-r[:0:-1], r]  # free-surface nodes
zc = -r  # contact nodes, downward from the surface
n_f = int(round(D / H_EL))
zf = zc[: n_f + 1]  # fault nodes: the top of the contact

els = bemcs.initialize_els()
els.x1 = np.r_[np.zeros(n_f), xs[:-1], np.zeros(len(zc) - 1)]
els.x2 = np.r_[np.zeros(n_f), xs[1:], np.zeros(len(zc) - 1)]
els.y1 = np.r_[zf[:-1], np.zeros(len(xs) - 1), zc[:-1]]
els.y2 = np.r_[zf[1:], np.zeros(len(xs) - 1), zc[1:]]
bemcs.standardize_els_geometry(els, reorder=False)

bc_type = np.array(["s"] * n_f + ["t"] * (len(xs) - 1) + ["h"] * (len(zc) - 1))
on_surface = bc_type == "t"
# the free surface: the modulus of the rock below it; the fault and the contact: MU1 (-n), MU2 (+n)
mu_minus = np.where(on_surface, np.where(els.x_centers < 0, MU1, MU2), MU1)
mu_plus = np.where(on_surface, mu_minus, MU2)
mesh = bemcs.Mesh(
    els,
    bc_type=bc_type,
    bc_value=np.where(bc_type == "s", slip_profile(-els.y_centers), 0.0),
    mu_minus=mu_minus,
    mu_plus=mu_plus,
    basis={"slip": "hat", "force": "trapezoid"},
)
print(mesh)

# %% solve
x = bemcs.solve(mesh, geometry=geometry)

# %% surface displacement
xo = np.r_[-np.linspace(4, 1e-3, 120), np.linspace(1e-3, 4, 120)]
u = bemcs.evaluate_fields(mesh, x, xo, np.full_like(xo, -1e-9), geometry=geometry)["uz"]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(xo, exact(xo), "k-", lw=1.5, label="exact (Rybicki 1971)")
ax.plot(xo[::6], u[::6], "o", color="C0", ms=4, label="BEM")
ax.plot(xo, S_halfspace(xo), "k--", lw=1, label="homogeneous half-space")
ax.axvline(0, color="0.6", lw=0.8, ls=":")
ax.set_xlabel("x / D")
ax.set_ylabel("surface displacement / surface slip")
ax.grid(alpha=0.3)
ax.legend()
ax.set_title(f"$\\mu_1$ = {MU1:g} (x < 0), $\\mu_2$ = {MU2:g} (x > 0)")
plt.show()

# %% displacement and stress around the fault
fig, axes, fields = bemcs.plot_fields(
    mesh, x, geometry=geometry, xlim=(-2, 2), ylim=(-2, 0), n=(161, 81), cmap="RdBu_r"
)
fig.set_size_inches(8, 10)
fig.tight_layout()
plt.show()
