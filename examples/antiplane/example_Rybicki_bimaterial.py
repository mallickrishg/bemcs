# %% [markdown]
# # Rybicki (1971), bimaterial: strike-slip fault near, and on, a vertical material contact (antiplane)
# **What this is:** a check of the heterogeneous solver against Rybicki's exact solution for a
# vertical strike-slip fault in a half-space made of two quarter-spaces (a lateral contrast, no
# layering), with the fault near the contact or on it (a bimaterial fault). Companion examples:
# `example_Rybicki_layered.py` (layering, no lateral contrast) and `example_layered_medium.py` (both
# together, several layers). (Formerly `BimaterialContact_Rybicki.py`.)
#
# A half-space (free surface y = 0) made of two quarter-spaces welded along a vertical contact at
# x = d: shear modulus MU1 for x < d and MU2 = 4 MU1 for x > d. A vertical strike-slip fault at x = 0
# breaks the surface, with slip 1 at the surface tapering smoothly to 0 at depth D. Two cases:
# - **fault near the contact**, d = 0.5 D;
# - **fault on the contact**, d = 0: a bimaterial fault (the slip and force elements coincide).
#
# **Exact solution** (Rybicki 1971). With S the homogeneous half-space field of the fault (fault plus
# free-surface image) and kappa = (MU1 - MU2) / (MU1 + MU2):
#
# - x < d: u = S(x, y) + kappa S(2d - x, y), the reflection off the contact;
# - x > d: u = (1 + kappa) S(x, y), the transmitted field.
#
# The contact mirror commutes with the free-surface mirror, so there is no infinite series. For d = 0
# this is u = 2 MU2/(MU1 + MU2) S on the MU1 side and 2 MU1/(MU1 + MU2) S on the MU2 side.
#
# The contact is a line of force elements (`'h'`); the free surface and the contact extend to
# |x|, depth = L with log-spaced elements.

# %% imports
import warnings

import numpy as np
import matplotlib.pyplot as plt
import bemcs

MU1, MU2 = 1.0, 4.0
D = 1.0  # fault depth
L = 1000.0  # extent of the free surface (|x| < L) and of the contact (depth < L)
H_EL = 0.01  # element length near the fault (|x| < 2, depth < 2)
geometry, mu = "antiplane", 1.0
CASES = {"fault near the contact (d = 0.5)": 0.5, "fault on the contact (d = 0)": 0.0}


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


def exact(x, dist, y=0.0, mu2=MU2):
    k = (MU1 - mu2) / (MU1 + mu2)
    return np.where(
        x < dist,
        S_halfspace(x, y) + k * S_halfspace(2 * dist - x, y),
        (1 + k) * S_halfspace(x, y),
    )


# %% the mesh


def graded(h, L, flat=2.0, ratio=1.08, h_max=5.0):
    """0, h, 2h, ... up to `flat`, then log-spaced (each element 8% longer) up to h_max, out to L."""
    r, step = list(np.arange(0.0, flat + 1e-12, h)), h
    while r[-1] < L:
        r.append(r[-1] + step)
        step = min(step * ratio, h_max)
    return np.array(r)


def contact_mesh(dist, h=H_EL, L=L):
    """Fault x = 0 (s), free surface y = 0 (t), contact x = dist (h). dist must be a multiple of h so
    that the surface has a node where the contact meets it."""
    r = graded(h, L)
    xs, zi = np.r_[-r[:0:-1], r], -r
    n_f = int(round(D / h))
    zf = np.linspace(0.0, -D, n_f + 1)
    els = bemcs.initialize_els()
    els.x1 = np.r_[np.zeros(n_f), xs[:-1], np.full(len(zi) - 1, dist)]
    els.x2 = np.r_[np.zeros(n_f), xs[1:], np.full(len(zi) - 1, dist)]
    els.y1 = np.r_[zf[:-1], np.zeros(len(xs) - 1), zi[:-1]]
    els.y2 = np.r_[zf[1:], np.zeros(len(xs) - 1), zi[1:]]
    bemcs.standardize_els_geometry(els, reorder=False)
    bc = np.array(["s"] * n_f + ["t"] * (len(xs) - 1) + ["h"] * (len(zi) - 1))
    side = np.where(els.x_centers < dist, MU1, MU2)
    # the fault and contact elements run downward, so their unit normal is +x: -n is the MU1 side
    mu_minus = np.where(bc == "t", side, MU1)
    mu_plus = np.where(
        bc == "t", side, np.where(bc == "h", MU2, MU2 if dist == 0 else MU1)
    )
    value = np.where(bc == "s", slip_profile(-els.y_centers), 0.0)
    return bemcs.Mesh(
        els,
        bc_type=bc,
        bc_value=value,
        mu_minus=mu_minus,
        mu_plus=mu_plus,
        basis={"slip": "hat", "force": "trapezoid"},
    )


def solve(mesh):
    with warnings.catch_warnings():
        # d = 0: every contact node lies on the fault, which the crossing check reports
        # (a false positive for coincident meshes, tasks.md P6)
        warnings.simplefilter("ignore", bemcs.TrapezoidCrossingWarning)
        return bemcs.solve(mesh, geometry=geometry, mu=mu)


# %% solve both cases, and a convergence study
xs = np.r_[-np.linspace(4, 1e-3, 120), np.linspace(1e-3, 4, 120)]
results = {}
for name, dist in CASES.items():
    mesh = contact_mesh(dist)
    x = solve(mesh)
    u = bemcs.evaluate_fields(
        mesh, x, xs, np.full_like(xs, -1e-9), geometry=geometry, mu=mu
    )["uz"]
    results[name] = (mesh, x, dist, u)
    print(f"{name}: {mesh}")

xc = np.r_[-np.geomspace(5, 0.02, 15), np.geomspace(0.02, 5, 15)]
h_list = [0.1, 0.05, 0.025]
conv = {}
for name, dist in CASES.items():
    conv[name] = []
    for h in h_list:
        m = contact_mesh(dist, h=h)
        xm = solve(m)
        um = bemcs.evaluate_fields(
            m, xm, xc, np.full_like(xc, -1e-9), geometry=geometry, mu=mu
        )["uz"]
        conv[name].append(np.abs(um - exact(xc, dist)).max())

# %% plot
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
ax = axes[0, 0]
for i, (name, (mesh, x, dist, u)) in enumerate(results.items()):
    ax.plot(xs, exact(xs, dist), "-", color=f"C{i}", lw=1.5, label=f"{name}: exact")
    ax.plot(xs[::6], u[::6], "o", color=f"C{i}", ms=3.5, label=f"{name}: BEM")
ax.plot(xs, exact(xs, 0.0, mu2=MU1), "k--", lw=1, label="homogeneous half-space")
for dist in set(CASES.values()):
    ax.axvline(dist, color="0.6", lw=0.8, ls=":")
ax.set_xlabel("x / D")
ax.set_ylabel("surface displacement / surface slip")
ax.grid(alpha=0.3)
ax.legend(fontsize=7)
ax.set_title(
    f"(a) surface displacement; $\\mu_1$ = {MU1:g} (x < d), $\\mu_2$ = {MU2:g} (x > d)",
    fontsize=10,
)

ax = axes[0, 1]
for i, (name, (mesh, x, dist, u)) in enumerate(results.items()):
    ax.semilogy(
        xs, np.abs(u - exact(xs, dist)), ".-", color=f"C{i}", ms=3, lw=0.8, label=name
    )
ax.set_xlabel("x / D")
ax.set_ylabel("|BEM - exact|")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)
ax.set_title(
    f"(b) error at the surface (element length {H_EL:g} D): smooth and broad",
    fontsize=10,
)

ax = axes[1, 0]
for i, (name, err) in enumerate(conv.items()):
    ax.loglog(h_list, err, "o-", color=f"C{i}", label=name)
h = np.array(h_list)
ax.loglog(h, conv[list(conv)[0]][0] * h / h[0], "k:", lw=1, label="first order")
ax.set_xlabel("element length near the fault / D")
ax.set_ylabel("max |BEM - exact| for |x| < 5 D")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)
ax.set_title("(c) convergence: first order (see the end of this script)", fontsize=10)

ax = axes[1, 1]
mesh, x, dist, _ = results["fault near the contact (d = 0.5)"]
_, ye, f = mesh.values_on_elements(x, "force", geometry=geometry, n_eval=11)
ye, f = np.ravel(ye), np.ravel(f)
keep = ye > -2.5
k = (MU1 - MU2) / (MU1 + MU2)
kappa = bemcs.interface_kappa(MU1, MU2, -1)
z = np.linspace(-2.5, -1e-4, 400)
e = 1e-6
f_exact = (
    -kappa
    * (1 - k)
    * (S_halfspace(dist - e, z) - S_halfspace(dist - 3 * e, z))
    / (2 * e)
)
ax.plot(f_exact, -z, "k-", lw=1.5, label="exact")
ax.plot(f[keep], -ye[keep], ".", color="C0", ms=3, label="BEM (trapezoids)")
ax.invert_yaxis()
ax.set_xlabel("force density on the contact (reference medium)")
ax.set_ylabel("depth / D")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
ax.set_title(
    "(d) fault near the contact: force density along the contact.\n"
    "The chain end at the free surface is open (a free value, not tapered)",
    fontsize=10,
)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Reading the figure.** The BEM matches the exact solution for both faults: 6e-3 (d = 0.5) and
# 2.3e-2 (d = 0) at the default element length, converging at first order.
#
# Where the contact meets the free surface, the force chain ends on slip elements. That end is open
# (`mesh.force_open_ends`): the default layout adds an open-end patch, a free force value at the tip
# node with its own collocation point at the top element, instead of tapering the force to zero
# there (panel d). The free surface stays an ordinary slip mesh, with the fault-surface triple junction
# handled by the slip elements alone. With a tapered end the error was 2.8x larger.
#
# The remaining first-order error comes from the corner where the force layer meets the free surface.
# Grading the elements towards that corner (3 levels of halving) cuts it to about 2e-3 for both faults
# (tasks.md P3). For d = 0 the slip and force elements coincide along the fault (tasks.md P6); since
# the fault's slip is prescribed and tapers smoothly at its bottom, no restart of the trapezoids is
# needed there.
