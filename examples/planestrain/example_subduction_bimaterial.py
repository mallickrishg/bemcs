# %% [markdown]
# # Subduction zone: coseismic slip on a bimaterial megathrust (plane strain)
# A thrust earthquake on a curved megathrust that is also a material contact: a stiff slab (mu = 3) under a
# compliant upper plate (mu = 1), with a compliant mantle (mu = 1) below the slab and a free surface with
# sigmoidal topography (ocean at -4 km, upper plate at +1 km). Lengths in km; slip is a Gaussian with unit
# peak, so displacements are in units of the peak slip and stresses in units of (modulus of 1) x (peak
# slip / km).
#
# The mesh is built by `ConstructMesh_subduction.py` (run it first, or this script runs it). It writes the
# three files read here:
# - `subduction.csv`: geometry, source type and moduli of every element;
# - `subduction_slip_connectivity.csv`: the slip hats, with their boundary conditions (prescribed slip on
#   the fault, traction-free on the free surface; the trench hats have a 4th element, `e4`);
# - `subduction_force_connectivity.csv`: the force trapezoids of the material interfaces (`h`).
#
# The megathrust is meshed twice, with slip and force elements on the same nodes: slip is prescribed, and
# the true traction is continuous across it.
#
# The script:
# 1. solves the problem, and checks the boundary conditions off the elements (free surface traction-free,
#    displacement and true traction continuous across the slab interfaces);
# 2. checks the truncation (L = 5000 vs 2500 km) and the element size near the fault (2 vs 1 km);
# 3. compares with a homogeneous medium (mu = 1 everywhere, same topography and fault): surface displacement,
#    and the traction change on the fault. On a straight fault in a homogeneous medium, slip does not change
#    the normal stress. Here the fault's curvature and the free surface already do. The contrast with the
#    slab (bimaterial coupling) about doubles that change, and raises the shear stress drop by ~50%;
# 4. plots the displacement and stress fields.
#
# **What to expect** (h = 2 km on the fault, 684 elements, ~30 s):
# - the free surface is traction-free to ~1e-5 (stresses near the fault ~0.04);
# - the slab interfaces are continuous to ~1e-5 in displacement and ~4e-4 (1% of the stress near the fault) in true
#   traction;
# - truncation and element size each change the surface displacement by ~0.2%;
# - the true traction on the fault jumps by up to ~1e-3 between its two sides. On the slab side the normal
#   traction shows a sawtooth from element to element: the trapezoid force basis between its collocation points.
#   It converges at first order: the jump is 1.3e-3, 5.4e-4 and 2.4e-4 for 2, 1 and 0.5 km elements, while the
#   normal traction at its peak stays 0.0139. The optional "nodal" force basis
#   (`basis={"force": "nodal"}`) smooths the sawtooth but does not reduce the jump.

# %% imports
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt

import bemcs
import ConstructMesh_subduction as C

plt.rcParams.update({"font.size": 11})
geometry, nu = "planestrain", C.NU

# %% read the mesh files (write them first if needed)
if not os.path.exists("subduction.csv"):
    C.write_files()
mesh = bemcs.Mesh.from_csv(
    "subduction.csv", basis=C.BASIS,
    connectivity={"slip": "subduction_slip_connectivity.csv", "force": "subduction_force_connectivity.csv"},
)
print(mesh)
t = C.element_table()  # the same elements, with their parts (surface, fault, slab top, slab bottom)
assert len(t) == mesh.n_els
els = mesh.els
part = t["part"].to_numpy()
fault = (part == "fault") & (t["label"] == "s").to_numpy()
surface = np.isin(part, ("surface_left", "surface_right"))

# %% solve
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    x, (A, b, rows) = bemcs.solve(mesh, geometry=geometry, nu=nu, return_system=True)
for w in caught:
    print(f"WARNING {w.category.__name__}: {w.message}")
print(f"{A.shape[0]} unknowns, condition number {np.linalg.cond(A):.3g}")


def fields(m, coefs, xo, yo):
    return bemcs.evaluate_fields(m, coefs, xo, yo, geometry=geometry, nu=nu)


def traction(f, nx, ny):
    """True traction on a surface with unit normal n (stresses from evaluate_fields are true stresses)."""
    return f["sxx"] * nx + f["sxy"] * ny, f["sxy"] * nx + f["syy"] * ny


# %% 1. boundary conditions off the elements
xc, yc = np.asarray(els.x_centers), np.asarray(els.y_centers)
nx, ny = np.asarray(els.x_normals), np.asarray(els.y_normals)
L_el = 2 * np.asarray(els.half_lengths)
near = np.abs(xc) < 400


def sides(sel, delta=1e-3):
    """Fields just off the centres of the selected elements, on the -n and +n sides."""
    d = delta * L_el[sel]
    return (fields(mesh, x, xc[sel] - d * nx[sel], yc[sel] - d * ny[sel]),
            fields(mesh, x, xc[sel] + d * nx[sel], yc[sel] + d * ny[sel]))


stress_scale = np.abs(np.r_[[fields(mesh, x, xc[fault], yc[fault] - 5.0)[k] for k in ("sxx", "syy", "sxy")]]).max()
below, _ = sides(surface & near)
ts = np.hypot(*traction(below, nx[surface & near], ny[surface & near]))
print(f"free surface (|x| < 400 km): max |traction| = {ts.max():.1e} "
      f"(stress near the fault ~{stress_scale:.2f})")
for name in ("slab_bottom", "slab_top"):
    sel = (part == name) & near
    fm, fp = sides(sel)
    du = max(np.abs(fp[k] - fm[k]).max() for k in ("ux", "uy"))
    dt = max(np.abs(a - c).max() for a, c in zip(traction(fm, nx[sel], ny[sel]), traction(fp, nx[sel], ny[sel])))
    print(f"{name} (|x| < 400 km): jump in displacement {du:.1e}, in true traction {dt:.1e}")
fm, fp = sides(fault)
slip = np.c_[fp["ux"] - fm["ux"], fp["uy"] - fm["uy"]]
tx, ty = ny[fault], -nx[fault]  # unit tangent of bemcs elements (n is its left normal)
s_shear = slip[:, 0] * tx + slip[:, 1] * ty
print(f"fault: slip just off the element centres vs prescribed, max error "
      f"{np.abs(s_shear - t['value'].to_numpy()[fault]).max():.1e}")
dt = max(np.abs(a - c).max() for a, c in zip(traction(fm, nx[fault], ny[fault]), traction(fp, nx[fault], ny[fault])))
print(f"fault: jump in true traction {dt:.1e}")

# %% 2. truncation and element size
xs = np.linspace(-200, 400, 121)
ys = C.topography(xs) - 1e-3  # just below the free surface


def surface_u(m, coefs):
    f = fields(m, coefs, xs, ys)
    return np.c_[f["ux"], f["uy"]]


u_surf = surface_u(mesh, x)
umax = np.abs(u_surf).max()
for label, kw in (("L = 2500 km", dict(L=2500.0)), ("element size 1 km near the fault", dict(h0=1.0))):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m2, _ = C.subduction_mesh(**kw)
        x2 = bemcs.solve(m2, geometry=geometry, nu=nu)
    print(f"{label}: surface displacement changes by {np.abs(surface_u(m2, x2) - u_surf).max() / umax:.1e} "
          f"(relative to its max {umax:.3f}), {m2.n_els} elements")

# %% 3. homogeneous medium: the slip elements only, mu = 1
keep = np.flatnonzero(t["label"].to_numpy() == "s")
els_h = bemcs.initialize_els()
els_h.x1, els_h.y1, els_h.x2, els_h.y2 = (np.asarray(getattr(els, k))[keep] for k in ("x1", "y1", "x2", "y2"))
bemcs.standardize_els_geometry(els_h, reorder=False)
bc_value = np.c_[t["value"].to_numpy()[keep], np.zeros(len(keep))]
homog = bemcs.Mesh(els_h, bc_type=t["BC_type"].to_numpy()[keep], bc_value=bc_value,
                   mu_minus=C.MU_OUT, mu_plus=C.MU_OUT, basis={"slip": "hat"})
x_h = bemcs.solve(homog, geometry=geometry, nu=nu)
u_homog = surface_u(homog, x_h)

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, gridspec_kw={"height_ratios": [1, 2, 2]})
ax = axes[0]
ax.fill_between(xs, C.topography(xs), -6, color="0.85")
ax.plot(xs, C.topography(xs), "k-")
ax.axvline(0, color="0.5", ls=":", lw=0.8)
ax.set_ylabel("elevation (km)")
ax.set_title("surface displacement (peak slip 1, thrust)")
for k, (ax, comp) in enumerate(zip(axes[1:], ("$u_x$", "$u_y$"))):
    ax.plot(xs, u_surf[:, k], "b-", label="bimaterial (slab mu = 3)")
    ax.plot(xs, u_homog[:, k], "k--", label="homogeneous (mu = 1)")
    ax.axvline(0, color="0.5", ls=":", lw=0.8)
    ax.set_ylabel(comp)
    ax.grid(alpha=0.3)
axes[1].legend()
axes[-1].set_xlabel("x (km); trench at 0")
fig.tight_layout()
plt.show()

# %% traction change on the fault: bimaterial coupling
s_dd = np.cumsum(L_el[fault]) - 0.5 * L_el[fault]  # down-dip distance of the fault element centres
idx_h = np.searchsorted(keep, np.flatnonzero(fault))  # the fault elements in the homogeneous mesh
d = 1e-3 * L_el[fault]
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for m, coefs, ls, name in ((mesh, x, "-", "bimaterial"), (homog, x_h, "--", "homogeneous")):
    for sgn, colour, side in ((-1, "tab:orange", "slab side (-n)"), (1, "tab:blue", "upper-plate side (+n)")):
        f = fields(m, coefs, xc[fault] + sgn * d * nx[fault], yc[fault] + sgn * d * ny[fault])
        Tx, Ty = traction(f, nx[fault], ny[fault])
        axes[0].plot(s_dd, Tx * tx + Ty * ty, ls, color=colour, label=f"{name}, {side}")
        axes[1].plot(s_dd, Tx * nx[fault] + Ty * ny[fault], ls, color=colour, label=f"{name}, {side}")
axes[0].set_ylabel("shear traction change\n(along the fault, down-dip)")
axes[1].set_ylabel("normal traction change\n(tension positive)")
for ax in axes:
    ax.grid(alpha=0.3)
    ax.axhline(0, color="0.5", lw=0.8)
axes[0].legend(fontsize=8)
axes[1].set_xlabel("down-dip distance from the trench (km)")
axes[0].set_title("traction change on the fault (true traction, continuous across it)")
fig.tight_layout()
plt.show()

# %% 4. fields
fig, axs, grid = bemcs.plot_fields(
    mesh, x, geometry=geometry, nu=nu, xlim=(-150, 300), ylim=(-120, 5), n=(181, 51),
    components=["ux", "uy", "sxy", "pressure"],
    derived={"pressure": lambda f: -(1 + nu) * (f["sxx"] + f["syy"]) / 3},
    mask=lambda X, Y: Y < C.topography(X),  # inside the rock
)
for ax in axs[:, 0]:
    ax.set_aspect("equal")
fig.set_size_inches(10, 13)
fig.tight_layout()
plt.show()
