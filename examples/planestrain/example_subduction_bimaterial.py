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
# The script solves the problem and compares it with a homogeneous medium (mu = 1 everywhere, same topography
# and fault): surface displacement, and the traction change on the fault. The slab contrast changes the
# normal stress on the fault (bimaterial coupling) and the shear stress drop. It then plots the displacement
# and stress fields.

# %% imports
import os

import numpy as np
import matplotlib.pyplot as plt

import bemcs
import ConstructMesh_subduction as C

plt.rcParams.update({"font.size": 11})
geometry, nu = "planestrain", C.NU

# %% read the mesh files (write them first if needed) and solve
if not os.path.exists("subduction.csv"):
    C.write_files()
mesh = bemcs.Mesh.from_csv(
    "subduction.csv",
    basis=C.BASIS,
    connectivity={
        "slip": "subduction_slip_connectivity.csv",
        "force": "subduction_force_connectivity.csv",
    },
)
print(mesh)
x = bemcs.solve(mesh, geometry=geometry, nu=nu)

# %% the same fault and topography in a homogeneous medium: the slip elements only, mu = 1
slip_els = np.flatnonzero(mesh.labels == "s")
els_h = bemcs.initialize_els()
for k in ("x1", "y1", "x2", "y2"):
    setattr(els_h, k, np.asarray(getattr(mesh.els, k))[slip_els])
bemcs.standardize_els_geometry(els_h, reorder=False)
homog = bemcs.Mesh(
    els_h,
    bc_type=mesh.bc_type[slip_els],
    bc_value=mesh.bc_value[slip_els],
    mu_minus=C.MU_OUT,
    mu_plus=C.MU_OUT,
    basis={"slip": "hat"},
)
x_h = bemcs.solve(homog, geometry=geometry, nu=nu)


def fields(m, coefs, xo, yo):
    return bemcs.evaluate_fields(m, coefs, xo, yo, geometry=geometry, nu=nu)


# %% surface displacement
xs = np.linspace(-200, 400, 121)
ys = C.topography(xs) - 1e-3  # just below the free surface
f_bi, f_h = fields(mesh, x, xs, ys), fields(homog, x_h, xs, ys)

fig, axes = plt.subplots(
    3, 1, figsize=(10, 9), sharex=True, gridspec_kw={"height_ratios": [1, 2, 2]}
)
ax = axes[0]
ax.fill_between(xs, C.topography(xs), -6, color="0.85")
ax.plot(xs, C.topography(xs), "k-")
ax.axvline(0, color="0.5", ls=":", lw=0.8)
ax.set_ylabel("elevation (km)")
ax.set_title("surface displacement (peak slip 1, thrust)")
for ax, comp, label in zip(axes[1:], ("ux", "uy"), ("$u_x$", "$u_y$")):
    ax.plot(xs, f_bi[comp], "b-", label="bimaterial (slab mu = 3)")
    ax.plot(xs, f_h[comp], "k--", label="homogeneous (mu = 1)")
    ax.axvline(0, color="0.5", ls=":", lw=0.8)
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)
axes[1].legend()
axes[-1].set_xlabel("x (km); trench at 0")
fig.tight_layout()
plt.show()

# %% traction change on the fault (upper-plate side; the true traction is continuous across the fault)
fault = mesh.bc_type[:, 0] == "s"
els = mesh.els
xc, yc = np.asarray(els.x_centers)[fault], np.asarray(els.y_centers)[fault]
nx, ny = np.asarray(els.x_normals)[fault], np.asarray(els.y_normals)[fault]
L_el = 2 * np.asarray(els.half_lengths)[fault]
tx, ty = (
    ny,
    -nx,
)  # unit tangent, down-dip (n is the element's left normal, pointing into the upper plate)
s_dd = np.cumsum(L_el) - 0.5 * L_el  # down-dip distance of the element centres
xo, yo = xc + 1e-3 * L_el * nx, yc + 1e-3 * L_el * ny

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for m, coefs, ls, name in (
    (mesh, x, "b-", "bimaterial"),
    (homog, x_h, "k--", "homogeneous"),
):
    f = fields(m, coefs, xo, yo)
    Tx, Ty = f["sxx"] * nx + f["sxy"] * ny, f["sxy"] * nx + f["syy"] * ny
    axes[0].plot(s_dd, Tx * tx + Ty * ty, ls, label=name)
    axes[1].plot(s_dd, Tx * nx + Ty * ny, ls, label=name)
axes[0].set_ylabel("shear traction change\n(down-dip)")
axes[1].set_ylabel("normal traction change\n(tension positive)")
for ax in axes:
    ax.grid(alpha=0.3)
    ax.axhline(0, color="0.5", lw=0.8)
axes[0].legend()
axes[1].set_xlabel("down-dip distance from the trench (km)")
axes[0].set_title("traction change on the fault")
fig.tight_layout()
plt.show()

# %% displacement and stress fields
fig, axs, grid = bemcs.plot_fields(
    mesh,
    x,
    geometry=geometry,
    nu=nu,
    xlim=(-150, 300),
    ylim=(-120, 5),
    n=(251, 101),
    components=["ux", "uy", "sxy", "pressure"],
    # pressure: the tension-positive mean stress (sxx + syy + szz) / 3, szz = nu (sxx + syy) in plane strain
    derived={"pressure": lambda f: (1 + nu) * (f["sxx"] + f["syy"]) / 3},
    mask=lambda X, Y: Y < C.topography(X),  # inside the rock
)
for ax in axs[:, 0]:
    ax.set_aspect("equal")
fig.set_size_inches(10, 13)
fig.tight_layout()
plt.show()

# %%
