# %% [markdown]
# # Dipping fault breaking a free surface: boundary conditions on a connectivity file
# A 60-degree dipping fault with prescribed slip reaches a traction-free surface with a Gaussian
# valley (plane strain). The fault meets the surface at a **triple junction**: the two surface
# elements on either side of the junction node and the top fault element share it.
#
# This example shows how to describe the problem with two files instead of one:
#
# - a **mesh file**: one row per element, geometry and moduli only (`x1, z1, x2, z2, mu_minus, mu_plus`);
# - a **connectivity file**: one row per slip *hat* (the slip basis used here), with the elements of
#   the hat and the boundary condition that the hat carries.
#
# **Connectivity rows.** Columns `e1, e2, e3, e4` are 0-based element indices (rows of the mesh file).
# `e2` is the centre: the hat has unit slip at the centre of `e2`, and the row's boundary condition is
# imposed there. `e1` and `e3` are the neighbours of `e2` at its start and end node. At a triple
# junction an element has a third neighbour, which goes in `e4`; everywhere else `e4 = -1`. Three
# rows here have one: the hats of the two surface elements next to the junction and of the top fault
# element. A 3-element chain cannot describe them (it would force the surface displacement to be
# continuous across the fault); bemcs warns (`HatConnectivityWarning`) if a junction row is missing
# its 4th element, or if a non-tip element has no row.
#
# **Boundary-condition columns** (plane strain, one type and value per component, as in a mesh file):
# `BC_type_1, value_1, BC_type_2, value_2`.
# - surface hats: `t, 0, t, 0` (traction free, global x and y);
# - fault hats: `s, s_s, s, 0` (local shear and normal slip at the centre of `e2`).
#
# Elements that are the centre of no row (the tip elements at the ends of the surface and at the
# bottom of the fault) get no boundary condition: the slip there tapers to zero at the open end. With
# the condition attached to the hat, there is no per-element value that is silently ignored.
#
# `mesh.connectivity_table(...)` writes the default layout of any mesh (with its boundary conditions)
# as a starting point; here the table is built by hand from `bemcs.one_ring_connectivity`.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import bemcs

plt.rcParams.update({"font.size": 12})

# %% Parameters
mu, nu = 1.0, 0.25
geometry = "planestrain"
topo_n = 50  # surface elements (even, so that x = 0 is a node)
topo_L = 50.0  # surface extent: x in [-L/2, L/2]
topo_amp, topo_sigma = -0.5, 1.0  # Gaussian valley
fault_n, fault_L, fault_dip = 6, 2.0, np.deg2rad(60.0)


def fault_slip(d):
    """Shear slip at down-dip distance d from the surface: -1 at the surface, tapering smoothly
    (cos^2) to 0 at the bottom of the fault."""
    return -np.cos(0.5 * np.pi * np.clip(d / fault_L, 0.0, 1.0)) ** 2


# %% Geometry: surface (left to right) and fault (from the junction down-dip)
xlog = np.logspace(-1.0, np.log10(topo_L / 2), topo_n // 2)
xt = np.r_[-xlog[::-1], 0.0, xlog]
yt = topo_amp * np.exp(-(xt**2) / (2 * topo_sigma**2))
d = np.linspace(0.0, fault_L, fault_n + 1)
xf, yf = d * np.cos(fault_dip), yt[topo_n // 2] - d * np.sin(fault_dip)  # starts at the valley floor

els = bemcs.initialize_els()
els.x1, els.y1 = np.r_[xt[:-1], xf[:-1]], np.r_[yt[:-1], yf[:-1]]
els.x2, els.y2 = np.r_[xt[1:], xf[1:]], np.r_[yt[1:], yf[1:]]
bemcs.standardize_els_geometry(els, reorder=False)
n_els = len(els.x1)
fault = np.arange(n_els) >= topo_n

# %% The two files
mesh_file = pd.DataFrame({"x1": els.x1, "z1": els.y1, "x2": els.x2, "z2": els.y2,
                          "mu_minus": mu, "mu_plus": mu})

conn = bemcs.one_ring_connectivity(els)  # one hat per non-tip element, junction rows 4 wide
table = pd.DataFrame(conn, columns=["e1", "e2", "e3", "e4"])
centre = table["e2"].to_numpy()
on_fault = fault[centre]
d_centre = np.hypot(els.x_centers[centre] - xf[0], els.y_centers[centre] - yf[0])
table["BC_type_1"] = np.where(on_fault, "s", "t")
table["value_1"] = np.where(on_fault, fault_slip(d_centre), 0.0)
table["BC_type_2"] = np.where(on_fault, "s", "t")
table["value_2"] = 0.0
print("connectivity rows at the triple junction:")
print(table[table["e4"] >= 0].to_string())

mesh_file.to_csv("freesurface_fault.csv", index=False)
table.to_csv("freesurface_fault_connectivity.csv", index=False)

# %% Read them back and solve
mesh = bemcs.Mesh.from_csv("freesurface_fault.csv", basis={"slip": "hat"},
                           connectivity={"slip": "freesurface_fault_connectivity.csv"})
print(mesh)
x, (A, b, rows) = bemcs.solve(mesh, geometry=geometry, mu=mu, nu=nu, return_system=True)
print(f"{A.shape[0]} unknowns (2 per hat), condition number {np.linalg.cond(A):.3g}")

fig, ax = plt.subplots(figsize=(10, 4))
bemcs.plot_mesh(mesh, ax=ax, show_moduli=False)
junction_rows = table[table["e4"] >= 0]
ax.plot(els.x_centers[junction_rows["e2"]], els.y_centers[junction_rows["e2"]], "o", mfc="none",
        mec="tab:red", ms=10, label="centres of the junction hats (rows with e4)")
ax.set_xlim(-2, 3)
ax.set_ylim(-2.5, 0.5)
ax.set_aspect("equal", adjustable="box")
ax.legend(loc="lower left", fontsize=9)
ax.set_title("mesh, unit normals, and the triple-junction hats")
plt.show()

# %% Check: the same problem with per-element conditions and bemcs's default layout
bc_type = np.where(fault, "s", "t")
bc_value = np.zeros((n_els, 2))
bc_value[centre, :] = table[["value_1", "value_2"]].to_numpy()
for name, basis in (("default one-ring hats", "hat"), ("quadratic elements", "quadratic")):
    ref = bemcs.Mesh(els, bc_type=bc_type, bc_value=bc_value, mu_minus=mu, mu_plus=mu,
                     basis={"slip": basis})
    x_ref = bemcs.solve(ref, geometry=geometry, mu=mu, nu=nu)
    xo, yo = np.linspace(-3, 3, 25), np.full(25, -1.5)
    u = bemcs.evaluate_fields(mesh, x, xo, yo, geometry=geometry, mu=mu, nu=nu)
    u_ref = bemcs.evaluate_fields(ref, x_ref, xo, yo, geometry=geometry, mu=mu, nu=nu)
    print(f"  vs {name}: max |du| = {max(np.abs(u[c] - u_ref[c]).max() for c in ('ux', 'uy')):.1e}")

# %% Slip on the fault, and the surface displacement jump at the junction
n_eval = 10
xs_, ys_, slip = mesh.values_on_elements(x, "slip", geometry=geometry, n_eval=n_eval, frame="local")
d_eval = np.hypot(xs_ - xf[0], ys_ - yf[0])
f_eval = np.repeat(fault, n_eval)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
ax = axes[0]
ax.plot(d_eval[f_eval], slip[f_eval, 0], "b-", label="$s_s$ (shear slip)")
ax.plot(d_eval[f_eval], slip[f_eval, 1], "g-", label="$s_n$ (opening)")
ax.plot(d_centre[on_fault], table["value_1"][on_fault], "ko", mfc="none",
        label="prescribed at the hat centres")
ax.set_xlabel("down-dip distance from the surface")
ax.set_ylabel("slip")
ax.grid(alpha=0.3)
ax.legend()
ax.set_title("fault slip (the bottom tip element has no row: slip tapers to 0)", fontsize=10)

# surface "slip" is the surface displacement (the medium is on one side only)
ax = axes[1]
s_eval = ~f_eval & (np.abs(xs_) < 3)
u_surf = mesh.values_on_elements(x, "slip", geometry=geometry, n_eval=n_eval, frame="global")[2]
for c, lab in enumerate(("$u_x$", "$u_y$")):
    ax.plot(xs_[s_eval], u_surf[s_eval, c], ".", ms=3, label=f"{lab} at the surface")
ax.axvline(0, color="0.6", lw=0.8, ls=":")
ax.set_xlabel("x")
ax.set_ylabel("surface displacement (= surface slip)")
ax.grid(alpha=0.3)
ax.legend()
ax.set_title("the surface displacement jumps by the fault slip at the junction", fontsize=10)
fig.tight_layout()
plt.show()

# %% Stress and displacement in the medium
width = 3.0
fig, axes, fields = bemcs.plot_fields(
    mesh, x, geometry=geometry, mu=mu, nu=nu, xlim=(-width, width), ylim=(-width, 0.0), n=(100, 100),
    components=["sxy"], cmap="coolwarm", mask=lambda X, Y: Y < np.interp(X, xt, yt),  # inside the medium
)
skip = (slice(None, None, 6), slice(None, None, 6))
axes[0, 0].quiver(fields["x"][skip], fields["y"][skip], fields["ux"][skip], fields["uy"][skip],
                  width=0.002, scale=10)
axes[0, 0].set_aspect("equal", adjustable="box")
fig.set_size_inches(10, 5)
plt.show()
