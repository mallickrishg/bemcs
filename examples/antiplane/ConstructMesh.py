# %% [markdown]
# # Mesh for the layered half-space example (`example_Rybicki_layered.py`)
# A compliant layer (shear modulus `MU1`, thickness `H`) over a stiff half-space (`MU2`), with a
# vertical strike-slip fault that breaks the free surface. Two faults:
# - `D_IN` < H: the fault stays inside the layer;
# - `D_CROSS` > H: the fault cuts through the layer-half-space interface.
#
# **Three files per mesh.** The elements are in a mesh file; the basis functions, and the boundary
# condition each one carries, are in two connectivity files:
#
# - `<name>.csv`, one row per element: `x1, z1, x2, z2, label, mu_minus, mu_plus` (geometry, source
#   type `s` slip or `f` force, and moduli);
# - `<name>_slip_connectivity.csv`, one row per slip hat: `e1, e2, e3, e4, BC_type, value`;
# - `<name>_force_connectivity.csv`, one row per force trapezoid: `e1, e2, e3, BC_type, value`.
#
# Element indices are 0-based rows of the mesh file. The centre of a patch is `e2`, and the row's
# boundary condition is imposed at the centre of `e2`:
# - fault hats, `s`: prescribed slip `slip_profile(depth)` (1 at the surface, tapering to 0 at depth D);
# - free-surface hats, `t`: traction-free;
# - interface trapezoids, `h`: material interface; elements run left to right, so the unit normal
#   points up: `mu_minus` = MU2 (below), `mu_plus` = MU1 (above).
#
# Elements that are the centre of no patch (tips, and the ends of the trapezoids) get no condition.
# The `label` column matters for the last interface element at x = +L: the stride-2 trapezoids leave
# it out (its force is zero), so no connectivity row says it is a force element.
#
# **The triple junction.** The fault breaks the free surface at x = 0, where three slip elements meet:
# the two surface elements on either side and the top fault element. Their hats have a third
# neighbour, listed in `e4` (`-1` on every other row). Without it the hat would force the surface
# displacement to be continuous across the fault (bemcs warns, `HatConnectivityWarning`).
#
# The free surface and the interface extend to |x| = L = 1000 H, far enough that the clamped (tapered)
# ends of these lines do not affect the comparison with the half-space solution near the fault (the
# example checks this). Away from the fault the elements are log-spaced: each is 8% longer than the
# previous one, up to 5 H. The cap keeps the surface and the interface (H apart) resolved; longer
# elements would put collocation points within 0.1 element lengths of the other line
# (`CollocationProximityWarning`).
#
# **Force trapezoids.** Here the trapezoid stride restarts at the node where the fault crosses the
# interface, so collocation points flank the crossing on both sides, and each side gets a free force
# value at the crossing (a half-hat);
# `mesh.split_force_trapezoids()` builds that layout. bemcs's default layout uses trapezoids on every
# other element from one end of the interface: that stride runs through the crossing, and `bemcs.solve`
# warns (`TrapezoidCrossingWarning`) because the interface then carries a spurious net force whose far
# field grows like log r. The example shows that case for comparison (`layered_mesh(..., lopsided=True)`).
#
# The fault does not get a node on the interface: it crosses it inside one of its elements, and the
# interface has a node there. (A node shared by the fault and the interface would be treated as an
# open, clamped node for both, forcing zero slip at the crossing.)
#
# Running this file writes the mesh files `fault_in_layer.csv` and `fault_through_interface.csv` and
# their connectivity files `*_slip_connectivity.csv` and `*_force_connectivity.csv`, read by
# `example_Rybicki_layered.py`, which also imports the parameters and `layered_mesh` from here.

# %% imports
import numpy as np
import pandas as pd
import bemcs

H = 1.0  # layer thickness
MU1, MU2 = 1.0, 4.0  # layer, half-space
D_IN, D_CROSS = 0.6, 1.63  # fault depths: inside the layer, through the interface
L = 1000.0  # half-width of the free surface and the interface (truncation)
H_MAX = 5.0  # longest element (log-spaced elements stop growing here)
H_EL = 0.05  # element length near the fault (free surface and interface: |x| < 2) and on the fault


def slip_profile(depth, D):
    """Slip cos^2(pi depth / 2D): 1 at the surface, zero slip and slope at depth D."""
    depth = np.asarray(depth, float)
    return np.where(depth < D, np.cos(0.5 * np.pi * depth / D) ** 2, 0.0)


def graded_nodes(h, L, ratio=1.08, flat=2.0, h_max=H_MAX):
    """Nodes on [-L, L], symmetric with a node at 0: spacing h for |x| < flat, then log-spaced
    (each element `ratio` times longer than the previous one) up to a length h_max."""
    x, step = list(np.arange(0.0, flat, h)), h
    while x[-1] < L:
        x.append(x[-1] + step)
        step = min(step * ratio, h_max)
    x = np.array(x)
    return np.r_[-x[:0:-1], x]


def mesh_table(D, h=H_EL, L=L):
    """One row per element: fault, free surface, interface (see the top of this file), with the
    boundary condition of every element. The files split it into the mesh file and the connectivity
    files (`write_files`)."""
    n_f = max(2, int(round(D / h)))
    zf = np.linspace(0.0, -D, n_f + 1)
    frac = (H / (D / n_f)) % 1.0
    if D > H and min(frac, 1 - frac) < 0.1:
        raise ValueError(f"the interface is within 0.1 element lengths of a fault node (D = {D}, h = {h})")
    xs = graded_nodes(h, L)
    n = len(xs) - 1
    zc = 0.5 * (zf[:-1] + zf[1:])
    mu_fault = np.where(zc > -H, MU1, MU2)  # not used by "s" rows; for plots and checks
    return pd.DataFrame({
        "x1": np.r_[np.zeros(n_f), xs[:-1], xs[:-1]],
        "z1": np.r_[zf[:-1], np.zeros(n), np.full(n, -H)],
        "x2": np.r_[np.zeros(n_f), xs[1:], xs[1:]],
        "z2": np.r_[zf[1:], np.zeros(n), np.full(n, -H)],
        "label": ["s"] * n_f + ["s"] * n + ["f"] * n,
        "BC_type": ["s"] * n_f + ["t"] * n + ["h"] * n,
        "value": np.r_[slip_profile(-zc, D), np.zeros(2 * n)],
        "mu_minus": np.r_[mu_fault, np.full(n, MU1), np.full(n, MU2)],
        "mu_plus": np.r_[mu_fault, np.full(n, MU1), np.full(n, MU1)],
    })


BASIS = {"slip": "hat", "force": "trapezoid"}  # both are patch bases: one connectivity file each


def _els(t):
    els = bemcs.initialize_els()
    els.x1, els.y1, els.x2, els.y2 = (t[c].to_numpy(float) for c in ("x1", "z1", "x2", "z2"))
    bemcs.standardize_els_geometry(els, reorder=False)
    return els


def connectivity_tables(t, lopsided=False):
    """The slip-hat and force-trapezoid connectivity tables (DataFrames, BC columns included) of the
    element table t. Slip: the one-ring hats, with the third branch of the triple junction in e4.
    Force: the trapezoids restarted at the fault crossing (`split_force_trapezoids`), or with
    lopsided=True bemcs's plain stride-2 layout through the crossing (comparison)."""
    m = bemcs.Mesh(_els(t), bc_type=t["BC_type"].to_numpy(), bc_value=t["value"].to_numpy(),
                   mu_minus=t["mu_minus"].to_numpy(), mu_plus=t["mu_plus"].to_numpy(), basis=BASIS)
    if not lopsided:
        m = bemcs.Mesh(m.els, bc_type=m.bc_type, bc_value=m.bc_value, mu_minus=m.mu_minus,
                       mu_plus=m.mu_plus, basis=BASIS, connectivity={"force": m.split_force_trapezoids()})
    return {k: m.connectivity_table(k, geometry="antiplane") for k in ("slip", "force")}


def layered_mesh(D, h=H_EL, L=L, lopsided=False):
    """The same mesh as a bemcs.Mesh, without going through files: geometry and moduli per element,
    boundary conditions on the connectivity rows (as in the files)."""
    t = mesh_table(D, h, L)
    return bemcs.Mesh(_els(t), labels=t["label"].to_numpy(),
                      mu_minus=t["mu_minus"].to_numpy(), mu_plus=t["mu_plus"].to_numpy(),
                      basis=BASIS, connectivity=connectivity_tables(t, lopsided))


def write_files(name, D):
    """<name>.csv (elements: geometry and moduli) and its two connectivity files."""
    t = mesh_table(D)
    t[["x1", "z1", "x2", "z2", "label", "mu_minus", "mu_plus"]].to_csv(f"{name}.csv", index=False)
    tables = connectivity_tables(t)
    for kind, table in tables.items():
        table.to_csv(f"{name}_{kind}_connectivity.csv", index=False)
    return t, tables


# %% write the mesh files
if __name__ == "__main__":
    for name, D in (("fault_in_layer", D_IN), ("fault_through_interface", D_CROSS)):
        t, tables = write_files(name, D)
        slip = tables["slip"]
        print(f"wrote {name}.csv, {name}_slip_connectivity.csv and {name}_force_connectivity.csv: fault "
              f"depth {D}, layer thickness {H}, mu {MU1} over {MU2}; {len(slip)} slip hats, "
              f"{len(tables['force'])} force trapezoids")
        print("  hats at the triple junction (e4 is the third branch):")
        print(slip[slip["e4"] >= 0].to_string(index=False))
