# %% [markdown]
# # Mesh for the layered half-space example
# A compliant layer (shear modulus `MU1`, thickness `H`) over a stiff half-space (`MU2`), with a
# vertical strike-slip fault that breaks the free surface. Two faults:
# - `D_IN` < H: the fault stays inside the layer;
# - `D_CROSS` > H: the fault cuts through the layer-half-space interface.
#
# Elements (one row each in the CSV, columns `x1, z1, x2, z2, BC_type, value, mu_minus, mu_plus`):
# - fault, `s`: prescribed slip `slip_profile(depth)` (1 at the surface, tapering to 0 at depth D);
# - free surface at z = 0, `t`: traction-free;
# - interface at z = -H, `h`: material interface; elements run left to right, so the unit normal
#   points up: `mu_minus` = MU2 (below), `mu_plus` = MU1 (above).
#
# The free surface and the interface extend to |x| = L = 1000 H, far enough that the clamped (tapered)
# ends of these lines do not affect the comparison with the half-space solution near the fault (the
# example checks this). Away from the fault the elements are log-spaced: each is 8% longer than the
# previous one, up to 5 H. The cap keeps the surface and the interface (H apart) resolved; longer
# elements would put collocation points within 0.1 element lengths of the other line
# (`CollocationProximityWarning`).
#
# **Force trapezoids are laid out symmetrically about the fault** (`force_connectivity`). The default
# layout (stride 2 from one end of the interface) puts a collocation point on one side of the fault
# crossing only; the problem is antisymmetric in x, and the lopsided layout gives the interface a
# spurious net force whose far field grows like log r (it then dominates the error, and grows with L).
# Restarting the stride at the crossing node (collocation points 1.5 elements away on both sides)
# keeps the discrete problem antisymmetric. The CSV cannot hold this connectivity: pass it to
# `bemcs.Mesh.from_csv(..., connectivity={"force": force_connectivity(table)})`.
#
# The fault does not get a node on the interface: it crosses it inside one of its elements, and the
# interface has a node there. (A node shared by the fault and the interface would be treated as an
# open, clamped node for both, forcing zero slip at the crossing.)
#
# Running this file writes `fault_in_layer.csv` and `fault_through_interface.csv`, read by
# `HeterogeneousElasticity_forces.py`, which also imports the parameters and `layered_mesh` from here.

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
    """One row per element: fault, free surface, interface (see the top of this file)."""
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
        "BC_type": ["s"] * n_f + ["t"] * n + ["h"] * n,
        "value": np.r_[slip_profile(-zc, D), np.zeros(2 * n)],
        "mu_minus": np.r_[mu_fault, np.full(n, MU1), np.full(n, MU2)],
        "mu_plus": np.r_[mu_fault, np.full(n, MU1), np.full(n, MU1)],
    })


BASIS = {"slip": "quadratic", "force": "trapezoid"}


def force_connectivity(table):
    """Force trapezoids (global element indices) mirror-symmetric about the interface node at x = 0:
    middle elements c + 1, c + 3, ... to the right and c - 2, c - 4, ... to the left, where c is the
    interface element starting at x = 0. The two trapezoids next to the crossing share no element."""
    iface = np.flatnonzero(np.asarray(table["BC_type"]) == "h")
    x1 = np.asarray(table["x1"], float)[iface]
    n, c = len(iface), int(np.flatnonzero(np.isclose(x1, 0.0))[0])
    mids = sorted(list(range(c + 1, n - 1, 2)) + list(range(c - 2, 0, -2)))
    return np.array([[iface[m - 1], iface[m], iface[m + 1]] for m in mids])


def layered_mesh(D, h=H_EL, L=L, symmetric=True):
    """The same mesh as a bemcs.Mesh, without going through a CSV file. symmetric=False uses the
    default trapezoid layout (for comparison)."""
    t = mesh_table(D, h, L)
    els = bemcs.initialize_els()
    els.x1, els.y1, els.x2, els.y2 = (t[c].to_numpy(float) for c in ("x1", "z1", "x2", "z2"))
    bemcs.standardize_els_geometry(els, reorder=False)
    return bemcs.Mesh(els, bc_type=t["BC_type"].to_numpy(), bc_value=t["value"].to_numpy(),
                      mu_minus=t["mu_minus"].to_numpy(), mu_plus=t["mu_plus"].to_numpy(), basis=BASIS,
                      connectivity={"force": force_connectivity(t)} if symmetric else None)


# %% write the two mesh files
if __name__ == "__main__":
    for name, D in (("fault_in_layer.csv", D_IN), ("fault_through_interface.csv", D_CROSS)):
        mesh_table(D).to_csv(name, index=False)
        print(f"wrote {name}: fault depth {D}, layer thickness {H}, mu {MU1} over {MU2}")
