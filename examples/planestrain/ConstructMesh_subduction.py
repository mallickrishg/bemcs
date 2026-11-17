# %% [markdown]
# # Mesh for the subduction-zone example (plane strain, bimaterial megathrust)
# Lengths in km. A stiff slab (shear modulus `MU_SLAB` = 3) subducts beneath an upper plate, and sits on a
# mantle below. Upper plate, mantle wedge and the mantle below the slab have `MU_OUT` = 1. Uniform Poisson's ratio.
#
# **Geometry**
# - **Free surface** (`t`): sigmoidal topography from `TOPO_LOW` = -4 (ocean) to `TOPO_HIGH` = +1 (upper plate),
#   rising over ~100 km landward of the trench (centre `TOPO_X0`, width `TOPO_W`). The trench is at x = 0.
# - **Megathrust** (slab top, from the trench down-dip): gently curved. The dip increases linearly with depth from
#   `DIP_TOP` = 10 deg at the trench to `DIP_BOT` = 20 deg at `FAULT_DEPTH` = 30 km below the trench. This part
#   is the fault: prescribed slip (`s`, a Gaussian in down-dip distance; thrust) **and** a material interface
#   (`h`, slab below, upper plate above). It is meshed twice, with slip and force elements on the same nodes
#   (tasks.md P6).
# - **Slab top below the fault**: a straight line at `DIP_BOT`, a material interface (`h`) only.
# - **Slab bottom** (`h`): the slab top offset by `SLAB_T` = 40 km, oceanward under the sea floor as well.
#   Oceanward of the trench the slab is the oceanic plate, with the sea floor as its top.
#
# The free surface and the two slab interfaces extend to `L` = 5000 km (in x, or along the slab), far enough
# to be comparable to a half-space solution. Element lengths grow with distance from the fault:
# `GROWTH` x distance, between `H0` on and near the fault and `H_MAX` far away. The cap keeps the parallel
# lines (sea floor and slab bottom, top and bottom of the slab: 40 km apart) resolved.
#
# **The trench** is a junction of four elements: the two surface elements on either side, and the top fault
# element, meshed as a slip element and a force element. The slip hats there are 4 wide (the `e4` column
# of the connectivity file). The force chain of the slab top starts on the free surface, an open end (non-zero
# force density at the surface). The force trapezoids restart there and at the bottom of the fault
# (`split_force_trapezoids`).
#
# Running this file writes three files, read by `example_subduction_bimaterial.py`:
# - `subduction.csv`: one row per element, `x1, z1, x2, z2, label, mu_minus, mu_plus`;
# - `subduction_slip_connectivity.csv`: one row per slip hat, `e1, e2, e3, e4, BC_type_1, value_1,
#   BC_type_2, value_2`;
# - `subduction_force_connectivity.csv`: one row per force trapezoid, `e1, e2, e3, BC_type_1, ...`.
#
# It also plots the mesh, for a check before solving.

# %% imports and parameters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path

import bemcs

MU_SLAB, MU_OUT, NU = 3.0, 1.0, 0.25

TOPO_LOW, TOPO_HIGH = -4.0, 1.0  # km: ocean floor, upper plate
TOPO_X0, TOPO_W = 60.0, 20.0  # sigmoid centre and width: 10-90 % of the rise over 4.4 TOPO_W ~ 90 km

DIP_TOP, DIP_BOT = np.deg2rad(10.0), np.deg2rad(20.0)
FAULT_DEPTH = 30.0  # depth range of the curved fault, below the trench
SLAB_T = 40.0  # slab thickness

SLIP_PEAK = 1.0  # peak thrust slip (results scale linearly with it)
SLIP_CENTRE, SLIP_SIGMA = 0.5, 1 / 7  # Gaussian centre and width, as fractions of the fault length

L = 5000.0  # extent of the free surface (|x|) and of the slab (along the slab)
H0 = 2.0  # element length on and near the fault
GROWTH = 0.08  # element length / distance from the fault, away from it
H_MAX = 100.0  # longest element

BASIS = {"slip": "hat", "force": "trapezoid"}


def topography(x):
    return TOPO_LOW + (TOPO_HIGH - TOPO_LOW) / (1.0 + np.exp(-(np.asarray(x, float) - TOPO_X0) / TOPO_W))


def dip_at(depth):
    """Fault dip at a depth below the trench: DIP_TOP at the trench to DIP_BOT at FAULT_DEPTH."""
    return DIP_TOP + (DIP_BOT - DIP_TOP) * np.clip(depth / FAULT_DEPTH, 0.0, 1.0)


def fault_curve(ds=0.01):
    """The curved fault, densely sampled from the trench to FAULT_DEPTH below it: (x, y)."""
    p = [np.array([0.0, topography(0.0)])]
    while p[-1][1] > topography(0.0) - FAULT_DEPTH:
        dip = dip_at(topography(0.0) - p[-1][1])
        p.append(p[-1] + ds * np.array([np.cos(dip), -np.sin(dip)]))
    p = np.array(p)
    # end exactly at FAULT_DEPTH (the last step overshoots by < ds)
    y_end = topography(0.0) - FAULT_DEPTH
    f = (p[-2, 1] - y_end) / (p[-2, 1] - p[-1, 1])
    p[-1] = p[-2] + f * (p[-1] - p[-2])
    return p


def arc_length(p):
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]


def slip_profile(s, s_fault):
    """Thrust slip (positive) at down-dip distance s along a fault of length s_fault: a Gaussian."""
    return SLIP_PEAK * np.exp(-0.5 * ((s / s_fault - SLIP_CENTRE) / SLIP_SIGMA) ** 2)


def distance_to(p, curve):
    """Distance from points p (n, 2) to the polyline curve (m, 2)."""
    a, b = curve[:-1], curve[1:]
    ab = b - a
    d = np.empty(len(p))
    for i, q in enumerate(np.atleast_2d(p)):
        t = np.clip(((q - a) * ab).sum(1) / (ab * ab).sum(1), 0.0, 1.0)
        d[i] = np.hypot(*(a + t[:, None] * ab - q).T).min()
    return d


def resample(curve, spacing):
    """Nodes along a dense polyline, marching from its first point with element lengths
    spacing(point). The last node is the curve's end point; the nodes are stretched evenly so
    that they end there exactly."""
    s = arc_length(curve)
    at = lambda v: np.c_[np.interp(v, s, curve[:, 0]), np.interp(v, s, curve[:, 1])]
    nodes = [0.0]
    while True:
        h = spacing(at(nodes[-1]))[0]
        if nodes[-1] + 1.5 * h >= s[-1]:
            break
        nodes.append(nodes[-1] + h)
    nodes = np.array(nodes + [s[-1]])
    # spread the shortfall of the last element over the whole line, in proportion
    nodes = nodes[0] + (nodes - nodes[0]) * (s[-1] - nodes[0]) / (nodes[-1] - nodes[0])
    out = at(nodes)
    out[0], out[-1] = curve[0], curve[-1]
    return out


def smooth_offset(curve, offset, width=10.0, ds=0.5):
    """The curve offset by `offset` to its right (below, for a curve running left to right), with
    normals from the tangent angle smoothed over `width` km (no corner at the trench)."""
    s = arc_length(curve)
    su = np.arange(0.0, s[-1] + ds, ds)
    pu = np.c_[np.interp(su, s, curve[:, 0]), np.interp(su, s, curve[:, 1])]
    th = np.unwrap(np.arctan2(*np.gradient(pu, axis=0)[:, ::-1].T))
    k = np.exp(-0.5 * (np.arange(-4 * width, 4 * width + ds, ds) / width) ** 2)
    pad = len(k) // 2
    th = np.convolve(np.r_[np.full(pad, th[0]), th, np.full(pad, th[-1])], k / k.sum(), mode="valid")
    return pu + offset * np.c_[np.sin(th), -np.cos(th)]


# %% the lines
def build_lines(L=L, h0=H0):
    """Dense curves and the meshed nodes of every line (extent L, element length h0 near the
    fault). Returns a dict of node arrays (n, 2), and the fault length."""
    fault = fault_curve()
    s_fault = arc_length(fault)[-1]
    far = lambda p: np.clip(GROWTH * distance_to(p, fault), h0, H_MAX)

    xs = np.r_[np.linspace(0.0, 500.0, 5001), np.linspace(500.0, L, 1001)[1:]]
    surf_right = np.c_[xs, topography(xs)]
    surf_left = np.c_[-xs, topography(-xs)]  # from the trench oceanward

    n_f = int(round(s_fault / h0))
    fault_nodes = resample(fault, lambda p: np.full(len(p), s_fault / n_f))
    end, dirn = fault[-1], np.array([np.cos(DIP_BOT), -np.sin(DIP_BOT)])
    ext = end + np.linspace(0.0, L - s_fault, 20001)[:, None] * dirn

    nodes = {
        "surface_left": resample(surf_left, far)[::-1],  # left to right, ends at the trench
        "surface_right": resample(surf_right, far),
        "fault": fault_nodes,
        "slab_top": resample(ext, far),
    }
    # slab bottom: the slab top (sea floor oceanward, fault, extension) offset by SLAB_T
    top = np.r_[surf_left[::-1], fault[1:], ext[1:]]
    bottom = smooth_offset(top, SLAB_T)
    k = np.argmin(distance_to(bottom, fault))  # mesh outward from the point nearest to the fault
    nodes["slab_bottom"] = np.r_[resample(bottom[k::-1], far)[:0:-1], resample(bottom[k:], far)]
    return nodes, s_fault


def slab_region(nodes):
    """The slab as a closed polygon: sea floor, fault and slab top, then the slab bottom back."""
    top = np.r_[nodes["surface_left"], nodes["fault"][1:], nodes["slab_top"][1:]]
    return Path(np.r_[top, nodes["slab_bottom"][::-1]])


def element_table(L=L, h0=H0):
    """One row per element, in the order: free surface (left, right), fault (slip), fault (force),
    slab top below the fault, slab bottom. Moduli on each side from the slab polygon; a free-surface
    element gets the modulus of the rock below it on both sides."""
    nodes, s_fault = build_lines(L, h0)
    parts = [("surface_left", "s", "t"), ("surface_right", "s", "t"), ("fault", "s", "s"),
             ("fault", "f", "h"), ("slab_top", "f", "h"), ("slab_bottom", "f", "h")]
    rows = []
    for name, label, bc in parts:
        p = nodes[name]
        rows.append(pd.DataFrame({"x1": p[:-1, 0], "z1": p[:-1, 1], "x2": p[1:, 0], "z2": p[1:, 1],
                                  "label": label, "BC_type": bc, "part": name}))
    t = pd.concat(rows, ignore_index=True)

    # slip on the fault slip elements, at their centres (thrust: negative local shear slip, since the
    # fault runs down-dip and its +n side is the hanging wall)
    on_fault = ((t.part == "fault") & (t.label == "s")).to_numpy()
    xc, zc = 0.5 * (t.x1 + t.x2).to_numpy(), 0.5 * (t.z1 + t.z2).to_numpy()
    s_c = np.cumsum(np.hypot(t.x2 - t.x1, t.z2 - t.z1)[on_fault]) - 0.5 * np.hypot(
        t.x2 - t.x1, t.z2 - t.z1)[on_fault]
    t["value"] = 0.0
    t.loc[on_fault, "value"] = -slip_profile(s_c.to_numpy(), s_fault)

    # moduli: sample just off each element on both sides
    slab = slab_region(nodes)
    dx, dz = (t.x2 - t.x1).to_numpy(), (t.z2 - t.z1).to_numpy()
    length = np.hypot(dx, dz)
    nx, nz = -dz / length, dx / length  # left normal (bemcs: +n)
    eps = 1e-4 * length
    mat = lambda x, z: np.where(slab.contains_points(np.c_[x, z]), MU_SLAB, MU_OUT)
    t["mu_minus"] = mat(xc - eps * nx, zc - eps * nz)
    t["mu_plus"] = mat(xc + eps * nx, zc + eps * nz)
    surf = (t.BC_type == "t").to_numpy()
    t.loc[surf, "mu_plus"] = t.loc[surf, "mu_minus"]  # rock below, air above: the rock's modulus
    t.attrs["s_fault"] = s_fault
    return t


def _els(t):
    els = bemcs.initialize_els()
    els.x1, els.y1, els.x2, els.y2 = (t[c].to_numpy(float) for c in ("x1", "z1", "x2", "z2"))
    bemcs.standardize_els_geometry(els, reorder=False)
    return els


def connectivity_tables(t):
    """Slip hats (one-ring, with the trench junction in e4) and force trapezoids (restarted at the
    trench and the bottom of the fault), with the boundary conditions on their rows."""
    bc_value = np.c_[t["value"].to_numpy(), np.zeros(len(t))]
    kw = dict(bc_type=t["BC_type"].to_numpy(), bc_value=bc_value, mu_minus=t["mu_minus"].to_numpy(),
              mu_plus=t["mu_plus"].to_numpy(), basis=BASIS, labels=t["label"].to_numpy())
    m = bemcs.Mesh(_els(t), **kw)
    m = bemcs.Mesh(m.els, **kw, connectivity={"force": m.split_force_trapezoids()})
    return m, {k: m.connectivity_table(k, geometry="planestrain") for k in ("slip", "force")}


def subduction_mesh(L=L, h0=H0):
    """The mesh as a bemcs.Mesh (without going through files), with geometry and moduli per element
    and the boundary conditions on the connectivity rows (as in the files). Returns (mesh, table)."""
    t = element_table(L, h0)
    _, tables = connectivity_tables(t)
    return bemcs.Mesh(_els(t), labels=t["label"].to_numpy(), mu_minus=t["mu_minus"].to_numpy(),
                      mu_plus=t["mu_plus"].to_numpy(), basis=BASIS, connectivity=tables), t


def write_files(name="subduction"):
    t = element_table()
    m, tables = connectivity_tables(t)
    t[["x1", "z1", "x2", "z2", "label", "mu_minus", "mu_plus"]].to_csv(f"{name}.csv", index=False)
    for kind, table in tables.items():
        table.to_csv(f"{name}_{kind}_connectivity.csv", index=False)
    return t, m, tables


# %% plots of the mesh
PART_COLOURS = {"surface_left": "tab:blue", "surface_right": "tab:blue", "fault": "k",
                "slab_top": "tab:red", "slab_bottom": "tab:orange"}
PART_NAMES = {"surface_left": "free surface (t)", "fault": "fault (s and h)",
              "slab_top": "slab top (h)", "slab_bottom": "slab bottom (h)"}


def _draw_lines(ax, t, nodes=True):
    for part, c in PART_COLOURS.items():
        sel = (t.part == part).to_numpy()
        x = np.c_[t.x1[sel], t.x2[sel], np.full(sel.sum(), np.nan)].ravel()
        z = np.c_[t.z1[sel], t.z2[sel], np.full(sel.sum(), np.nan)].ravel()
        ax.plot(x, z, "-", color=c, lw=1.2, label=PART_NAMES.get(part))
        if nodes:
            ax.plot(t.x1[sel], t.z1[sel], ".", color=c, ms=3)


def plot_geometry(t=None, ax=None, xlim=(-200, 350), ylim=(-160, 10)):
    """The region of interest at true scale: the mesh lines (dots: nodes) over the shear modulus."""
    t = element_table() if t is None else t
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 5))
    xg, zg = np.meshgrid(np.linspace(*xlim, 551), np.linspace(*ylim, 171))
    mu = np.where(slab_region(build_lines()[0]).contains_points(np.c_[xg.ravel(), zg.ravel()]),
                  MU_SLAB, MU_OUT).reshape(xg.shape)
    mu = np.where(zg > topography(xg), np.nan, mu)
    im = ax.pcolormesh(xg, zg, mu, cmap="Greys", vmin=0, vmax=4, shading="auto", alpha=0.5)
    _draw_lines(ax, t)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.figure.colorbar(im, ax=ax, label=r"$\mu$", shrink=0.7)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("z (km)")
    ax.set_title("region of interest (true scale)")
    return ax


def plot_review(t, m, tables):
    fault = ((t.part == "fault") & (t.label == "s")).to_numpy()
    colours = PART_COLOURS

    fig = plt.figure(figsize=(14, 13))
    ax = fig.add_subplot(3, 2, (1, 2))
    _draw_lines(ax, t, nodes=True)
    ax.set_xlim(-L * 1.02, L * 1.02)
    ax.set_title(f"whole mesh: {len(t)} elements (dots: nodes); vertical exaggeration none")
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=8)

    plot_geometry(t, ax=fig.add_subplot(3, 2, (3, 4)))
    xc, zc = 0.5 * (t.x1 + t.x2), 0.5 * (t.z1 + t.z2)

    ax = fig.add_subplot(3, 2, 5)
    bemcs.plot_mesh(m, ax=ax, show_moduli=True, offset=0.4)
    ax.set_xlim(-15, 30)
    ax.set_ylim(-12, 2)
    ax.set_aspect("equal")
    junction = tables["slip"][tables["slip"]["e4"] >= 0]
    ax.plot(xc[junction["e2"]], zc[junction["e2"]], "o", mfc="none", mec="tab:green", ms=11,
            label="junction hat centres (rows with e4)")
    ax.legend(loc="lower left", fontsize=7)
    ax.set_title(r"trench: normals and $\mu$ on each side of interface/surface elements", fontsize=10)

    ax = fig.add_subplot(3, 2, 6)
    length = np.hypot(t.x2 - t.x1, t.z2 - t.z1)
    nodes_fault = np.r_[t.x1[fault].to_numpy()[:1], t.x2[fault].to_numpy()], np.r_[
        t.z1[fault].to_numpy()[:1], t.z2[fault].to_numpy()]
    d = distance_to(np.c_[xc, zc], np.c_[nodes_fault])
    for part, c in colours.items():
        sel = (t.part == part).to_numpy()
        ax.loglog(np.maximum(d[sel], 0.1), length[sel], ".", color=c, ms=4)
    ax.set_xlabel("distance from the fault (km)")
    ax.set_ylabel("element length (km)")
    ax.grid(alpha=0.3, which="both")
    s = np.cumsum(length[fault]) - 0.5 * length[fault]
    ins = ax.inset_axes([0.08, 0.55, 0.4, 0.38])
    ins.plot(s, -t.value[fault], "k.-", ms=3)
    ins.set_xlabel("down-dip distance (km)", fontsize=7)
    ins.set_ylabel("thrust slip", fontsize=7)
    ins.tick_params(labelsize=6)
    ax.set_title("element length vs distance from the fault; inset: prescribed slip", fontsize=10)
    fig.tight_layout()
    return fig


# %% write the files and plot
if __name__ == "__main__":
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t, m, tables = write_files()
    for w in caught:
        print(f"WARNING {w.category.__name__}: {w.message}")
    counts = t.groupby(["part", "label"], sort=False).size()
    print(f"wrote subduction.csv and its two connectivity files: {len(t)} elements")
    print(counts.to_string())
    print(f"fault length {t.attrs['s_fault']:.1f} km, {len(tables['slip'])} slip hats, "
          f"{len(tables['force'])} force trapezoids")
    print("slip hats at the trench junction:")
    print(tables["slip"][tables["slip"]["e4"] >= 0].to_string(index=False))
    close = bemcs.close_collocation_points(m.els)
    print(f"collocation points within 0.1 element lengths of another element: {len(close['element'])}")
    fig = plot_review(t, m, tables)
    fig.savefig("subduction_mesh.png", dpi=130)
    plt.show()
