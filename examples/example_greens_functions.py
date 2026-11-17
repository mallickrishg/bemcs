# %% [markdown]
# # Green's functions of a heterogeneous mesh: `bemcs.get_greens_functions`
# The fault ("s" elements) is the source; everything else in the mesh (here a free surface and a
# material interface) is solved for. `get_greens_functions` returns, for unit slip on each slip hat
# of the fault, the displacement and stress at your observation points, so that the fields for any
# slip distribution are a matrix product. This is what inversions of geodetic data, or
# earthquake-cycle models, need.
#
# The same mesh file works in **antiplane** (strike-slip, out-of-plane displacement uz) and in
# **plane strain** (dip-slip and opening, in-plane displacements ux, uy). This script writes one mesh
# file and builds the Green's functions for both geometries from it:
#
# ```python
# gf = bemcs.get_greens_functions("mesh.csv", x_obs, y_obs, geometry="planestrain", nu=0.25)
# gf.fields["uy"]      # (n_obs, n_sources): uy at every observation point for unit slip at each source
# gf.fields["sxy"]     # true stresses too, e.g. on the fault itself (stress-interaction kernels)
# gf.slip_average      # (n_fault_elements, n_components, n_sources): each source as per-element slip
# u = gf(a)            # all fields for hat amplitudes a (one per source)
# u = gf(gf.amplitudes(slip))   # ... or for slip values at the sources' collocation points
# ```
#
# **Model.** A fault dipping at 60 degrees breaks the free surface and reaches depth D = 1. A
# compliant layer (shear modulus 1) lies over a stiffer half-space (modulus 3) below depth 1.5.
# Lengths are in units of D.

# %% imports
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bemcs

HERE = (
    os.path.dirname(os.path.abspath(__file__))
    if "__file__" in globals()
    else os.getcwd()
)
MESH_FILE = os.path.join(HERE, "example_greens_functions_mesh.csv")

DIP, D = np.deg2rad(60.0), 1.0  # fault dip and depth
H_LAYER, MU_LAYER, MU_BELOW = 1.5, 1.0, 3.0  # layer thickness and moduli
NU = 0.25  # Poisson's ratio (plane strain only)
H_EL = 0.05  # element length on the fault and near it

# %% [markdown]
# ## 1. The mesh file
# One row per element: end points, the shear modulus on the element's -n and +n side, and the
# boundary condition. The shorthand `BC_type, value` (the same type on both components) makes the
# file valid for both geometries. The values do not matter for Green's functions: the slip on "s"
# elements is the unknown amplitude of each source, and "t" values are set to zero.
#
# - fault: `s` (prescribed slip; the sources);
# - free surface: `t` (traction-free), modulus of the layer;
# - interface: `h` (material interface); elements run left to right, so the unit normal points up:
#   mu_minus = the half-space below, mu_plus = the layer above.
#
# The free surface and the interface extend to |x| = 200 with log-spaced elements. No connectivity
# file is given, so bemcs builds the basis patches: slip hats (including the triple junction where
# the fault meets the free surface) and force trapezoids.


def graded(h, L=200.0, flat=2.0, ratio=1.08, h_max=5.0):
    """Nodes on [-L, L]: spacing h for |x| < flat, then each element 8% longer, up to h_max."""
    r, step = list(np.round(np.arange(0.0, flat + 1e-12, h), 12)), h
    while r[-1] < L:
        r.append(r[-1] + step)
        step = min(step * ratio, h_max)
    r = np.array(r)
    return np.r_[-r[:0:-1], r]


def write_mesh_file(path, h=H_EL):
    s = np.linspace(
        0.0, D / np.sin(DIP), int(round(D / np.sin(DIP) / h)) + 1
    )  # along the fault
    xf, zf = s * np.cos(DIP), -s * np.sin(DIP)
    xs = graded(h)
    n_f, n_s = len(s) - 1, len(xs) - 1
    rows = {
        "x1": np.r_[xf[:-1], xs[:-1], xs[:-1]],
        "z1": np.r_[zf[:-1], np.zeros(n_s), np.full(n_s, -H_LAYER)],
        "x2": np.r_[xf[1:], xs[1:], xs[1:]],
        "z2": np.r_[zf[1:], np.zeros(n_s), np.full(n_s, -H_LAYER)],
        "mu_minus": np.r_[
            np.full(n_f, MU_LAYER), np.full(n_s, MU_LAYER), np.full(n_s, MU_BELOW)
        ],
        "mu_plus": np.r_[
            np.full(n_f, MU_LAYER), np.full(n_s, MU_LAYER), np.full(n_s, MU_LAYER)
        ],
        "BC_type": ["s"] * n_f + ["t"] * n_s + ["h"] * n_s,
        "value": np.zeros(n_f + 2 * n_s),
    }
    pd.DataFrame(rows).to_csv(path, index=False)
    return n_f, n_s


n_f, n_s = write_mesh_file(MESH_FILE)
print(
    f"wrote {MESH_FILE}: {n_f} fault, {n_s} free-surface and {n_s} interface elements"
)

# %% [markdown]
# ## 2. Green's functions for both geometries
# Observation points:
# - the free surface (just below it, inside the medium), where geodetic data are;
# - the fault itself, at the centre of every fault element (the sources' collocation points). There
#   the useful output is the traction on the fault (shear and normal stress on its plane), which is
#   continuous across the fault: the stress-interaction kernels that earthquake-cycle models need.
#   (The displacement jumps across the fault, so on the fault it is the average of its two sides.)
#
# The call is the same for both geometries; `nu` is only used in plane strain.

mesh = bemcs.Mesh.from_csv(MESH_FILE)
fault = np.flatnonzero(mesh.bc_type[:, 0] == "s")  # fault elements, top to bottom
els = mesh.els
xf, yf = np.asarray(els.x_centers)[fault], np.asarray(els.y_centers)[fault]
nx, ny = np.asarray(els.x_normals)[fault], np.asarray(els.y_normals)[fault]
sx, sy = np.asarray(els.x_shears)[fault], np.asarray(els.y_shears)[fault]
depth_f = -yf

x_surf = np.r_[-np.geomspace(6, 0.01, 60), np.geomspace(0.01, 6, 60)]
x_obs = np.r_[x_surf, xf]
y_obs = np.r_[np.full_like(x_surf, -1e-9), yf]
surface, on_fault = slice(0, len(x_surf)), slice(len(x_surf), None)

gfs = {}
for geometry in ("antiplane", "planestrain"):
    gfs[geometry] = bemcs.get_greens_functions(
        MESH_FILE, x_obs, y_obs, geometry=geometry, nu=NU
    )
    print(gfs[geometry])
gf_a, gf_p = gfs["antiplane"], gfs["planestrain"]


# the stress-interaction kernels: traction on the fault plane at each fault element, per source
K_a = gf_a.tractions(nx, ny, index=on_fault)  # (n_fault, n_sources): shear
K_p = gf_p.tractions(nx, ny, sx, sy, index=on_fault)  # (n_fault, 2, n_sources): shear, normal
print("fault stress kernels: antiplane", K_a.shape, " plane strain", K_p.shape)

# %% [markdown]
# ## 3. What a source is
# A source is one slip hat with unit amplitude, in one slip component. A hat is a smooth,
# non-negative bump over three fault elements: 1 at the centre of its centre element, 0 (with zero
# slope) at its two outer nodes. In antiplane there is one component (strike-slip); in plane strain
# two (0: along the fault, i.e. dip-slip; 1: normal, i.e. opening), so twice as many sources.
# `gf.sources` says which element and component each one is; `gf.slip_average` gives the average
# slip it puts on every fault element (the per-element slip most codes use), and
# `gf.slip_coefficients` the quadratic coefficients behind the smooth shape.
#
# Neighbouring hats overlap (a hat is 1/6 at its neighbours' centres), so the amplitudes are not
# the slip at the hat centres. `gf.amplitudes(slip)` converts slip values at the sources'
# collocation points into amplitudes.

print(
    "antiplane sources:",
    gf_a.n_sources,
    "  plane strain sources:",
    gf_p.n_sources,
    "(per component:",
    np.bincount(gf_p.sources["component"]),
    ")",
)
pick = [gf_a.n_sources // 4, gf_a.n_sources // 2, gf_a.n_sources - 1]  # buried sources


def plane_strain_source(element, component):
    return np.flatnonzero(
        (gf_p.sources["element"] == element) & (gf_p.sources["component"] == component)
    )[0]


# distance down the fault: of each fault element's end nodes, and along it
s_nodes = np.r_[0.0, np.cumsum(2 * np.asarray(els.half_lengths)[fault])]


def source_slip(gf, j, n_eval=20):
    """Slip along the fault (local components) for source j alone, and its element averages."""
    a = np.zeros(gf.n_sources)
    a[j] = 1.0
    xe, ye, v = gf.mesh.values_on_elements(
        gf.solution(a), "slip", geometry=gf.geometry, n_eval=n_eval, frame="local"
    )
    slip_els = gf.mesh.elements("slip")
    keep = np.isin(np.repeat(slip_els, n_eval), fault)
    dist = np.hypot(
        np.asarray(xe)[keep] - els.x1[fault[0]], np.asarray(ye)[keep] - els.y1[fault[0]]
    )
    v = np.asarray(v).reshape(len(xe), -1)[keep]
    return dist, v, gf.slip_average[:, :, j]


def plot_source(ax, gf, j, color, component=0, label="", style="-"):
    dist, v, avg = source_slip(gf, j)
    ax.plot(dist, v[:, component], style, color=color, lw=1.5, label=label)
    ax.stairs(
        avg[:, component],
        s_nodes,
        color=color,
        lw=1,
        ls=style,
        alpha=0.7,
        baseline=None,
    )


fig, ax = plt.subplots(figsize=(7, 4.5))
X, Y = np.meshgrid(np.linspace(-3, 3, 241), np.linspace(-2.5, 0, 101))
ax.pcolormesh(
    X,
    Y,
    np.where(Y > -H_LAYER, MU_LAYER, MU_BELOW),
    cmap="Greys",
    vmin=0,
    vmax=5,
    shading="auto",
)
for e in range(mesh.n_els):
    ax.plot(
        [els.x1[e], els.x2[e]],
        [els.y1[e], els.y2[e]],
        {"s": "r-", "t": "k-", "h": "b-"}[mesh.bc_type[e, 0]],
        lw=2,
    )
ax.plot(x_surf, np.zeros_like(x_surf), "|", color="C2", ms=7)
ax.plot(xf, yf, "o", color="C2", ms=3.5, mfc="none")
ax.text(1.5, -0.6, rf"layer $\mu$ = {MU_LAYER:g}", fontsize=9)
ax.text(1.5, -2.1, rf"half-space $\mu$ = {MU_BELOW:g}", fontsize=9)
for label, style in (
    ("fault (s): the sources", "r-"),
    ("free surface (t)", "k-"),
    ("interface (h)", "b-"),
):
    ax.plot([], [], style, label=label)
ax.plot(
    [], [], "o", color="C2", mfc="none", label="observation points (surface, fault)"
)
ax.set_xlim(-3, 3)
ax.set_ylim(-2.5, 0.2)
ax.set_aspect("equal")
ax.set_xlabel("x / D")
ax.set_ylabel("depth / D")
ax.legend(fontsize=7, loc="lower left")
ax.set_title("the mesh file (one file, both geometries)", fontsize=10)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Using them: fields for a slip distribution
# The fault slips 1 at the surface, tapering smoothly to 0 at depth D: sampled at the sources'
# collocation points (depth `-gf.sources["y"]`) and turned into hat amplitudes. Plane strain:
# dip-slip only, so the opening sources get 0. For comparison, the same Green's functions in a
# homogeneous medium (the same mesh file with the layer's modulus everywhere).


def taper(depth):
    return np.cos(0.5 * np.pi * np.clip(depth, 0, D) / D) ** 2


amp_a = gf_a.amplitudes(taper(-gf_a.sources["y"]))
amp_p = gf_p.amplitudes(
    np.where(gf_p.sources["component"] == 0, taper(-gf_p.sources["y"]), 0.0)
)
fa, fp = gf_a(amp_a), gf_p(amp_p)

homog_file = os.path.join(HERE, "example_greens_functions_mesh_homogeneous.csv")
pd.read_csv(MESH_FILE).assign(mu_minus=MU_LAYER, mu_plus=MU_LAYER).to_csv(
    homog_file, index=False
)
gh_a = bemcs.get_greens_functions(homog_file, x_obs, y_obs, geometry="antiplane")
gh_p = bemcs.get_greens_functions(
    homog_file, x_obs, y_obs, geometry="planestrain", nu=NU
)
ha, hp = gh_a(amp_a), gh_p(amp_p)
Kh_a = gh_a.tractions(nx, ny, index=on_fault)
Kh_p = gh_p.tractions(nx, ny, sx, sy, index=on_fault)

# %% [markdown]
# ## 5. Antiplane (strike-slip)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax = axes[0, 0]
for i, j in enumerate(pick):
    plot_source(
        ax,
        gf_a,
        j,
        f"C{i}",
        label=f"source {j} (centre depth {-gf_a.sources['y'][j]:.2f})",
    )
ax.set_xlabel("distance down the fault / D")
ax.set_ylabel("slip (strike-slip)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
ax.set_title(
    "(a) sources: slip along the fault (lines) and element averages\n"
    "(steps, gf.slip_average[:, 0, j])",
    fontsize=10,
)
ax = axes[0, 1]
for i, j in enumerate(pick):
    ax.plot(
        x_surf, gf_a.fields["uz"][surface, j], "-", color=f"C{i}", label=f"source {j}"
    )
ax.set_xlabel("x / D")
ax.set_ylabel("uz per unit source")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
ax.set_title('(b) columns: gf.fields["uz"][surface, j]', fontsize=10)
ax = axes[1, 0]
ax.plot(
    x_surf, fa["uz"][surface], "-", color="C3", lw=2, label="layered: gf(amplitudes)"
)
ax.plot(x_surf, ha["uz"][surface], ":", color="k", label="homogeneous")
ax.set_xlabel("x / D")
ax.set_ylabel("uz")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
ax.set_title("(c) surface displacement for the tapered slip", fontsize=10)
ax = axes[1, 1]
ax.plot(K_a @ amp_a, depth_f, "o-", ms=3, color="C3", label="layered")
ax.plot(
    Kh_a @ amp_a,
    depth_f,
    "o:",
    ms=3,
    color="k",
    label="homogeneous",
)
ax.invert_yaxis()
ax.set_xlabel("shear traction on the fault")
ax.set_ylabel("depth / D")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
ax.set_title("(d) stress change on the fault (stress drop where it slips)", fontsize=10)
fig.suptitle('antiplane: geometry="antiplane"', fontsize=11)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Plane strain (dip-slip and opening)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax = axes[0, 0]
for c, (name, color, k) in enumerate(
    (("dip-slip", "C0", pick[0]), ("opening", "C3", pick[1]))
):
    j = plane_strain_source(gf_a.sources["element"][k], c)
    plot_source(
        ax,
        gf_p,
        j,
        color,
        component=0,
        style="-",
        label=f"{name} source {j}: shear slip",
    )
    plot_source(
        ax,
        gf_p,
        j,
        color,
        component=1,
        style="--",
        label=f"{name} source {j}: normal slip (opening)",
    )
ax.set_xlabel("distance down the fault / D")
ax.set_ylabel("slip")
ax.grid(alpha=0.3)
ax.legend(fontsize=7)
ax.set_title(
    "(e) a dip-slip and an opening source: both slip components along the\n"
    "fault (lines) and element averages (steps, gf.slip_average[:, c, j])",
    fontsize=10,
)
ax = axes[0, 1]
for i, j in enumerate(pick):
    jp = plane_strain_source(gf_a.sources["element"][j], 0)
    ax.plot(
        x_surf,
        gf_p.fields["uy"][surface, jp],
        "-",
        color=f"C{i}",
        label=f"dip-slip source {jp}",
    )
ax.set_xlabel("x / D")
ax.set_ylabel("uy per unit source")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
ax.set_title('(f) columns: gf.fields["uy"][surface, j]', fontsize=10)
ax = axes[1, 0]
for name, c in (("ux", "C0"), ("uy", "C3")):
    ax.plot(
        x_surf,
        fp[name][surface],
        "-",
        color=c,
        lw=2,
        label=f"{name}, layered: gf(amplitudes)",
    )
    ax.plot(x_surf, hp[name][surface], ":", color=c, label=f"{name}, homogeneous")
ax.set_xlabel("x / D")
ax.set_ylabel("displacement")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
ax.set_title("(g) surface displacement for the tapered dip-slip", fontsize=10)
ax = axes[1, 1]
shear, normal = (K_p @ amp_p).T  # K_p @ amp_p is (n_fault, 2): shear, normal
shear_h, normal_h = (Kh_p @ amp_p).T
ax.plot(shear, depth_f, "o-", ms=3, color="C3", label="shear, layered")
ax.plot(
    shear_h, depth_f, "o:", ms=3, color="C3", mfc="none", label="shear, homogeneous"
)
ax.plot(normal, depth_f, "s-", ms=3, color="C0", label="normal, layered")
ax.plot(
    normal_h, depth_f, "s:", ms=3, color="C0", mfc="none", label="normal, homogeneous"
)
ax.invert_yaxis()
ax.set_xlabel("traction on the fault")
ax.set_ylabel("depth / D")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
ax.set_title("(h) stress change on the fault: shear and normal", fontsize=10)
fig.suptitle('plane strain: geometry="planestrain"', fontsize=11)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Reading the figures.**
# - (a, e) A source is one hat: a smooth, non-negative bump over three elements (two where the fault
#   ends, e.g. at the free surface), in one slip component only (in e, the other component of each
#   source is zero all along the fault); the steps are its element averages.
#   A slip distribution is a weighted sum of hats.
# - (b, f) Each column of `gf.fields` is the response to one source; deeper sources give broader,
#   smaller surface signals.
# - (c, g) `gf(amplitudes)` gives the fields for a whole slip distribution. The compliant layer
#   concentrates the surface deformation near the fault compared with a homogeneous medium.
# - (d, h) On the fault, the traction is the stress change the slip causes: a stress drop where the
#   fault slips most. The rows of `gf.fields` at the fault points, projected on the fault plane, are
#   the stress-interaction kernels of the fault: `gf.tractions(nx, ny, sx, sy, index=...)` (`K_a`, `K_p`
#   above: fault elements x sources).
#
# **Other outputs**: `gf.slip_coefficients` (the quadratic coefficients on each fault element),
# `gf.point_slip` (slip at each source's collocation point per unit amplitude), `gf.matrix` and
# `gf.coefficients` (the BEM system and its solution for every source), `gf.solution(a)` (the
# coefficient vector for `mesh.values_on_elements`), and `components=[...]` to keep only some
# fields for large problems.
