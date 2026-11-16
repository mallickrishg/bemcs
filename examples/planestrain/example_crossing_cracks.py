# %% [markdown]
# # Crossing cracks without remeshing
# Several straight cracks, each meshed on its own, cross each other: where two cracks intersect,
# their elements simply overlap in space (no node at the crossing, no junction). Each crack is
# traction-free under a uniform remote stress, so the boundary condition on every element is
# `t = -sigma_inf . n`.
#
# With quadratic hats every crack's stress is bounded everywhere (the singular parts of its
# elements cancel), so the crossings are harmless. The random tangle is drawn so that the
# crossings never close a loop: a closed loop of cracks would cut out a block bounded only by
# traction-free faces, which is free to translate (a rigid-body mode), not a problem we study.
#
# The last part is a separate test with two cracks, A and B (panels c and d): a node of B moves
# onto a collocation point (element centre) of A, to show when float64 round-off takes over.

# %% imports
import numpy as np
import matplotlib.pyplot as plt
import bemcs

SIG = np.array([[0.3, 1.0], [1.0, -0.4]])  # remote stress (sxx, sxy; sxy, syy)
geometry, mu, nu = "planestrain", 1.0, 0.25


def crack_mesh(segments, n_el):
    """Mesh each crack (start, end) with n_el elements; no nodes at crossings."""
    x1, y1, x2, y2 = [], [], [], []
    for a, b in segments:
        pts = np.linspace(a, b, n_el + 1)
        x1 += list(pts[:-1, 0])
        y1 += list(pts[:-1, 1])
        x2 += list(pts[1:, 0])
        y2 += list(pts[1:, 1])
    els = bemcs.initialize_els()
    els.x1, els.y1, els.x2, els.y2 = (np.array(v) for v in (x1, y1, x2, y2))
    bemcs.standardize_els_geometry(els, reorder=False)
    return els


def traction_free_mesh(els, basis):
    """Traction-free crack faces under the remote stress: BEM traction = -sigma_inf . n."""
    n = np.c_[els.x_normals, els.y_normals]
    return bemcs.Mesh(
        els,
        bc_type=["t"] * len(els.x1),
        bc_value=-n @ SIG,
        mu_minus=mu,
        mu_plus=mu,
        basis={"slip": basis},
    )


def crossing(a, b, c, d):
    """Intersection point of segments ab and cd, or None."""
    r, s = b - a, d - c
    den = r[0] * s[1] - r[1] * s[0]
    if den == 0:
        return None
    t = ((c - a)[0] * s[1] - (c - a)[1] * s[0]) / den
    u = ((c - a)[0] * r[1] - (c - a)[1] * r[0]) / den
    return a + t * r if (0 < t < 1 and 0 < u < 1) else None


def has_loop(pairs, n):
    """True if the crossing graph has a cycle (the cracks then enclose a block)."""
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra == rb:
            return True
        parent[ra] = rb
    return False


def random_tangle(rng, n_cracks=5, min_crossings=4):
    """Random straight cracks with at least min_crossings crossings and no closed loops."""
    while True:
        segs = []
        for _ in range(n_cracks):
            c = rng.uniform(-1, 1, 2)
            t = rng.uniform(0, np.pi)
            half = 0.5 * rng.uniform(1.5, 2.5) * np.array([np.cos(t), np.sin(t)])
            segs.append((c - half, c + half))
        pairs = [
            (i, j)
            for i in range(n_cracks)
            for j in range(i + 1, n_cracks)
            if crossing(*segs[i], *segs[j]) is not None
        ]
        if len(pairs) >= min_crossings and not has_loop(pairs, n_cracks):
            return segs, pairs


# %% a random tangle: solve with one-ring hats and with the general form
segs, pairs = random_tangle(np.random.default_rng(20126))
els = crack_mesh(segs, n_el=6)
print(
    "crossings:",
    len(pairs),
    "  junctions found by label_nodes:",
    len(bemcs.label_nodes(els)[2]),
)

hat = traction_free_mesh(els, "hat")
general = traction_free_mesh(els, "quadratic")
x_hat, (A_hat, _, _) = bemcs.solve(
    hat, geometry=geometry, mu=mu, nu=nu, return_system=True
)
x_gen = bemcs.solve(general, geometry=geometry, mu=mu, nu=nu)
print("condition number (hats): %.1f" % np.linalg.cond(A_hat))

# %% a node of crack B approaching a collocation point (element centre) of crack A
P = np.array([0.125, 0.0])  # centre of element 4 of crack A (8 elements on [-1, 1])
L = 0.25  # element length of crack A
DIR_B = np.array([np.cos(1.1), np.sin(1.1)])
APPROACH = np.array([np.cos(0.3), np.sin(0.3)])


def node_near_collocation(d, dr, approach=APPROACH):
    """Quadratic slip coefficients when node 4 of crack B is at distance d from P."""
    node = P + d * approach
    els_ab = crack_mesh(
        [
            (np.array([-1.0, 0.0]), np.array([1.0, 0.0])),
            (node - 4 * 0.22 * DIR_B, node + 4 * 0.22 * DIR_B),
        ],
        n_el=8,
    )
    mesh_ab = traction_free_mesh(els_ab, "hat")
    x = bemcs.solve(mesh_ab, geometry=geometry, mu=mu, nu=nu, dr=dr)
    return mesh_ab.element_coefficients(x, "slip", geometry=geometry)


ds = np.logspace(-1, -16, 31)
limit = node_near_collocation(10**-9.75, 0.0)  # converged solution
other_side = node_near_collocation(
    1e-8, 0.0, approach=-APPROACH
)  # limit from B's other side
err = {
    dr: [
        np.abs(node_near_collocation(d, dr) - limit).max() / np.abs(limit).max()
        for d in ds
    ]
    for dr in (0.0, 1e-9)
}

# %% plot all four panels (one cell, so the figure is complete when it is shown)
fig = plt.figure(figsize=(14, 12))

# (a) total maximum shear stress (remote + cracks)
ax = fig.add_subplot(2, 2, 1)
X, Y = np.meshgrid(np.linspace(-2.2, 2.2, 320), np.linspace(-2.2, 2.2, 320))
f = bemcs.evaluate_fields(hat, x_hat, X, Y, geometry=geometry, mu=mu, nu=nu)
sxx, syy, sxy = f["sxx"] + SIG[0, 0], f["syy"] + SIG[1, 1], f["sxy"] + SIG[0, 1]
tau = np.sqrt(0.25 * (sxx - syy) ** 2 + sxy**2)
tau0 = np.sqrt(0.25 * (SIG[0, 0] - SIG[1, 1]) ** 2 + SIG[0, 1] ** 2)
im = ax.pcolormesh(X, Y, tau / tau0, cmap="magma", vmin=0, vmax=3, shading="auto")
n_el = len(els.x1) // len(segs)
for k, (a, b) in enumerate(segs):
    e = slice(k * n_el, (k + 1) * n_el)
    ax.plot([a[0], b[0]], [a[1], b[1]], "-", color=f"C{k}", lw=2)
    ax.plot(els.x1[e], els.y1[e], ".", color=f"C{k}", ms=4)  # nodes
    # unit normals at the element centres: they point to the + side of the crack
    ax.quiver(
        els.x_centers[e],
        els.y_centers[e],
        els.x_normals[e],
        els.y_normals[e],
        color=f"C{k}",
        angles="xy",
        scale_units="xy",
        scale=4.5,
        width=0.006,
    )
    # start of the crack (position 0 in panel (b))
    ax.plot(*a, "s", color=f"C{k}", mec="w", ms=7)
    # label a quarter of the way along, on the - side (opposite to the normals)
    nk = np.array([els.x_normals[e][0], els.y_normals[e][0]])
    ax.text(
        *(a + 0.25 * (b - a) - 0.22 * nk),
        str(k),
        color="w",
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(boxstyle="circle,pad=0.2", fc=f"C{k}", ec="w"),
    )
for i, j in pairs:
    ax.plot(*crossing(*segs[i], *segs[j]), "wo", ms=8, mfc="none", mew=1.2)
fig.colorbar(im, ax=ax, label="max shear stress / remote value")
ax.set_aspect("equal")
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)
ax.set_title(
    f"(a) {len(segs)} cracks, {len(pairs)} crossings (circles), no remeshing\n"
    "squares: crack start; arrows: unit normals (+ side)",
    fontsize=10,
)

# (b) slip along each crack: hats (lines) and general form (dots) agree
ax = fig.add_subplot(2, 2, 2)
n_eval = 20
for mesh_k, x, style in ((hat, x_hat, "-"), (general, x_gen, ".")):
    _, _, v = mesh_k.values_on_elements(
        x, "slip", geometry=geometry, n_eval=n_eval, frame="local"
    )
    v = v.reshape(len(segs), n_el * n_eval, 2)
    s = np.linspace(0, 1, n_el * n_eval)
    for k in range(len(segs)):
        ax.plot(
            s,
            v[k, :, 0],
            style,
            color=f"C{k}",
            ms=3,
            lw=1.2,
            label=f"crack {k}" if style == "-" else None,
        )
ax.plot([], [], "k-", label="one-ring hats")
ax.plot([], [], "k.", label="general form")
ax.set_xlabel("position along crack from its start (square in (a)), normalised")
ax.set_ylabel("shear slip")
ax.grid(alpha=0.3)
ax.legend(fontsize=7, ncol=2)
ax.set_title(
    "(b) shear slip along each crack:\nhats and general form agree", fontsize=10
)

# (c) a separate two-crack test: crack A and crack B, with a node of B near a
# collocation point P (element centre) of A; d is exaggerated here
D_SKETCH = 0.08
ax = fig.add_subplot(2, 2, 3)
node = P + D_SKETCH * APPROACH
els_ab = crack_mesh(
    [
        (np.array([-1.0, 0.0]), np.array([1.0, 0.0])),
        (node - 4 * 0.22 * DIR_B, node + 4 * 0.22 * DIR_B),
    ],
    n_el=8,
)
for k, (name, color) in enumerate((("A", "k"), ("B", "C3"))):
    e = slice(8 * k, 8 * (k + 1))
    ax.plot(np.r_[els_ab.x1[e], els_ab.x2[e][-1]], np.r_[els_ab.y1[e], els_ab.y2[e][-1]],
            "-", color=color, lw=2, marker="|", ms=10, mew=1.5)
    ax.quiver(els_ab.x_centers[e], els_ab.y_centers[e], els_ab.x_normals[e],
              els_ab.y_normals[e], color=color, angles="xy", scale_units="xy",
              scale=6, width=0.005)
    ax.text(els_ab.x2[e][-1] + 0.06, els_ab.y2[e][-1], f"crack {name}", color=color,
            fontsize=11, fontweight="bold", va="center")
ax.plot(els_ab.x_centers[:8], els_ab.y_centers[:8], "k.", ms=5)
ax.plot(*P, "*", color="C0", ms=14, zorder=5, label="P: collocation point (centre of A's element 4)")
ax.plot(*node, "o", color="C3", mec="w", ms=9, zorder=5, label="node 4 of B")
ax.annotate("", xy=node, xytext=P, arrowprops=dict(arrowstyle="<->", color="C0", lw=1.2))
ax.set_aspect("equal")
ax.set_xlim(-1.15, 1.35)
ax.set_ylim(-0.95, 1.0)
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="lower right")
# zoom on P and B's node
axz = ax.inset_axes([0.03, 0.55, 0.4, 0.42])
w = 1.6 * D_SKETCH
for k, color in enumerate(("k", "C3")):
    e = slice(8 * k, 8 * (k + 1))
    axz.plot(np.r_[els_ab.x1[e], els_ab.x2[e][-1]], np.r_[els_ab.y1[e], els_ab.y2[e][-1]],
             "-", color=color, lw=2, marker="|", ms=14, mew=1.5)
axz.plot(*P, "*", color="C0", ms=16, zorder=5)
axz.plot(*node, "o", color="C3", mec="w", ms=10, zorder=5)
axz.annotate("", xy=node, xytext=P, arrowprops=dict(arrowstyle="<->", color="C0", lw=1.5))
axz.text(*(P + 0.5 * D_SKETCH * APPROACH + np.array([0.0, -0.025])), "d", color="C0", fontsize=12)
axz.set_xlim(P[0] - w, P[0] + w)
axz.set_ylim(P[1] - w, P[1] + w)
axz.set_aspect("equal")
axz.set_xticks([])
axz.set_yticks([])
ax.indicate_inset_zoom(axz, edgecolor="0.4")
ax.set_title(
    "(c) separate test, not the tangle above: node 4 of crack B moves\n"
    f"towards P along a fixed direction (d = {D_SKETCH} drawn, exaggerated);"
    " ticks: nodes, arrows: normals",
    fontsize=10,
)

# (d) how the solution behaves as d -> 0
ax = fig.add_subplot(2, 2, 4)
for dr, style, label in (
    (0.0, "o-", "collocation at the element centre (dr = 0)"),
    (1e-9, "s-", "default offset (dr = 1e-9 element lengths)"),
):
    ax.loglog(ds / L, np.maximum(err[dr], 1e-17), style, ms=3.5, lw=1, label=label)
ax.loglog(
    ds / L, 1e-16 / (ds / L), "k:", lw=1, label=r"float64 floor $\sim \epsilon\,L/d$"
)
ax.axvline(1e-9, color="0.5", lw=0.8)
ax.text(7e-10, 2e-11, "d = dr L", fontsize=8, color="0.4", rotation=90)
jump = np.abs(other_side - limit).max() / np.abs(limit).max()
ax.axhline(jump, color="C1", lw=0.8, ls="--")
ax.text(
    1e-2, jump * 1.6, "limit from the other side of crack B", fontsize=7, color="C1"
)
ax.set_xlabel("distance d from B's node to A's collocation point (element lengths of A)")
ax.set_ylabel("relative distance from the converged solution")
ax.set_ylim(1e-12, 1e3)
ax.invert_xaxis()
ax.legend(fontsize=7, loc="upper left")
ax.set_title(
    "(d) setup (c) as d -> 0: converges until round-off at dr = 0;\n"
    "with the default offset the sampling point crosses to B's other side",
    fontsize=10,
)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Reading panel (d).** With collocation exactly at the element centre (`dr = 0`) the solution
# converges to a finite limit as the node approaches (each crack's stress is bounded at its own
# nodes), until float64 round-off between B's element kernels (each ~1/d near the node) takes over
# near 1e-12 to 1e-13 element lengths; it blows up at 1e-16, and `bemcs.solve` raises an error when
# the node sits exactly on the collocation point. With the default offset the sampling point sits
# 1e-9 element lengths off the element: below that distance it is on the other side of crack B's
# line, where the in-plane stress differs by a jump proportional to the slip gradient, so the
# solution switches to the limit reached from that side (dashed line) and stays finite.
