"""Figures of the slip and force bases for docs/theory/continuous_sources.md.

Run from the repository root:  python docs/figures/make_basis_figures.py
Every curve is computed with the package's own basis transforms, so the figures follow the code.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import bemcs
from bemcs import bemAssembly as BA

OUT = Path(__file__).resolve().parent
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7a5cc4"]
GREY, INK, SHADE = "#b8b7ae", "#3d3d3a", "#ecebe6"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK})


def straight_fault(n, length=1.0):
    xs = np.arange(n + 1.0) * length
    els = bemcs.initialize_els()
    els.x1, els.y1, els.x2, els.y2 = xs[:-1], np.zeros(n), xs[1:], np.zeros(n)
    bemcs.standardize_els_geometry(els, reorder=False)
    return els


def profile(els, coefs, kind, n_eval=80):
    """x positions and values along a straight horizontal mesh, element by element."""
    x, _, v = BA.values_on_elements(els, coefs, kind=kind, geometry="antiplane", n_eval=n_eval)
    n = len(els.x1)
    return x.reshape(n, n_eval), v.reshape(n, n_eval)


def element_guides(ax, els, ymin, ymax, shade=None):
    for e in range(len(els.x1)):
        if shade is not None and e in shade:
            ax.axvspan(els.x1[e], els.x2[e], color=SHADE, zorder=0)
    for xb in np.r_[els.x1, els.x2[-1]]:
        ax.axvline(xb, color=GREY, lw=0.8, ls=":", zorder=0)
    ax.axhline(0, color=GREY, lw=0.8, zorder=0)
    ax.set_ylim(ymin, ymax)


def tidy(ax, xlabel="x", ylabel="slip"):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# -----------------------------------------------------------------------------------------
# Figure 1: slip, general form (element coefficients + continuity conditions)
# -----------------------------------------------------------------------------------------
def fig_slip_general():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1.6]})

    # (a) the three quadratic shape functions of one element (half-length w = 1)
    ax = axes[0]
    s = np.linspace(-1, 1, 200)
    P = bemcs.slip_functions(s, 1.0)
    nodes = np.array([-2 / 3, 0, 2 / 3])
    for k in range(3):
        ax.plot(s, P[:, k], color=COLORS[k], lw=2, label=f"shape function {k + 1}")
    ax.plot(s, P.sum(axis=1), color=INK, lw=2.5, label="sum (uniform slip)")
    for k in range(3):
        ax.plot(nodes[k], 1.0, "o", ms=8, color=COLORS[k], mec=INK, zorder=5)
    ax.plot(nodes, np.zeros(3), "o", ms=7, mfc="white", mec=INK, zorder=5,
            label="coefficient positions")
    ax.axhline(0, color=GREY, lw=0.8, zorder=0)
    ax.set_xticks([-1, -2 / 3, 0, 2 / 3, 1], ["-w", "-2w/3", "0", "2w/3", "w"])
    ax.set_ylim(-0.5, 2.3)
    ax.set_title("(a) one element: three quadratic shape functions", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    tidy(ax, "position along element s")

    # (b) a 5-element fault: per-element quadratic slip, continuous with continuous gradient,
    #     clamped (zero slip and slope) at both tips
    ax = axes[1]
    els = straight_fault(5)
    xc = els.x_centers
    target = np.exp(-((xc - 2.5) ** 2) / 1.2)  # centre slips to match; any values work
    T = np.asarray(BA.one_ring_hat_coefficients(els, geometry="antiplane"))
    hats = BA.hat_elements(els)
    C = np.array([np.r_[BA.values_on_elements(els, T[:, p], kind="slip", geometry="antiplane",
                                                n_eval=1)[2]] for p in range(T.shape[1])]).T
    amp = np.linalg.solve(C[hats], target[hats])
    c = T @ amp  # a slip field that satisfies every continuity condition
    X, V = profile(els, c, "slip")
    for e in range(5):
        ax.plot(X[e], V[e], color=COLORS[e % len(COLORS)], lw=2.5, label=f"element {e + 1}")
        w = els.half_lengths[e]
        ax.plot(xc[e] + nodes * w, c[3 * e:3 * e + 3], "o", ms=6, color=COLORS[e % len(COLORS)],
                mec=INK, zorder=5)
    tips = [els.x1[0], els.x2[-1]]
    ax.plot(tips, [0, 0], "s", ms=8, mfc="white", mec=INK, zorder=6, label="clamped tips")
    element_guides(ax, els, -0.1, 1.25)
    ax.set_title("(b) general form: 3 coefficients per element, continuity imposed by extra rows",
                 color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT / "slip_general.png", dpi=160)
    plt.close(fig)


# -----------------------------------------------------------------------------------------
# Figure 2: slip, reduced form (one-ring hats), including a triple junction
# -----------------------------------------------------------------------------------------
def fig_slip_reduced():
    fig = plt.figure(figsize=(12, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.75])

    # (a) every one-ring hat on a 7-element fault (tip elements carry none)
    ax = fig.add_subplot(gs[0, :2])
    els = straight_fault(7)
    T = np.asarray(BA.one_ring_hat_coefficients(els, geometry="antiplane"))
    hats = BA.hat_elements(els)
    for p, e in enumerate(hats):
        X, V = profile(els, T[:, p], "slip")
        nz = np.abs(V).max(axis=1) > 0
        col = COLORS[p % len(COLORS)]
        ax.plot(X[nz].ravel(), V[nz].ravel(), color=col, lw=2)
        ax.plot(els.x_centers[e], 1, "o", ms=6, color=col, mec=INK, zorder=5)
        ax.text(els.x_centers[e], 1.06, f"hat {p + 1}", color=col, ha="center", fontsize=8)
    element_guides(ax, els, -0.05, 1.2, shade=[0, len(els.x1) - 1])
    ax.text(els.x_centers[0], 0.55, "tip element:\nno hat", ha="center", fontsize=8, color=INK)
    ax.text(els.x_centers[-1], 0.55, "tip element:\nno hat", ha="center", fontsize=8, color=INK)
    ax.set_title("(a) reduced form: one hat per non-tip element; each equals 1 at its own centre "
                 "and is 2/3 at its element's ends", color=INK, fontsize=10)
    tidy(ax)

    # (b) inside one element: the pieces of the three hats that overlap it
    ax = fig.add_subplot(gs[0, 2])
    e0 = 3
    names = {e0 - 1: "hat of left neighbour", e0: "own hat", e0 + 1: "hat of right neighbour"}
    for k, (e, name) in enumerate(names.items()):
        p = int(np.where(hats == e)[0][0])
        X, V = profile(els, T[:, p], "slip")
        ax.plot(X[e0] - els.x_centers[e0], V[e0], color=COLORS[[0, 1, 2][k]], lw=2.5, label=name)
    Xs, Vs = profile(els, T.sum(axis=1), "slip")
    ax.plot(Xs[e0] - els.x_centers[e0], Vs[e0], color=INK, lw=1.5, ls="--", label="sum of all hats (= 4/3)")
    ax.axvspan(-0.5, 0.5, color=SHADE, zorder=0)
    ax.axhline(0, color=GREY, lw=0.8, zorder=0)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.05, 1.85)
    ax.set_xticks([-0.5, 0, 0.5], ["-w", "0", "w"])
    ax.set_title("(b) one element: the three hats that overlap it", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    tidy(ax, "position along element s")

    # (c)-(e) a triple junction: main fault A -> J -> B, splay C out of J at 40 degrees
    t = np.deg2rad(40)

    def branch(p0, p1, n):
        xs, ys = np.linspace(p0[0], p1[0], n + 1), np.linspace(p0[1], p1[1], n + 1)
        return [(xs[i], ys[i], xs[i + 1], ys[i + 1]) for i in range(n)]

    segs = np.array(branch((-4, 0), (0, 0), 4) + branch((0, 0), (4, 0), 4)
                    + branch((0, 0), (4 * np.cos(t), 4 * np.sin(t)), 4))
    tj = bemcs.initialize_els()
    tj.x1, tj.y1, tj.x2, tj.y2 = segs.T
    bemcs.standardize_els_geometry(tj, reorder=False)
    Tp = np.asarray(BA.one_ring_hat_coefficients(tj, geometry="planestrain"))
    hats_tj = list(BA.hat_elements(tj))
    n_eval = 40
    for k, (e, label) in enumerate(((3, "A"), (4, "B"), (8, "C"))):
        ax = fig.add_subplot(gs[1, k])
        p = hats_tj.index(e)
        c = Tp[:, :, p, 0].reshape(-1)  # unit shear slip at the centre of element e
        x, y, v = BA.values_on_elements(tj, c, kind="slip", geometry="planestrain", n_eval=n_eval,
                                        frame="local")
        x, y, v = x.reshape(-1, n_eval), y.reshape(-1, n_eval), v.reshape(-1, n_eval, 2)
        for j in range(len(tj.x1)):
            ax.plot([tj.x1[j], tj.x2[j]], [tj.y1[j], tj.y2[j]], color=GREY, lw=1.5, zorder=1)
            ax.plot([tj.x1[j], tj.x2[j]], [tj.y1[j], tj.y2[j]], "|", color=GREY, ms=5, zorder=1)
            for comp, col in ((0, COLORS[0]), (1, COLORS[1])):
                vals = v[j, :, comp]
                if np.abs(vals).max() < 1e-12:
                    continue
                ox, oy = x[j] + vals * tj.x_normals[j], y[j] + vals * tj.y_normals[j]
                ax.fill(np.r_[x[j], ox[::-1]], np.r_[y[j], oy[::-1]], color=col, alpha=0.18, lw=0)
                ax.plot(ox, oy, color=col, lw=2)
        ax.plot(tj.x_centers[e], tj.y_centers[e], "o", ms=7, mfc="white", mec=INK, mew=1.5, zorder=5)
        ax.plot(0, 0, "D", ms=6, color=INK, zorder=5)
        for lab, (lx, ly) in {"A": (-4.3, -0.4), "B": (4.2, -0.4),
                              "C": (4.3 * np.cos(t), 4.3 * np.sin(t) + 0.2)}.items():
            ax.text(lx, ly, lab, fontsize=11, weight="bold", color=INK, ha="center")
        ax.set_aspect("equal")
        ax.set_xlim(-4.6, 4.6)
        ax.set_ylim(-0.9, 3.0)
        ax.axis("off")
        ax.set_title(f"({'cde'[k]}) junction hat of the {label} element next to J", color=INK,
                     fontsize=10)
    fig.text(0.5, 0.02, "(c)-(e): unit shear slip at the centre of the circled element; slip drawn as an "
             "offset along each element's normal (blue: shear, orange: normal, local frame); diamond: "
             "junction J. The splay hat (e) puts equal and opposite slip on A and B (slip conservation).",
             ha="center", fontsize=8.5, color="#6b6a63", wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "slip_reduced.png", dpi=160)
    plt.close(fig)


# -----------------------------------------------------------------------------------------
# Figure 3: force, linear elements and trapezoids
# -----------------------------------------------------------------------------------------
def fig_force():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1, 1.6]})

    ax = axes[0]
    s = np.linspace(-1, 1, 100)
    ax.plot(s, (1 - s) / 2, color=COLORS[0], lw=2, label="ramp 1")
    ax.plot(s, (1 + s) / 2, color=COLORS[1], lw=2, label="ramp 2")
    ax.plot(s, np.ones_like(s), color=INK, lw=2.5, label="sum (uniform force)")
    ax.axhline(0, color=GREY, lw=0.8, zorder=0)
    ax.set_xticks([-1, 0, 1], ["-w", "0", "w"])
    ax.set_ylim(-0.1, 1.45)
    ax.set_title("(a) one element: two linear ramps", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    tidy(ax, "position along element s", "force")

    ax = axes[1]
    els = straight_fault(9)
    F = np.asarray(BA.trapezoid_coefficients(els, geometry="antiplane"))
    conn = BA.trapezoid_connectivity(els)
    for p in range(F.shape[1]):
        X, V = profile(els, F[:, p], "force")
        nz = np.abs(V).max(axis=1) > 0
        col = COLORS[p % len(COLORS)]
        ax.plot(X[nz].ravel(), V[nz].ravel(), color=col, lw=2.5)
        mid = conn[p, 1]
        ax.text(els.x_centers[mid], 1.06, f"trapezoid {p + 1}", color=col, ha="center", fontsize=8)
    element_guides(ax, els, -0.05, 1.25, shade=list(conn[:, 1]))
    ax.set_title("(b) trapezoids: centred on every other element; neighbours share one element",
                 color=INK, fontsize=10)
    tidy(ax, "x", "force")
    fig.tight_layout()
    fig.savefig(OUT / "force_bases.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    fig_slip_general()
    fig_slip_reduced()
    fig_force()
    print("wrote", *sorted(p.name for p in OUT.glob("*.png")))
