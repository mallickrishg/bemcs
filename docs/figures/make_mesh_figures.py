"""Figure of a triple-junction connectivity file for docs/user_guide/mesh_input.md.

Run from the repository root:  python docs/figures/make_mesh_figures.py
The rows and boundary conditions are written by the package (`one_ring_connectivity`,
`Mesh.connectivity_table`), so the figure and the table in the docs follow the code.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import bemcs

OUT = Path(__file__).resolve().parent
INK, MUTED, GREY, SHADE = "#3d3d3a", "#7a7974", "#b8b7ae", "#ecebe6"
BC_COLORS = {"s": "#2a78d6", "t_local": "#eb6834", "t": "#1baf7a"}
plt.rcParams.update({"font.size": 10, "text.color": INK})


def line(x0, y0, x1, y1, n):
    t = np.linspace(0, 1, n + 1)
    return np.c_[x0 + (x1 - x0) * t[:-1], y0 + (y1 - y0) * t[:-1],
                 x0 + (x1 - x0) * t[1:], y0 + (y1 - y0) * t[1:]]


def junction_mesh():
    """Main fault A (left) and B (right) with a splay C, 4 elements each; A slip, B local
    traction, C global traction."""
    seg = np.r_[line(-1, 0, 0, 0, 4), line(0, 0, 1, 0, 4), line(0, 0, 0.3, -1, 4)]
    els = bemcs.initialize_els()
    els.x1, els.y1, els.x2, els.y2 = (seg[:, j].copy() for j in range(4))
    bemcs.standardize_els_geometry(els, reorder=False)
    n = len(seg)
    bc_type = np.array([["s", "s"]] * 4 + [["t_local", "t_local"]] * 4 + [["t", "t"]] * 4)
    bc_value = np.array([[1.0, 0.0]] * 4 + [[0.0, 0.0]] * 8)
    mesh = bemcs.Mesh(els, basis={"slip": "hat"}, bc_type=bc_type, bc_value=bc_value,
                      mu_minus=np.ones(n), mu_plus=np.ones(n))
    return els, mesh, bc_type


def draw_mesh(ax, els, bc_type, tips, highlight=None):
    """Elements coloured by BC type (tips grey); with `highlight` (a connectivity row), only
    the row's support is drawn in colour and its centre is thicker."""
    n = len(els.x1)
    for e in range(n):
        color = GREY if e in tips else BC_COLORS[bc_type[e, 0]]
        lw, alpha = 3.0, 1.0
        if highlight is not None:
            row = [r for r in highlight if r >= 0]
            if e not in row:
                color, alpha = GREY, 0.45
            lw = 6.0 if e == highlight[1] else 3.0
        ax.plot([els.x1[e], els.x2[e]], [els.y1[e], els.y2[e]], color=color, lw=lw, alpha=alpha,
                solid_capstyle="butt")
    nodes = np.unique(np.r_[np.c_[els.x1, els.y1], np.c_[els.x2, els.y2]], axis=0)
    ax.plot(nodes[:, 0], nodes[:, 1], "o", ms=3.5, mfc="white", mec=INK, mew=0.8, zorder=3)
    ax.plot(0, 0, "D", ms=7, mfc=INK, mec="white", mew=1.0, zorder=4)
    ax.set_aspect("equal")
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.15, 0.32)
    ax.axis("off")


def label_elements(ax, els, tips, highlight=None):
    for e in range(len(els.x1)):
        dx, dy = els.x2[e] - els.x1[e], els.y2[e] - els.y1[e]
        nx, ny = -dy / np.hypot(dx, dy), dx / np.hypot(dx, dy)
        off = 0.1
        if highlight is not None and e not in highlight:
            continue
        ax.text(els.x_centers[e] + off * nx, els.y_centers[e] + off * ny, str(e), ha="center",
                va="center", fontsize=9, color=MUTED if e in tips else INK)


def fig_junction_rows():
    els, mesh, bc_type = junction_mesh()
    conn = mesh.patch_connectivity("slip")
    centres = set(conn[:, 1].tolist())
    tips = [e for e in range(len(els.x1)) if e not in centres]
    junction_rows = [r for r in conn.tolist() if r[3] >= 0]

    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1], hspace=0.12, wspace=0.05)
    ax = fig.add_subplot(gs[0, :])
    draw_mesh(ax, els, bc_type, tips)
    label_elements(ax, els, tips)
    for name, (x, y) in {"A": (-0.55, 0.2), "B": (0.55, 0.2), "C": (0.02, -0.72)}.items():
        ax.text(x, y, name, fontsize=12, fontweight="bold", ha="center")
    ax.annotate("triple junction\n(no row of its own)", (0, 0), (0.85, -0.45), fontsize=9, ha="center",
                color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    handles = [plt.Line2D([], [], color=BC_COLORS[k], lw=3, label=f'"{k}"') for k in BC_COLORS]
    handles.append(plt.Line2D([], [], color=GREY, lw=3, label="tip: no row, no condition"))
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9, title="BC_type of the row",
              title_fontsize=9)
    ax.set_title("(a) Elements by boundary condition", loc="left", fontsize=10)

    for k, row in enumerate(junction_rows):
        axk = fig.add_subplot(gs[1, k])
        draw_mesh(axk, els, bc_type, tips, highlight=row)
        label_elements(axk, els, tips, highlight=row)
        e1, e2, e3, e4 = row
        bc = bc_type[e2, 0]
        axk.set_title(f"({'bcd'[k]}) row e1,e2,e3,e4 = {e1},{e2},{e3},{e4}\nBC_type {bc} at the centre of {e2}",
                      loc="left", fontsize=9)
    fig.savefig(OUT / "junction_connectivity.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(mesh.connectivity_table("slip", geometry="planestrain").to_csv(index=False))


if __name__ == "__main__":
    fig_junction_rows()
