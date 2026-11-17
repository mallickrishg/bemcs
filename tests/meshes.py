"""Mesh builders (and their observation points) shared by several test modules.

Every setup that more than one test file uses lives here, so that test modules never
import each other. Exact solutions for these setups are in `reference_solutions.py`;
reference kernels in `reference_kernels.py`.

Setups (parameters as module constants, prefixed by the setup):

- layered half-space, antiplane (Rybicki): a vertical fault from the surface, a compliant
  layer over a stiff half-space (`layered_mesh`, LAYERED_*);
- vertical contact, antiplane (Rybicki): a fault near or on a vertical contact
  (`contact_mesh`, CONTACT_*);
- circular inclusion, plane strain (Muskhelishvili) (`inclusion_mesh`, INCLUSION_*);
- buried fault near a bimaterial interface, plane strain (Dundurs) (`dundurs_mesh`,
  DUNDURS_*);
- fault crossing a bimaterial interface, plane strain (tasks.md P2) (`crossing_mesh`,
  CROSSING_*);
- fault below a straight interface (`fault_and_interface`);
- fault breaking a free surface, plane strain (`free_surface_fault`);
- a chain of unit elements with a kink (`kinked_chain`).
"""

import numpy as np

import bemcs


def _els(x1, y1, x2, y2):
    els = bemcs.initialize_els()
    els.x1, els.y1, els.x2, els.y2 = (np.asarray(v, float) for v in (x1, y1, x2, y2))
    bemcs.standardize_els_geometry(els, reorder=False)
    return els


def graded(h, L=300.0, flat=2.0, ratio=1.08, h_max=5.0, centre=0.0):
    """Nodes on [centre - L, centre + L], symmetric about centre: spacing h for |x - centre| <=
    flat, then log-spaced (each element `ratio` times longer) up to h_max."""
    r, step = list(np.arange(0.0, flat + 1e-12, h)), h
    while r[-1] < L:
        r.append(r[-1] + step)
        step = min(step * ratio, h_max)
    r = np.round(np.array(r), 12)
    return centre + np.r_[-r[:0:-1], r]


# ----------------------------------------------------------------------------
# layered half-space, antiplane (Rybicki 1971)
# ----------------------------------------------------------------------------

LAYERED_H, LAYERED_MU1, LAYERED_MU2 = 1.0, 1.0, 4.0  # compliant layer over a stiff half-space
LAYERED_SURFACE = np.r_[-np.geomspace(5, 0.02, 12), np.geomspace(0.02, 5, 12)]  # surface points


def layered_slip(depth, D):
    """Slip cos^2(pi depth / 2D): 1 at the surface, zero slip and slope at depth D."""
    return np.where(depth < D, np.cos(0.5 * np.pi * depth / D) ** 2, 0.0)


def layered_mesh(D, h, L=300.0, ratio=1.08, h_max=5.0, lopsided=False):
    """Vertical fault from the surface to depth D (no node on the interface), free surface
    and interface on [-L, L]: spacing h for |x| < 2, then log-spaced up to h_max. Force
    trapezoids restarting at the fault crossing (`Mesh.split_force_trapezoids`, what a user
    would put in the connectivity file), or with lopsided=True no connectivity: the plain
    default stride, which runs through the crossing."""
    H, MU1, MU2 = LAYERED_H, LAYERED_MU1, LAYERED_MU2
    xs, step = list(np.arange(0.0, 2.0, h)), h
    while xs[-1] < L:
        xs.append(xs[-1] + step)
        step = min(step * ratio, h_max)
    xs = np.r_[-np.array(xs)[:0:-1], xs]
    n_f, n = max(2, int(round(D / h))), len(xs) - 1
    zf = np.linspace(0.0, -D, n_f + 1)
    assert np.abs(zf + H).min() > 0.1 * D / n_f
    zc = 0.5 * (zf[:-1] + zf[1:])
    els = bemcs.initialize_els()
    els.x1, els.x2 = np.r_[np.zeros(n_f), xs[:-1], xs[:-1]], np.r_[np.zeros(n_f), xs[1:], xs[1:]]
    els.y1 = np.r_[zf[:-1], np.zeros(n), np.full(n, -H)]
    els.y2 = np.r_[zf[1:], np.zeros(n), np.full(n, -H)]
    bemcs.standardize_els_geometry(els, reorder=False)
    mu_f = np.where(zc > -H, MU1, MU2)
    conn = None
    if not lopsided:
        conn = {"force": layered_mesh(D, h, L, ratio, h_max, lopsided=True).split_force_trapezoids()}
    return bemcs.Mesh(els, bc_type=["s"] * n_f + ["t"] * n + ["h"] * n, connectivity=conn,
                      bc_value=np.r_[layered_slip(-zc, D), np.zeros(2 * n)],
                      mu_minus=np.r_[mu_f, np.full(n, MU1), np.full(n, MU2)],
                      mu_plus=np.r_[mu_f, np.full(n, MU1), np.full(n, MU1)],
                      basis={"slip": "quadratic", "force": "trapezoid"})


# ----------------------------------------------------------------------------
# vertical contact, antiplane (Rybicki 1971)
# ----------------------------------------------------------------------------

CONTACT_MU1, CONTACT_MU2, CONTACT_D = 1.0, 4.0, 1.0
CONTACT_SURFACE = np.r_[-np.geomspace(5, 0.02, 12), np.geomspace(0.02, 5, 12)]  # surface points


def contact_mesh(dist, h, L=300.0, ratio=1.08, h_max=5.0):
    """Fault at x = 0 (0..CONTACT_D), free surface on [-L, L], vertical contact at x = dist from
    the surface to depth L; spacing h near the fault, log-spaced beyond (dist a multiple of h)."""
    MU1, MU2, D = CONTACT_MU1, CONTACT_MU2, CONTACT_D
    r, step = list(np.arange(0.0, 2.0 + 1e-12, h)), h
    while r[-1] < L:
        r.append(r[-1] + step)
        step = min(step * ratio, h_max)
    r = np.array(r)
    xs, zi = np.r_[-r[:0:-1], r], -r
    assert np.isclose(xs, dist).any()
    n_f = int(round(D / h))
    zf = np.linspace(0.0, -D, n_f + 1)
    els = _els(np.r_[np.zeros(n_f), xs[:-1], np.full(len(zi) - 1, dist)],
               np.r_[zf[:-1], np.zeros(len(xs) - 1), zi[:-1]],
               np.r_[np.zeros(n_f), xs[1:], np.full(len(zi) - 1, dist)],
               np.r_[zf[1:], np.zeros(len(xs) - 1), zi[1:]])
    bc = np.array(["s"] * n_f + ["t"] * (len(xs) - 1) + ["h"] * (len(zi) - 1))
    side = np.where(els.x_centers < dist, MU1, MU2)
    # vertical elements run downward: unit normal +x, so -n is the mu1 side
    mu_minus = np.where(bc == "t", side, MU1)
    mu_plus = np.where(bc == "t", side, np.where(bc == "h", MU2, MU2 if dist == 0 else MU1))
    value = np.where(bc == "s", np.cos(0.5 * np.pi * (-els.y_centers) / D) ** 2, 0.0)
    return bemcs.Mesh(els, bc_type=bc, bc_value=value, mu_minus=mu_minus, mu_plus=mu_plus,
                      basis={"slip": "hat", "force": "trapezoid"})


# ----------------------------------------------------------------------------
# circular inclusion, plane strain (Muskhelishvili)
# ----------------------------------------------------------------------------

INCLUSION_A, INCLUSION_NU, INCLUSION_MU_M = 1.0, 0.27, 1.0
INCLUSION_STRESS = (0.6, -0.3, 0.45)  # remote true stress in the matrix (sxx, syy, sxy)


def inclusion_mesh(n, mu_i):
    """n elements, clockwise (unit normal out of the inclusion: -n side = inclusion),
    closed exactly (the last node is the first)."""
    t = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    x1, y1 = INCLUSION_A * np.cos(-t), INCLUSION_A * np.sin(-t)
    els = _els(x1, y1, np.roll(x1, -1), np.roll(y1, -1))
    return bemcs.Mesh(els, bc_type=["h"] * n, mu_minus=mu_i, mu_plus=INCLUSION_MU_M, basis={"force": "trapezoid"})


def ring(r, n=12):
    """n observation points on a circle of radius r."""
    th = np.linspace(0.1, 2 * np.pi, n + 1)[:-1]
    return r * np.cos(th), r * np.sin(th)


# ----------------------------------------------------------------------------
# buried fault near a bimaterial interface, plane strain (Dundurs 1969)
# ----------------------------------------------------------------------------

DUNDURS_NU, DUNDURS_MU1 = 0.25, 1.0
DUNDURS_SLIP = (1.0, 0.5)  # local (t, n) slip amplitudes on the fault


def dundurs_fault(case):
    """(centre, angle, half-length) of the "horizontal" or "dipping" fault."""
    if case == "horizontal":
        return np.array([0.0, -0.6]), 0.0, 0.5
    return np.array([0.0, -0.9]), np.deg2rad(40.0), 0.5


def dundurs_mesh(case, h, mu2):
    """Fault ("s", hats, cos^2 slip) below the interface y = 0 ("h", out to 300); material 1
    (DUNDURS_MU1) below, mu2 above."""
    c, ang, a = dundurs_fault(case)
    t = np.array([np.cos(ang), np.sin(ang)])
    n_f = int(round(2 * a / h))
    xi = np.linspace(-a, a, n_f + 1)
    P = c + xi[:, None] * t
    xs = graded(h)
    n_i = len(xs) - 1
    els = _els(np.r_[P[:-1, 0], xs[:-1]], np.r_[P[:-1, 1], np.zeros(n_i)],
               np.r_[P[1:, 0], xs[1:]], np.r_[P[1:, 1], np.zeros(n_i)])
    profile = np.cos(0.5 * np.pi * (0.5 * (xi[:-1] + xi[1:])) / a) ** 2
    value = np.r_[np.c_[DUNDURS_SLIP[0] * profile, DUNDURS_SLIP[1] * profile], np.zeros((n_i, 2))]
    # interface elements run left to right: -n is below (material 1)
    return bemcs.Mesh(els, bc_type=["s"] * n_f + ["h"] * n_i, bc_value=value, mu_minus=DUNDURS_MU1,
                      mu_plus=np.r_[np.full(n_f, DUNDURS_MU1), np.full(n_i, mu2)],
                      basis={"slip": "hat", "force": "trapezoid"})


def dundurs_points(case):
    """Observation points around the fault and above the interface, away from both."""
    c, _, _ = dundurs_fault(case)
    th = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    P = np.r_[c + 0.9 * np.c_[np.cos(th), np.sin(th)], np.c_[np.linspace(-2, 2, 5), np.full(5, 0.5)]]
    P = P[np.abs(P[:, 1]) > 0.2]
    return P[:, 0], P[:, 1]


# ----------------------------------------------------------------------------
# fault crossing a bimaterial interface, plane strain (tasks.md P2)
# ----------------------------------------------------------------------------

CROSSING_NU, CROSSING_MU1 = 0.25, 1.0
CROSSING_SLIP = (1.0, 0.5)  # local (t, n) slip amplitudes
CROSSING_XC, CROSSING_A_LO, CROSSING_A_HI = 0.15, 0.7, 0.5  # crossing (XC, 0); fault s in [-A_LO, A_HI]


def crossing_frame(dip):
    """(crossing point, unit vector along the fault) for a fault dipping at `dip` degrees."""
    d = np.deg2rad(dip)
    return np.array([CROSSING_XC, 0.0]), np.array([np.cos(d), np.sin(d)])


def crossing_profile(s):
    """(slip, d slip / ds) of the cos^2 profile along the fault, s from -A_LO to A_HI."""
    c, a = 0.5 * (CROSSING_A_HI - CROSSING_A_LO), 0.5 * (CROSSING_A_HI + CROSSING_A_LO)
    return np.cos(0.5 * np.pi * (s - c) / a) ** 2, -(np.pi / (2 * a)) * np.sin(np.pi * (s - c) / a)


def crossing_mesh(dip, h, mu2, force="trapezoid", half_hats=True):
    """Fault ("s", hats) through the interface ("h", spacing h near the fault, log-spaced to
    300); the crossing is a node of both. Trapezoids from `split_force_trapezoids`;
    half_hats=False drops the half-hats at the crossing (force tapered to zero there)."""
    C, t = crossing_frame(dip)
    A_LO, A_HI = CROSSING_A_LO, CROSSING_A_HI
    s = np.r_[np.linspace(-A_LO, 0, int(round(A_LO / h)) + 1)[:-1], np.linspace(0, A_HI, int(round(A_HI / h)) + 1)]
    P = C + s[:, None] * t
    P[s == 0] = C
    xs = graded(h, centre=CROSSING_XC)
    n_f, n_i = len(s) - 1, len(xs) - 1
    els = _els(np.r_[P[:-1, 0], xs[:-1]], np.r_[P[:-1, 1], np.zeros(n_i)],
               np.r_[P[1:, 0], xs[1:]], np.r_[P[1:, 1], np.zeros(n_i)])
    sc = 0.5 * (s[:-1] + s[1:])
    prof = crossing_profile(sc)[0]
    mu_f = np.where(sc > 0, mu2, CROSSING_MU1)
    kw = dict(bc_type=["s"] * n_f + ["h"] * n_i,
              bc_value=np.r_[np.c_[CROSSING_SLIP[0] * prof, CROSSING_SLIP[1] * prof], np.zeros((n_i, 2))],
              mu_minus=np.r_[mu_f, np.full(n_i, CROSSING_MU1)], mu_plus=np.r_[mu_f, np.full(n_i, mu2)],
              basis={"slip": "hat", "force": force})
    mesh = bemcs.Mesh(els, **kw)
    if force == "trapezoid":
        conn = mesh.split_force_trapezoids()
        if not half_hats:
            conn = conn[~((conn[:, 0] == conn[:, 1]) & (conn[:, 1] == conn[:, 2]))]
        mesh = bemcs.Mesh(els, connectivity={"force": conn}, **kw)
    return mesh


# ----------------------------------------------------------------------------
# other geometries
# ----------------------------------------------------------------------------


def fault_and_interface(n_fault=6, L=40.0, n_int=800, y_top=-0.6, y_bot=-1.8):
    """Vertical fault (in the lower material) below a straight interface along y = 0
    (elements left to right, normal +y). Returns (els, n_fault, n_int)."""
    yf = np.linspace(y_top, y_bot, n_fault + 1)
    xi = np.linspace(-L, L, n_int + 1)
    return _els(
        np.r_[np.zeros(n_fault), xi[:-1]], np.r_[yf[:-1], np.zeros(n_int)],
        np.r_[np.zeros(n_fault), xi[1:]], np.r_[yf[1:], np.zeros(n_int)],
    ), n_fault, n_int


def free_surface_fault(h=0.1, dip=45.0, L=50.0):
    """Plane strain, homogeneous: a dipping fault breaking a free surface (triple junction)."""
    d = np.deg2rad(dip)
    s = np.linspace(0.0, 1.0, int(round(1 / h)) + 1)
    P = np.c_[s * np.cos(d), -s * np.sin(d)]
    xs = graded(h, L=L, h_max=2.0)
    n_f, n_s = len(s) - 1, len(xs) - 1
    els = _els(np.r_[P[:-1, 0], xs[:-1]], np.r_[P[:-1, 1], np.zeros(n_s)],
               np.r_[P[1:, 0], xs[1:]], np.r_[P[1:, 1], np.zeros(n_s)])
    prof = np.cos(0.5 * np.pi * 0.5 * (s[:-1] + s[1:])) ** 2
    return bemcs.Mesh(els, bc_type=["s"] * n_f + ["t"] * n_s,
                      bc_value=np.r_[np.c_[prof, 0.3 * prof], np.zeros((n_s, 2))],
                      mu_minus=1.0, mu_plus=1.0, basis={"slip": "hat"})


def kinked_chain(kink_deg):
    """5 unit elements; a kink of kink_deg at the node shared by elements 1 and 2."""
    k = np.deg2rad(kink_deg)
    pts = [np.array([-2.0, 0.0]), np.array([-1.0, 0.0]), np.array([0.0, 0.0])]
    for _ in range(3):
        pts.append(pts[-1] + np.array([np.cos(k), np.sin(k)]))
    return _els([p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]],
                [p[0] for p in pts[1:]], [p[1] for p in pts[1:]])
