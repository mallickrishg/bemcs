"""Plane strain: a fault crossing a bonded bimaterial interface (tasks.md P2).

Reference: the Dundurs (1969) edge dislocation near the interface, integrated over the
fault's slip profile. A dislocation above the interface is the mirror image of one below
with the materials swapped (`_dislocation`), so a fault that runs through the interface
has an exact solution in the full space.

Measured (stress at points >= 0.15 from the interface and the fault, relative to the
largest; h = 0.1, 0.05, 0.025; fault through (0.15, 0), dips 90/60/30 degrees):
- no contrast: 2.7e-4 -> 1.3e-5 -> 6.7e-7 (all dips), the interface carries no force;
- mu2/mu1 = 4 and 1/4, trapezoids tapered to zero at the crossing (the earlier layout):
  first order and large, 5e-2 to 1e-1 at h = 0.025, growing as the dip gets gentler;
- with a free half-hat on each side of the crossing (the default `split_force_trapezoids`
  layout now): 3e-3 to 5e-3 at h = 0.025, the level of a fault near the interface without
  a crossing (test_dundurs.py). The nodal basis: 1e-3 to 2e-3.
"""

import warnings

import numpy as np
import pytest

import bemcs
import bemcs.bemAssembly as GF
from reference_solutions import edge_dislocation_bimaterial
from meshes import CROSSING_A_HI as A_HI, CROSSING_A_LO as A_LO, CROSSING_MU1 as MU1  # noqa: E402
from meshes import CROSSING_NU as NU, CROSSING_SLIP as SLIP, CROSSING_XC as XC  # noqa: E402
from meshes import crossing_frame, crossing_mesh, crossing_profile, graded  # noqa: E402

def _dislocation(X, Y, x0, y0, b, mu1, mu2):
    """Stresses of a dislocation on either side of the interface (mu1 below, mu2 above)."""
    if y0 < 0:
        return edge_dislocation_bimaterial(X, Y, x0, y0, b, mu1, mu2, NU)[0]
    # mirror y -> -y: the materials swap, and the reflection reverses the Burgers circuit
    s, _ = edge_dislocation_bimaterial(X, -Y, x0, -y0, (-b[0], b[1]), mu2, mu1, NU)
    return np.array([s[0], s[1], -s[2]])


def _exact_stress(X, Y, dip, mu2, n_quad=60):
    C, t = crossing_frame(dip)
    e = SLIP[0] * t + SLIP[1] * np.array([-t[1], t[0]])
    g, w = np.polynomial.legendre.leggauss(n_quad)
    out = 0.0
    for lo, hi in ((-A_LO, 0.0), (0.0, A_HI)):  # split at the crossing (slip density is smooth)
        for gi, wi in zip(g, w):
            s = 0.5 * (lo + hi) + 0.5 * (hi - lo) * gi
            p = C + s * t
            out = out + 0.5 * (hi - lo) * wi * _dislocation(X, Y, p[0], p[1], tuple(-crossing_profile(s)[1] * e), MU1, mu2)
    return out


def _points(dip):
    C, t = crossing_frame(dip)
    th = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    P = np.r_[C + 0.9 * np.c_[np.cos(th), np.sin(th)], C + 0.4 * np.c_[np.cos(th), np.sin(th)],
              np.c_[np.linspace(-2, 2, 9), np.full(9, 0.5)], np.c_[np.linspace(-2, 2, 9), np.full(9, -0.5)]]
    along, perp = (P - C) @ t, np.abs((P - C) @ np.array([-t[1], t[0]]))
    off_fault = (perp > 0.15) | (along < -A_LO - 0.15) | (along > A_HI + 0.15)
    keep = (np.abs(P[:, 1]) > 0.15) & off_fault
    return P[keep, 0], P[keep, 1]


def _error(dip, h, mu2, **kw):
    X, Y = _points(dip)
    ref = _exact_stress(X, Y, dip, mu2)
    mesh = crossing_mesh(dip, h, mu2, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bemcs.TrapezoidCrossingWarning)
        x = bemcs.solve(mesh, geometry="planestrain", mu=1.0, nu=NU)
    f = bemcs.evaluate_fields(mesh, x, X, Y, geometry="planestrain", mu=1.0, nu=NU,
                              mu_field=lambda a, b: np.where(b > 0, mu2, MU1))
    return np.abs(np.array([f["sxx"], f["syy"], f["sxy"]]) - ref).max() / np.abs(ref).max()


# ----------------------------------------------------------------------------
# the reference
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("b", [(1.0, 0.0), (0.0, 1.0)])
def test_mirrored_dislocation_matches_the_direct_one_without_contrast(b):
    X, Y = np.array([0.7, -0.4, 1.1]), np.array([0.9, -0.5, 2.0])
    np.testing.assert_allclose(_dislocation(X, Y, 0.2, 0.6, b, 1.0, 1.0),
                               edge_dislocation_bimaterial(X, Y - 1.2, 0.2, -0.6, b, 1.0, 1.0, NU)[0], atol=1e-14)


@pytest.mark.parametrize("y0", [-1e-4, 1e-4])
def test_mirrored_dislocation_keeps_tractions_continuous(y0):
    xi = np.linspace(-2, 2, 9)
    up = _dislocation(xi, np.full(9, 1e-9), 0.1, y0, (1.0, 0.5), MU1, 4.0)
    down = _dislocation(xi, np.full(9, -1e-9), 0.1, y0, (1.0, 0.5), MU1, 4.0)
    np.testing.assert_allclose(up[1:], down[1:], atol=1e-6)


def test_reference_constants_stay_accurate_next_to_the_interface():
    """The image constants are fitted by least squares; with column scaling the fit stays at
    round-off down to 1e-6 from the interface (it was 1e-2 at 1e-4 without it, which spoiled
    the integral across the crossing)."""
    from reference_solutions import edge_dislocation_bimaterial_constants
    for y0 in (-1e-3, -1e-4, -1e-6):
        assert edge_dislocation_bimaterial_constants(complex(0.1, y0), complex(1, 0.5), 1.0, 4.0, NU)[2] < 1e-10


# ----------------------------------------------------------------------------
# layout
# ----------------------------------------------------------------------------


def test_split_layout_has_a_half_hat_on_each_side_of_the_crossing():
    mesh = crossing_mesh(60, 0.1, 4.0)
    conn = mesh.connectivity["force"]
    half = conn[(conn[:, 0] == conn[:, 1]) & (conn[:, 1] == conn[:, 2]), 0]
    assert len(half) == 2
    ends = {(round(float(mesh.els.x1[e]), 9), round(float(mesh.els.x2[e]), 9)) for e in half}
    assert ends == {(round(XC - 0.1, 9), XC), (XC, round(XC + 0.1, 9))}
    # each half-hat is 1 at the crossing node: ramp 1 (start node) right of it, ramp 0 left of it
    sub, _, sconn = mesh.subset("force")
    T = mesh.transform("force", geometry="planestrain")
    for p, row in enumerate(sconn):
        if row[0] == row[1] == row[2]:
            e = int(row[0])
            k = 1 if np.isclose(sub.x1[e], XC) else 0
            np.testing.assert_allclose(T[2 * e + k, :, p, :], np.eye(2))
            np.testing.assert_allclose(T[2 * e + 1 - k, :, p, :], 0.0)


def test_interior_half_hat_needs_the_break_to_find_its_free_node():
    els = bemcs.initialize_els()
    els.x1, els.x2 = np.arange(6.0), np.arange(1.0, 7.0)
    els.y1 = els.y2 = np.zeros(6)
    bemcs.standardize_els_geometry(els, reorder=False)
    conn = GF.trapezoid_connectivity(els, breaks=[[3.0, 0.0]])
    assert [2, 2, 2] in conn.tolist() and [3, 3, 3] in conn.tolist()
    GF.trapezoid_coefficients(els, conn, geometry="planestrain", breaks=[[3.0, 0.0]])
    with pytest.raises(ValueError, match="chain tip or a break point"):
        GF.trapezoid_coefficients(els, conn, geometry="planestrain")


# ----------------------------------------------------------------------------
# accuracy
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("dip", [90, 60, 30])
def test_no_contrast_matches_the_dislocation_integral(dip):
    assert _error(dip, 0.05, MU1) < 5e-5  # measured 1.3e-5, 1.0e-5, 1.1e-5


@pytest.mark.parametrize("dip", [90, 30])
@pytest.mark.parametrize("mu2", [4.0, 0.25])
def test_half_hats_at_the_crossing_converge(dip, mu2):
    e1, e2 = _error(dip, 0.1, mu2), _error(dip, 0.05, mu2)
    assert e2 < 1.8e-2 and e1 / e2 > 1.8  # measured 4.8e-3 .. 1.6e-2, ratios 2.5 .. 3.6


def test_tapering_to_zero_at_the_crossing_is_much_worse():
    assert 10 * _error(60, 0.05, 4.0) < _error(60, 0.05, 4.0, half_hats=False)  # 7.5e-3 vs 1.4e-1


# ----------------------------------------------------------------------------
# the refinement warning
# ----------------------------------------------------------------------------


def _refined(dip, h, mu2, M):
    """Like `crossing_mesh`, with force elements h / M within |x - XC| < 1."""
    base = crossing_mesh(dip, h, mu2, force="nodal")
    els0, nf = base.els, len(base.elements("slip"))
    r = np.round(np.arange(0, 1.0 + 1e-12, h / M), 12)
    tail = graded(h)
    r = np.r_[r, tail[tail > 1.0 + 1e-9]]
    xs = XC + np.r_[-r[:0:-1], r]
    els = bemcs.initialize_els()
    els.x1, els.y1 = np.r_[els0.x1[:nf], xs[:-1]], np.r_[els0.y1[:nf], np.zeros(len(xs) - 1)]
    els.x2, els.y2 = np.r_[els0.x2[:nf], xs[1:]], np.r_[els0.y2[:nf], np.zeros(len(xs) - 1)]
    bemcs.standardize_els_geometry(els, reorder=False)
    n_i = len(xs) - 1
    kw = dict(bc_type=["s"] * nf + ["h"] * n_i, bc_value=np.r_[base.bc_value[:nf], np.zeros((n_i, 2))],
              mu_minus=np.r_[base.mu_minus[:nf], np.full(n_i, MU1)], mu_plus=np.r_[base.mu_plus[:nf], np.full(n_i, mu2)],
              basis={"slip": "hat", "force": "trapezoid"})
    return bemcs.Mesh(els, connectivity={"force": bemcs.Mesh(els, **kw).split_force_trapezoids()}, **kw)


def test_solve_recommends_refinement_at_a_crossing():
    with pytest.warns(bemcs.CrossingRefinementWarning, match=r"\(0.15, 0\) \[force 0.1, fault 0.1\]"):
        bemcs.solve(crossing_mesh(60, 0.1, 4.0), geometry="planestrain", mu=1.0, nu=NU)
    with warnings.catch_warnings():
        warnings.simplefilter("error", bemcs.CrossingRefinementWarning)
        bemcs.solve(crossing_mesh(60, 0.1, 1.0), geometry="planestrain", mu=1.0, nu=NU)  # no contrast
        bemcs.solve(_refined(60, 0.1, 4.0, 4), geometry="planestrain", mu=1.0, nu=NU)  # refined
