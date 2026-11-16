"""Crossing cracks without triple junctions (no remeshing at intersections).

Each crack is meshed on its own; where cracks cross, their elements simply
overlap in space. With the hat basis each crack's stress is bounded everywhere
(singularities cancel between its own elements), so crossings are harmless,
except numerically when a node of one crack comes within float64 resolution of
a collocation point (element centre) of another crack.
"""

import numpy as np
import pytest

import bemcs

SIG = np.array([[0.3, 1.0], [1.0, -0.4]])  # remote plane-strain stress
SIG_AP = np.array([0.8, -0.5])  # remote antiplane stress (sxz, syz)


def _cracks(segments, n_el):
    x1, y1, x2, y2 = [], [], [], []
    for a, b in segments:
        pts = np.linspace(a, b, n_el + 1)
        x1 += list(pts[:-1, 0]); y1 += list(pts[:-1, 1])
        x2 += list(pts[1:, 0]); y2 += list(pts[1:, 1])
    els = bemcs.initialize_els()
    els.x1, els.y1, els.x2, els.y2 = (np.array(v) for v in (x1, y1, x2, y2))
    bemcs.standardize_els_geometry(els, reorder=False)
    return els


def _mesh(els, geometry, basis):
    """Traction-free crack faces under a remote stress: BEM traction = -sigma_inf . n."""
    n = np.c_[els.x_normals, els.y_normals]
    t = -n @ SIG if geometry == "planestrain" else -(n @ SIG_AP)
    return bemcs.Mesh(els, bc_type=["t"] * len(els.x1), bc_value=t, mu_minus=1.0, mu_plus=1.0,
                      basis={"slip": basis})


def _intersect(a, b, c, d):
    def cross(u, v):
        return u[0] * v[1] - u[1] * v[0]
    r, s = b - a, d - c
    den = cross(r, s)
    if den == 0:
        return False
    t, u = cross(c - a, s) / den, cross(c - a, r) / den
    return 0 < t < 1 and 0 < u < 1


def _random_segments(seed, n=5):
    rng = np.random.default_rng(seed)
    segs = []
    for _ in range(n):
        c = rng.uniform(-1, 1, 2)
        half = 0.5 * rng.uniform(1.5, 2.5) * np.array([np.cos(t := rng.uniform(0, np.pi)), np.sin(t)])
        segs.append((c - half, c + half))
    return segs


def _crossing_pairs(segs):
    n = len(segs)
    return [(i, j) for i in range(n) for j in range(i + 1, n) if _intersect(*segs[i], *segs[j])]


def _is_forest(pairs, n):
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[ra] = rb
    return True


def _solve_both(segs, geometry):
    els = _cracks(segs, n_el=6)
    _, _, triple = bemcs.label_nodes(els)
    assert not triple  # crossings are not meshed: no junctions
    kw = dict(geometry=geometry, mu=1.0, nu=0.25, proximity_tol=None)  # checked separately
    hat, general = _mesh(els, geometry, "hat"), _mesh(els, geometry, "quadratic")
    xh, (Ah, bh, _) = bemcs.solve(hat, return_system=True, **kw)
    xg = bemcs.solve(general, **kw)
    np.testing.assert_allclose(Ah @ xh, bh, atol=1e-10 * np.abs(bh).max())
    ch = hat.element_coefficients(xh, "slip", geometry=geometry)
    cg = general.element_coefficients(xg, "slip", geometry=geometry)
    np.testing.assert_allclose(ch, cg, atol=1e-8 * np.abs(cg).max())  # same answer, both forms
    return hat, xh, ch, np.linalg.cond(Ah)


@pytest.mark.parametrize("seed", [7, 37])
@pytest.mark.parametrize("geometry", ["antiplane", "planestrain"])
def test_random_crossing_cracks_are_stable(seed, geometry):
    """4 crossings, no enclosed blocks (the crossing graph is a tree): well conditioned,
    slip of the order of the remote load, bounded stresses right at the crossings."""
    segs = _random_segments(seed)
    pairs = _crossing_pairs(segs)
    assert len(pairs) == 4 and _is_forest(pairs, 5)
    hat, xh, ch, cond = _solve_both(segs, geometry)
    assert cond < 1e3  # measured 20 to 690 for these tangles
    assert np.isfinite(ch).all() and np.abs(ch).max() < 5
    # stresses stay bounded within 1e-6 of each crossing point
    pts = []
    for i, j in pairs:
        (a, b), (c, d) = segs[i], segs[j]
        r, s_ = b - a, d - c
        t = ((c - a)[0] * s_[1] - (c - a)[1] * s_[0]) / (r[0] * s_[1] - r[1] * s_[0])
        pts.append(a + t * r + 1e-6 * np.array([0.6, 0.8]))
    pts = np.array(pts)
    f = bemcs.evaluate_fields(hat, xh, pts[:, 0], pts[:, 1], geometry=geometry, mu=1.0, nu=0.25)
    for v in f.values():
        assert np.isfinite(v).all() and np.abs(v).max() < 50


def test_crossing_cracks_that_enclose_blocks_slip_a_lot_but_consistently():
    """Three mutually crossing cracks cut out a triangle bounded only by traction-free
    faces: a nearly detached block (close to a rigid-body mode). The slip is large
    (~20x the load) - a property of the geometry, not of the basis: both forms agree."""
    segs = _random_segments(3)  # 7 crossings; triangles e.g. cracks (1, 3, 4)
    assert not _is_forest(_crossing_pairs(segs), 5)
    _, _, ch, cond = _solve_both(segs, "planestrain")
    assert np.abs(ch).max() > 10
    assert cond < 1e4


# ----------------------------------------------------------------------------
# a node of crack B approaching a collocation point (element centre) of crack A
# ----------------------------------------------------------------------------

P = np.array([0.125, 0.0])  # centre of element 4 of crack A (8 elements on [-1, 1])
DIR_B = np.array([np.cos(1.1), np.sin(1.1)])
APPROACH = np.array([np.cos(0.3), np.sin(0.3)])


def _node_near_collocation(d, dr, approach=APPROACH):
    node = P + d * approach  # node 4 of crack B
    els = _cracks([(np.array([-1.0, 0.0]), np.array([1.0, 0.0])),
                   (node - 4 * 0.22 * DIR_B, node + 4 * 0.22 * DIR_B)], n_el=8)
    mesh = _mesh(els, "planestrain", "hat")
    x = bemcs.solve(mesh, geometry="planestrain", mu=1.0, nu=0.25, dr=dr, proximity_tol=None)
    return mesh.element_coefficients(x, "slip", geometry="planestrain")


def test_node_approaching_a_collocation_point_converges_then_hits_the_float64_floor():
    """Collocation exactly at element centres (dr = 0). Analytically the solution has a
    finite limit (B's hat-basis stress is bounded at its nodes), and it converges as
    d -> 0, until ~eps L / d round-off between B's element kernels (each ~1/d near
    the node) dominates: lost by d ~ 1e-13 L, blown up by 1e-16 L, and the kernels
    are not finite when the node sits exactly on the collocation point."""
    ds = [1e-4, 1e-6, 1e-8, 1e-10]
    c = [_node_near_collocation(d, 0.0) for d in ds]
    changes = [np.abs(b - a).max() / np.abs(a).max() for a, b in zip(c[:-1], c[1:])]
    assert changes[0] < 1e-2 and all(b < a for a, b in zip(changes[:-1], changes[1:]))
    assert changes[-1] < 1e-5  # converged to ~1e-6 by d = 1e-10
    blown = _node_near_collocation(1e-16, 0.0)
    assert np.abs(blown - c[-1]).max() > 1.0
    with pytest.raises(FloatingPointError, match="collocation point"):
        _node_near_collocation(0.0, 0.0)


def test_default_collocation_offset_keeps_the_solve_stable():
    """With the default offset (1e-9 element lengths along A's normal) the sampling point
    never gets closer than ~2e-10 to B's node: the solve stays finite for all d,
    including a node exactly at the element centre.

    Below d ~ dr L the offset, not d, sets where the sampling point is: here it ends up
    on the other side of crack B's line. The in-plane stress jumps across a slipping
    crack (in proportion to the slip gradient), so the solution switches to the limit
    reached from that side: it matches the dr = 0 solution with B's node approaching
    from the opposite direction.
    """
    other_side = _node_near_collocation(1e-8, 0.0, approach=-APPROACH)
    same_side = _node_near_collocation(1e-8, 0.0)
    assert np.abs(other_side - same_side).max() > 0.1 * np.abs(same_side).max()  # a real jump
    for d in (1e-12, 1e-15, 0.0):
        c = _node_near_collocation(d, 1e-9)
        assert np.isfinite(c).all()
        np.testing.assert_allclose(c, other_side, atol=1e-5 * np.abs(other_side).max())


# ----------------------------------------------------------------------------
# crack B passing exactly through a collocation point of crack A: only the mesh
# of B changes (its tips are fixed, its node next to P slides along B)
# ----------------------------------------------------------------------------


def _crack_through_collocation(n, s, dr=1e-9):
    """n elements per crack; A on [-1, 1]; P = centre of A's element n/2; B (length 1.76)
    passes through P with fixed tips; node n/2 of B sits at P + s L DIR_B (L = A's element
    length). Returns A's shear slip at its element centres and the condition number."""
    L = 2.0 / n
    p = np.array([-1 + (n // 2 + 0.5) * L, 0.0])
    b0, b1, node = p - 0.88 * DIR_B, p + 0.88 * DIR_B, p + s * L * DIR_B
    h = n // 2
    b = np.r_[np.linspace(b0, node, h + 1)[:-1], np.linspace(node, b1, h + 1)]
    a = np.linspace([-1.0, 0.0], [1.0, 0.0], n + 1)
    els = bemcs.initialize_els()
    els.x1, els.y1 = np.r_[a[:-1, 0], b[:-1, 0]], np.r_[a[:-1, 1], b[:-1, 1]]
    els.x2, els.y2 = np.r_[a[1:, 0], b[1:, 0]], np.r_[a[1:, 1], b[1:, 1]]
    bemcs.standardize_els_geometry(els, reorder=False)
    mesh = _mesh(els, "planestrain", "hat")
    x, (A, _, _) = bemcs.solve(mesh, geometry="planestrain", mu=1.0, nu=0.25, dr=dr,
                               return_system=True, proximity_tol=None)
    c = mesh.element_coefficients(x, "slip", geometry="planestrain").reshape(2 * n, 3, 2)
    return c[:n, 1, 0], np.linalg.cond(A)


def _rel(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


def test_crack_through_a_collocation_point_converges_and_stays_finite():
    """B always passes through P, so the offset sampling point stays on one side of B:
    no jump (unlike the approach in the tests above), finite down to node-on-P."""
    c = [_crack_through_collocation(8, s)[0] for s in (1e-3, 1e-6, 1e-12, 0.0)]
    assert all(np.isfinite(v).all() for v in c)
    assert _rel(c[1], c[0]) < 1e-3  # measured 2.4e-4
    assert _rel(c[3], c[1]) < 1e-5  # measured ~6e-7


def test_remeshing_a_crack_through_a_collocation_point_is_discretization_error():
    """The geometry is fixed, so the exact answer does not depend on where B's node sits.
    The discrete one does (the collocation point is on B, where the in-plane stress
    takes two values), but the dependence shrinks with refinement."""
    change = {n: _rel(_crack_through_collocation(n, 0.2)[0], _crack_through_collocation(n, 0.0)[0])
              for n in (8, 64)}
    assert change[8] > 1e-2  # measured 3.1e-2
    assert change[64] < 0.4 * change[8]  # measured 7.8e-3


def test_collocation_point_on_another_crack_degrades_conditioning():
    """At s = 0.44 n / (n - 1) the centre of B's element next to the sliding node lands
    on A (at P itself): two traction conditions sampled at the crossing point."""
    _, cond_ok = _crack_through_collocation(8, 0.2)
    _, cond_bad = _crack_through_collocation(8, 0.44 * 8 / 7)
    assert cond_ok < 1e3  # measured 50
    assert cond_bad > 1e6 * cond_ok  # measured 4e10


# ----------------------------------------------------------------------------
# the proximity check behind CollocationProximityWarning
# ----------------------------------------------------------------------------


def test_proximity_check_flags_close_collocation_points_in_random_tangles():
    """Seed 7: one crossing lies 0.078 element lengths from a collocation point of
    another crack; seed 37: none closer than 0.1. solve warns accordingly."""
    m7 = _mesh(_cracks(_random_segments(7), n_el=6), "planestrain", "hat")
    close = m7.close_collocation_points()
    assert close["element"].tolist() == [26] and close["other"].tolist() == [17]
    assert 0.07 < close["distance"][0] < 0.09
    with pytest.warns(bemcs.CollocationProximityWarning, match=r"26 -> 17"):
        bemcs.solve(m7, geometry="planestrain", mu=1.0, nu=0.25)
    m37 = _mesh(_cracks(_random_segments(37), n_el=6), "planestrain", "hat")
    assert len(m37.close_collocation_points()["element"]) == 0
    assert len(m7.close_collocation_points(tol=0.05)["element"]) == 0


def test_proximity_check_flags_both_measured_trouble_spots():
    """A crack through a collocation point (distance 0) and a node of B next to A's
    collocation point are both reported; proximity_tol=None silences solve."""
    import warnings

    # B through P, its node 0.2 L away along B: A's element 4 is on B
    n, L = 8, 0.25
    p = np.array([0.125, 0.0])
    b = np.r_[np.linspace(p - 0.88 * DIR_B, p + 0.2 * L * DIR_B, 5)[:-1],
              np.linspace(p + 0.2 * L * DIR_B, p + 0.88 * DIR_B, 5)]
    a = np.linspace([-1.0, 0.0], [1.0, 0.0], n + 1)
    els = bemcs.initialize_els()
    els.x1, els.y1 = np.r_[a[:-1, 0], b[:-1, 0]], np.r_[a[:-1, 1], b[:-1, 1]]
    els.x2, els.y2 = np.r_[a[1:, 0], b[1:, 0]], np.r_[a[1:, 1], b[1:, 1]]
    bemcs.standardize_els_geometry(els, reorder=False)
    mesh = _mesh(els, "planestrain", "hat")
    close = mesh.close_collocation_points()
    pairs = set(zip(close["element"].tolist(), close["other"].tolist()))
    assert (4, 11) in pairs and close["distance"].min() < 1e-12
    with pytest.warns(bemcs.CollocationProximityWarning):
        bemcs.solve(mesh, geometry="planestrain", mu=1.0, nu=0.25)
    with warnings.catch_warnings():
        warnings.simplefilter("error", bemcs.CollocationProximityWarning)
        bemcs.solve(mesh, geometry="planestrain", mu=1.0, nu=0.25, proximity_tol=None)


def test_proximity_check_ignores_neighbours_kinks_and_junctions():
    """Elements sharing a node are never reported: a sharply kinked fault, a triple
    junction, and a fault ending on a free surface at a shared node."""
    def els_from(segments):
        e = bemcs.initialize_els()
        e.x1, e.y1 = np.array([s[0][0] for s in segments]), np.array([s[0][1] for s in segments])
        e.x2, e.y2 = np.array([s[1][0] for s in segments]), np.array([s[1][1] for s in segments])
        bemcs.standardize_els_geometry(e, reorder=False)
        return e

    o = np.array([0.0, 0.0])
    kink = [(np.array([-1.0, 0.0]), o), (o, np.array([-0.98, 0.2]))]  # ~170 degree kink
    junction = [(np.array([-1.0, 0.0]), o), (o, np.array([1.0, 0.0])), (o, np.array([0.99, 0.12]))]
    surface = [(np.array([-1.0, 0.0]), o), (o, np.array([1.0, 0.0])), (np.array([0.05, -1.0]), o)]
    for segs in (kink, junction, surface):
        close = bemcs.close_collocation_points(els_from(segs))
        assert len(close["element"]) == 0
    # but a fault tip in the middle of a surface element is reported
    tip_mid = [(np.array([-1.0, 0.0]), np.array([1.0, 0.0])), (np.array([0.0, -1.0]), np.array([0.0, -1e-3]))]
    close = bemcs.close_collocation_points(els_from(tip_mid))
    assert close["element"].tolist() == [0] and close["other"].tolist() == [1]
