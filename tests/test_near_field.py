"""Near-field behaviour of the singularity-free patch bases: hat (quadratic slip)
and trapezoid (linear force), in plane strain and antiplane.

Each element kernel is singular at its end points. In a hat or trapezoid patch
the singular parts of neighbouring elements cancel, so on a co-planar mesh the
stress stays bounded up to every node of the patch. These tests check, while
approaching each node of the patch support down to 1e-8 element lengths:

1. the exact stress (the same formulas evaluated with 50-digit mpmath) is
   bounded and converges on a straight mesh;
2. the float64 kernels reproduce it to the float64 floor ~1e-16 L/d. This
   requires the endpoint-relative evaluation of the log/arctan terms; the
   original closed forms were O(1) to 1e6 wrong here;
3. at a kink the stress is log-singular (geometric, discrete curvature), not a
   power law, and float64 still resolves it;
4. controls: a single quadratic node basis (no patch) is 1/d-singular and a
   single linear force ramp is log-singular, so the tests can detect a
   singularity.
"""

import numpy as np
import mpmath as mp
import pytest

import bemcs
import bemcs.bemAssembly as GF
from meshes import kinked_chain
from reference_kernels import mp_version

mp.mp.dps = 50
MU, NU = 1.3, 0.27
CONN = np.array([[1, 2, 3]])  # one patch on elements 1, 2, 3 of the 5-element chain
DISTANCES = [1e-2, 1e-4, 1e-6, 1e-8]  # element length is 1
ANGLES = [60.0, 135.0, -100.0, 179.9]  # from the element direction; last: grazing the line
# (179.9 deg keeps the point >= 1.7e-11 off the line at d = 1e-8, i.e. outside the
# on-element tolerance, where the kernels return the two-sided average instead)
NODES = ["start of el1", "el1/el2", "el2/el3", "end of el3"]  # the kink (if any) is at el1/el2
GEOMETRIES = ["planestrain", "antiplane"]
KINDS = ["slip", "force"]

_SLIP_PS = mp_version(GF._quadratic_slip_f)
_FORCE_PS = mp_version(GF._linear_force_g)
_SLIP_AP = mp_version(GF._antiplane_slip_point)
_FORCE_AP = mp_version(GF._antiplane_force_point)
_MIRROR = [2, 1, 0, 5, 4, 3, 8, 7, 6]
_MIRROR_SIGN = [1, 1, 1, -1, -1, -1, 1, 1, 1]


def _m(v):
    return mp.mpf(np.asarray(v, dtype=float).item())


def _nonzero(v):
    return v if v != 0 else mp.mpf("1e-45")


# ----------------------------------------------------------------------------
# 50-digit reference: stress of a basis T (element dof -> one patch direction)
# ----------------------------------------------------------------------------


def _element_stress_mp(geometry, kind, els, e, xo, yo):
    """Global stress of every element dof of element e at (xo, yo), 50 digits.

    Returns {(node, l): (s1, s2[, s3])} with l the source direction (t, n) in
    plane strain ([sxx, syy, sxy]) and l = 0 in antiplane ([sxz, syz]).
    """
    R = np.array([[_m(v) for v in row] for row in els.rot_mats[e]], dtype=object)
    w = _m(els.half_lengths[e])
    X, Y = _m(xo), _m(yo)
    out = {}
    if kind == "slip":
        xc = R[0, 0] * (X - _m(els.x_centers[e])) + R[1, 0] * (Y - _m(els.y_centers[e]))
        s = 1 if xc >= 0 else -1
        ex, ey = (els.x2[e], els.y2[e]) if s > 0 else (els.x1[e], els.y1[e])
        dx, dy = X - _m(ex), Y - _m(ey)
        xi = _nonzero(R[0, 0] * dx + R[1, 0] * dy)
        y = R[0, 1] * dx + R[1, 1] * dy
        if geometry == "planestrain":
            f = np.empty((7, 3), dtype=object)
            _SLIP_PS(xi, y, w, _m(NU), s, f)
            for node in range(3):
                for l in range(2):
                    loc = GF._slip_local.py_func(f, node, y, _m(MU), _m(NU), 1 - l, l)
                    out[(node, l)] = GF._to_global.py_func(R, *loc)[2:]
        else:
            vv = _SLIP_AP(s * xi, y, w)
            v = vv if s > 0 else [_MIRROR_SIGN[k] * vv[_MIRROR[k]] for k in range(9)]
            for node in range(3):
                gx, gy = _m(MU) * v[3 + node], _m(MU) * v[6 + node]
                out[(node, 0)] = (R[0, 0] * gx + R[0, 1] * gy, R[1, 0] * gx + R[1, 1] * gy)
        return out
    dx, dy = X - _m(els.x_centers[e]), Y - _m(els.y_centers[e])
    x = R[0, 0] * dx + R[1, 0] * dy
    y = R[0, 1] * dx + R[1, 1] * dy
    if geometry == "planestrain":
        g = np.empty((2, 2, 5), dtype=object)
        _FORCE_PS(x, y, w, _m(MU), _m(NU), g)
        for r in range(2):
            for l in range(2):
                out[(r, l)] = GF._to_global.py_func(R, *g[r, l])[2:]
    else:
        v = _FORCE_AP(x, y, w)
        for r in range(2):
            out[(r, 0)] = (
                R[0, 0] * v[2 + r] + R[0, 1] * v[4 + r],
                R[1, 0] * v[2 + r] + R[1, 1] * v[4 + r],
            )
    return out


def _basis_stress_mp(geometry, kind, els, T, xo, yo):
    """Stress of sum_j T[j, l] * kernel_j at (xo, yo); T is (n_dof, n_dir)."""
    n_nodes = 3 if kind == "slip" else 2
    total = None
    for e in range(len(els.x1)):
        if not np.any(T[n_nodes * e : n_nodes * (e + 1)]):
            continue
        for (node, l), sig in _element_stress_mp(geometry, kind, els, e, xo, yo).items():
            c = T[n_nodes * e + node, l]
            if c == 0:
                continue
            contrib = [_m(c) * v for v in sig]
            total = contrib if total is None else [a + b for a, b in zip(total, contrib)]
    return np.array([float(v) for v in total])


# ----------------------------------------------------------------------------
# geometry of the approach
# ----------------------------------------------------------------------------


def _node(els, name):
    """Node coordinates and the direction of the element leaving/entering it."""
    e, end = {"start of el1": (1, 0), "el1/el2": (1, 1), "el2/el3": (2, 1), "end of el3": (3, 1)}[name]
    p = np.array([els.x2[e], els.y2[e]]) if end else np.array([els.x1[e], els.y1[e]])
    return p, np.arctan2(els.y_shears[e], els.x_shears[e])


def _points(els, name):
    p, theta = _node(els, name)
    return [
        (angle, d, p + d * np.array([np.cos(theta + np.deg2rad(angle)), np.sin(theta + np.deg2rad(angle))]))
        for angle in ANGLES
        for d in DISTANCES
    ]


def _patch_T(geometry, kind, els):
    basis = "hat" if kind == "slip" else "trapezoid"
    T = GF.basis_transform(els, kind=kind, basis=basis, geometry=geometry, connectivity=CONN)
    return T[:, None] if geometry == "antiplane" else T[:, :, 0, 0]  # patch 0, direction t


def _float_stress(geometry, kind, els, x, y):
    fn = bemcs.get_slip_kernels if kind == "slip" else bemcs.get_force_kernels
    basis = "hat" if kind == "slip" else "trapezoid"
    _, S, _ = fn(
        np.array([x]), np.array([y]), els, geometry=geometry, mu=MU, nu=NU,
        basis=basis, connectivity=CONN,
    )
    return S[0, 0] if geometry == "antiplane" else S[0, 0, :, 0]


# ----------------------------------------------------------------------------
# exact and float64 values, computed once per (geometry, kind, kink)
# ----------------------------------------------------------------------------

_CACHE = {}


def _sweep(geometry, kind, kink):
    key = (geometry, kind, kink)
    if key not in _CACHE:
        els = kinked_chain(kink)
        T = _patch_T(geometry, kind, els)
        rows = []
        for name in NODES:
            for angle, d, p in _points(els, name):
                ref = _basis_stress_mp(geometry, kind, els, T, *p)
                got = _float_stress(geometry, kind, els, *p)
                rows.append((name, angle, d, ref, got))
        _CACHE[key] = rows
    return _CACHE[key]


def _by_node_angle(rows):
    out = {}
    for name, angle, d, ref, got in rows:
        out.setdefault((name, angle), []).append((d, ref, got))
    return out


# ----------------------------------------------------------------------------
# 1. bounded on a straight (co-planar) mesh
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_patch_stress_is_bounded_near_every_support_node_of_a_straight_fault(geometry, kind):
    for (name, angle), series in _by_node_angle(_sweep(geometry, kind, 0.0)).items():
        mags = [np.abs(ref).max() for _, ref, _ in series]
        refs = [ref for _, ref, _ in series]
        # converging: the change per two decades is small and shrinking (a bounded
        # field approaches its node value like d or d log d; a singular one grows)
        changes = [np.abs(b - a).max() / mags[-1] for a, b in zip(refs[1:], refs[2:])]
        assert changes[-1] < 1e-3 and changes[-1] < changes[0], (
            f"{name} at {angle} deg: {mags} (changes {changes})"
        )
        assert max(mags) < 1.2 * max(mags[0], 1.0), f"{name} at {angle} deg grows: {mags}"


# ----------------------------------------------------------------------------
# 2. float64 reproduces the exact values down to 1e-8 L (straight and kinked)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("kink", [0.0, 30.0])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_patch_stress_float64_matches_exact_near_nodes(geometry, kind, kink):
    worst = []
    for name, angle, d, ref, got in _sweep(geometry, kind, kink):
        scale = np.abs(ref).max()
        err = np.abs(got - ref).max() / scale
        tol = 1e-12 + 2e-15 / d  # float64 floor: ~eps L / d survives the cancellation
        worst.append((err / tol, name, angle, d, err))
    ratio, name, angle, d, err = max(worst)
    assert ratio < 1, f"{name}, {angle} deg, d={d:g}: relative error {err:.1e}"


# ----------------------------------------------------------------------------
# 3. at a kink: log singularity (geometric), not a power law
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_kink_node_is_log_singular_and_other_nodes_stay_bounded(geometry, kind):
    series = _by_node_angle(_sweep(geometry, kind, 30.0))
    for (name, angle), s in series.items():
        mags = np.array([np.abs(ref).max() for _, ref, _ in s])  # d = 1e-2 ... 1e-8
        if name == "el1/el2" and angle != ANGLES[-1]:
            # logarithmic: S(d) ~ A log d + B, so the stress vector changes by the same
            # (non-zero) amount every two decades; a power law would multiply instead
            refs = [ref for _, ref, _ in s]
            step1, step2 = refs[2] - refs[1], refs[3] - refs[2]  # 1e-4 -> 1e-6 -> 1e-8
            assert np.abs(step1).max() > 0.05 * mags[1], f"{name}, {angle} deg: no growth {mags}"
            assert np.abs(step2 - step1).max() < 0.05 * np.abs(step1).max(), (
                f"{name}, {angle} deg: increments {step1} vs {step2}"
            )
        elif name != "el1/el2":
            change = np.abs(s[-1][1] - s[-2][1]).max() / mags[-1]
            assert change < 1e-3, f"{name}, {angle} deg: {mags}"


# ----------------------------------------------------------------------------
# 4. controls: without a patch the element kernels are singular at the node
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_control_single_quadratic_node_is_one_over_d(geometry):
    els = kinked_chain(0.0)
    T = np.zeros((15, 1 if geometry == "antiplane" else 2))
    T[3 * 2 + 2, 0] = 1.0  # end node of element 2, alone
    p, theta = _node(els, "el2/el3")
    mags = []
    for d in (1e-4, 1e-6):
        q = p + d * np.array([np.cos(theta + 1.0), np.sin(theta + 1.0)])
        mags.append(np.abs(_basis_stress_mp(geometry, "slip", els, T, *q)).max())
    assert 90 < mags[1] / mags[0] < 110  # 100x per two decades: 1/d


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_control_single_force_ramp_is_log_singular(geometry):
    els = kinked_chain(0.0)
    T = np.zeros((10, 1 if geometry == "antiplane" else 2))
    T[2 * 2 + 0, 0] = 1.0  # ramp 0 of element 2: force 1 at its end node, 0 beyond
    p, theta = _node(els, "el2/el3")
    mags = []
    for d in (1e-2, 1e-4, 1e-6, 1e-8):
        q = p + d * np.array([np.cos(theta + 1.0), np.sin(theta + 1.0)])
        mags.append(np.abs(_basis_stress_mp(geometry, "force", els, T, *q)).max())
    steps = np.diff(mags)
    assert np.all(steps > 0.1) and steps.max() / steps.min() < 1.1, mags
