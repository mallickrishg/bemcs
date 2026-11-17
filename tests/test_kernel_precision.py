"""Floating-point precision of the plane-strain kernels near element endpoints
(tasks.md issue 16).

Each formula (`_quadratic_slip_f`, `_linear_force_g`) is evaluated twice: in
float64, and with 50-digit mpmath numbers by executing the *same source* with
an mpmath stand-in for numpy. The difference is pure rounding error.

The assembly tests sum the stress of a hat basis over its elements while
approaching a node shared by two elements. Each element's contribution grows
like 1/d there while the sum stays bounded on a straight fault. This is where
float64 truncation in the log/atan terms used to blow up and inflate
condition numbers.
"""


import mpmath as mp
import numpy as np
import pytest

import bemcs
import bemcs.bemAssembly as GF
from meshes import kinked_chain  # noqa: E402
from reference_kernels import mp_version  # noqa: E402

mp.mp.dps = 50
A, NU, MU = 0.7, 0.27, 1.3


_SLIP_MP = mp_version(GF._quadratic_slip_f)
_FORCE_MP = mp_version(GF._linear_force_g)


def _nonzero(v):
    v = mp.mpf(v)
    return v if v != 0 else mp.mpf("1e-45")  # exactly above an endpoint: avoid 0/0


def _slip_rel_error(xi, y, s):
    lo = np.zeros((7, 3))
    with np.errstate(divide="ignore"):
        GF._quadratic_slip_f.py_func(np.float64(xi), np.float64(y), A, NU, np.float64(s), lo)
    hi = np.empty((7, 3), dtype=object)
    _SLIP_MP(_nonzero(xi), mp.mpf(y), mp.mpf(A), mp.mpf(NU), s, hi)
    hi = np.array(hi.tolist(), dtype=float)
    return np.abs(lo - hi).max() / np.abs(hi).max()


def _force_rel_error(x, y):
    lo = np.zeros((2, 2, 5))
    with np.errstate(divide="ignore"):
        GF._linear_force_g.py_func(np.float64(x), np.float64(y), A, MU, NU, lo)
    hi = np.empty((2, 2, 5), dtype=object)
    _FORCE_MP(mp.mpf(x), mp.mpf(y), mp.mpf(A), mp.mpf(MU), mp.mpf(NU), hi)
    hi = np.array(hi.tolist(), dtype=float)
    return np.abs(lo - hi).max() / np.abs(hi).max()


ANGLES = [0.0, 0.7, np.pi / 2, 2.5, np.pi - 1e-3]  # around the endpoint, incl. on the line
DISTANCES = [1e-1, 1e-2, 1e-4, 1e-6, 1e-8]  # in units of the half-length a


# ----------------------------------------------------------------------------
# one element
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("d", DISTANCES)
@pytest.mark.parametrize("angle", ANGLES)
@pytest.mark.parametrize("s", [1.0, -1.0])
def test_slip_formula_precision_near_endpoint(d, angle, s):
    # angle measured from the outward direction of the endpoint x = s a
    xi = s * d * A * np.cos(angle)
    y = d * A * np.sin(angle) + (1e-3 * d * A if angle == 0.0 else 0.0)
    assert _slip_rel_error(xi, y, s) < 1e-12


@pytest.mark.parametrize("r", [1.5, 10.0, 100.0])
def test_slip_formula_precision_mid_and_far_field(r):
    # far-field cancellation (polynomial x log) grows with r: ~1e-10 at 100 a
    for t in np.linspace(0.1, 3.0, 7):
        x, y = r * A * np.cos(t), r * A * np.sin(t)
        s = 1.0 if x >= 0 else -1.0
        assert _slip_rel_error(x - s * A, y, s) < 1e-9


@pytest.mark.parametrize("d, tol", [(1e-2, 1e-13), (1e-4, 1e-12), (1e-6, 1e-11), (1e-8, 1e-9)])
@pytest.mark.parametrize("angle", ANGLES[1:4])
def test_force_formula_precision_near_endpoint(d, tol, angle):
    x = A + d * A * np.cos(angle)
    y = d * A * np.sin(angle)
    assert _force_rel_error(x, y) < tol


# ----------------------------------------------------------------------------
# sum over elements: hat basis near a shared node
# ----------------------------------------------------------------------------


def _hat_stress_mp(els, T, xo, yo):
    """sxx, syy, sxy of sum_j T[j, l] * kernel_j at one point, in 50 digits."""
    out = [mp.mpf(0)] * 3
    for e in range(len(els.x1)):
        if not np.any(T[3 * e : 3 * e + 3]):
            continue
        R = np.array([[mp.mpf(v) for v in row] for row in els.rot_mats[e]], dtype=object)
        xc = R[0, 0] * (mp.mpf(xo) - mp.mpf(els.x_centers[e])) + R[1, 0] * (
            mp.mpf(yo) - mp.mpf(els.y_centers[e])
        )
        s = 1 if xc >= 0 else -1
        ex, ey = (els.x2[e], els.y2[e]) if s > 0 else (els.x1[e], els.y1[e])
        dx, dy = mp.mpf(xo) - mp.mpf(ex), mp.mpf(yo) - mp.mpf(ey)
        xi = R[0, 0] * dx + R[1, 0] * dy
        yl = R[0, 1] * dx + R[1, 1] * dy
        f = np.empty((7, 3), dtype=object)
        _SLIP_MP(_nonzero(xi), yl, mp.mpf(els.half_lengths[e]), mp.mpf(NU), s, f)
        for node in range(3):
            for l in range(2):
                c = T[3 * e + node, l]
                if c == 0:
                    continue
                loc = GF._slip_local.py_func(f, node, yl, mp.mpf(MU), mp.mpf(NU), 1 - l, l)
                g = GF._to_global.py_func(R, *loc)
                out = [o + mp.mpf(c) * gi for o, gi in zip(out, g[2:])]
    return np.array([float(v) for v in out])


@pytest.mark.parametrize("kink_deg", [0.0, 30.0])
@pytest.mark.parametrize("d", [1e-2, 1e-4, 1e-6, 1e-8])
def test_hat_stress_near_shared_node_has_no_float64_blowup(kink_deg, d):
    els = kinked_chain(kink_deg)
    T = GF.hat_coefficients(els, np.array([[1, 2, 3]]), geometry="planestrain")[:, :, 0, 0]
    node = np.array([els.x2[1], els.y2[1]])
    n = np.array([els.x_normals[1] + els.x_normals[2], els.y_normals[1] + els.y_normals[2]])
    p = node + d * n / np.linalg.norm(n)
    ref = _hat_stress_mp(els, T, *p)
    _, S, _ = bemcs.get_slip_kernels(
        np.array([p[0]]), np.array([p[1]]), els, geometry="planestrain", mu=MU, nu=NU
    )
    got = np.einsum("oscl,sl->oc", S, T)[0]
    # Float64 floor: each element contributes ~1/d near the node and the sum is
    # O(1), so ~eps/d survives the cancellation between elements (1e-8 at
    # d = 1e-8). Before the endpoint rewrite this error was ~1e6 at d = 1e-8.
    assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-12 + 2e-15 / d
