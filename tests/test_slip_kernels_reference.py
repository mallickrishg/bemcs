"""The quadratic slip kernels against an independent reference: quadrature of the
point-dislocation solution (tasks.md T1, "quadratic slip formula itself").

Reference (tests/reference_kernels.py): textbook edge (plane strain) and screw
(antiplane) dislocations. A quadratic displacement jump s(xi) on an element equals
point dislocations s(a) at +a and -s(-a) at -a plus a continuous density -s'(xi),
integrated here by composite Gauss-Legendre. None of it uses package code.

The reference formulas are first checked against elasticity itself (Hooke's law
from the displacement gradient, equilibrium, the jump across the cut), so a typo
in a textbook formula cannot cancel against the same typo in the package.

Sign convention fixed here: slip = u(+n side) - u(-n side), with n the element
normal (the tangent rotated by +90 degrees).
"""

import numpy as np
import pytest

import bemcs
import bemcs.bemAssembly as GF
from reference_kernels import (
    edge_dislocation_planestrain,
    quadratic_slip_by_dislocations,
    screw_dislocation_antiplane,
)

MU, NU = 1.3, 0.27
LAM = 2 * MU * NU / (1 - 2 * NU)
H = 1e-6  # finite-difference step


# ----------------------------------------------------------------------------
# the reference itself: elasticity, equilibrium, jump
# ----------------------------------------------------------------------------


def _random_points(n=30, seed=0):
    x, y = np.random.default_rng(seed).uniform(-2, 2, (2, n))
    return x, y


def _derivatives(f, x, y):
    return (f(x + H, y) - f(x - H, y)) / (2 * H), (f(x, y + H) - f(x, y - H)) / (2 * H)


@pytest.mark.parametrize("burgers", [(1.0, 0.0), (0.0, 1.0), (0.6, -0.8)])
def test_edge_dislocation_reference_is_an_elastic_field(burgers):
    def f(x, y):
        return edge_dislocation_planestrain(x, y, *burgers, MU, NU)

    x, y = _random_points()
    dx, dy = _derivatives(f, x, y)
    exx, eyy, exy = dx[0], dy[1], 0.5 * (dy[0] + dx[1])
    s = f(x, y)
    scale = np.abs(s[2:]).max()
    hooke = np.array([LAM * (exx + eyy) + 2 * MU * exx, LAM * (exx + eyy) + 2 * MU * eyy, 2 * MU * exy])
    np.testing.assert_allclose(hooke, s[2:], atol=1e-8 * scale)
    np.testing.assert_allclose(dx[2] + dy[4], 0, atol=1e-7 * scale)  # d sxx/dx + d sxy/dy
    np.testing.assert_allclose(dx[4] + dy[3], 0, atol=1e-7 * scale)  # d sxy/dx + d syy/dy
    xs = np.array([-1.3, -0.2, 0.2, 1.3])
    jump = f(xs, np.full(4, 1e-13)) - f(xs, np.full(4, -1e-13))
    np.testing.assert_allclose(jump[:2, :2].T, [burgers, burgers], atol=1e-12)  # cut: x < 0
    np.testing.assert_allclose(jump[:2, 2:], 0, atol=1e-12)
    # the stress does not see the cut
    np.testing.assert_allclose(jump[2:], 0, atol=1e-9 * scale)


def test_screw_dislocation_reference_is_an_elastic_field():
    def f(x, y):
        return screw_dislocation_antiplane(x, y, 1.0, MU)

    x, y = _random_points()
    dx, dy = _derivatives(f, x, y)
    s = f(x, y)
    scale = np.abs(s[1:]).max()
    np.testing.assert_allclose(MU * dx[0], s[1], atol=1e-8 * scale)
    np.testing.assert_allclose(MU * dy[0], s[2], atol=1e-8 * scale)
    np.testing.assert_allclose(dx[1] + dy[2], 0, atol=1e-7 * scale)
    xs = np.array([-1.3, 1.3])
    jump = f(xs, np.full(2, 1e-13)) - f(xs, np.full(2, -1e-13))
    np.testing.assert_allclose(jump[0], [1.0, 0.0], atol=1e-12)


# ----------------------------------------------------------------------------
# quadratic slip element kernels vs dislocation quadrature
# ----------------------------------------------------------------------------


def _element(angle, length=1.4, centre=(0.3, -0.2)):
    t = np.array([np.cos(angle), np.sin(angle)])
    c = np.asarray(centre)
    els = bemcs.initialize_els()
    els.x1, els.y1 = np.array([c[0] - 0.5 * length * t[0]]), np.array([c[1] - 0.5 * length * t[1]])
    els.x2, els.y2 = np.array([c[0] + 0.5 * length * t[0]]), np.array([c[1] + 0.5 * length * t[1]])
    bemcs.standardize_els_geometry(els, reorder=False)
    return els


def _local_points(els, xl, yl):
    """Global coordinates of points given in element 0's local frame."""
    R = np.asarray(els.rot_mats[0])
    return (els.x_centers[0] + R[0, 0] * xl + R[0, 1] * yl,
            els.y_centers[0] + R[1, 0] * xl + R[1, 1] * yl)


# local (x, y) in units of the half-length a: near the element (0.05 a off it, above
# and below), around and beyond the tips (incl. 1e-3 a off the line extension), far
_XL = np.array([0.0, 0.5, -0.8, 0.95, -1.05, 1.0, 1.5, -2.5, 3.0, 0.2, -6.0, 40.0, -150.0])
_YL = np.array([0.05, -0.05, 0.3, -0.1, 0.05, 0.4, 1e-3, -1e-3, 2.0, -4.0, 5.0, 25.0, -100.0])


def _tolerance(els, e, xo, yo):
    """Per-point tolerance. Near field ~1e-14; in the far field both the reference sum
    and the closed form lose digits to cancellation (a quadratic slip patch has zero net
    Burgers vector, so its field is a small difference of ~a/r terms): measured
    ~1e-12 at 8 a, 1e-9 at 47 a, 7e-8 at 180 a, i.e. ~eps (r/a)^3."""
    r = np.hypot(xo - els.x_centers[e], yo - els.y_centers[e]) / els.half_lengths[e]
    return 5e-14 * np.maximum(1.0, r) ** 3


def _check(got, ref, tol, what):
    err = np.abs(got - ref).max(axis=0) / np.abs(ref).max(axis=0)  # per point, rel. to its field
    assert (err < tol).all(), (what, err.max(), np.argmax(err / tol))


def _global_fields(U, S, k, direction, geometry):
    if geometry == "planestrain":
        return np.r_[U[:, k, :, direction].T, S[:, k, :, direction].T]
    return np.r_[U[:, k, :].T, S[:, k, :].T]


@pytest.mark.parametrize("geometry", ["planestrain", "antiplane"])
@pytest.mark.parametrize("angle", [0.0, 0.7, np.pi / 2, 2.6])
@pytest.mark.parametrize("use_numba", [False, pytest.param(True, marks=pytest.mark.numba)])
def test_quadratic_slip_element_matches_dislocation_quadrature(geometry, angle, use_numba):
    """Every basis node and slip direction, displacement and stress, near and far."""
    els = _element(angle)
    a = els.half_lengths[0]
    xo, yo = _local_points(els, _XL * a, _YL * a)
    U, S = GF.element_kernels_slip(xo, yo, els, geometry=geometry, mu=MU, nu=NU, use_numba=use_numba)
    for k in range(3):
        for d in range(2 if geometry == "planestrain" else 1):
            ref = quadratic_slip_by_dislocations(els, 0, xo, yo, k, d, geometry, MU, NU)
            _check(_global_fields(U, S, k, d, geometry), ref, _tolerance(els, 0, xo, yo), (k, d))


@pytest.mark.parametrize("geometry", ["planestrain", "antiplane"])
def test_quadratic_slip_sign_convention_is_plus_minus_minus(geometry):
    """slip = u(+n) - u(-n), directly at the element: the Lagrange-node test
    (test_get_kernels.py) checks only the magnitude of the jump."""
    els = _element(0.7)
    a = els.half_lengths[0]
    xo, yo = _local_points(els, np.array([0.1 * a, 0.1 * a]), np.array([1e-7, -1e-7]))
    U, _ = GF.element_kernels_slip(xo, yo, els, geometry=geometry, mu=MU, nu=NU)
    s_centre = 1 - 2.25 * 0.1**2  # centre-node basis 1 - (9/4) (s/a)^2 at s = 0.1 a
    if geometry == "antiplane":
        np.testing.assert_allclose(U[0, 1, 0] - U[1, 1, 0], s_centre, atol=1e-6)
    else:
        t = np.array([els.x_shears[0], els.y_shears[0]])
        n = np.array([els.x_normals[0], els.y_normals[0]])
        for d, vec in enumerate((t, n)):
            np.testing.assert_allclose((U[0, 1, :, d] - U[1, 1, :, d]) @ vec, s_centre, atol=1e-6)


@pytest.mark.parametrize("geometry", ["planestrain", "antiplane"])
def test_quadratic_slip_kernels_on_a_curved_mesh_match_dislocation_quadrature(els_curved, geometry):
    """Level 2 (`get_slip_kernels`, basis "quadratic"): each column of a 4-element curved
    mesh is the dislocation quadrature over that element alone."""
    xo, yo = np.array([-1.0, 1.0, 3.0, 5.0, 2.2, 7.5]), np.array([0.9, -0.6, 1.4, -1.6, 0.05, 2.0])
    U, S, _ = bemcs.get_slip_kernels(xo, yo, els_curved, geometry=geometry, mu=MU, nu=NU,
                                        basis="quadratic")
    for e in range(len(els_curved.x1)):
        for k in range(3):
            for d in range(2 if geometry == "planestrain" else 1):
                ref = quadratic_slip_by_dislocations(els_curved, e, xo, yo, k, d, geometry, MU, NU)
                _check(_global_fields(U, S, 3 * e + k, d, geometry), ref,
                       _tolerance(els_curved, e, xo, yo), (e, k, d))
