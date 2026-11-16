"""Plane strain: a buried fault near a bonded bimaterial interface, against the exact
edge-dislocation solution (Dundurs 1969), integrated over the fault's slip profile.

Setup: interface y = 0 (force elements "h", out to +-300, log-spaced), material 1 (mu1 = 1)
below, material 2 above; a buried fault in material 1 with prescribed (smoothly tapered)
shear and opening slip ("s", hats). No background, no free surface.

Measured (true stresses, relative to the largest, h = 0.1 -> 0.05 -> 0.025 near the fault):
- no contrast: 2.1e-4, 9.3e-6, 5.3e-7 (validates the reference, signs and quadrature);
- mu2/mu1 = 4 and 1/4: 2.4e-2 -> 1.0e-2 -> 5.4e-3 and 1.8e-2 -> 6.6e-3 -> 2.7e-3 (horizontal
  fault), similar for a 40-degree dipping fault: about first order.
Cause (tested below): in plane strain a tangential force on a straight line produces a normal
traction through a 1/x kernel (and vice versa). The trapezoids hold the force constant over
each collocated element, so the odd error f'(x - c) there contributes O(h) to the traction at
the collocation point. Linear interpolation of the same density is O(h^2). Antiplane has no
such coupling along the line, which is why it converges at h^2 there (tasks.md P1b).
"""

import numpy as np
import pytest

import bemcs
import bemcs.bemAssembly as GF
from reference_solutions import (
    edge_dislocation_bimaterial,
    edge_dislocation_bimaterial_constants,
    edge_dislocation_bimaterial_displacement,
)

NU, MU1 = 0.25, 1.0
X0, Y0 = 0.3, -0.8
SLIP = (1.0, 0.5)  # local (t, n) slip amplitudes on the fault


# ----------------------------------------------------------------------------
# the reference: a point dislocation near the interface
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("mu2", [1.0, 4.0, 0.25])
@pytest.mark.parametrize("b", [(1.0, 0.0), (0.0, 1.0)])
def test_dislocation_reference_is_continuous_across_the_interface(mu2, b):
    _, c, residual = edge_dislocation_bimaterial_constants(complex(X0, Y0), complex(*b), MU1, mu2, NU)
    assert residual < 1e-12  # the image ansatz is complete
    if mu2 == MU1:
        assert max(abs(v) for v in c[0].values()) < 1e-12  # no contrast: no images
    xi = np.linspace(-3, 3, 7)
    above, _ = edge_dislocation_bimaterial(xi, np.full(7, 1e-9), X0, Y0, b, MU1, mu2, NU)
    below, _ = edge_dislocation_bimaterial(xi, np.full(7, -1e-9), X0, Y0, b, MU1, mu2, NU)
    np.testing.assert_allclose(above[1:], below[1:], atol=1e-7)  # syy, sxy continuous


@pytest.mark.parametrize("b", [(1.0, 0.0), (0.0, 1.0), (0.6, -0.8)])
def test_dislocation_reference_is_an_elastic_field_with_burgers_vector_b(b):
    mu2, h = 4.0, 1e-6
    yy, e = np.array([-1.5]), 1e-9  # the cut runs straight down from the dislocation
    jump = (edge_dislocation_bimaterial_displacement(np.array([X0 - e]), yy, X0, Y0, b, MU1, mu2, NU)
            - edge_dislocation_bimaterial_displacement(np.array([X0 + e]), yy, X0, Y0, b, MU1, mu2, NU))
    np.testing.assert_allclose(jump.ravel(), b, atol=1e-8)
    X, Y = np.array([1.7, -0.9, 0.4]), np.array([-0.3, 0.5, 1.2])

    def U(x, y):
        return edge_dislocation_bimaterial_displacement(x, y, X0, Y0, b, MU1, mu2, NU)

    def S(x, y):
        return edge_dislocation_bimaterial(x, y, X0, Y0, b, MU1, mu2, NU)[0]

    dux, duy = (U(X + h, Y) - U(X - h, Y)) / (2 * h), (U(X, Y + h) - U(X, Y - h)) / (2 * h)
    mu = np.where(Y > 0, mu2, MU1)
    lam = 2 * mu * NU / (1 - 2 * NU)
    exx, eyy, exy = dux[0], duy[1], 0.5 * (duy[0] + dux[1])
    s = S(X, Y)
    np.testing.assert_allclose(lam * (exx + eyy) + 2 * mu * exx, s[0], atol=1e-8)
    np.testing.assert_allclose(lam * (exx + eyy) + 2 * mu * eyy, s[1], atol=1e-8)
    np.testing.assert_allclose(2 * mu * exy, s[2], atol=1e-8)
    dsx, dsy = (S(X + h, Y) - S(X - h, Y)) / (2 * h), (S(X, Y + h) - S(X, Y - h)) / (2 * h)
    np.testing.assert_allclose(dsx[0] + dsy[2], 0, atol=1e-8)
    np.testing.assert_allclose(dsx[2] + dsy[1], 0, atol=1e-8)


def test_dislocation_reference_limits_free_surface_and_rigid_boundary():
    xi = np.linspace(-3, 3, 13)
    s_free, _ = edge_dislocation_bimaterial(xi, np.full(13, -1e-9), X0, Y0, (1.0, 0.3), MU1, 1e-8, NU)
    assert np.abs(s_free[1:]).max() < 1e-6  # traction-free surface
    u_rigid = edge_dislocation_bimaterial_displacement(xi, np.full(13, -1e-9), X0, Y0, (1.0, 0.3), MU1, 1e8, NU)
    assert np.abs(u_rigid - u_rigid[:, :1]).max() < 1e-6  # the interface does not move


# ----------------------------------------------------------------------------
# a buried fault near the interface: BEM vs the dislocation integral
# ----------------------------------------------------------------------------


def _fault(case):
    if case == "horizontal":
        return np.array([0.0, -0.6]), 0.0, 0.5  # centre, angle, half-length
    return np.array([0.0, -0.9]), np.deg2rad(40.0), 0.5


def _exact_stress(X, Y, case, mu2, n_quad=100):
    """Slip cos^2(pi xi / 2a) e (e = SLIP in the fault's local frame) as a dislocation density
    -s'(xi) e along the fault."""
    c, ang, a = _fault(case)
    t = np.array([np.cos(ang), np.sin(ang)])
    n = np.array([-t[1], t[0]])
    e = SLIP[0] * t + SLIP[1] * n
    g, w = np.polynomial.legendre.leggauss(n_quad)
    out = 0.0
    for gi, wi in zip(g, w):
        xi = a * gi
        ds = -(np.pi / (2 * a)) * np.sin(np.pi * xi / a)
        p = c + xi * t
        out = out + a * wi * edge_dislocation_bimaterial(X, Y, p[0], p[1], tuple(-ds * e), MU1, mu2, NU)[0]
    return out


def _graded(h, L=300.0, flat=2.0, ratio=1.08, h_max=5.0):
    r, step = list(np.arange(0.0, flat + 1e-12, h)), h
    while r[-1] < L:
        r.append(r[-1] + step)
        step = min(step * ratio, h_max)
    r = np.array(r)
    return np.r_[-r[:0:-1], r]


def _mesh(case, h, mu2):
    c, ang, a = _fault(case)
    t = np.array([np.cos(ang), np.sin(ang)])
    n_f = int(round(2 * a / h))
    xi = np.linspace(-a, a, n_f + 1)
    P = c + xi[:, None] * t
    xs = _graded(h)
    n_i = len(xs) - 1
    els = bemcs.initialize_els()
    els.x1, els.y1 = np.r_[P[:-1, 0], xs[:-1]], np.r_[P[:-1, 1], np.zeros(n_i)]
    els.x2, els.y2 = np.r_[P[1:, 0], xs[1:]], np.r_[P[1:, 1], np.zeros(n_i)]
    bemcs.standardize_els_geometry(els, reorder=False)
    profile = np.cos(0.5 * np.pi * (0.5 * (xi[:-1] + xi[1:])) / a) ** 2
    value = np.r_[np.c_[SLIP[0] * profile, SLIP[1] * profile], np.zeros((n_i, 2))]
    # interface elements run left to right: -n is below (material 1)
    return bemcs.Mesh(els, bc_type=["s"] * n_f + ["h"] * n_i, bc_value=value, mu_minus=MU1,
                      mu_plus=np.r_[np.full(n_f, MU1), np.full(n_i, mu2)],
                      basis={"slip": "hat", "force": "trapezoid"})


def _points(case):
    c, _, _ = _fault(case)
    th = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    P = np.r_[c + 0.9 * np.c_[np.cos(th), np.sin(th)], np.c_[np.linspace(-2, 2, 5), np.full(5, 0.5)]]
    P = P[np.abs(P[:, 1]) > 0.2]
    return P[:, 0], P[:, 1]


def _error(case, h, mu2):
    X, Y = _points(case)
    ref = _exact_stress(X, Y, case, mu2)
    mesh = _mesh(case, h, mu2)
    x = bemcs.solve(mesh, geometry="planestrain", mu=1.0, nu=NU)
    f = bemcs.evaluate_fields(mesh, x, X, Y, geometry="planestrain", mu=1.0, nu=NU,
                              mu_field=lambda x_, y_: np.where(y_ > 0, mu2, MU1))
    got = np.array([f["sxx"], f["syy"], f["sxy"]])
    return np.abs(got - ref).max() / np.abs(ref).max()


@pytest.mark.parametrize("case", ["horizontal", "dipping"])
def test_no_contrast_matches_the_dislocation_integral(case):
    """Validates the reference, the sign convention (slip = u(+n) - u(-n) <-> Burgers vector)
    and the quadrature: no images, the BEM interface carries no force."""
    assert _error(case, 0.05, MU1) < 1e-4  # measured 9.3e-6 / 7.9e-6


@pytest.mark.parametrize("case", ["horizontal", "dipping"])
@pytest.mark.parametrize("mu2", [4.0, 0.25])
def test_fault_near_a_bimaterial_interface_converges_to_dundurs(case, mu2):
    e1, e2 = _error(case, 0.1, mu2), _error(case, 0.05, mu2)
    assert e2 < 1.5e-2  # measured 4e-3 to 1e-2
    assert e1 / e2 > 1.8  # about first order (see the module docstring)


def test_trapezoids_are_first_order_consistent_in_plane_strain_on_a_straight_line():
    """The cause of the first-order convergence above: the traction at the collocation points
    from a smooth force density represented by trapezoids converges at O(h), the same density
    linearly interpolated at O(h^2)."""

    def f_t(x):
        return np.exp(-x**2) * np.cos(2 * x)

    def f_n(x):
        return 0.5 * np.exp(-(x - 0.3) ** 2)

    def line(h, L=8.0):
        xs = np.arange(-L, L + 1e-12, h)
        els = bemcs.initialize_els()
        els.x1, els.x2 = xs[:-1], xs[1:]
        els.y1, els.y2 = np.zeros(len(xs) - 1), np.zeros(len(xs) - 1)
        bemcs.standardize_els_geometry(els, reorder=False)
        return els

    def traction_linear(els, xo, yo):
        _, S, _ = bemcs.get_force_kernels(xo, yo, els, geometry="planestrain", mu=1.0, nu=NU, basis="linear")
        c = np.zeros((len(els.x1) * 2, 2))  # ramp 0: end node, ramp 1: start node
        c[0::2, 0], c[1::2, 0], c[0::2, 1], c[1::2, 1] = f_t(els.x2), f_t(els.x1), f_n(els.x2), f_n(els.x1)
        s = np.einsum("oscl,sl->oc", S, c)
        return np.c_[s[:, 2], s[:, 1]]

    fine = line(0.1 / 32)
    err_trap, err_lin = [], []
    for h in (0.1, 0.05):
        els = line(h)
        conn = GF.trapezoid_connectivity(els)
        cm = els.x_centers[conn[:, 1]]
        xo = cm[np.abs(cm) < 2]
        yo = np.full_like(xo, -1e-9)
        ref = traction_linear(fine, xo, yo)
        _, St, _ = bemcs.get_force_kernels(xo, yo, els, geometry="planestrain", mu=1.0, nu=NU, basis="trapezoid",
                                           connectivity=conn)
        s = np.einsum("oscl,sl->oc", St, np.c_[f_t(cm), f_n(cm)])
        err_trap.append(np.abs(np.c_[s[:, 2], s[:, 1]] - ref).max())
        err_lin.append(np.abs(traction_linear(els, xo, yo) - ref).max())
    assert 1.7 < err_trap[0] / err_trap[1] < 2.6  # first order: 1.25e-2 -> 5.8e-3
    assert err_lin[0] / err_lin[1] > 3.5  # second order: 3.8e-3 -> 9.6e-4
