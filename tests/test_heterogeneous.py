"""Heterogeneous media.

- Plane-strain compliance used to turn stresses into strains (tasks.md issue 1).
- Antiplane strike-slip fault in a layered half-space against the exact image series
  (Rybicki 1971), for a fault inside the layer and one through the interface (the
  setup of examples/antiplane/example_Rybicki_layered.py).

The heterogeneous solver is also tested in test_solve.py (bimaterial image solution,
traction continuity across a plane-strain interface, independence of the sampled side)
and test_validation.py (rigid/free limits, mirror symmetry).
"""

import numpy as np

import bemcs.bemAssembly as GF

MU, NU = 1.3, 0.27


def test_compliance_inverts_plane_strain_hooke():
    rng = np.random.default_rng(0)
    exx, eyy, exy = rng.normal(size=(3, 5))
    lam = 2 * MU * NU / (1 - 2 * NU)
    sxx = 2 * MU * exx + lam * (exx + eyy)
    syy = 2 * MU * eyy + lam * (exx + eyy)
    sxy = 2 * MU * exy
    got = GF.plane_strain_compliance(sxx, syy, sxy, MU, NU)
    np.testing.assert_allclose(got, (exx, eyy, exy), atol=1e-14)
    # the old nu/(1+nu) (plane-stress) form differs: issue 1
    nup = NU / (1 + NU)
    old_exx = ((1 - nup) * sxx - nup * syy) / (2 * MU)
    assert np.abs(old_exx - exx).max() > 1e-3


# ----------------------------------------------------------------------------
# strike-slip fault in a layered half-space vs the image series (antiplane)
# ----------------------------------------------------------------------------

import pytest  # noqa: E402

import bemcs  # noqa: E402
from meshes import (CONTACT_D, CONTACT_MU1, CONTACT_MU2, CONTACT_SURFACE, LAYERED_H, LAYERED_MU1,
                    LAYERED_MU2, LAYERED_SURFACE, contact_mesh, layered_mesh)  # noqa: E402
from reference_solutions import (contact_displacement, halfspace_fault_displacement,  # noqa: E402
                                 layered_image_series)


def _image_series(x, D, n_quad=300):
    return layered_image_series(x, D, LAYERED_H, LAYERED_MU1, LAYERED_MU2, n_quad)


def _contact_exact(x, dist, y=0.0):
    return contact_displacement(x, dist, y, CONTACT_D, CONTACT_MU1, CONTACT_MU2)


def _S_halfspace(x, y=0.0):
    return halfspace_fault_displacement(x, y, CONTACT_D)


def _surface_error(D, h, **kw):
    mesh = layered_mesh(D, h, **kw)
    x, (_, _, rows) = bemcs.solve(mesh, geometry="antiplane", mu=1.0, return_system=True)
    u = bemcs.evaluate_fields(mesh, x, LAYERED_SURFACE, np.full_like(LAYERED_SURFACE, -1e-9),
                              geometry="antiplane", mu=1.0)["uz"]
    symmetric_part = np.abs(u[:12][::-1] + u[12:]).max()  # u is odd in x
    return np.abs(u - _image_series(LAYERED_SURFACE, D)).max(), symmetric_part, mesh, x, rows


def test_image_series_branches_agree_at_the_interface():
    """The two branches of the reference meet at d = H, and the jump at the fault trace
    is the surface slip (here 1) on both."""
    x = np.array([-0.3, 0.3, 2.0])
    D_just = [LAYERED_H - 1e-9, LAYERED_H + 1e-9]
    u = [_image_series(x, D, n_quad=400) for D in D_just]
    np.testing.assert_allclose(u[0], u[1], atol=1e-6)
    eps = np.array([-1e-9, 1e-9])
    for D in (0.6, 1.63):
        jump = np.diff(_image_series(eps, D))[0]
        np.testing.assert_allclose(jump, 1.0, atol=1e-6)


@pytest.mark.parametrize("D", [0.6, 1.63])
def test_layered_half_space_matches_the_image_series(D):
    """Inside the layer and through the interface, with the lines out to 300 H
    (log-spaced) and trapezoids restarting at the crossing: both ~2.5e-5."""
    err, sym, *_ = _surface_error(D, 0.05)
    # measured 2.4e-5 (in the layer) and 6.1e-5 (through the interface). The split layout has a
    # free half-hat on each side of the crossing (needed in plane strain, tasks.md P2); here the
    # force there is zero by antisymmetry, and tapering to zero gave 2.6e-5 (both -> ~2e-5 at h/2)
    assert err < (5e-5 if D < LAYERED_H else 8e-5)
    # through the interface the split layout is mirror-symmetric about the crossing; inside
    # the layer there is no crossing and the plain stride leaves a harmless ~2e-6 asymmetry
    assert sym < (1e-8 if D > LAYERED_H else 1e-5)


def test_lopsided_trapezoids_at_a_crossing_give_a_spurious_net_force():
    """Without a connectivity, the plain stride-2 layout runs through the crossing and has
    a collocation point on one side of it only, and solve warns (TrapezoidCrossingWarning):
    the antisymmetry breaks, the interface carries a net force, and its log far field
    offsets the surface displacement (percent level, growing with L)."""
    with pytest.warns(bemcs.TrapezoidCrossingWarning, match="split_force_trapezoids"):
        err, sym, *_ = _surface_error(1.63, 0.1, L=100.0, lopsided=True)
    assert sym > 1e-2 and err > 5e-3  # measured 3.3e-2 and 1.7e-2 (split: <1e-8, 8e-5)
    mesh = layered_mesh(1.63, 0.1, L=100.0)
    np.testing.assert_allclose(mesh.fault_crossings, [[0.0, -LAYERED_H]], atol=1e-12)


def test_traction_is_continuous_at_interface_collocation_points():
    _, _, mesh, x, rows = _surface_error(1.63, 0.1, L=100.0)
    xk = mesh.els.x_centers[rows["element"][rows["type"] == "h"]]
    xk = xk[np.abs(xk) < 2]
    mf = lambda X, Y: np.where(Y > -LAYERED_H, LAYERED_MU1, LAYERED_MU2)  # noqa: E731
    kw = dict(geometry="antiplane", mu=1.0, mu_field=mf)
    above = bemcs.evaluate_fields(mesh, x, xk, np.full_like(xk, -LAYERED_H + 1e-7), **kw)["syz"]
    below = bemcs.evaluate_fields(mesh, x, xk, np.full_like(xk, -LAYERED_H - 1e-7), **kw)["syz"]
    assert np.abs(above - below).max() < 1e-4 * np.abs(above).max()


# ----------------------------------------------------------------------------
# strike-slip fault next to (or on) a vertical material contact (Rybicki 1971)
# ----------------------------------------------------------------------------
# Half-space y < 0; mu1 for x < d, mu2 for x > d; vertical fault at x = 0 from the surface to depth CONTACT_D.
# With S the homogeneous half-space field of the fault (fault + free-surface image) and
# k = (mu1 - mu2) / (mu1 + mu2), the exact solution is u = S(x) + k S(2d - x) for x < d and
# (1 + k) S(x) for x > d: the contact mirror commutes with the surface mirror, so no series.
# d = 0 is a fault on the contact (a bimaterial fault): u = 2 mu2/(mu1 + mu2) S on the mu1 side,
# 2 mu1/(mu1 + mu2) S on the mu2 side.

def _contact_error(dist, h):
    import warnings

    mesh = contact_mesh(dist, h)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bemcs.TrapezoidCrossingWarning)  # d = 0: false positive (tasks.md P6)
        x = bemcs.solve(mesh, geometry="antiplane", mu=1.0)
    u = bemcs.evaluate_fields(mesh, x, CONTACT_SURFACE, np.full_like(CONTACT_SURFACE, -1e-9), geometry="antiplane", mu=1.0)["uz"]
    return np.abs(u - _contact_exact(CONTACT_SURFACE, dist)).max(), mesh, x


@pytest.mark.parametrize("dist", [0.5, 0.0])
def test_contact_reference_is_continuous_across_the_contact(dist):
    """u and mu du/dx continuous at x = dist (at depth), and the jump across the fault is the slip."""
    y, e = np.array([-0.3, -1.7]), 1e-7
    k_l, k_r = _contact_exact(np.full(2, dist - e), dist, y), _contact_exact(np.full(2, dist + e), dist, y)
    if dist > 0:
        np.testing.assert_allclose(k_l, k_r, atol=1e-6)
    h = 1e-5
    grad_l = (_contact_exact(np.full(2, dist - e), dist, y) - _contact_exact(np.full(2, dist - e - h), dist, y)) / h
    grad_r = (_contact_exact(np.full(2, dist + e + h), dist, y) - _contact_exact(np.full(2, dist + e), dist, y)) / h
    np.testing.assert_allclose(CONTACT_MU1 * grad_l, CONTACT_MU2 * grad_r, rtol=1e-3)
    jump = _contact_exact(np.array([1e-9]), dist) - _contact_exact(np.array([-1e-9]), dist)
    np.testing.assert_allclose(jump, 1.0, atol=1e-6)  # surface slip = 1


@pytest.mark.parametrize("dist, tol", [(0.5, 1e-2), (0.0, 4e-2)])
def test_fault_near_and_on_a_vertical_contact_converges(dist, tol):
    """The contact meets the free surface, where the force chain has an open end (a free force
    value at the tip, collocated at the tip element). First order: 1.1e-2 -> 6.1e-3 (d = 0.5) and
    4.7e-2 -> 2.3e-2 (d = 0) for h = 0.1 -> 0.05. With a tapered end it was 3.3e-2 -> 1.7e-2 and
    1.3e-1 -> 6.8e-2; with a constant extrapolation over the tip element 1.3e-2 -> 6.4e-3 and
    6.0e-2 -> 2.5e-2. The rest is the corner (tasks.md P3)."""
    e1, *_ = _contact_error(dist, 0.1)
    e2, mesh, x = _contact_error(dist, 0.05)
    assert e2 < tol and 1.6 < e1 / e2 < 2.8


def test_contact_force_is_exact_at_depth_and_open_at_the_free_surface():
    _, mesh, x = _contact_error(0.5, 0.05)
    np.testing.assert_allclose(mesh.force_open_ends, [[0.5, 0.0]], atol=1e-12)
    xe, ye, f = mesh.values_on_elements(x, "force", geometry="antiplane", n_eval=3)
    ye, f = np.ravel(ye)[1::3], np.ravel(f)[1::3]
    order = np.argsort(-ye)
    ye, f = ye[order], f[order]
    k = (CONTACT_MU1 - CONTACT_MU2) / (CONTACT_MU1 + CONTACT_MU2)
    kappa = bemcs.interface_kappa(CONTACT_MU1, CONTACT_MU2, -1)
    e = 1e-6
    dudx = (1 - k) * (_S_halfspace(0.5 - e, ye) - _S_halfspace(0.5 - 3 * e, ye)) / (2 * e)  # mu1 side
    f_exact = -kappa * dudx  # antiplane row f + kappa T = 0, T = mu0/mu1 * (mu1 du/dx), mu0 = mu1 = 1
    np.testing.assert_allclose(f[:8], f_exact[:8], atol=6e-3)  # including the top element (was ~half)
