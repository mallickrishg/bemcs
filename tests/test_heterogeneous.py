"""Plane-strain heterogeneous solver: hat slip + trapezoid force with
equivalent body forces on material interfaces (tasks.md 2.2, T1).

The checks recompute each row type from the solved fields with the public
kernel API, so they test the assembly, not just the solver against itself.
"""

import numpy as np
import pytest

import bemcs
import bemcs.bemAssembly as GF

MU, NU = 1.3, 0.27
DR = -1e-9


def _layered(fault_dip_deg=0.0):
    """3 slip elements (a fault, optionally dipping) above 3 interface elements."""
    d = np.deg2rad(fault_dip_deg)
    xs = np.array([-1.5, -0.5, 0.5, 1.5])
    fx, fy = xs * np.cos(d), xs * np.sin(d)
    els = bemcs.initialize_els()
    els.x1 = np.r_[fx[:-1], xs[:-1]]
    els.y1 = np.r_[fy[:-1], np.full(3, -1.0)]
    els.x2 = np.r_[fx[1:], xs[1:]]
    els.y2 = np.r_[fy[1:], np.full(3, -1.0)]
    bemcs.standardize_els_geometry(els, reorder=False)
    return els, np.array([[0, 1, 2]]), np.array([[3, 4, 5]])


def _fields(els, conn_s, conn_f, coefs_s, coefs_f, x, y):
    mesh = bemcs.Mesh(
        els, ["s", "s", "s", "f", "f", "f"], basis={"slip": "hat", "force": "trapezoid"},
        connectivity={"slip": conn_s, "force": conn_f},
    )
    U, S, _ = bemcs.get_kernels(x, y, mesh, geometry="planestrain", mu=MU, nu=NU)
    c = np.concatenate((coefs_s, coefs_f))
    n = len(np.atleast_1d(x))
    return [S[:, :, k, :].reshape(n, -1) @ c for k in range(3)]


def _projected_strain(els, j, sxx, syy, sxy):
    exx, eyy, exy = GF.plane_strain_compliance(sxx, syy, sxy, MU, NU)
    nx, ny = els.x_normals[j], els.y_normals[j]
    psi = np.array([exx * nx + exy * ny, exy * nx + eyy * ny]).ravel()
    return psi, (exx + eyy).item(), np.array([nx, ny])


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


@pytest.mark.parametrize("dip", [0.0, 20.0])
def test_alpha_zero_gives_no_force_and_the_prescribed_slip(dip):
    els, cs, cf = _layered(dip)
    slip = np.array([[0.7, -0.2]])
    coefs_s, coefs_f = GF.solve_planestrain_heterogeneous(
        els, cs, ["s"], slip, cf, np.array([0.0]), mu=MU, nu=NU
    )
    np.testing.assert_allclose(coefs_f, 0.0, atol=1e-14)
    # slip at the centre of the collocation element (local frame) equals the BC
    mesh = bemcs.Mesh(els, ["s", "s", "s", "f", "f", "f"], basis={"slip": "hat", "force": "trapezoid"},
                      connectivity={"slip": cs, "force": cf})
    x = np.concatenate((coefs_s, coefs_f))
    _, _, v = mesh.values_on_elements(x, "slip", geometry="planestrain", n_eval=3, frame="local")
    np.testing.assert_allclose(v[3 * 1 + 1], slip[0], atol=1e-12)


@pytest.mark.parametrize("dip", [0.0, 20.0])
@pytest.mark.parametrize("alpha", [0.5, -0.8])
def test_h_rows_hold_for_the_solved_field(dip, alpha):
    """f + alpha Psi~(s, f) = 0 at the force collocation point, recomputed from the fields."""
    els, cs, cf = _layered(dip)
    coefs_s, coefs_f = GF.solve_planestrain_heterogeneous(
        els, cs, ["s"], np.array([[1.0, 0.3]]), cf, np.array([alpha]), mu=MU, nu=NU
    )
    j = cf[0, 1]
    x = els.x_centers[j] + DR * els.x_normals[j]
    y = els.y_centers[j] + DR * els.y_normals[j]
    psi, ekk, n = _projected_strain(els, j, *_fields(els, cs, cf, coefs_s, coefs_f, x, y))
    psi_tilde = psi + NU / (1 - 2 * NU) * n * ekk
    np.testing.assert_allclose(coefs_f + alpha * psi_tilde, 0.0, atol=1e-12)
    assert np.abs(coefs_f).max() > 1e-4  # a non-trivial body force


@pytest.mark.parametrize("dip", [0.0, 20.0])
def test_t_rows_prescribe_the_projected_strain(dip):
    els, cs, cf = _layered(dip)
    target = np.array([[0.02, -0.01]])
    coefs_s, coefs_f = GF.solve_planestrain_heterogeneous(
        els, cs, ["t"], target, cf, np.array([0.4]), mu=MU, nu=NU
    )
    j = cs[0, 1]
    x = els.x_centers[j] + DR * els.x_normals[j]
    y = els.y_centers[j] + DR * els.y_normals[j]
    psi, _, _ = _projected_strain(els, j, *_fields(els, cs, cf, coefs_s, coefs_f, x, y))
    np.testing.assert_allclose(psi, target[0], atol=1e-10)
