"""The reference modulus `mu` of the kernels is optional (default 1): the moduli are mesh data,
and `evaluate_fields` scales stresses to the material of each point (`Mesh.modulus_at`)."""

import numpy as np
import pytest

import bemcs

MU1, MU2, NU = 1.0, 4.0, 0.25


def _layered(geometry):
    """Free surface y = 0 (t), interface y = -1 (h, mu MU1 above, MU2 below), and a vertical
    fault from the surface to y = -0.6 (s), all on |x| < 10."""
    xs = np.linspace(-10, 10, 81)
    zf = np.linspace(0.0, -0.6, 7)
    n, nf = len(xs) - 1, len(zf) - 1
    els = bemcs.initialize_els()
    els.x1 = np.r_[np.zeros(nf), xs[:-1], xs[:-1]]
    els.x2 = np.r_[np.zeros(nf), xs[1:], xs[1:]]
    els.y1 = np.r_[zf[:-1], np.zeros(n), np.full(n, -1.0)]
    els.y2 = np.r_[zf[1:], np.zeros(n), np.full(n, -1.0)]
    bemcs.standardize_els_geometry(els, reorder=False)
    below = els.y_normals[-1] > 0  # interface normal points up: mu_minus is below
    labels = ["s"] * (nf + n) + ["f"] * n
    bc_type = ["s"] * nf + ["t"] * n + ["h"] * n
    bc_value = np.zeros((len(labels), 2))
    bc_value[:nf, 0] = np.cos(0.5 * np.pi * np.linspace(0.05, 0.55, nf) / 0.6) ** 2
    mu_minus = np.r_[np.full(nf + n, MU1), np.full(n, MU2 if below else MU1)]
    mu_plus = np.r_[np.full(nf + n, MU1), np.full(n, MU1 if below else MU2)]
    return bemcs.Mesh(els, labels=labels, bc_type=bc_type, bc_value=bc_value, mu_minus=mu_minus,
                      mu_plus=mu_plus, basis={"slip": "hat", "force": "trapezoid"})


@pytest.mark.parametrize("geometry", ["antiplane", "planestrain"])
def test_solution_does_not_depend_on_the_reference_modulus(geometry):
    mesh = _layered(geometry)
    nu = NU if geometry == "planestrain" else None
    xo, yo = np.linspace(-3, 3, 12), np.r_[np.full(6, -0.5), np.full(6, -1.7)]  # off the fault
    x = bemcs.solve(mesh, geometry=geometry, nu=nu)  # mu left out: 1
    f = bemcs.evaluate_fields(mesh, x, xo, yo, geometry=geometry, nu=nu)
    mu_true = lambda x_, y_: np.where(y_ > -1.0, MU1, MU2)
    x3 = bemcs.solve(mesh, geometry=geometry, mu=3.7, nu=nu)
    f3 = bemcs.evaluate_fields(mesh, x3, xo, yo, geometry=geometry, mu=3.7, nu=nu, mu_field=mu_true)
    for k in f:
        assert np.abs(f[k] - f3[k]).max() <= 1e-12 * np.abs(f3[k]).max(), k
    # "reference" gives the unscaled reference stresses (mu = 1 here)
    fr = bemcs.evaluate_fields(mesh, x, xo, yo, geometry=geometry, nu=nu, mu_field="reference")
    s = [k for k in f if k.startswith("s")][0]
    np.testing.assert_allclose(fr[s] * mu_true(xo, yo), f[s], rtol=1e-13, atol=1e-15)


def test_modulus_at_layers():
    mesh = _layered("antiplane")
    X, Y = np.meshgrid(np.linspace(-9, 9, 37), np.linspace(-3, -0.01, 30))
    np.testing.assert_array_equal(mesh.modulus_at(X, Y), np.where(Y > -1.0, MU1, MU2))
    # fault moduli are not trusted (they do not enter solve): a wrong value is ignored
    mesh.mu_minus[:6] = mesh.mu_plus[:6] = 99.0
    assert (mesh.modulus_at([0.01, -0.01], [-0.3, -0.3]) == MU1).all()


def test_modulus_at_corners_of_an_inclusion():
    """Square inclusion (mu 5) in a medium (mu 1): points whose nearest mesh point is a corner."""
    c = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]], float)
    els = bemcs.initialize_els()
    els.x1, els.y1, els.x2, els.y2 = c[:-1, 0], c[:-1, 1], c[1:, 0], c[1:, 1]
    bemcs.standardize_els_geometry(els, reorder=False)
    outward = els.x_normals[0] * 0 + els.y_normals[0] < 0  # bottom edge: outward is -y
    mm, mp = (5.0, 1.0) if outward else (1.0, 5.0)
    mesh = bemcs.Mesh(els, labels=["f"] * 4, bc_type=["h"] * 4, mu_minus=mm, mu_plus=mp)
    d = 0.3
    xo = np.array([1 + d, 1 - d, -1 - d, -1 + d, 1 + d, 0.0, 0.0])
    yo = np.array([1 + d, 1 - d, -1 - d, -1 + d, 1 - 0.1, 3.0, 0.2])
    np.testing.assert_array_equal(mesh.modulus_at(xo, yo), [1, 5, 1, 5, 1, 1, 5])


def test_background_carries_its_reference_modulus():
    with pytest.raises(ValueError, match="mu_medium"):
        bemcs.uniform_background((1.0, 0.0), geometry="antiplane")
    bg = bemcs.uniform_background((1.0, 0.0), geometry="antiplane", mu_medium=MU1)
    assert bg.mu == 1.0
    bg3 = bemcs.uniform_background((1.0, 0.0), geometry="antiplane", mu=3.0, mu_medium=MU1)
    mesh = _layered("antiplane")
    with pytest.raises(ValueError, match="reference modulus of the background"):
        bemcs.solve(mesh, geometry="antiplane", mu=2.0, background=bg3)
    xo, yo = np.array([2.0, 2.0]), np.array([-0.5, -2.0])
    f = bemcs.evaluate_fields(mesh, bemcs.solve(mesh, geometry="antiplane", background=bg), xo, yo,
                              geometry="antiplane", background=bg)
    f3 = bemcs.evaluate_fields(mesh, bemcs.solve(mesh, geometry="antiplane", background=bg3), xo, yo,
                               geometry="antiplane", background=bg3)
    for k in f:
        np.testing.assert_allclose(f[k], f3[k], rtol=1e-12, atol=1e-14)
