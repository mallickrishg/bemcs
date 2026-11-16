"""Heterogeneous media.

- Plane-strain compliance used to turn stresses into strains (tasks.md issue 1).
- Antiplane strike-slip fault in a layered half-space against the exact image series
  (Rybicki 1971), for a fault inside the layer and one through the interface (the
  setup of examples/antiplane/HeterogeneousElasticity_forces.py).

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

H_LAYER, MU1, MU2 = 1.0, 1.0, 4.0  # compliant layer over a stiff half-space


def _slip(depth, D):
    return np.where(depth < D, np.cos(0.5 * np.pi * depth / D) ** 2, 0.0)


def _layered_mesh(D, h, L=30.0, ratio=1.1):
    """Vertical fault from the surface to depth D (no node on the interface), free surface
    and interface on [-L, L] with spacing h for |x| < 2, then growing."""
    xs, step = list(np.arange(0.0, 2.0, h)), h
    while xs[-1] < L:
        xs.append(xs[-1] + step)
        step *= ratio
    xs = np.r_[-np.array(xs)[:0:-1], xs]
    n_f, n = max(2, int(round(D / h))), len(xs) - 1
    zf = np.linspace(0.0, -D, n_f + 1)
    assert np.abs(zf + H_LAYER).min() > 0.1 * D / n_f
    zc = 0.5 * (zf[:-1] + zf[1:])
    els = bemcs.initialize_els()
    els.x1, els.x2 = np.r_[np.zeros(n_f), xs[:-1], xs[:-1]], np.r_[np.zeros(n_f), xs[1:], xs[1:]]
    els.y1 = np.r_[zf[:-1], np.zeros(n), np.full(n, -H_LAYER)]
    els.y2 = np.r_[zf[1:], np.zeros(n), np.full(n, -H_LAYER)]
    bemcs.standardize_els_geometry(els, reorder=False)
    mu_f = np.where(zc > -H_LAYER, MU1, MU2)
    return bemcs.Mesh(els, bc_type=["s"] * n_f + ["t"] * n + ["h"] * n,
                      bc_value=np.r_[_slip(-zc, D), np.zeros(2 * n)],
                      mu_minus=np.r_[mu_f, np.full(n, MU1), np.full(n, MU2)],
                      mu_plus=np.r_[mu_f, np.full(n, MU1), np.full(n, MU1)],
                      basis={"slip": "quadratic", "force": "trapezoid"})


def _image_series(x, D, n_quad=300):
    """Exact surface displacement (Rybicki 1971; Segall 2010 ch. 5), integrated over the
    slip profile: unit slip from 0 to d gives B(d) + sum k^m [B(2mH + d) - B(2mH - d)] for
    d < H and (1 - k) sum k^m B(d + 2mH) for d > H, B = atan(d / x) / pi."""
    k = (MU1 - MU2) / (MU1 + MU2)
    m_max = int(np.ceil(np.log(1e-17) / np.log(abs(k)))) + 1

    def B(d):
        return np.arctan(d / x) / np.pi

    def G(d):
        if d < H_LAYER:
            return B(d) + sum(k**m * (B(2 * m * H_LAYER + d) - B(2 * m * H_LAYER - d)) for m in range(1, m_max))
        return (1 - k) * sum(k**m * B(d + 2 * m * H_LAYER) for m in range(m_max))

    u = np.zeros_like(x)
    g, w = np.polynomial.legendre.leggauss(n_quad)
    edges = [0.0, D] if D < H_LAYER else [0.0, H_LAYER, D]
    for lo, hi in zip(edges[:-1], edges[1:]):
        for d, wd in zip(0.5 * (lo + hi) + 0.5 * (hi - lo) * g, 0.5 * (hi - lo) * w):
            u += wd * 0.5 * np.pi / D * np.sin(np.pi * d / D) * G(d)
    return u


_XS = np.r_[-np.geomspace(5, 0.02, 12), np.geomspace(0.02, 5, 12)]


def _surface_error(D, h):
    mesh = _layered_mesh(D, h)
    x, (_, _, rows) = bemcs.solve(mesh, geometry="antiplane", mu=1.0, return_system=True)
    u = bemcs.evaluate_fields(mesh, x, _XS, np.full_like(_XS, -1e-9), geometry="antiplane", mu=1.0)["uz"]
    return np.abs(u - _image_series(_XS, D)).max(), mesh, x, rows


def test_image_series_branches_agree_at_the_interface():
    """The two branches of the reference meet at d = H, and the jump at the fault trace
    is the surface slip (here 1) on both."""
    x = np.array([-0.3, 0.3, 2.0])
    D_just = [H_LAYER - 1e-9, H_LAYER + 1e-9]
    u = [_image_series(x, D, n_quad=400) for D in D_just]
    np.testing.assert_allclose(u[0], u[1], atol=1e-6)
    eps = np.array([-1e-9, 1e-9])
    for D in (0.6, 1.63):
        jump = np.diff(_image_series(eps, D))[0]
        np.testing.assert_allclose(jump, 1.0, atol=1e-6)


def test_fault_inside_the_layer_matches_the_image_series():
    err, *_ = _surface_error(0.6, 0.05)
    assert err < 1e-4  # measured 5e-5, limited by truncating the lines at |x| = 30


def test_fault_through_the_interface_converges_at_first_order():
    """The interface force density steps where the fault crosses it; continuous
    trapezoids converge at first order there (tasks.md)."""
    e1, *_ = _surface_error(1.63, 0.1)
    e2, mesh, x, rows = _surface_error(1.63, 0.05)
    assert e2 < 0.03 and 1.6 < e1 / e2 < 2.4  # measured 3.1e-2 -> 1.55e-2
    # the true traction is still continuous at the interface collocation points
    xk = mesh.els.x_centers[rows["element"][rows["type"] == "h"]]
    xk = xk[np.abs(xk) < 2]
    mf = lambda X, Y: np.where(Y > -H_LAYER, MU1, MU2)  # noqa: E731
    kw = dict(geometry="antiplane", mu=1.0, mu_field=mf)
    above = bemcs.evaluate_fields(mesh, x, xk, np.full_like(xk, -H_LAYER + 1e-7), **kw)["syz"]
    below = bemcs.evaluate_fields(mesh, x, xk, np.full_like(xk, -H_LAYER - 1e-7), **kw)["syz"]
    assert np.abs(above - below).max() < 1e-4 * np.abs(above).max()
