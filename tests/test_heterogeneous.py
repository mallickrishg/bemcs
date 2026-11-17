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

H_LAYER, MU1, MU2 = 1.0, 1.0, 4.0  # compliant layer over a stiff half-space


def _slip(depth, D):
    return np.where(depth < D, np.cos(0.5 * np.pi * depth / D) ** 2, 0.0)


def _layered_mesh(D, h, L=300.0, ratio=1.08, h_max=5.0, lopsided=False):
    """Vertical fault from the surface to depth D (no node on the interface), free surface
    and interface on [-L, L]: spacing h for |x| < 2, then log-spaced up to h_max. Force
    trapezoids restarting at the fault crossing (`Mesh.split_force_trapezoids`, what a user
    would put in the connectivity file), or with lopsided=True no connectivity: the plain
    default stride, which runs through the crossing."""
    xs, step = list(np.arange(0.0, 2.0, h)), h
    while xs[-1] < L:
        xs.append(xs[-1] + step)
        step = min(step * ratio, h_max)
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
    conn = None
    if not lopsided:
        conn = {"force": _layered_mesh(D, h, L, ratio, h_max, lopsided=True).split_force_trapezoids()}
    return bemcs.Mesh(els, bc_type=["s"] * n_f + ["t"] * n + ["h"] * n, connectivity=conn,
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


def _surface_error(D, h, **kw):
    mesh = _layered_mesh(D, h, **kw)
    x, (_, _, rows) = bemcs.solve(mesh, geometry="antiplane", mu=1.0, return_system=True)
    u = bemcs.evaluate_fields(mesh, x, _XS, np.full_like(_XS, -1e-9), geometry="antiplane", mu=1.0)["uz"]
    symmetric_part = np.abs(u[:12][::-1] + u[12:]).max()  # u is odd in x
    return np.abs(u - _image_series(_XS, D)).max(), symmetric_part, mesh, x, rows


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


@pytest.mark.parametrize("D", [0.6, 1.63])
def test_layered_half_space_matches_the_image_series(D):
    """Inside the layer and through the interface, with the lines out to 300 H
    (log-spaced) and trapezoids restarting at the crossing: both ~2.5e-5."""
    err, sym, *_ = _surface_error(D, 0.05)
    # measured 2.4e-5 (in the layer) and 6.1e-5 (through the interface). The split layout has a
    # free half-hat on each side of the crossing (needed in plane strain, tasks.md P2); here the
    # force there is zero by antisymmetry, and tapering to zero gave 2.6e-5 (both -> ~2e-5 at h/2)
    assert err < (5e-5 if D < H_LAYER else 8e-5)
    # through the interface the split layout is mirror-symmetric about the crossing; inside
    # the layer there is no crossing and the plain stride leaves a harmless ~2e-6 asymmetry
    assert sym < (1e-8 if D > H_LAYER else 1e-5)


def test_lopsided_trapezoids_at_a_crossing_give_a_spurious_net_force():
    """Without a connectivity, the plain stride-2 layout runs through the crossing and has
    a collocation point on one side of it only, and solve warns (TrapezoidCrossingWarning):
    the antisymmetry breaks, the interface carries a net force, and its log far field
    offsets the surface displacement (percent level, growing with L)."""
    with pytest.warns(bemcs.TrapezoidCrossingWarning, match="split_force_trapezoids"):
        err, sym, *_ = _surface_error(1.63, 0.1, L=100.0, lopsided=True)
    assert sym > 1e-2 and err > 5e-3  # measured 3.3e-2 and 1.7e-2 (split: <1e-8, 8e-5)
    mesh = _layered_mesh(1.63, 0.1, L=100.0)
    np.testing.assert_allclose(mesh.fault_crossings, [[0.0, -H_LAYER]], atol=1e-12)


def test_traction_is_continuous_at_interface_collocation_points():
    _, _, mesh, x, rows = _surface_error(1.63, 0.1, L=100.0)
    xk = mesh.els.x_centers[rows["element"][rows["type"] == "h"]]
    xk = xk[np.abs(xk) < 2]
    mf = lambda X, Y: np.where(Y > -H_LAYER, MU1, MU2)  # noqa: E731
    kw = dict(geometry="antiplane", mu=1.0, mu_field=mf)
    above = bemcs.evaluate_fields(mesh, x, xk, np.full_like(xk, -H_LAYER + 1e-7), **kw)["syz"]
    below = bemcs.evaluate_fields(mesh, x, xk, np.full_like(xk, -H_LAYER - 1e-7), **kw)["syz"]
    assert np.abs(above - below).max() < 1e-4 * np.abs(above).max()


# ----------------------------------------------------------------------------
# strike-slip fault next to (or on) a vertical material contact (Rybicki 1971)
# ----------------------------------------------------------------------------
# Half-space y < 0; mu1 for x < d, mu2 for x > d; vertical fault at x = 0 from the surface to depth D_V.
# With S the homogeneous half-space field of the fault (fault + free-surface image) and
# k = (mu1 - mu2) / (mu1 + mu2), the exact solution is u = S(x) + k S(2d - x) for x < d and
# (1 + k) S(x) for x > d: the contact mirror commutes with the surface mirror, so no series.
# d = 0 is a fault on the contact (a bimaterial fault): u = 2 mu2/(mu1 + mu2) S on the mu1 side,
# 2 mu1/(mu1 + mu2) S on the mu2 side.

MU_V1, MU_V2, D_V = 1.0, 4.0, 1.0


def _S_halfspace(x, y=0.0, n_quad=300):
    """Homogeneous half-space displacement of the tapered fault at x = 0 (slip cos^2 profile, jump
    +s across x = 0, the fault elements' -n side being x < 0)."""
    g, w = np.polynomial.legendre.leggauss(n_quad)
    d, wd = 0.5 * D_V * (g + 1), 0.5 * D_V * w
    neg_ds = 0.5 * np.pi / D_V * np.sin(np.pi * d / D_V)
    yp = -np.asarray(y, float)  # depth
    return sum(wi * si * (np.arctan((di - yp) / x) + np.arctan((di + yp) / x)) / (2 * np.pi)
               for di, wi, si in zip(d, wd, neg_ds))


def _contact_exact(x, dist, y=0.0):
    k = (MU_V1 - MU_V2) / (MU_V1 + MU_V2)
    return np.where(x < dist, _S_halfspace(x, y) + k * _S_halfspace(2 * dist - x, y), (1 + k) * _S_halfspace(x, y))


def _contact_mesh(dist, h, L=300.0, ratio=1.08, h_max=5.0):
    """Fault at x = 0 (0..D_V), free surface on [-L, L], vertical contact at x = dist from the surface
    to depth L; spacing h near the fault, log-spaced beyond (dist a multiple of h)."""
    r, step = list(np.arange(0.0, 2.0 + 1e-12, h)), h
    while r[-1] < L:
        r.append(r[-1] + step)
        step = min(step * ratio, h_max)
    r = np.array(r)
    xs, zi = np.r_[-r[:0:-1], r], -r
    assert np.isclose(xs, dist).any()
    n_f = int(round(D_V / h))
    zf = np.linspace(0.0, -D_V, n_f + 1)
    x1 = np.r_[np.zeros(n_f), xs[:-1], np.full(len(zi) - 1, dist)]
    x2 = np.r_[np.zeros(n_f), xs[1:], np.full(len(zi) - 1, dist)]
    y1 = np.r_[zf[:-1], np.zeros(len(xs) - 1), zi[:-1]]
    y2 = np.r_[zf[1:], np.zeros(len(xs) - 1), zi[1:]]
    els = bemcs.initialize_els()
    els.x1, els.x2, els.y1, els.y2 = x1, x2, y1, y2
    bemcs.standardize_els_geometry(els, reorder=False)
    bc = np.array(["s"] * n_f + ["t"] * (len(xs) - 1) + ["h"] * (len(zi) - 1))
    side = np.where(els.x_centers < dist, MU_V1, MU_V2)
    # vertical elements run downward: unit normal +x, so -n is the mu1 side
    mu_minus = np.where(bc == "t", side, MU_V1)
    mu_plus = np.where(bc == "t", side, np.where(bc == "h", MU_V2, MU_V2 if dist == 0 else MU_V1))
    value = np.where(bc == "s", np.cos(0.5 * np.pi * (-els.y_centers) / D_V) ** 2, 0.0)
    return bemcs.Mesh(els, bc_type=bc, bc_value=value, mu_minus=mu_minus, mu_plus=mu_plus,
                      basis={"slip": "hat", "force": "trapezoid"})


_XV = np.r_[-np.geomspace(5, 0.02, 12), np.geomspace(0.02, 5, 12)]


def _contact_error(dist, h):
    import warnings

    mesh = _contact_mesh(dist, h)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bemcs.TrapezoidCrossingWarning)  # d = 0: false positive (tasks.md P6)
        x = bemcs.solve(mesh, geometry="antiplane", mu=1.0)
    u = bemcs.evaluate_fields(mesh, x, _XV, np.full_like(_XV, -1e-9), geometry="antiplane", mu=1.0)["uz"]
    return np.abs(u - _contact_exact(_XV, dist)).max(), mesh, x


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
    np.testing.assert_allclose(MU_V1 * grad_l, MU_V2 * grad_r, rtol=1e-3)
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
    k = (MU_V1 - MU_V2) / (MU_V1 + MU_V2)
    kappa = bemcs.interface_kappa(MU_V1, MU_V2, -1)
    e = 1e-6
    dudx = (1 - k) * (_S_halfspace(0.5 - e, ye) - _S_halfspace(0.5 - 3 * e, ye)) / (2 * e)  # mu1 side
    f_exact = -kappa * dudx  # antiplane row f + kappa T = 0, T = mu0/mu1 * (mu1 du/dx), mu0 = mu1 = 1
    np.testing.assert_allclose(f[:8], f_exact[:8], atol=6e-3)  # including the top element (was ~half)
