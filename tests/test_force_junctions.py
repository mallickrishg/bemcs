"""Junctions of force elements: layer interfaces meeting a vertical material contact
(examples/antiplane/example_layered_medium.py).

Exact check: one layer over a half-space on the left, the same stack times c on the right, the
fault on the contact. Then u = 2c/(1+c) S for x < 0 and 2/(1+c) S for x > 0, with S Rybicki's
layered image series (no lateral contrast).
"""

import warnings

import numpy as np
import pytest

import bemcs

D, H1 = 1.0, 0.5
XC = np.r_[-np.geomspace(5, 0.02, 12), np.geomspace(0.02, 5, 12)]


def _graded(h, L, h_max, flat=1.0, ratio=1.08):
    r, step = list(np.round(np.arange(0.0, flat + 1e-12, h), 12)), h
    while r[-1] < L:
        r.append(r[-1] + step)
        step = min(step * ratio, h_max)
    return np.array(r)


def _mesh(mu_left, mu_right, depths, h, L=100.0, h_max=2.5):
    depths = np.asarray(depths, float)
    r = _graded(h, L, h_max)
    n_f = int(round(D / h))
    zf = -np.round(np.linspace(0.0, D, n_f + 1), 12)

    def mu_at(stack, z):
        return np.asarray(stack, float)[np.searchsorted(depths, -np.asarray(z), side="right")]

    rows = []

    def line(x, y, bc, mm, mp, v=0.0):
        n = len(x) - 1
        rows.append((x[:-1], y[:-1], x[1:], y[1:], [bc] * n, np.broadcast_to(mm, n), np.broadcast_to(mp, n),
                     np.broadcast_to(v, n)))

    zm = 0.5 * (zf[:-1] + zf[1:])
    line(np.zeros(n_f + 1), zf, "s", mu_at(mu_left, zm), mu_at(mu_right, zm),
         np.cos(0.5 * np.pi * -zm / D) ** 2)
    xs = np.r_[-r[:0:-1], r]
    top = np.where(0.5 * (xs[:-1] + xs[1:]) < 0, mu_left[0], mu_right[0])
    line(xs, np.zeros_like(xs), "t", top, top)
    zc = -r
    zcm = 0.5 * (zc[:-1] + zc[1:])
    line(np.zeros_like(zc), zc, "h", mu_at(mu_left, zcm), mu_at(mu_right, zcm))
    for k, d in enumerate(depths):
        line(-r[::-1], np.full(len(r), -d), "h", mu_left[k + 1], mu_left[k])
        line(r, np.full(len(r), -d), "h", mu_right[k + 1], mu_right[k])
    cols = [np.concatenate([np.asarray(rw[i]) for rw in rows]) for i in range(8)]
    els = bemcs.initialize_els()
    els.x1, els.y1, els.x2, els.y2 = (c.astype(float) for c in cols[:4])
    bemcs.standardize_els_geometry(els, reorder=False)
    return bemcs.Mesh(els, bc_type=cols[4], bc_value=cols[7].astype(float), mu_minus=cols[5].astype(float),
                      mu_plus=cols[6].astype(float), basis={"slip": "hat", "force": "trapezoid"})


def _image_series(x, mu1, mu2, n_quad=200):
    k = (mu1 - mu2) / (mu1 + mu2)
    m_max = int(np.ceil(np.log(1e-17) / np.log(abs(k)))) + 1

    def B(d):
        return np.arctan(d / x) / np.pi

    def G(d):
        if d < H1:
            return B(d) + sum(k**m * (B(2 * m * H1 + d) - B(2 * m * H1 - d)) for m in range(1, m_max))
        return (1 - k) * sum(k**m * B(d + 2 * m * H1) for m in range(m_max))

    u = np.zeros_like(x)
    g, w = np.polynomial.legendre.leggauss(n_quad)
    for lo, hi in ((0.0, H1), (H1, D)):
        for d, wd in zip(0.5 * (lo + hi) + 0.5 * (hi - lo) * g, 0.5 * (hi - lo) * w):
            u += wd * 0.5 * np.pi / D * np.sin(np.pi * d / D) * G(d)
    return u


def _error(c, h, **kw):
    mesh = _mesh([1.0, 4.0], [c, 4.0 * c], [H1], h)
    x = bemcs.solve(mesh, geometry="antiplane", mu=1.0, **kw)
    u = bemcs.evaluate_fields(mesh, x, XC, np.full_like(XC, -1e-9), geometry="antiplane", mu=1.0)["uz"]
    exact = np.where(XC < 0, 2 * c / (1 + c), 2 / (1 + c)) * _image_series(XC, 1.0, 4.0)
    return np.abs(u - exact).max()


def test_junction_branches_end_open():
    mesh = _mesh([1.0, 4.0], [2.0, 8.0], [H1], 0.1)
    np.testing.assert_allclose(mesh.force_junctions, [[0.0, -H1]])
    conn = mesh.connectivity["force"]
    half = conn[(conn[:, 0] == conn[:, 1]) & (conn[:, 1] == conn[:, 2]), 0]
    els = mesh.els
    at_junction = [e for e in half if np.isclose(min(np.hypot(els.x1[e], els.y1[e] + H1),
                                                     np.hypot(els.x2[e], els.y2[e] + H1)), 0.0)]
    assert len(at_junction) == 4  # one free end per branch: contact above/below, left, right
    assert any(np.isclose(els.y1[e], 0.0) for e in half)  # and the contact's end at the free surface


def test_lateral_contrast_on_a_layered_medium_matches_the_exact_solution():
    e1, e2 = _error(2.0, 0.1), _error(2.0, 0.1 / 3)  # contact segment surface-junction: 5, 15 elements
    assert e2 < 1.2e-2 and e1 / e2 > 2.5  # first order (example, L = 300: 2.5e-2 -> 8.6e-3)


def test_without_lateral_contrast_the_junction_is_a_plain_crossing():
    assert _error(1.0, 0.1) < 1e-3  # measured 6.9e-4 (h = 0.1, lines to 100)


def test_even_segments_between_breaks_warn_about_pinned_force():
    """Surface-junction segment of 10 elements (even): the node next to one end has no patch."""
    with pytest.warns(bemcs.PinnedForceWarning, match=r"\(0, -0.45\)|\(0, -0.05\)"):
        bemcs.solve(_mesh([1.0, 4.0], [2.0, 8.0], [H1], 0.05), geometry="antiplane", mu=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", bemcs.PinnedForceWarning)
        bemcs.solve(_mesh([1.0, 4.0], [2.0, 8.0], [H1], 0.1), geometry="antiplane", mu=1.0)  # 5: odd
