"""Slip and force meshes that overlap: a fault or crack lying on a material interface.

Where slip elements ("s" prescribed slip, or "t" traction-loaded) coincide with force
elements ("h" interface), and where the overlap ends, the code must:
- solve (topology is per source type, so coincident nodes are fine);
- not warn inside an overlap, but report overlap ends where the force regime changes
  (`Mesh.trapezoid_restart_points`, `TrapezoidCrossingWarning`);
- converge to the exact solution.

Exact reference (antiplane): a crack on a bimaterial interface under remote shear tau.
The bonded bimaterial carries a uniform tau that satisfies the interface conditions, so
the crack problem is the crack faces loaded with -tau, and the crack slip is
tau (1/mu1 + 1/mu2) sqrt(a^2 - x^2) (homogeneous: 2 tau / mu sqrt(...)).
"""

import warnings

import numpy as np
import pytest

import bemcs
import bemcs.bemAssembly as GF

A, TAU, SIG, NU = 1.0, 1.0, 1.0, 0.25


def _graded(h, L=300.0, flat=2.0, ratio=1.08, h_max=5.0):
    r, step = list(np.arange(0.0, flat + 1e-12, h)), h
    while r[-1] < L:
        r.append(r[-1] + step)
        step = min(step * ratio, h_max)
    r = np.array(r)
    return np.r_[-r[:0:-1], r]


def _interface_crack(h, mu_below, mu_above, geometry, split=True):
    """Crack on [-A, A] ("t", faces loaded with -remote traction) on a straight interface
    y = 0 ("h", out to +-300, log-spaced). Elements run left to right: -n is below."""
    xi = _graded(h)
    xc = xi[np.abs(xi) <= A + 1e-12]
    nc, ni = len(xc) - 1, len(xi) - 1
    els = bemcs.initialize_els()
    els.x1, els.x2 = np.r_[xc[:-1], xi[:-1]], np.r_[xc[1:], xi[1:]]
    els.y1, els.y2 = np.zeros(nc + ni), np.zeros(nc + ni)
    bemcs.standardize_els_geometry(els, reorder=False)
    if geometry == "antiplane":
        value = np.r_[np.full(nc, -TAU), np.zeros(ni)]
    else:
        value = np.r_[np.tile([-TAU, -SIG], (nc, 1)), np.zeros((ni, 2))]
    kw = dict(bc_type=["t"] * nc + ["h"] * ni, bc_value=value, mu_minus=mu_below, mu_plus=mu_above,
              basis={"slip": "hat", "force": "trapezoid"})
    mesh = bemcs.Mesh(els, **kw)
    if split:
        mesh = bemcs.Mesh(els, connectivity={"force": mesh.split_force_trapezoids()}, **kw)
    return mesh


def _crack_slip(mesh, x, geometry, xq):
    xe, _, s = mesh.values_on_elements(x, "slip", geometry=geometry, n_eval=40, frame="local")
    xe = np.ravel(xe)
    o = np.argsort(xe)
    s = np.asarray(s).reshape(len(xe), -1)[o]
    return np.stack([np.interp(xq, xe[o], s[:, k]) for k in range(s.shape[1])], axis=1)


XQ = np.linspace(-0.8, 0.8, 9)


def _antiplane_error(h, mu_below, mu_above, split=True):
    mesh = _interface_crack(h, mu_below, mu_above, "antiplane", split)
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        x = bemcs.solve(mesh, geometry="antiplane", mu=1.0)
    exact = TAU * (1 / mu_below + 1 / mu_above) * np.sqrt(A**2 - XQ**2)
    slip = _crack_slip(mesh, x, "antiplane", XQ)[:, 0]
    return np.abs(slip - exact).max() / exact.max(), {w.category for w in wl}


# ----------------------------------------------------------------------------
# classification of nodes where the meshes touch
# ----------------------------------------------------------------------------


def test_overlap_interior_is_not_a_crossing_but_overlap_ends_are():
    mesh = _interface_crack(0.25, 4.0, 1.0, "antiplane", split=False)
    n_crack = int(np.sum(mesh.bc_type[:, 0] == "t"))
    coincident = GF.coincident_force_elements(mesh.els, mesh.labels)
    assert len(coincident) == n_crack  # the interface elements under the crack
    np.testing.assert_allclose(mesh.fault_crossings, [[-A, 0.0], [A, 0.0]])  # the tips only
    np.testing.assert_allclose(mesh.trapezoid_restart_points(), [[-A, 0.0], [A, 0.0]])
    assert len(mesh.force_open_ends) == 0  # the interface continues past the crack


def test_overlap_ends_of_prescribed_slip_faults_need_no_restart():
    """A prescribed-slip ("s") fault tapers smoothly where it ends on the interface (zero slip
    and slope): no restart there, and no warning. The same geometry as a traction ("t")
    crack restarts there."""
    mesh_t = _interface_crack(0.25, 4.0, 1.0, "antiplane", split=False)
    bc = np.where(mesh_t.bc_type == "t", "s", mesh_t.bc_type)
    mesh_s = bemcs.Mesh(mesh_t.els, bc_type=bc[:, 0], bc_value=np.r_[np.full(8, 0.5), np.zeros(len(bc) - 8)],
                        mu_minus=4.0, mu_plus=1.0, basis={"slip": "hat", "force": "trapezoid"})
    assert len(mesh_s.fault_crossings) == 2 and len(mesh_s.trapezoid_restart_points()) == 0
    with warnings.catch_warnings():
        warnings.simplefilter("error", bemcs.TrapezoidCrossingWarning)
        bemcs.solve(mesh_s, geometry="antiplane", mu=1.0)


def test_label_nodes_on_the_whole_overlapping_mesh_raises_but_solve_does_not_use_it():
    """bemcs.label_nodes cannot label 2 slip + 2 force elements at one node; the solver works
    on each source type separately, so it is unaffected."""
    mesh = _interface_crack(0.25, 4.0, 1.0, "antiplane")
    with pytest.raises(Exception, match="overlapping nodes|more than 3 lines"):
        bemcs.label_nodes(mesh.els)
    x = bemcs.solve(mesh, geometry="antiplane", mu=1.0)
    assert np.isfinite(x).all()


# ----------------------------------------------------------------------------
# antiplane interface crack vs exact
# ----------------------------------------------------------------------------


def test_antiplane_interface_crack_matches_the_exact_slip():
    """With the stride restarted at the crack tips, the contrast adds almost nothing to the
    error of the crack itself (which comes from the clamped tips vs the sqrt crack tip):
    4x contrast 1.4e-1, 6.4e-2, 3.0e-2 vs no contrast 1.2e-1, 5.8e-2, 2.9e-2."""
    e_hom = [_antiplane_error(h, 1.0, 1.0)[0] for h in (0.1, 0.05)]
    e_bi = [_antiplane_error(h, 4.0, 1.0)[0] for h in (0.1, 0.05)]
    assert e_bi[1] < 1.25 * e_hom[1] and e_bi[1] < 0.08
    assert e_bi[0] / e_bi[1] > 1.8  # converges (first order, as the crack alone)


def test_antiplane_interface_crack_warns_without_the_split():
    err_default, cats = _antiplane_error(0.05, 4.0, 1.0, split=False)
    assert bemcs.TrapezoidCrossingWarning in cats
    err_split, cats = _antiplane_error(0.05, 4.0, 1.0, split=True)
    assert bemcs.TrapezoidCrossingWarning not in cats
    assert err_split < 0.85 * err_default  # measured 6.4e-2 vs 8.2e-2


# ----------------------------------------------------------------------------
# plane strain
# ----------------------------------------------------------------------------


def test_planestrain_coincident_meshes_without_contrast_give_griffith():
    """Equal moduli: the interface carries no force, and the crack slip is Griffith's,
    2 (1 - nu) / mu * sqrt(a^2 - x^2) times tau (shear) and sigma (opening)."""
    mesh = _interface_crack(0.05, 1.0, 1.0, "planestrain")
    x = bemcs.solve(mesh, geometry="planestrain", mu=1.0, nu=NU)
    np.testing.assert_allclose(x[mesh.columns("force", geometry="planestrain")], 0.0, atol=1e-12)
    slip = _crack_slip(mesh, x, "planestrain", XQ)
    g = 2 * (1 - NU) * np.sqrt(A**2 - XQ**2)
    assert np.abs(slip[:, 0] - TAU * g).max() / (TAU * g).max() < 0.07  # measured 5.8e-2 (the crack tips)
    assert np.abs(slip[:, 1] - SIG * g).max() / (SIG * g).max() < 0.07


def test_planestrain_interface_crack_converges_and_keeps_traction_continuous():
    """4x contrast (no simple exact solution: the near-tip field oscillates). The crack slip
    converges (changes 7.3e-2, 3.2e-2 for h = 0.1 -> 0.05 -> 0.025) and the true traction is
    continuous at the collocation points of the bonded interface."""
    slips = {}
    for h in (0.1, 0.05, 0.025):
        mesh = _interface_crack(h, 4.0, 1.0, "planestrain")
        x, (_, _, rows) = bemcs.solve(mesh, geometry="planestrain", mu=1.0, nu=NU, return_system=True)
        slips[h] = _crack_slip(mesh, x, "planestrain", XQ)
    d1 = np.abs(slips[0.1] - slips[0.05]).max()
    d2 = np.abs(slips[0.05] - slips[0.025]).max()
    assert d2 < 0.6 * d1
    xk = mesh.els.x_centers[rows["element"][rows["type"] == "h"]]
    xk = xk[(np.abs(xk) > A) & (np.abs(xk) < 3)]

    def mu_field(x_, y_):
        return np.where(y_ < 0, 4.0, 1.0)

    kw = dict(geometry="planestrain", mu=1.0, nu=NU, mu_field=mu_field)
    above = bemcs.evaluate_fields(mesh, x, xk, np.full_like(xk, 1e-7), **kw)
    below = bemcs.evaluate_fields(mesh, x, xk, np.full_like(xk, -1e-7), **kw)
    scale = np.abs(above["syy"]).max()
    assert np.abs(above["syy"] - below["syy"]).max() < 1e-4 * scale
    assert np.abs(above["sxy"] - below["sxy"]).max() < 1e-4 * scale
