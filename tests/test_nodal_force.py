"""The nodal force basis (`basis={"force": "nodal"}`): continuous piecewise-linear force with
one unknown (a global force vector) per node, collocated at the nodes (tasks.md P1d).

- Straight interfaces: second order in plane strain (the trapezoids are first order there:
  the force is constant over their collocated element, which the tangential-normal 1/x
  coupling turns into an O(h) error). Dundurs case: 1.0e-2 -> 2.6e-3 -> 6.5e-4.
- Break points (fault crossings, open force ends, overlap ends of non-"s" slip) split the
  force automatically: a half-hat on each side, collocated at its element's centre.
- Kinks (curved interfaces as polygons): a node is sampled `kink_offset` element lengths off
  the interface (the straight-element force layer is log-singular at a kink node). First order,
  better than the trapezoids.
- Overlapping slip and force meshes: an integer number M >= 1 of force elements per slip element
  is recommended; otherwise `OverlapRefinementWarning`.
"""

import warnings

import numpy as np
import pytest

import bemcs
import bemcs.bemAssembly as GF
from reference_solutions import circular_inclusion_planestrain
from meshes import (CONTACT_D, CONTACT_MU1, CONTACT_MU2, CONTACT_SURFACE, DUNDURS_MU1, DUNDURS_NU, DUNDURS_SLIP,
                    INCLUSION_A, INCLUSION_MU_M, INCLUSION_NU, INCLUSION_STRESS, LAYERED_H, LAYERED_MU1,
                    LAYERED_MU2, LAYERED_SURFACE, contact_mesh, dundurs_fault, dundurs_mesh, dundurs_points,
                    inclusion_mesh, layered_mesh, ring)
from reference_solutions import contact_displacement, dundurs_fault_stress, layered_image_series


def _rebase(mesh, force_basis="nodal"):
    return bemcs.Mesh(mesh.els, bc_type=mesh.bc_type, bc_value=mesh.bc_value, mu_minus=mesh.mu_minus,
                      mu_plus=mesh.mu_plus, basis={**mesh.basis, "force": force_basis})


def _line(n, x0=0.0, y=0.0, dx=1.0):
    els = bemcs.initialize_els()
    els.x1, els.x2 = x0 + dx * np.arange(n), x0 + dx * np.arange(1, n + 1)
    els.y1, els.y2 = np.full(n, y), np.full(n, y)
    bemcs.standardize_els_geometry(els, reorder=False)
    return els


# ----------------------------------------------------------------------------
# layout and transform
# ----------------------------------------------------------------------------


def test_layout_of_a_straight_chain():
    lay = GF.nodal_force_layout(_line(5))
    assert list(lay["kind"]) == ["node"] * 4  # interior nodes; tapered tips have no unknown
    np.testing.assert_allclose(lay["x"], [1, 2, 3, 4])
    np.testing.assert_allclose(np.c_[lay["nx"], lay["ny"]], np.tile([0.0, 1.0], (4, 1)))
    assert np.all(lay["kink"] == 0)


def test_breaks_give_half_hats_collocated_at_element_centres():
    lay = GF.nodal_force_layout(_line(6), breaks=[[3.0, 0.0]])  # interior break
    ends = lay["kind"] == "end"
    np.testing.assert_allclose(lay["x"][ends], [2.5, 3.5])
    lay = GF.nodal_force_layout(_line(6), breaks=[[0.0, 0.0]])  # break at a chain tip: open end
    assert list(lay["kind"]).count("end") == 1 and lay["x"][lay["kind"] == "end"][0] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="two half-hats"):
        GF.nodal_force_layout(_line(6), breaks=[[2.0, 0.0], [3.0, 0.0]])  # both ends of one element


def test_nodal_transform_is_the_unit_global_force_at_a_node():
    els = bemcs.initialize_els()  # a kink at x = 2: the third element goes up-right
    els.x1, els.x2 = np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0])
    els.y1, els.y2 = np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    bemcs.standardize_els_geometry(els, reorder=False)
    lay = GF.nodal_force_layout(els)
    assert lay["kink"][1] > 0.1
    T = GF.nodal_force_coefficients(els, lay, geometry="planestrain")  # (2n, 2 local, n_dof, 2 global)
    R = np.asarray(els.rot_mats[2])
    # node at x = 2 is element 1's end (ramp 0) and element 2's start (ramp 1)
    np.testing.assert_allclose(T[2 * 2 + 1, :, 1, :], R.T)
    np.testing.assert_allclose(T[2 * 1 + 0, :, 1, :], np.eye(2))


# ----------------------------------------------------------------------------
# accuracy
# ----------------------------------------------------------------------------


def _dundurs_error(case, h, mu2, force_basis):
    X, Y = dundurs_points(case)
    ref = dundurs_fault_stress(X, Y, *dundurs_fault(case), DUNDURS_SLIP, DUNDURS_MU1, mu2, DUNDURS_NU)
    mesh = _rebase(dundurs_mesh(case, h, mu2), force_basis)
    x = bemcs.solve(mesh, geometry="planestrain", mu=1.0, nu=DUNDURS_NU)
    f = bemcs.evaluate_fields(mesh, x, X, Y, geometry="planestrain", mu=1.0, nu=DUNDURS_NU,
                              mu_field=lambda x_, y_: np.where(y_ > 0, mu2, DUNDURS_MU1))
    return np.abs(np.array([f["sxx"], f["syy"], f["sxy"]]) - ref).max() / np.abs(ref).max()


@pytest.mark.parametrize("case", ["horizontal", "dipping"])
@pytest.mark.parametrize("mu2", [4.0, 0.25])
def test_nodal_force_is_second_order_on_the_dundurs_case(case, mu2):
    e1, e2 = _dundurs_error(case, 0.1, mu2, "nodal"), _dundurs_error(case, 0.05, mu2, "nodal")
    assert e2 < 3e-3 and e1 / e2 > 3.3  # measured 2.6e-3 / 1.1e-3 / 1.4e-3 / 9.2e-4, ratios ~3.9
    assert e2 < 0.5 * _dundurs_error(case, 0.05, mu2, "trapezoid")


def test_nodal_force_restarts_at_a_fault_crossing_automatically():
    """Antiplane layered half-space, fault through the interface: no connectivity needed."""
    errs = []
    for h in (0.1, 0.05):
        mesh = _rebase(layered_mesh(1.63, h, lopsided=True))  # the plain mesh; nodal ignores trapezoid layouts
        x = bemcs.solve(mesh, geometry="antiplane", mu=1.0)
        u = bemcs.evaluate_fields(mesh, x, LAYERED_SURFACE, np.full_like(LAYERED_SURFACE, -1e-9), geometry="antiplane", mu=1.0)["uz"]
        errs.append(np.abs(u - layered_image_series(LAYERED_SURFACE, 1.63, LAYERED_H, LAYERED_MU1, LAYERED_MU2)).max())
    assert errs[1] < 5e-5  # measured 1.1e-4 -> 3.3e-5


def test_nodal_force_on_a_curved_interface_needs_the_kink_offset():
    X, Y = np.r_[0.0, ring(2.5)[0]], np.r_[0.0, ring(2.5)[1]]
    ref = circular_inclusion_planestrain(X, Y, INCLUSION_A, INCLUSION_STRESS, INCLUSION_MU_M, 4.0, INCLUSION_NU)

    def err(n, **kw):
        mesh = _rebase(inclusion_mesh(n, 4.0))
        bg = bemcs.uniform_background(INCLUSION_STRESS, geometry="planestrain", mu=1.0, nu=INCLUSION_NU, mu_medium=INCLUSION_MU_M)
        x = bemcs.solve(mesh, geometry="planestrain", mu=1.0, nu=INCLUSION_NU, background=bg, **kw)
        f = bemcs.evaluate_fields(mesh, x, X, Y, geometry="planestrain", mu=1.0, nu=INCLUSION_NU, background=bg,
                                  mu_field=lambda a, b: np.where(np.hypot(a, b) < INCLUSION_A, 4.0, INCLUSION_MU_M))
        return np.abs(np.array([f["sxx"], f["syy"], f["sxy"]]) - ref[2:]).max()

    e64, e128 = err(64), err(128)
    assert e128 < 2e-3 and e64 / e128 > 1.6  # measured 2.4e-3 -> 1.3e-3 (incl. r = 0.6): first order
    assert err(64, kink_offset=0.0) > 10 * e64  # sampling at the kink nodes themselves fails


def test_nodal_force_at_a_contact_meeting_the_free_surface():
    """Rybicki: the same as the trapezoids with an open end (corner-limited, tasks.md P3)."""
    mesh = _rebase(contact_mesh(0.5, 0.05))
    x = bemcs.solve(mesh, geometry="antiplane", mu=1.0)
    u = bemcs.evaluate_fields(mesh, x, CONTACT_SURFACE, np.full_like(CONTACT_SURFACE, -1e-9), geometry="antiplane", mu=1.0)["uz"]
    assert np.abs(u - contact_displacement(CONTACT_SURFACE, 0.5, 0.0, CONTACT_D, CONTACT_MU1, CONTACT_MU2)).max() < 8e-3  # measured 6.1e-3


# ----------------------------------------------------------------------------
# overlapping meshes with a denser force mesh
# ----------------------------------------------------------------------------

A_C, TAU = 1.0, 1.0


def _crack_on_interface(h, M, mu_below=4.0, mu_above=1.0, L=300.0):
    """Crack ("t", spacing h) on [-1, 1] on an interface ("h", spacing h / M on [-2, 2], then
    log-spaced). Rounded coordinates make shared nodes identical."""
    xc = np.round(np.linspace(-A_C, A_C, int(round(2 * A_C / h)) + 1), 12)
    hf = h / M
    inner = np.round(np.linspace(-2.0, 2.0, int(round(4.0 / hf)) + 1), 12)
    r, step = [inner[-1]], hf
    while r[-1] < L:
        r.append(r[-1] + step)
        step = min(step * 1.08, 5.0)
    out = np.array(r[1:])
    xi = np.r_[-out[::-1], inner, out]
    nc, ni = len(xc) - 1, len(xi) - 1
    els = bemcs.initialize_els()
    els.x1, els.x2 = np.r_[xc[:-1], xi[:-1]], np.r_[xc[1:], xi[1:]]
    els.y1, els.y2 = np.zeros(nc + ni), np.zeros(nc + ni)
    bemcs.standardize_els_geometry(els, reorder=False)
    return bemcs.Mesh(els, bc_type=["t"] * nc + ["h"] * ni, bc_value=np.r_[np.full(nc, -TAU), np.zeros(ni)],
                      mu_minus=mu_below, mu_plus=mu_above, basis={"slip": "hat", "force": "nodal"})


def _crack_error(mesh, mu_below=4.0, mu_above=1.0):
    x = bemcs.solve(mesh, geometry="antiplane", mu=1.0)
    xq = np.linspace(-0.8, 0.8, 9)
    exact = TAU * (1 / mu_below + 1 / mu_above) * np.sqrt(A_C**2 - xq**2)
    xe, _, s = mesh.values_on_elements(x, "slip", geometry="antiplane", n_eval=40)
    xe, s = np.ravel(xe), np.ravel(s)
    o = np.argsort(xe)
    return np.abs(np.interp(xq, xe[o], s[o]) - exact).max() / exact.max()


def test_denser_force_mesh_under_a_crack_makes_the_contrast_free():
    """M = 2: the error equals that of the same crack without contrast (5.8e-2 at h = 0.05,
    from the crack tips); M = 1: 8.1e-2. No warnings for integer M (no proximity warning
    either, although force nodes lie inside slip elements)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", bemcs.OverlapRefinementWarning)
        warnings.simplefilter("error", bemcs.CollocationProximityWarning)
        e_m2 = _crack_error(_crack_on_interface(0.05, 2))
        e_m1 = _crack_error(_crack_on_interface(0.05, 1))
        e_hom = _crack_error(_crack_on_interface(0.05, 2, 1.0, 1.0), 1.0, 1.0)
    assert e_m2 < 1.05 * e_hom and e_m2 < 0.8 * e_m1


def test_non_integer_force_refinement_warns():
    with pytest.warns(bemcs.OverlapRefinementWarning, match="integer number of force elements"):
        mesh = _crack_on_interface(0.05, 1.5)
    assert len(GF.overlap_refinement_issues(mesh.els, mesh.labels)) > 0
    assert _crack_error(mesh) < 0.08  # measured 6.4e-2: worse than M = 2, but stable here
