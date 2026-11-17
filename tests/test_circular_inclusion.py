"""Plane-strain heterogeneous solver at finite contrast: a circular inclusion under a
uniform remote stress against the exact (Muskhelishvili) solution. Also checks the
background-field machinery (`uniform_background`) and the near-coincident-node
warning (`MeshGapWarning`), which this case motivated.

Measured (stiff inclusion mu_i/mu_m = 4, soft 1/4, nu = 0.27, default trapezoids):
- stress error at the centre 5.5e-3 -> 2.1e-3 (stiff) and 6.6e-4 -> 4.0e-4 (soft)
  from 64 to 128 elements; 2.5e-4 -> 1.0e-4 (stiff) at 5 a;
- displacement error ~1e-3 near the interface, falling with distance.
The trapezoid basis converges at about first order on this curved, closed interface
(the antiplane inclusion behaves the same way).
"""

import warnings

import numpy as np
import pytest

import bemcs
from reference_solutions import (
    circular_inclusion_constants,
    circular_inclusion_planestrain,
    inclusion_stress_closed_form,
)
from meshes import INCLUSION_A as A, INCLUSION_MU_M as MU_M, INCLUSION_NU as NU  # noqa: E402
from meshes import INCLUSION_STRESS as STRESS, inclusion_mesh, ring  # noqa: E402

def _bem_fields(n, mu_i, X, Y):
    mesh = inclusion_mesh(n, mu_i)
    bg = bemcs.uniform_background(STRESS, geometry="planestrain", mu=1.0, nu=NU, mu_medium=MU_M)
    x = bemcs.solve(mesh, geometry="planestrain", mu=1.0, nu=NU, background=bg)

    def mu_field(x_, y_):
        return np.where(np.hypot(x_, y_) < A, mu_i, MU_M)

    f = bemcs.evaluate_fields(mesh, x, X, Y, geometry="planestrain", mu=1.0, nu=NU, mu_field=mu_field,
                              background=bg)
    return np.array([f[k] for k in ("ux", "uy", "sxx", "syy", "sxy")])


# ----------------------------------------------------------------------------
# the reference itself
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("mu_i", [4.0, 0.25, 1.0, 1e6])
def test_exact_solution_is_continuous_and_matches_the_closed_form(mu_i):
    _, residual = circular_inclusion_constants(A, STRESS, MU_M, mu_i, NU)
    assert residual < 1e-12  # the potentials satisfy continuity exactly on the circle
    X, Y = ring(0.5)
    f = circular_inclusion_planestrain(X, Y, A, STRESS, MU_M, mu_i, NU)
    np.testing.assert_allclose(f[2:], np.repeat(inclusion_stress_closed_form(STRESS, MU_M, mu_i, NU)[:, None],
                                                len(X), axis=1), rtol=1e-12)
    far = circular_inclusion_planestrain(np.array([1e5]), np.array([3e4]), A, STRESS, MU_M, mu_i, NU)
    np.testing.assert_allclose(far[2:, 0], STRESS, atol=1e-9)


def test_exact_solution_gives_kirsch_for_a_hole_and_is_elastic():
    f = circular_inclusion_planestrain(np.array([0.0, 1.0 + 1e-12]), np.array([1.0 + 1e-12, 0.0]),
                                       A, (1.0, 0.0, 0.0), MU_M, 1e-10, NU)
    np.testing.assert_allclose([f[2, 0], f[3, 1]], [3.0, -1.0], atol=1e-8)  # hoop stress
    h, mu_i = 1e-6, 4.0
    X, Y = np.array([1.7, -2.3, 0.3]), np.array([0.4, 1.1, -0.2])  # outside, outside, inside

    def F(x, y):
        return circular_inclusion_planestrain(x, y, A, STRESS, MU_M, mu_i, NU)

    dx, dy = (F(X + h, Y) - F(X - h, Y)) / (2 * h), (F(X, Y + h) - F(X, Y - h)) / (2 * h)
    mu = np.where(np.hypot(X, Y) < A, mu_i, MU_M)
    lam = 2 * mu * NU / (1 - 2 * NU)
    exx, eyy, exy = dx[0], dy[1], 0.5 * (dy[0] + dx[1])
    s = F(X, Y)
    np.testing.assert_allclose(lam * (exx + eyy) + 2 * mu * exx, s[2], atol=1e-8)
    np.testing.assert_allclose(lam * (exx + eyy) + 2 * mu * eyy, s[3], atol=1e-8)
    np.testing.assert_allclose(2 * mu * exy, s[4], atol=1e-8)
    np.testing.assert_allclose(dx[2] + dy[4], 0, atol=1e-7)
    np.testing.assert_allclose(dx[4] + dy[3], 0, atol=1e-7)


# ----------------------------------------------------------------------------
# BEM vs exact
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("mu_i, tol_in, tol_out", [(4.0, 1e-2, 1.5e-3), (0.25, 2e-3, 1.5e-3)])
def test_circular_inclusion_matches_the_exact_solution(mu_i, tol_in, tol_out):
    """128 elements: stresses inside (r <= 0.6 a) and outside (r >= 1.5 a), and
    displacements, against the exact solution."""
    X, Y = np.r_[ring(0.0, 1)[0], ring(0.3)[0], ring(0.6)[0], ring(1.5)[0], ring(2.5)[0], ring(5.0)[0]], \
        np.r_[ring(0.0, 1)[1], ring(0.3)[1], ring(0.6)[1], ring(1.5)[1], ring(2.5)[1], ring(5.0)[1]]
    got = _bem_fields(128, mu_i, X, Y)
    ref = circular_inclusion_planestrain(X, Y, A, STRESS, MU_M, mu_i, NU)
    inside = np.hypot(X, Y) < A
    assert np.abs(got[2:, inside] - ref[2:, inside]).max() < tol_in  # stiff: 7.6e-3, soft: 7.6e-4
    assert np.abs(got[2:, ~inside] - ref[2:, ~inside]).max() < tol_out  # 8.6e-4 and 9.4e-4
    assert np.abs(got[:2] - ref[:2]).max() < 1e-3  # 5.2e-4 and 7.5e-4


def test_circular_inclusion_converges():
    X, Y = np.r_[0.0, ring(2.5)[0]], np.r_[0.0, ring(2.5)[1]]
    ref = circular_inclusion_planestrain(X, Y, A, STRESS, MU_M, 4.0, NU)
    err = [np.abs(_bem_fields(n, 4.0, X, Y)[2:] - ref[2:]).max() for n in (32, 64, 128)]
    assert err[0] > err[1] > err[2]
    assert err[0] / err[2] > 3.0  # measured 1.6e-2, 5.5e-3, 2.1e-3 (about first order or better)


def test_no_contrast_gives_the_background_exactly():
    X, Y = np.r_[0.0, 0.4, 2.0], np.r_[0.0, -0.3, 1.0]
    got = _bem_fields(32, MU_M, X, Y)
    bg = bemcs.uniform_background(STRESS, geometry="planestrain", mu=1.0, nu=NU)(X, Y)
    np.testing.assert_allclose(got, np.array([bg[k] for k in ("ux", "uy", "sxx", "syy", "sxy")]),
                               atol=1e-13)


# ----------------------------------------------------------------------------
# background fields
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("geometry", ["planestrain", "antiplane"])
def test_background_equals_prescribing_minus_remote_traction(geometry):
    """A traction-free crack under a remote stress: prescribing t = -sigma_inf . n with
    no background, or t = 0 with the remote stress as a background, is the same solve."""
    els = bemcs.initialize_els()
    xs = np.linspace(-1, 1, 9)
    els.x1, els.x2 = xs[:-1] * np.cos(0.4), xs[1:] * np.cos(0.4)
    els.y1, els.y2 = xs[:-1] * np.sin(0.4), xs[1:] * np.sin(0.4)
    bemcs.standardize_els_geometry(els, reorder=False)
    n = np.c_[els.x_normals, els.y_normals]
    if geometry == "planestrain":
        sig = np.array([[0.3, 1.0], [1.0, -0.4]])
        t, stress = -n @ sig, (0.3, -0.4, 1.0)
    else:
        sig = np.array([0.8, -0.5])
        t, stress = -(n @ sig), sig
    kw = dict(mu_minus=1.0, mu_plus=1.0, basis={"slip": "hat"})
    m1 = bemcs.Mesh(els, bc_type=["t"] * 8, bc_value=t, **kw)
    m2 = bemcs.Mesh(els, bc_type=["t"] * 8, bc_value=np.zeros_like(t), **kw)
    x1 = bemcs.solve(m1, geometry=geometry, mu=1.0, nu=0.25)
    bg = bemcs.uniform_background(stress, geometry=geometry, mu=1.0, nu=0.25)
    x2 = bemcs.solve(m2, geometry=geometry, mu=1.0, nu=0.25, background=bg)
    np.testing.assert_allclose(x2, x1, atol=1e-12 * np.abs(x1).max())


def test_uniform_background_scales_to_the_medium():
    bg = bemcs.uniform_background((2.0, 0.0, 0.0), geometry="planestrain", mu=1.0, nu=0.25, mu_medium=4.0)
    f = bg(np.array([1.0]), np.array([0.0]))
    assert f["sxx"][0] == pytest.approx(0.5)  # reference stress: true stress * mu / mu_medium
    # displacement: plane-strain strain of the true stress in the true medium
    assert f["ux"][0] == pytest.approx(2.0 * (1 - 0.25**2) / (2 * 4.0 * (1 + 0.25)))
    with pytest.raises(ValueError, match="plane-strain background"):
        bemcs.uniform_background((1.0, 0.0), geometry="planestrain", mu=1.0, nu=0.25)


# ----------------------------------------------------------------------------
# the meshing mistake this case exposed: a loop that is not exactly closed
# ----------------------------------------------------------------------------


def test_nearly_closed_loop_warns_and_degrades():
    t = np.linspace(0, 2 * np.pi, 33)  # cos/sin(2 pi) are not exactly (1, 0): a 2e-16 gap
    els = bemcs.initialize_els()
    els.x1, els.y1 = np.cos(-t[:-1]), np.sin(-t[:-1])
    els.x2, els.y2 = np.cos(-t[1:]), np.sin(-t[1:])
    bemcs.standardize_els_geometry(els, reorder=False)
    with pytest.warns(bemcs.MeshGapWarning, match="nearly coincide"):
        mesh = bemcs.Mesh(els, bc_type=["h"] * 32, mu_minus=4.0, mu_plus=MU_M, basis={"force": "trapezoid"})
    # treated as an open chain: one trapezoid short, and the inclusion stress far off
    assert len(mesh.split_force_trapezoids()) == 15
    bg = bemcs.uniform_background(STRESS, geometry="planestrain", mu=1.0, nu=NU)
    x = bemcs.solve(mesh, geometry="planestrain", mu=1.0, nu=NU, background=bg)
    f = bemcs.evaluate_fields(mesh, x, np.array([0.0]), np.array([0.0]), geometry="planestrain", mu=1.0, nu=NU,
                              mu_field=lambda x_, y_: np.full(np.shape(x_), 4.0), background=bg)
    exact = inclusion_stress_closed_form(STRESS, MU_M, 4.0, NU)
    assert abs(f["sxx"][0] - exact[0]) > 0.05  # vs 1.6e-2 for the closed loop
    # an exactly closed loop does not warn
    with warnings.catch_warnings():
        warnings.simplefilter("error", bemcs.MeshGapWarning)
        inclusion_mesh(32, 4.0)
