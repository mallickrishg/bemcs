"""T1 validation of the heterogeneous (equivalent body force) formulation.

- Limits of a bimaterial interface, against an independent homogeneous BEM
  solve on the same line: mu_plus / mu_minus -> infinity is a rigid boundary
  (u = 0) and -> 0 a free surface (traction-free). Compared by stresses below
  the interface (displacements of a dislocation decay slowly, so they also
  reflect the different truncation of the two finite meshes). Default
  trapezoids converge to first order in h.
- Orientation invariance: reversing the interface elements (n -> -n) while
  swapping mu_minus <-> mu_plus describes the same material and must give the
  same fields.
- Mirror symmetry: the bimaterial reflected about the interface, with the
  materials following the reflection, gives mirrored fields.

The exact antiplane image solution and plane-strain traction continuity are in
test_solve.py.
"""

import numpy as np
import pytest

import bemcs
from test_solve import _fault_and_interface

NU = 0.27
POINTS = (np.array([0.7, -0.5, 0.3, -2.0, 1.2]), np.array([-1.0, -0.3, -2.4, -2.5, -0.05]))
STRESSES = {"antiplane": ("sxz", "syz"), "planestrain": ("sxx", "syy", "sxy")}
DISPS = {"antiplane": ("uz",), "planestrain": ("ux", "uy")}


def _slip_bc(n_f, n_i, geometry):
    value = np.zeros((n_f + n_i, 2))
    value[:n_f, 0] = 1.0
    if geometry == "planestrain":
        value[:n_f, 1] = 0.3
    return value


def _stresses(mesh, x, geometry):
    f = bemcs.evaluate_fields(mesh, x, *POINTS, geometry=geometry, mu=1.0, nu=NU)
    return np.stack([f[k] for k in STRESSES[geometry]])


def _limit_mismatch(geometry, ratio, n_int):
    els, n_f, n_i = _fault_and_interface(L=10.0, n_int=n_int)
    value = _slip_bc(n_f, n_i, geometry)
    het = bemcs.Mesh(
        els, bc_type=["s"] * n_f + ["h"] * n_i, bc_value=value, mu_minus=1.0,
        mu_plus=np.r_[np.ones(n_f), np.full(n_i, ratio)], basis={"slip": "hat", "force": "trapezoid"},
    )
    boundary = "u" if ratio > 1 else "t"  # rigid: u = 0; free: traction-free
    hom = bemcs.Mesh(els, bc_type=["s"] * n_f + [boundary] * n_i, bc_value=value,
                     mu_minus=1.0, mu_plus=1.0, basis={"slip": "hat"})
    kw = dict(geometry=geometry, mu=1.0, nu=NU)
    a = _stresses(het, bemcs.solve(het, **kw), geometry)
    b = _stresses(hom, bemcs.solve(hom, **kw), geometry)
    return np.abs(a - b).max() / np.abs(b).max()


@pytest.mark.parametrize("limit, ratio", [("rigid", 1e4), ("free surface", 1e-4)])
@pytest.mark.parametrize("geometry", ["antiplane", "planestrain"])
def test_high_contrast_limits_match_rigid_and_free_boundaries(geometry, limit, ratio):
    coarse = _limit_mismatch(geometry, ratio, 200)  # h = 0.1
    fine = _limit_mismatch(geometry, ratio, 400)  # h = 0.05
    # measured (h = 0.1 -> 0.05): plane strain rigid 5.0e-3 -> 2.5e-3, free 7.3e-2 -> 3.1e-2;
    # antiplane rigid ~1.3e-3, free 5.2e-3 -> 4.2e-3 (a floor from the reference solve)
    assert fine < (0.05 if geometry == "planestrain" else 0.01)
    if geometry == "planestrain":
        assert fine < 0.6 * coarse  # first-order convergence


def _bimaterial(geometry, mu_below, mu_above, *, reverse_interface=False, mirror=False, n_int=200):
    els, n_f, n_i = _fault_and_interface(L=10.0, n_int=n_int)
    x1, y1, x2, y2 = (np.array(getattr(els, k)) for k in ("x1", "y1", "x2", "y2"))
    if reverse_interface:  # interface elements run right to left: normal points down
        sl = slice(n_f, n_f + n_i)
        x1[sl], x2[sl] = x2[sl].copy(), x1[sl].copy()
        y1[sl], y2[sl] = y2[sl].copy(), y1[sl].copy()
    if mirror:  # reflect about y = 0 and reverse every element (keeps the orientation sense)
        x1, y1, x2, y2 = x2, -y2, x1, -y1
    e = bemcs.initialize_els()
    e.x1, e.y1, e.x2, e.y2 = x1, y1, x2, y2
    bemcs.standardize_els_geometry(e, reorder=False)
    up = e.y_normals[n_f:] > 0  # interface normal: +n is above the interface
    mu_minus_i = np.where(up, mu_below, mu_above) if not mirror else np.where(up, mu_above, mu_below)
    mu_plus_i = np.where(up, mu_above, mu_below) if not mirror else np.where(up, mu_below, mu_above)
    mu_fault = mu_below
    value = _slip_bc(n_f, n_i, geometry)
    if mirror and geometry == "planestrain":
        # reflected and reversed elements: t' = -R t, n' = R n, so the mirror image of the
        # original slip has the opposite local shear component
        value[:, 0] *= -1
    mesh = bemcs.Mesh(
        e, bc_type=["s"] * n_f + ["h"] * n_i, bc_value=value,
        mu_minus=np.r_[np.full(n_f, mu_fault), mu_minus_i], mu_plus=np.r_[np.full(n_f, mu_fault), mu_plus_i],
        basis={"slip": "hat", "force": "trapezoid"},
    )
    return mesh, bemcs.solve(mesh, geometry=geometry, mu=1.0, nu=NU)


@pytest.mark.parametrize("geometry", ["antiplane", "planestrain"])
def test_reversing_interface_elements_and_swapping_moduli_changes_nothing(geometry):
    """(d) mu1 <-> mu2 symmetry of the sign handling: the same material described with
    the opposite interface normal."""
    m1, x1 = _bimaterial(geometry, 4.0, 1.5)
    m2, x2 = _bimaterial(geometry, 4.0, 1.5, reverse_interface=True)
    kw = dict(geometry=geometry, mu=1.0, nu=NU)
    f1 = bemcs.evaluate_fields(m1, x1, *POINTS, **kw)
    f2 = bemcs.evaluate_fields(m2, x2, *POINTS, **kw)
    for k in DISPS[geometry] + STRESSES[geometry]:
        np.testing.assert_allclose(f2[k], f1[k], atol=1e-10 * np.abs(f1[k]).max())


@pytest.mark.parametrize("geometry", ["antiplane", "planestrain"])
def test_mirrored_bimaterial_gives_mirrored_fields(geometry):
    """Reflect the whole problem about the interface (y -> -y, materials follow):
    the fields are the reflection of the original ones."""
    m1, x1 = _bimaterial(geometry, 4.0, 1.5)
    m2, x2 = _bimaterial(geometry, 4.0, 1.5, mirror=True)
    kw = dict(geometry=geometry, mu=1.0, nu=NU)
    X, Y = POINTS
    f1 = bemcs.evaluate_fields(m1, x1, X, Y, **kw)
    f2 = bemcs.evaluate_fields(m2, x2, X, -Y, **kw)
    # reflection R = diag(1, -1): u'(x) = R u(R x) and sigma' = R sigma R (plane strain:
    # ux, sxx, syy keep their sign, uy and sxy flip); antiplane u_z'(x) = u_z(R x), so
    # u_z and sxz keep their sign and syz flips
    signs = {"uz": 1, "sxz": 1, "syz": -1, "ux": 1, "uy": -1, "sxx": 1, "syy": 1, "sxy": -1}
    for k in DISPS[geometry] + STRESSES[geometry]:
        np.testing.assert_allclose(f2[k], signs[k] * f1[k], atol=1e-9 * np.abs(f1[k]).max(), err_msg=k)
