"""`get_greens_functions`: Green's functions of a whole (heterogeneous) mesh.

One source per prescribed-slip row (unit slip at a hat's collocation point, zero at the
others, every other condition homogeneous). Checked against `solve` + `evaluate_fields`.
"""

import os
import warnings

import numpy as np
import pytest

import bemcs
from meshes import (CROSSING_NU, LAYERED_H, LAYERED_MU1, LAYERED_MU2, crossing_mesh, free_surface_fault,
                    layered_mesh)
from reference_solutions import layered_image_series

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples", "antiplane")


def _quiet(f, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*a, **k)


def _hats(mesh):
    """The same mesh with the hat slip basis (get_greens_functions' sources are slip hats)."""
    return bemcs.Mesh(mesh.els, labels=mesh.labels, bc_type=mesh.bc_type, bc_value=mesh.bc_value,
                      mu_minus=mesh.mu_minus, mu_plus=mesh.mu_plus, basis={**mesh.basis, "slip": "hat"},
                      connectivity={k: v for k, v in mesh.connectivity.items() if k == "force" and v is not None})


def _with_slip(mesh, slip_rows, values):
    """The mesh with prescribed slip `values` at the source rows and zero on every other condition."""
    v = np.zeros_like(mesh.bc_value)
    v[slip_rows] = values
    return bemcs.Mesh(mesh.els, labels=mesh.labels, bc_type=mesh.bc_type, bc_value=v, mu_minus=mesh.mu_minus,
                      mu_plus=mesh.mu_plus, basis=mesh.basis, connectivity=mesh.connectivity)


CASES = {
    "antiplane layered, fault through the interface": (lambda: _hats(layered_mesh(1.63, 0.1)), "antiplane", {},
                                                       (np.linspace(-3, 3, 13), np.full(13, -0.3))),
    "planestrain bimaterial, fault crossing the interface": (lambda: crossing_mesh(60, 0.1, 4.0), "planestrain",
                                                             {"nu": CROSSING_NU},
                                                             (np.linspace(-2, 2, 9), np.full(9, 0.4))),
    "planestrain fault breaking a free surface": (free_surface_fault, "planestrain", {"nu": 0.25},
                                                  (np.linspace(-2, 2, 9), np.full(9, -0.6))),
}


@pytest.mark.parametrize("case", list(CASES))
def test_greens_functions_reproduce_the_full_solve(case):
    make, geometry, kw, (xo, yo) = CASES[case]
    mesh = make()
    gf = _quiet(bemcs.get_greens_functions, mesh, xo, yo, geometry=geometry, **kw)
    x = _quiet(bemcs.solve, mesh, geometry=geometry, **kw)
    ref = bemcs.evaluate_fields(mesh, x, xo, yo, geometry=geometry, **kw)
    s = gf.amplitudes(mesh.bc_value[gf.sources["element"], gf.sources["component"]])
    got = gf(s)
    for name in ref:
        np.testing.assert_allclose(got[name], ref[name], atol=1e-9 * np.abs(ref[name]).max())
    np.testing.assert_allclose(gf.solution(s), x, atol=1e-10 * np.abs(x).max())
    # the fault slip of that solution, as element coefficients and averages
    c = mesh.element_coefficients(x, "slip", geometry=geometry).reshape(len(mesh.elements("slip")), 3, -1)
    pos = np.searchsorted(mesh.elements("slip"), gf.fault_elements)
    np.testing.assert_allclose(gf.slip_coefficients @ s, c[pos], atol=1e-12)
    np.testing.assert_allclose(gf.slip_average @ s, np.einsum("k,ekc->ec", [3 / 8, 1 / 4, 3 / 8], c[pos]),
                               atol=1e-12)


@pytest.mark.parametrize("case", list(CASES)[:2])
def test_a_column_is_a_solve_with_one_unit_source(case):
    make, geometry, kw, (xo, yo) = CASES[case]
    mesh = make()
    gf = _quiet(bemcs.get_greens_functions, mesh, xo, yo, geometry=geometry, **kw)
    j = gf.n_sources // 2
    rows = (gf.sources["element"], gf.sources["component"])
    one = _with_slip(mesh, rows, gf.point_slip[:, j])  # the slip hat j alone puts on the sources
    x = _quiet(bemcs.solve, one, geometry=geometry, **kw)
    ref = bemcs.evaluate_fields(one, x, xo, yo, geometry=geometry, **kw)
    for name in ref:
        np.testing.assert_allclose(gf.fields[name][:, j], ref[name], atol=1e-9 * np.abs(ref[name]).max())
    assert gf.point_slip[j, j] == pytest.approx(1.0)  # unit amplitude: slip 1 at its own centre


@pytest.mark.parametrize("case", list(CASES))
def test_every_source_is_one_non_negative_hat(case):
    """A source is one hat: non-negative slip on (at most) three fault elements, in its own
    component only; its element averages agree with the slip along the elements."""
    make, geometry, kw, (xo, yo) = CASES[case]
    mesh = make()
    gf = _quiet(bemcs.get_greens_functions, mesh, xo, yo, geometry=geometry, **kw)
    vals = gf.slip_coefficients  # node values of the quadratic slip (n_fault, 3, pe, n_sources)
    assert vals.min() > -1e-12
    for j in range(gf.n_sources):
        c = gf.sources["component"][j]
        on = np.flatnonzero(np.abs(vals[:, :, c, j]).max(axis=1) > 1e-12)
        assert len(on) <= 3 and gf.sources["element"][j] in gf.fault_elements[on]  # 2 at the fault's ends
        if vals.shape[2] == 2:
            assert np.abs(vals[:, :, 1 - c, j]).max() < 1e-12  # the other component is zero
    np.testing.assert_allclose(np.diag(gf.point_slip), 1.0)


def test_from_files_with_inferred_connectivity_matches_rybicki():
    """The < 10-line workflow: a mesh file and its connectivity files (BCs on the rows)."""
    path = os.path.join(EXAMPLES, "fault_through_interface")
    if not os.path.exists(path + ".csv"):
        pytest.skip("run examples/antiplane/ConstructMesh.py first")
    x_obs = np.r_[-np.geomspace(5, 0.02, 10), np.geomspace(0.02, 5, 10)]
    gf = _quiet(bemcs.get_greens_functions, path + ".csv", x_obs, np.full_like(x_obs, -1e-9),
                geometry="antiplane",
                connectivity=[path + "_force_connectivity.csv", path + "_slip_connectivity.csv"])
    depth = -gf.sources["y"]
    D = 1.63
    u = gf(gf.amplitudes(np.where(depth < D, np.cos(0.5 * np.pi * depth / D) ** 2, 0.0)))["uz"]
    assert np.abs(u - layered_image_series(x_obs, D, LAYERED_H, LAYERED_MU1, LAYERED_MU2)).max() < 1e-4  # measured 6e-5 (h = 0.05)


def test_options_and_errors():
    mesh = _hats(layered_mesh(0.6, 0.1))
    xo, yo = np.array([0.5, 1.0]), np.array([-0.2, -1.5])
    gf = _quiet(bemcs.get_greens_functions, mesh, xo, yo, geometry="antiplane", components=["uz"])
    assert list(gf.fields) == ["uz"] and gf.fields["uz"].shape == (2, gf.n_sources)
    true = _quiet(bemcs.get_greens_functions, mesh, xo, yo, geometry="antiplane")
    ref = _quiet(bemcs.get_greens_functions, mesh, xo, yo, geometry="antiplane", mu_obs="reference")
    mu = mesh.modulus_at(xo, yo)
    np.testing.assert_allclose(true.fields["sxz"], mu[:, None] * ref.fields["sxz"])
    assert mu[0] != mu[1]  # one point in the layer, one in the half-space
    with pytest.raises(ValueError, match="unknown components"):
        bemcs.get_greens_functions(mesh, xo, yo, geometry="antiplane", components=["ux"])
    with pytest.raises(ValueError, match="hat slip basis"):
        bemcs.get_greens_functions(layered_mesh(0.6, 0.1), xo, yo, geometry="antiplane")  # quadratic
    with pytest.raises(ValueError, match="a Mesh already has them"):
        bemcs.get_greens_functions(mesh, xo, yo, geometry="antiplane", connectivity={"slip": "x.csv"})


def test_tractions_project_the_stress_kernels():
    """gf.tractions = sigma . n per source; local frame with the tangent; K @ a = traction of gf(a)."""
    mesh = crossing_mesh(60, 0.1, 4.0)
    xo, yo = np.array([-0.5, 0.7, 1.2]), np.array([0.4, -0.5, 0.6])
    nx, ny = np.array([0.0, 0.6, 1.0]), np.array([1.0, 0.8, 0.0])
    sx, sy = ny, -nx
    gf = _quiet(bemcs.get_greens_functions, mesh, xo, yo, geometry="planestrain", nu=CROSSING_NU)
    a = np.random.default_rng(0).normal(size=gf.n_sources)
    f = gf(a)
    tx, ty = f["sxx"] * nx + f["sxy"] * ny, f["sxy"] * nx + f["syy"] * ny
    np.testing.assert_allclose(gf.tractions(nx, ny) @ a, np.c_[tx, ty], atol=1e-12)
    np.testing.assert_allclose(gf.tractions(nx, ny, sx, sy) @ a, np.c_[tx * sx + ty * sy, tx * nx + ty * ny],
                               atol=1e-12)
    np.testing.assert_allclose(gf.tractions(nx[1:], ny[1:], index=slice(1, None)) @ a, np.c_[tx, ty][1:], atol=1e-12)
    anti = _quiet(bemcs.get_greens_functions, _hats(layered_mesh(0.6, 0.1)), xo, -np.abs(yo), geometry="antiplane")
    fa = anti(np.ones(anti.n_sources))
    np.testing.assert_allclose(anti.tractions(nx, ny) @ np.ones(anti.n_sources), fa["sxz"] * nx + fa["syz"] * ny,
                               atol=1e-12)
    only_u = _quiet(bemcs.get_greens_functions, mesh, xo, yo, geometry="planestrain", nu=CROSSING_NU,
                    components=["ux"])
    with pytest.raises(ValueError, match="stress components"):
        only_u.tractions(nx, ny)
