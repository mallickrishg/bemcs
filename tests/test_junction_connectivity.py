"""Explicit slip-hat connectivity with triple-junction rows (4 element columns) and boundary
conditions given per connectivity row instead of per element."""

import warnings

import numpy as np
import pandas as pd
import pytest

import bemcs
import bemcs.bemAssembly as GF

MU, NU = 1.0, 0.25


def _surface_fault():
    """Free surface y = 0 on [-4, 4] (8 elements, node at 0) and a fault dipping from (0, 0)
    (4 elements): the fault breaks the surface at a triple junction."""
    xs = np.linspace(-4, 4, 9)
    d = np.linspace(0, 2, 5)
    xf, yf = 0.5 * d, -d
    els = bemcs.initialize_els()
    els.x1, els.y1 = np.r_[xs[:-1], xf[:-1]], np.r_[np.zeros(8), yf[:-1]]
    els.x2, els.y2 = np.r_[xs[1:], xf[1:]], np.r_[np.zeros(8), yf[1:]]
    bemcs.standardize_els_geometry(els, reorder=False)
    bc_type = np.array(["t"] * 8 + ["s"] * 4)
    bc_value = np.zeros((12, 2))
    bc_value[8:, 0] = [1.0, 1.0, 0.7, 0.3]
    return els, bc_type, bc_value


def _solve_ux(mesh, geometry):
    x = bemcs.solve(mesh, geometry=geometry, mu=MU, nu=NU if geometry == "planestrain" else None)
    xo, yo = np.linspace(-3, 3, 9), np.full(9, -0.7)
    f = bemcs.evaluate_fields(mesh, x, xo, yo, geometry=geometry, mu=MU,
                              nu=NU if geometry == "planestrain" else None)
    return f["uz" if geometry == "antiplane" else "ux"]


def test_one_ring_connectivity_has_a_fourth_column_at_the_junction():
    els, _, _ = _surface_fault()
    conn = bemcs.one_ring_connectivity(els)
    assert conn.shape[1] == 4
    junction = conn[conn[:, 3] >= 0]
    # the hats of the two surface elements at x = 0 and of the top fault element
    assert sorted(junction[:, 1].tolist()) == [3, 4, 8]
    assert (conn[conn[:, 3] < 0, 3] == -1).all()


@pytest.mark.parametrize("geometry", ["antiplane", "planestrain"])
def test_explicit_junction_rows_equal_the_general_solve(geometry):
    els, bc_type, bc_value = _surface_fault()
    ref = bemcs.Mesh(els, bc_type=bc_type, bc_value=bc_value, mu_minus=MU, mu_plus=MU)
    conn = bemcs.one_ring_connectivity(els)
    mesh = bemcs.Mesh(els, bc_type=bc_type, bc_value=bc_value, mu_minus=MU, mu_plus=MU,
                      basis={"slip": "hat"}, connectivity={"slip": conn})
    assert np.abs(_solve_ux(mesh, geometry) - _solve_ux(ref, geometry)).max() < 1e-12


@pytest.mark.parametrize("geometry", ["antiplane", "planestrain"])
def test_boundary_conditions_on_connectivity_rows(geometry, tmp_path):
    """Mesh file with geometry and moduli only, BCs in the connectivity file: same solution."""
    els, bc_type, bc_value = _surface_fault()
    ref = bemcs.Mesh(els, bc_type=bc_type, bc_value=bc_value, mu_minus=MU, mu_plus=MU,
                     basis={"slip": "hat"})
    table = ref.connectivity_table("slip", geometry=geometry)
    assert ("BC_type" in table) == (geometry == "antiplane")
    table.to_csv(tmp_path / "conn.csv", index=False)
    pd.DataFrame({"x1": els.x1, "z1": els.y1, "x2": els.x2, "z2": els.y2,
                  "mu_minus": MU, "mu_plus": MU}).to_csv(tmp_path / "mesh.csv", index=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mesh = bemcs.Mesh.from_csv(tmp_path / "mesh.csv", basis={"slip": "hat"},
                                   connectivity={"slip": tmp_path / "conn.csv"})
    centres = table["e2"].to_numpy()
    assert (mesh.bc_type[centres, 0] == bc_type[centres]).all()
    # tips: no row, bookkeeping type from their row
    assert mesh.bc_type[0, 0] == "t" and mesh.bc_type[11, 0] == "s"
    assert np.abs(_solve_ux(mesh, geometry) - _solve_ux(ref, geometry)).max() < 1e-12


def test_dataframe_and_headerless_file(tmp_path):
    els, bc_type, bc_value = _surface_fault()
    conn = bemcs.one_ring_connectivity(els)
    np.savetxt(tmp_path / "c.csv", conn, fmt="%d", delimiter=",")
    read, bc = GF._read_connectivity(tmp_path / "c.csv")
    assert bc is None and (read == conn).all()
    df = pd.DataFrame(conn, columns=["e1", "e2", "e3", "e4"])
    df["BC_type"] = bc_type[conn[:, 1]]
    df["value"] = bc_value[conn[:, 1], 0]
    mesh = bemcs.Mesh(els, mu_minus=MU, mu_plus=MU, basis={"slip": "hat"}, connectivity={"slip": df})
    assert np.allclose(mesh.bc_value[conn[:, 1], 0], bc_value[conn[:, 1], 0])


def test_missing_junction_branch_warns():
    els, bc_type, bc_value = _surface_fault()
    conn = bemcs.one_ring_connectivity(els)[:, :3]  # drop the 4th column
    with pytest.warns(bemcs.HatConnectivityWarning, match="must span its neighbours"):
        bemcs.Mesh(els, bc_type=bc_type, bc_value=bc_value, basis={"slip": "hat"},
                   connectivity={"slip": conn})


def test_missing_hat_warns_and_duplicate_centre_raises():
    els, bc_type, bc_value = _surface_fault()
    conn = bemcs.one_ring_connectivity(els)
    with pytest.warns(bemcs.HatConnectivityWarning, match="no row is centred on element"):
        bemcs.Mesh(els, bc_type=bc_type, basis={"slip": "hat"}, connectivity={"slip": conn[1:]})
    with pytest.raises(ValueError, match="more than one row"):
        bemcs.Mesh(els, bc_type=bc_type, basis={"slip": "hat"},
                   connectivity={"slip": np.r_[conn, conn[:1]]})


def test_conflicting_boundary_conditions_raise():
    els, bc_type, bc_value = _surface_fault()
    ref = bemcs.Mesh(els, bc_type=bc_type, bc_value=bc_value, basis={"slip": "hat"})
    table = ref.connectivity_table("slip", geometry="planestrain")
    with pytest.raises(ValueError, match="differs from bc_type"):
        bemcs.Mesh(els, bc_type=bc_type, bc_value=0 * bc_value, basis={"slip": "hat"},
                   connectivity={"slip": table})


def test_elements_without_a_condition_raise():
    els, _, _ = _surface_fault()
    table = bemcs.Mesh(els, bc_type=["t"] * 8 + ["s"] * 4, basis={"slip": "hat"}).connectivity_table(
        "slip", geometry="antiplane")
    with pytest.raises(ValueError, match="in no connectivity row"):
        bemcs.Mesh(els, basis={"slip": "hat"}, connectivity={"slip": table[table.e2 > 3]})


def test_force_trapezoids_take_no_junction_column():
    with pytest.raises(ValueError, match="only slip hats"):
        GF._check_connectivity(np.array([[0, 1, 2, 3]]), 5, "force")


def test_label_column_covers_a_force_element_in_no_trapezoid(tmp_path):
    """A force chain of 4 elements: stride-2 trapezoids leave the last one out. With BCs on the
    connectivity rows, the mesh file's `label` column says it is a force element."""
    els = bemcs.initialize_els()
    xs = np.linspace(-2, 2, 5)
    els.x1, els.x2, els.y1, els.y2 = xs[:-1], xs[1:], np.full(4, -1.0), np.full(4, -1.0)
    bemcs.standardize_els_geometry(els, reorder=False)
    pd.DataFrame({"e1": [0], "e2": [1], "e3": [2], "BC_type": ["h"], "value": [0.0]}).to_csv(
        tmp_path / "f.csv", index=False)
    mesh_file = pd.DataFrame({"x1": els.x1, "z1": els.y1, "x2": els.x2, "z2": els.y2,
                              "mu_minus": 2.0, "mu_plus": 1.0})
    mesh_file.to_csv(tmp_path / "m.csv", index=False)
    with pytest.raises(ValueError, match="in no connectivity row"):
        bemcs.Mesh.from_csv(tmp_path / "m.csv", basis={"force": "trapezoid"},
                            connectivity={"force": tmp_path / "f.csv"})
    mesh_file.insert(4, "label", "f")
    mesh_file.to_csv(tmp_path / "m.csv", index=False)
    mesh = bemcs.Mesh.from_csv(tmp_path / "m.csv", basis={"force": "trapezoid"},
                               connectivity={"force": tmp_path / "f.csv"})
    assert (mesh.labels == "f").all() and (mesh.bc_type == "h").all()
