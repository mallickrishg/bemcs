"""Displacement boundary conditions: "u" (alias "u_global", global x/y) and "u_local"
(local shear/normal components of the element), and the "t_global" alias of "t".

Setup: a buried fault ("s") in a full space and a dipping line of slip elements carrying a
displacement condition. With the line's displacements prescribed as those of the fault
alone, the exact answer is no slip on the line, and the global and local forms of the same
condition are the same equations (rows rotated by R^T).
"""

import numpy as np
import pytest

import bemcs

NU = 0.25
DIP = np.deg2rad(35.0)


def _els(n_line=40):
    xf = np.linspace(-0.5, 0.5, 11)  # horizontal fault at y = -1
    s = np.linspace(-2.0, 2.0, n_line + 1)  # dipping line through the origin
    xl, yl = s * np.cos(DIP), s * np.sin(DIP)
    els = bemcs.initialize_els()
    els.x1, els.y1 = np.r_[xf[:-1], xl[:-1]], np.r_[np.full(10, -1.0), yl[:-1]]
    els.x2, els.y2 = np.r_[xf[1:], xl[1:]], np.r_[np.full(10, -1.0), yl[1:]]
    bemcs.standardize_els_geometry(els, reorder=False)
    return els, 10, n_line


def _fault_slip(n_f):
    xc = np.linspace(-0.45, 0.45, n_f)
    p = np.cos(0.5 * np.pi * xc / 0.5) ** 2
    return np.c_[p, 0.4 * p]


def _line_displacement(els, n_f, geometry, side=-1, dr=1e-9):
    """Displacement of the fault alone at the line's centres, on side `side`."""
    fault = bemcs.Mesh(_sub(els, n_f), bc_type=["s"] * n_f,
                       bc_value=_fault_slip(n_f), mu_minus=1.0, mu_plus=1.0, basis={"slip": "hat"})
    x = bemcs.solve(fault, geometry=geometry, mu=1.0, nu=NU)
    xc, yc = np.asarray(els.x_centers)[n_f:], np.asarray(els.y_centers)[n_f:]
    nx, ny = np.asarray(els.x_normals)[n_f:], np.asarray(els.y_normals)[n_f:]
    L = 2 * np.asarray(els.half_lengths)[n_f:]
    f = bemcs.evaluate_fields(fault, x, xc + side * dr * L * nx, yc + side * dr * L * ny,
                              geometry=geometry, mu=1.0, nu=NU)
    return np.c_[f["ux"], f["uy"]] if geometry == "planestrain" else np.c_[f["uz"], np.zeros(len(xc))]


def _sub(els, n):
    e = bemcs.initialize_els()
    e.x1, e.y1, e.x2, e.y2 = (np.asarray(getattr(els, k))[:n] for k in ("x1", "y1", "x2", "y2"))
    bemcs.standardize_els_geometry(e, reorder=False)
    return e


def _to_local(els, n_f, u):
    sx, sy = np.asarray(els.x_shears)[n_f:], np.asarray(els.y_shears)[n_f:]
    nx, ny = np.asarray(els.x_normals)[n_f:], np.asarray(els.y_normals)[n_f:]
    return np.c_[u[:, 0] * sx + u[:, 1] * sy, u[:, 0] * nx + u[:, 1] * ny]


def _solve(els, n_f, n_l, bc, value, geometry, **kw):
    mesh = bemcs.Mesh(els, bc_type=["s"] * n_f + [bc] * n_l, bc_value=np.r_[_fault_slip(n_f), value],
                      mu_minus=1.0, mu_plus=1.0, basis={"slip": "hat"})
    return mesh, bemcs.solve(mesh, geometry=geometry, mu=1.0, nu=NU, **kw)


def _line_slip(mesh, x, n_f, geometry):
    xe, _, s = mesh.values_on_elements(x, "slip", geometry=geometry, n_eval=5)
    s = np.asarray(s).reshape(len(mesh.elements("slip")), 5, -1)
    return np.abs(s[n_f:]).max()


@pytest.mark.parametrize("geometry", ["planestrain", "antiplane"])
def test_global_and_local_displacement_conditions_agree(geometry):
    els, n_f, n_l = _els()
    u = _line_displacement(els, n_f, geometry)
    _, xg = _solve(els, n_f, n_l, "u", u, geometry)
    ul = _to_local(els, n_f, u) if geometry == "planestrain" else u
    mesh, xl = _solve(els, n_f, n_l, "u_local", ul, geometry)
    np.testing.assert_allclose(xl, xg, atol=1e-10 * np.abs(xg).max())
    # the line reproduces the fault's own field: (almost) no slip on it
    assert _line_slip(mesh, xl, n_f, geometry) < 1e-6


def test_global_aliases_are_the_plain_types():
    els, n_f, n_l = _els()
    u = _line_displacement(els, n_f, "planestrain")
    mesh_a, xa = _solve(els, n_f, n_l, "u_global", u, "planestrain")
    _, xb = _solve(els, n_f, n_l, "u", u, "planestrain")
    np.testing.assert_array_equal(xa, xb)
    assert (mesh_a.bc_type[n_f:] == "u").all()  # stored under the plain name
    mesh_t = bemcs.Mesh(els, bc_type=["s"] * n_f + ["t_global"] * n_l, mu_minus=1.0, mu_plus=1.0,
                        bc_value=np.r_[_fault_slip(n_f), np.zeros((n_l, 2))], basis={"slip": "hat"})
    assert (mesh_t.bc_type[n_f:] == "t").all()
    with pytest.raises(ValueError, match="unknown bc_type"):
        bemcs.Mesh(els, bc_type=["s"] * n_f + ["u_globl"] * n_l, mu_minus=1.0, mu_plus=1.0)


def test_mixed_local_components_make_a_roller():
    """Roller: ["t_local", "u_local"] with values (0, 0), i.e. zero shear traction
    (component 1, along the element) and zero normal displacement (component 2). Checked
    between the collocation points on the sampled (-n) side: measured 4e-5 (normal
    displacement) and 1.2e-3 (shear traction) relative to the fault's own field there."""
    els, n_f, n_l = _els(80)
    bc = np.array([["s", "s"]] * n_f + [["t_local", "u_local"]] * n_l)
    mesh = bemcs.Mesh(els, bc_type=bc, bc_value=np.r_[_fault_slip(n_f), np.zeros((n_l, 2))],
                      mu_minus=1.0, mu_plus=1.0, basis={"slip": "hat"})
    x = bemcs.solve(mesh, geometry="planestrain", mu=1.0, nu=NU)
    idx = np.arange(n_f + 20, n_f + 60)  # away from the line's tips
    xq = np.asarray(els.x1)[idx] + 0.3 * (np.asarray(els.x2)[idx] - np.asarray(els.x1)[idx])
    yq = np.asarray(els.y1)[idx] + 0.3 * (np.asarray(els.y2)[idx] - np.asarray(els.y1)[idx])
    nx, ny = np.asarray(els.x_normals)[idx], np.asarray(els.y_normals)[idx]
    sx, sy = np.asarray(els.x_shears)[idx], np.asarray(els.y_shears)[idx]
    f = bemcs.evaluate_fields(mesh, x, xq - 1e-7 * nx, yq - 1e-7 * ny, geometry="planestrain", mu=1.0, nu=NU)
    u_n = f["ux"] * nx + f["uy"] * ny
    tx, ty = f["sxx"] * nx + f["sxy"] * ny, f["sxy"] * nx + f["syy"] * ny
    t_s = tx * sx + ty * sy
    # scales: the displacement and traction of the fault alone at the same points
    u_ref = _line_displacement(els, n_f, "planestrain")[20:60]
    fault = bemcs.Mesh(_sub(els, n_f), bc_type=["s"] * n_f, bc_value=_fault_slip(n_f), mu_minus=1.0, mu_plus=1.0,
                       basis={"slip": "hat"})
    g = bemcs.evaluate_fields(fault, bemcs.solve(fault, geometry="planestrain", mu=1.0, nu=NU), xq, yq,
                              geometry="planestrain", mu=1.0, nu=NU)
    t_ref = np.hypot(g["sxx"] * nx + g["sxy"] * ny, g["sxy"] * nx + g["syy"] * ny).max()
    assert np.abs(u_n).max() < 2e-4 * np.abs(u_ref).max()
    assert np.abs(t_s).max() < 5e-3 * t_ref
    # and it is a different problem from the clamped line (the line moves tangentially)
    u_s = f["ux"] * sx + f["uy"] * sy
    assert np.abs(u_s).max() > 10 * np.abs(u_n).max()


def test_local_displacement_with_a_background_field():
    """The background's displacement moves to the right-hand side in the rotated frame too."""
    els, n_f, n_l = _els()
    bg = bemcs.uniform_background([0.3, -0.2, 0.1], geometry="planestrain", mu=1.0, nu=NU)
    u = _line_displacement(els, n_f, "planestrain")
    _, xg = _solve(els, n_f, n_l, "u", u, "planestrain", background=bg)
    _, xl = _solve(els, n_f, n_l, "u_local", _to_local(els, n_f, u), "planestrain", background=bg)
    np.testing.assert_allclose(xl, xg, atol=1e-10 * np.abs(xg).max())


# ----------------------------------------------------------------------------
# where rows are sampled: on the element unless the sampled quantity jumps there
# ----------------------------------------------------------------------------


def _sides(mesh):
    import bemcs.bemAssembly as GF
    rec = GF._collocation_points(mesh)
    return GF._needs_side(mesh, rec, rec["owner"].astype(int)), rec["owner"].astype(int)


def test_rows_move_off_the_element_only_where_the_sampled_quantity_jumps():
    els, n_f, n_l = _els()
    cases = {  # line type -> needs the side? (slip "u" jumps; slip "t" on its own plane does not)
        "u": True, "u_local": True, "t": False, "t_local": False,
    }
    for bc, expected in cases.items():
        mesh = bemcs.Mesh(els, bc_type=["s"] * n_f + [bc] * n_l, bc_value=np.zeros((n_f + n_l, 2)),
                          mu_minus=1.0, mu_plus=1.0, basis={"slip": "hat"})
        need, owner = _sides(mesh)
        assert (need[owner >= n_f] == expected).all(), bc
    # force elements: displacement continuous, traction ("h") jumps
    for bc, expected in {"u": False, "h": True}.items():
        mesh = bemcs.Mesh(els, labels=["s"] * n_f + ["f"] * n_l, bc_type=["s"] * n_f + [bc] * n_l,
                          bc_value=np.zeros((n_f + n_l, 2)), mu_minus=1.0, mu_plus=np.r_[np.ones(n_f), np.full(n_l, 2.0)],
                          basis={"slip": "hat", "force": "trapezoid"})
        need, owner = _sides(mesh)
        assert (need[owner >= n_f] == expected).all(), bc


def test_a_crack_through_a_collocation_point_moves_that_row_off_the_element():
    """Traction on A's plane jumps across an oblique crack B through A's collocation point."""
    els = bemcs.initialize_els()
    els.x1, els.y1 = np.array([-1.0, 0.0, 0.5, 0.5]), np.zeros(4)
    els.x2, els.y2 = np.array([0.0, 1.0, 0.5, 0.5]), np.zeros(4)
    els.y1[2:], els.y2[2:] = [-1.0, 0.0], [0.0, 1.0]  # B: vertical through (0.5, 0), A's 2nd centre
    bemcs.standardize_els_geometry(els, reorder=False)
    mesh = bemcs.Mesh(els, bc_type=["t"] * 4, bc_value=np.zeros((4, 2)), mu_minus=1.0, mu_plus=1.0,
                      basis={"slip": "quadratic"})
    need, owner = _sides(mesh)
    assert need[owner == 1].all() and not need[owner == 0].any()


def test_free_surface_rows_on_the_surface_give_the_offset_solution():
    """Sampling a free surface on the element (now) or 1e-9 L off it (before) is the same
    problem: the traction of a slip layer is continuous on its own plane."""
    els, n_f, n_l = _els()
    mesh, x = _solve(els, n_f, n_l, "t", np.zeros((n_l, 2)), "planestrain")
    import bemcs.bemAssembly as GF
    orig = GF._needs_side
    try:
        GF._needs_side = lambda m, r, c, tol=1e-9: np.ones(len(c), dtype=bool)
        _, x_off = _solve(els, n_f, n_l, "t", np.zeros((n_l, 2)), "planestrain")
    finally:
        GF._needs_side = orig
    np.testing.assert_allclose(x, x_off, atol=1e-6 * np.abs(x).max())
