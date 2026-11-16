"""Unified solver (`bemcs.solve`) on meshes with boundary conditions and the
shear modulus on both sides of every element.

Physics checks independent of the implementation:
- antiplane: a fault near a straight bimaterial interface against the exact
  image solution;
- plane strain: the true traction (mu_side / mu0) S.n is continuous across a
  material interface, and the solution does not depend on the sampled side;
- homogeneous: general (element DOFs + continuity rows) and reduced (one-ring
  hats) forms give the same slip.
"""

import numpy as np
import pandas as pd
import pytest

import bemcs
import bemcs.bemAssembly as GF

MU0, NU = 1.0, 0.27


def _els(x1, y1, x2, y2):
    els = bemcs.initialize_els()
    els.x1, els.y1, els.x2, els.y2 = (np.asarray(v, float) for v in (x1, y1, x2, y2))
    bemcs.standardize_els_geometry(els, reorder=False)
    return els


def _fault_and_interface(n_fault=6, L=40.0, n_int=800, y_top=-0.6, y_bot=-1.8):
    """Vertical fault (in the lower material) below a straight interface along y = 0
    (elements left to right, normal +y)."""
    yf = np.linspace(y_top, y_bot, n_fault + 1)
    xi = np.linspace(-L, L, n_int + 1)
    return _els(
        np.r_[np.zeros(n_fault), xi[:-1]], np.r_[yf[:-1], np.zeros(n_int)],
        np.r_[np.zeros(n_fault), xi[1:]], np.r_[yf[1:], np.zeros(n_int)],
    ), n_fault, n_int


def _bimaterial_mesh(mu_below, mu_above, geometry):
    # antiplane: long interface for the image solution; plane strain: smaller (speed)
    size = dict(L=40.0, n_int=800) if geometry == "antiplane" else dict(L=10.0, n_int=400)
    els, n_f, n_i = _fault_and_interface(**size)
    bc_type = ["s"] * n_f + ["h"] * n_i
    slip = np.zeros((n_f + n_i, 2))
    slip[:n_f, 0] = 1.0  # unit shear slip on the fault
    if geometry == "planestrain":
        slip[:n_f, 1] = 0.3  # and some opening
    mu_minus = np.r_[np.full(n_f, mu_below), np.full(n_i, mu_below)]  # interface: -n side is below
    mu_plus = np.r_[np.full(n_f, mu_below), np.full(n_i, mu_above)]
    return bemcs.Mesh(
        els, bc_type=bc_type, bc_value=slip, mu_minus=mu_minus, mu_plus=mu_plus,
        basis={"slip": "hat", "force": "trapezoid"},
    ), n_f


# ----------------------------------------------------------------------------
# antiplane: exact image solution
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("side", [-1, 1])
def test_antiplane_bimaterial_matches_image_solution(side):
    mu1, mu2 = 5.0, 2.0  # below (fault side) and above
    k = (mu1 - mu2) / (mu1 + mu2)
    mesh, n_f = _bimaterial_mesh(mu1, mu2, "antiplane")
    x = bemcs.solve(mesh, geometry="antiplane", mu=MU0, side=side)
    n_s = mesh.n_src("slip")
    X = np.array([0.7, -0.5, 1.5, 0.3, -2.0, 0.05])
    Y = np.array([-1.0, -0.3, 0.8, 0.4, -2.5, -0.05])
    U, _, _ = bemcs.get_kernels(X, Y, mesh, geometry="antiplane", mu=MU0)
    u = U[:, :, 0] @ x
    Ub, _, _ = bemcs.get_kernels(X, -Y, mesh, geometry="antiplane", mu=MU0)
    slip_only = U[:, :n_s, 0] @ x[:n_s]  # homogeneous field of the same slip
    image = Ub[:, :n_s, 0] @ x[:n_s]
    exact = np.where(Y < 0, slip_only + k * image, (1 + k) * slip_only)
    # discretization error, second order in the interface element size h and the same
    # from either side: 1.8e-3, 3.5e-4, 9.0e-5 for h = 0.2, 0.1 (this mesh), 0.05
    assert np.abs(u - exact).max() < 5e-4 * np.abs(exact).max()


# ----------------------------------------------------------------------------
# plane strain: traction continuity across the interface, side independence
# ----------------------------------------------------------------------------


def _traction_mismatch(n_int, where):
    """Relative jump of the true traction across a plane-strain interface (y = 0)
    at the collocation centres or at quarter points of all interface elements."""
    mu1, mu2 = 5.0, 2.0
    els, n_f, n_i = _fault_and_interface(L=10.0, n_int=n_int)
    value = np.zeros((n_f + n_i, 2))
    value[:n_f] = [1.0, 0.3]
    mesh = bemcs.Mesh(
        els, bc_type=["s"] * n_f + ["h"] * n_i, bc_value=value,
        mu_minus=np.r_[np.full(n_f, mu1), np.full(n_i, mu1)],
        mu_plus=np.r_[np.full(n_f, mu1), np.full(n_i, mu2)],
        basis={"slip": "hat", "force": "trapezoid"},
    )
    x = bemcs.solve(mesh, geometry="planestrain", mu=MU0, nu=NU)
    sub, idx, _ = mesh.subset("force")
    near = [e for e in range(n_f, n_f + n_i) if abs(els.x_centers[e]) < 2]
    if where == "collocation":
        mids = set(idx[GF.patch_elements(sub, "trapezoid")].tolist())
        xs = np.array([els.x_centers[e] for e in near if e in mids])
    else:
        xs = np.array([els.x_centers[e] + 0.25 * (20.0 / n_int) for e in near])
    kw = dict(geometry="planestrain", mu=MU0, nu=NU)
    _, Sb, _ = bemcs.get_kernels(xs, np.full_like(xs, -1e-7), mesh, **kw)
    _, Sa, _ = bemcs.get_kernels(xs, np.full_like(xs, 1e-7), mesh, **kw)
    n = np.zeros_like(xs), np.ones_like(xs)

    def true_traction(S, mu_side):
        return mu_side / MU0 * np.stack(
            [GF.kernel_matrix(GF.traction_kernels(S, *n)[:, :, c : c + 1]) @ x for c in (0, 1)]
        )

    tb, ta = true_traction(Sb, mu1), true_traction(Sa, mu2)
    ref_jump = np.abs(tb / mu1 - ta / mu2).max() / (np.abs(tb).max() / mu1)
    return np.abs(tb - ta).max() / np.abs(tb).max(), ref_jump


def test_planestrain_true_traction_is_continuous_at_collocation_points():
    """The 'h' rows make the true traction continuous where they are imposed, while
    the reference-medium traction jumps by ~150%."""
    mismatch, ref_jump = _traction_mismatch(400, "collocation")
    assert ref_jump > 1.0
    assert mismatch < 1e-5


def test_planestrain_traction_continuity_converges_between_collocation_points():
    """Between collocation points the default trapezoids (force constant on every
    other element) satisfy the interface condition to first order in h."""
    coarse, _ = _traction_mismatch(200, "quarter")
    fine, _ = _traction_mismatch(400, "quarter")
    assert fine < 0.1
    assert fine < 0.6 * coarse


@pytest.mark.parametrize("geometry", ["antiplane", "planestrain"])
def test_solution_does_not_depend_on_the_sampled_side(geometry):
    mesh, _ = _bimaterial_mesh(4.0, 1.5, geometry)
    xm = bemcs.solve(mesh, geometry=geometry, mu=MU0, nu=NU, side=-1)
    xp = bemcs.solve(mesh, geometry=geometry, mu=MU0, nu=NU, side=1)
    np.testing.assert_allclose(xm, xp, atol=1e-6 * np.abs(xm).max())


@pytest.mark.parametrize("geometry", ["antiplane", "planestrain"])
def test_no_contrast_gives_no_interface_force(geometry):
    mesh, _ = _bimaterial_mesh(3.0, 3.0, geometry)
    x = bemcs.solve(mesh, geometry=geometry, mu=MU0, nu=NU)
    np.testing.assert_allclose(x[mesh.columns("force", geometry=geometry)], 0, atol=1e-14)


def test_interface_kappa():
    assert GF.interface_kappa(5.0, 2.0, -1) == pytest.approx(1 - 5 / 2)
    assert GF.interface_kappa(5.0, 2.0, 1) == pytest.approx(2 / 5 - 1)
    assert GF.interface_kappa(3.0, 3.0, 1) == 0


# ----------------------------------------------------------------------------
# homogeneous: general and reduced forms agree
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("geometry", ["antiplane", "planestrain"])
def test_general_and_one_ring_forms_agree(geometry):
    """Free surface (traction-free) with a dipping fault of prescribed slip."""
    xs = np.linspace(-10, 10, 41)
    d = np.deg2rad(60)
    fl = np.linspace(0, 2.0, 7)
    els = _els(
        np.r_[xs[:-1], 0.3 + fl[:-1] * np.cos(d)], np.r_[np.zeros(40), -0.2 - fl[:-1] * np.sin(d)],
        np.r_[xs[1:], 0.3 + fl[1:] * np.cos(d)], np.r_[np.zeros(40), -0.2 - fl[1:] * np.sin(d)],
    )
    n = len(els.x1)
    bc_type = ["t"] * 40 + ["s"] * 6
    value = np.zeros((n, 2))
    value[40:, 0] = [0.0, 1.0, 1.0, 0.8, 0.4, 0.0]
    common = dict(bc_type=bc_type, bc_value=value, mu_minus=1.0, mu_plus=1.0)
    general = bemcs.Mesh(els, basis={"slip": "quadratic"}, **common)
    reduced = bemcs.Mesh(els, basis={"slip": "hat"}, **common)
    xg = bemcs.solve(general, geometry=geometry, mu=MU0, nu=NU)
    xr = bemcs.solve(reduced, geometry=geometry, mu=MU0, nu=NU)
    cg = general.element_coefficients(xg, "slip", geometry=geometry)
    cr = reduced.element_coefficients(xr, "slip", geometry=geometry)
    np.testing.assert_allclose(cg, cr, atol=1e-9 * np.abs(cg).max())
    # the prescribed slip holds at the centres of non-tip elements
    _, _, v = general.values_on_elements(xg, "slip", geometry=geometry, n_eval=3, frame="local")
    centre = v.reshape(n, 3, -1)[:, 1, 0] if geometry == "planestrain" else v.reshape(n, 3)[:, 1]
    np.testing.assert_allclose(centre[41:45], value[41:45, 0], atol=1e-10)


# ----------------------------------------------------------------------------
# input checks, CSV, plots
# ----------------------------------------------------------------------------


def test_solve_requires_moduli_on_both_sides():
    els, n_f, n_i = _fault_and_interface(n_int=20, L=2.0)
    mesh = bemcs.Mesh(els, bc_type=["s"] * n_f + ["h"] * n_i, bc_value=np.ones(n_f + n_i),
                      basis={"slip": "hat", "force": "trapezoid"})
    with pytest.raises(ValueError, match="both sides"):
        bemcs.solve(mesh, geometry="antiplane", mu=1.0)


def test_solve_requires_trapezoid_force():
    els, n_f, n_i = _fault_and_interface(n_int=20, L=2.0)
    mesh = bemcs.Mesh(els, bc_type=["s"] * n_f + ["h"] * n_i, bc_value=np.ones(n_f + n_i),
                      basis={"slip": "hat"}, mu_minus=1.0, mu_plus=2.0)
    with pytest.raises(ValueError, match="trapezoid"):
        bemcs.solve(mesh, geometry="antiplane", mu=1.0)


def test_mesh_bc_validation():
    els, n_f, n_i = _fault_and_interface(n_int=20, L=2.0)
    with pytest.raises(ValueError, match="unknown bc_type"):
        bemcs.Mesh(els, bc_type=["q"] * (n_f + n_i))
    with pytest.raises(ValueError, match="not in"):  # 's' on a force element
        bemcs.Mesh(els, labels=["s"] * n_f + ["f"] * n_i, bc_type=["s"] * (n_f + n_i))


def test_mesh_from_csv(tmp_path):
    rows = pd.DataFrame({
        "x1": [0.0, 1.0, 2.0], "z1": [0.0, 0.0, 0.0], "x2": [1.0, 2.0, 3.0], "z2": [0.0, 0.0, 0.0],
        "BC_type": ["s", "s", "h"], "value": [1.0, 0.5, 0.0], "value2": [0.0, 0.1, 0.0],
        "mu_minus": [1.0, 1.0, 2.0], "mu_plus": [1.0, 1.0, 3.0],
    })
    path = tmp_path / "mesh.csv"
    rows.to_csv(path, index=False)
    mesh = bemcs.Mesh.from_csv(path)
    np.testing.assert_array_equal(mesh.labels, ["s", "s", "f"])
    np.testing.assert_allclose(mesh.bc_value[1], [0.5, 0.1])
    np.testing.assert_allclose(mesh.mu_plus, [1.0, 1.0, 3.0])


@pytest.mark.parametrize("geometry", ["antiplane", "planestrain"])
def test_plots_run(geometry):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mesh, _ = _bimaterial_mesh(2.0, 1.0, geometry)
    x = bemcs.solve(mesh, geometry=geometry, mu=MU0, nu=NU)
    bemcs.plot_mesh(mesh)
    fig, axes, fields = bemcs.plot_fields(
        mesh, x, geometry=geometry, mu=MU0, nu=NU, xlim=(-3, 3), ylim=(-3, 1), n=(20, 15),
        mu_field=lambda X, Y: np.where(Y < 0, 2.0, 1.0),
    )
    expected = {"antiplane": {"uz", "sxz", "syz"}, "planestrain": {"ux", "uy", "sxx", "syy", "sxy"}}[geometry]
    assert expected <= set(fields) and len(axes) == len(expected)
    plt.close("all")


# ----------------------------------------------------------------------------
# coefficient layout and helpers
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("slip_basis", ["quadratic", "hat"])
@pytest.mark.parametrize("geometry", ["antiplane", "planestrain"])
def test_layout_and_helpers(geometry, slip_basis):
    els, n_f, n_i = _fault_and_interface(n_int=20, L=2.0)
    pe = 1 if geometry == "antiplane" else 2
    mesh = bemcs.Mesh(
        els, bc_type=["s"] * n_f + ["h"] * n_i, bc_value=np.r_[np.ones(n_f), np.zeros(n_i)],
        mu_minus=2.0, mu_plus=np.r_[np.full(n_f, 2.0), np.full(n_i, 1.0)],
        basis={"slip": slip_basis, "force": "trapezoid"},
    )
    x, layout = bemcs.solve(mesh, geometry=geometry, mu=MU0, nu=NU, return_info=True)
    assert layout.n == x.size == mesh.n_columns(geometry=geometry)
    blocks = layout.split(x)
    assert blocks["slip"].shape == (mesh.n_src("slip"), pe)
    assert blocks["force"].shape == (mesh.n_src("force"), pe)
    np.testing.assert_array_equal(np.r_[blocks["slip"].ravel(), blocks["force"].ravel()], x)
    assert set(layout.kind[layout.columns["slip"]]) == {"slip"}
    assert set(layout.direction) == ({"z"} if pe == 1 else {"t", "n"})
    if slip_basis == "quadratic":
        np.testing.assert_array_equal(layout.node[layout.columns["slip"]][::pe], np.tile([0, 1, 2], n_f))
        np.testing.assert_array_equal(layout.element[layout.columns["slip"]][::pe], np.repeat(np.arange(n_f), 3))
    else:  # hats sit on the non-tip fault elements
        np.testing.assert_array_equal(layout.element[layout.columns["slip"]][::pe], np.arange(1, n_f - 1))
    assert "x[0:" in str(layout)
    # node values: slip at s = -w, 0, +w equals evaluation along the element at those points
    nv = mesh.node_values(x, "slip", geometry=geometry)
    assert nv.shape == (n_f, 3, pe)
    np.testing.assert_allclose(nv[1:-1, 1, 0], 1.0, atol=1e-10)  # prescribed slip at non-tip centres
    c = mesh.element_coefficients(x, "slip", geometry=geometry).reshape(n_f, 3, pe)
    w = els.half_lengths[:n_f]
    ends = np.array([core_slip(-1.0), core_slip(1.0)])  # Lagrange basis at s = -w, +w
    np.testing.assert_allclose(nv[:, 0], np.einsum("k,ekc->ec", ends[0], c), atol=1e-12)
    np.testing.assert_allclose(nv[:, 2], np.einsum("k,ekc->ec", ends[1], c), atol=1e-12)
    # clamped tips: zero slip at the open ends of the fault
    np.testing.assert_allclose(nv[0, 0], 0, atol=1e-10)
    np.testing.assert_allclose(nv[-1, 2], 0, atol=1e-10)
    fv = mesh.node_values(x, "force", geometry=geometry)
    assert fv.shape == (n_i, 2, pe)


def core_slip(t):
    return bemcs.slip_functions(np.array([t]), 1.0)[0]


def test_mesh_from_csv_per_component(tmp_path):
    """Plane strain: one BC type and value per component; mixed frames are allowed."""
    rows = pd.DataFrame({
        "x1": [0.0, 1.0, 2.0], "z1": [0.0, 0.0, 0.0], "x2": [1.0, 2.0, 3.0], "z2": [0.0, 0.0, 0.0],
        "BC_type_1": ["s", "t", "h"], "value_1": [1.0, 0.0, 0.0],
        "BC_type_2": ["s", "s", "h"], "value_2": [0.2, 0.0, 0.0],
        "mu_minus": [1.0, 1.0, 2.0], "mu_plus": [1.0, 1.0, 3.0],
    })
    path = tmp_path / "mesh.csv"
    rows.to_csv(path, index=False)
    mesh = bemcs.Mesh.from_csv(path)
    np.testing.assert_array_equal(mesh.bc_type, [["s", "s"], ["t", "s"], ["h", "h"]])
    np.testing.assert_allclose(mesh.bc_value, [[1.0, 0.2], [0.0, 0.0], [0.0, 0.0]])
    np.testing.assert_array_equal(mesh.labels, ["s", "s", "f"])
    bad = rows.copy()
    bad.loc[2, "BC_type_2"] = "t"  # 'h' on one component only
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match="'h' applies to both components"):
        bemcs.Mesh.from_csv(path)
    both = rows.assign(BC_type=["s", "s", "h"])
    both.to_csv(path, index=False)
    with pytest.raises(ValueError, match="not both"):
        bemcs.Mesh.from_csv(path)


def test_mixed_component_bcs_solve():
    """Per-component types in plane strain: traction-free in x and zero normal slip in y
    on a free surface element gives the same answer as setting each row by hand."""
    els, n_f, n_i = _fault_and_interface(n_int=20, L=2.0)
    n = n_f + n_i
    bc_type = np.array([["s", "s"]] * n_f + [["t", "s"]] * n_i)
    value = np.zeros((n, 2))
    value[:n_f, 0] = 1.0
    mesh = bemcs.Mesh(els, bc_type=bc_type, bc_value=value, mu_minus=1.0, mu_plus=1.0,
                      basis={"slip": "hat"})
    x, (A, b, rows) = bemcs.solve(mesh, geometry="planestrain", mu=1.0, nu=NU, return_system=True)
    np.testing.assert_allclose(A @ x, b, atol=1e-12)
    surf = rows["element"] >= n_f
    assert set(rows["type"][surf & (rows["component"] == 0)]) == {"t"}
    assert set(rows["type"][surf & (rows["component"] == 1)]) == {"s"}
    # the normal slip on the surface elements is zero at their centres (non-tip elements)
    nv = mesh.node_values(x, "slip", geometry="planestrain")
    np.testing.assert_allclose(nv[n_f + 1 : n - 1, 1, 1], 0, atol=1e-10)


def test_plot_mesh_marks_every_modulus_used_by_the_solve():
    """Interface elements: a dot on each side; traction elements: one square at the
    centre; slip-BC elements: nothing (their rows do not use mu)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    els, n_f, n_i = _fault_and_interface(n_int=10, L=2.0)
    xs = np.linspace(-2, 2, 9)  # free surface above the interface
    e2 = _els(np.r_[els.x1, xs[:-1]], np.r_[els.y1, np.full(8, 1.0)],
              np.r_[els.x2, xs[1:]], np.r_[els.y2, np.full(8, 1.0)])
    n_t = 8
    mesh = bemcs.Mesh(
        e2, bc_type=["s"] * n_f + ["h"] * n_i + ["t"] * n_t, bc_value=np.zeros(n_f + n_i + n_t),
        mu_minus=np.r_[np.full(n_f, 3.0), np.full(n_i, 3.0), np.full(n_t, 1.5)],
        mu_plus=np.r_[np.full(n_f, 3.0), np.full(n_i, 1.5), np.full(n_t, 1.5)],
    )
    ax = bemcs.plot_mesh(mesh, show_normals=False)
    sizes = sorted(len(c.get_offsets()) for c in ax.collections)
    assert sizes == [n_t, 2 * n_i]
    values = sorted(np.unique(np.concatenate([np.asarray(c.get_array()) for c in ax.collections])))
    assert values == [1.5, 3.0]
    plt.close("all")
