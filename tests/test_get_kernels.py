"""Tests for the three-level kernel API (tasks.md section 7):

level 3  element_kernels_slip/force, basis_transform, hat/trapezoid coefficients
level 2  get_slip_kernels, get_force_kernels
level 1  Mesh, get_kernels

Comparisons with independent references (analytic constant slip, point-force
quadrature) live in test_force_kernels_reference.py and here. The comparisons
with the old per-case functions (`test_matches_old_function`) are temporary and
go away with the old functions in stage 3.
"""

import numpy as np
import pytest

import bemcs
import bemcs.bemAssembly as GF
import bemcs.bemcs as core
import reference_kernels as ref

MU, NU = 1.3, 0.3
CONN = np.array([[0, 1, 2], [1, 2, 3]])
GEOMETRIES = ["antiplane", "planestrain"]
LEVEL2 = {"slip": bemcs.get_slip_kernels, "force": bemcs.get_force_kernels}

# (kind, basis, n_src on the 4-element test mesh)
BASES = [
    ("slip", "quadratic", 12),
    ("slip", "constant", 4),
    ("slip", "hat", 2),
    ("force", "linear", 8),
    ("force", "constant", 4),
    ("force", "trapezoid", 2),
]
CASES = [(g, k, b, n) for g in GEOMETRIES for (k, b, n) in BASES]
CASE_IDS = ["-".join(c[:3]) for c in CASES]


def _level2(els, obs, geometry, kind, basis, mu=MU, **kw):
    conn = CONN if basis in ("hat", "trapezoid") else None
    return LEVEL2[kind](
        *obs, els, geometry=geometry, mu=mu, nu=NU, basis=basis, connectivity=conn, **kw
    )


def _subset(els, idx):
    sub = bemcs.initialize_els()
    for name in ("x1", "y1", "x2", "y2"):
        setattr(sub, name, np.asarray(getattr(els, name))[idx].copy())
    bemcs.standardize_els_geometry(sub, reorder=False)
    return sub


# ============================================================================
# level 3: element kernels and basis transforms
# ============================================================================


def _jump_across_center(els, e, kind_fn, geometry, eps=1e-7):
    """Displacement just above minus just below the centre of element e, per source."""
    xc, yc = els.x_centers[e], els.y_centers[e]
    nx, ny = els.x_normals[e], els.y_normals[e]
    xo = np.array([xc + eps * nx, xc - eps * nx])
    yo = np.array([yc + eps * ny, yc - eps * ny])
    U, _ = kind_fn(xo, yo, els, geometry=geometry, mu=MU, nu=NU)
    return U[0] - U[1]


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_quadratic_slip_nodes_are_lagrange(els_curved, geometry):
    """At an element centre, only the central node's basis has a displacement jump,
    and the jump is a unit slip (along t or n in plane strain)."""
    e = 1
    jump = _jump_across_center(els_curved, e, GF.element_kernels_slip, geometry)
    cols = slice(3 * e, 3 * e + 3)
    others = np.delete(jump, np.r_[cols], axis=0)
    np.testing.assert_allclose(others, 0, atol=1e-6)  # other elements: continuous
    if geometry == "antiplane":
        np.testing.assert_allclose(np.abs(jump[cols, 0]), [0, 1, 0], atol=1e-6)
    else:
        t = np.array([els_curved.x_shears[e], els_curved.y_shears[e]])
        n = np.array([els_curved.x_normals[e], els_curved.y_normals[e]])
        for l, (along, across) in enumerate(((t, n), (n, t))):
            j = jump[cols, :, l]  # (node, [ux, uy])
            np.testing.assert_allclose(np.abs(j @ along), [0, 1, 0], atol=1e-6)
            np.testing.assert_allclose(j @ across, 0, atol=1e-6)


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_force_kernels_are_continuous_across_element(els_curved, geometry):
    jump = _jump_across_center(els_curved, 1, GF.element_kernels_force, geometry)
    np.testing.assert_allclose(jump, 0, atol=1e-6)


@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize("kind", ["slip", "force"])
def test_element_and_constant_transforms(els_curved, geometry, kind):
    n_nodes = 3 if kind == "slip" else 2
    elem_basis = "quadratic" if kind == "slip" else "linear"
    T_id = GF.basis_transform(els_curved, kind=kind, basis=elem_basis, geometry=geometry)
    T_c = GF.basis_transform(els_curved, kind=kind, basis="constant", geometry=geometry)
    if geometry == "antiplane":
        np.testing.assert_array_equal(T_id, np.eye(4 * n_nodes))
        np.testing.assert_array_equal(T_c, np.kron(np.eye(4), np.ones((n_nodes, 1))))
    else:
        assert T_id.shape == (4 * n_nodes, 2, 4 * n_nodes, 2)
        assert T_c.shape == (4 * n_nodes, 2, 4, 2)
        for l in (0, 1):  # no mixing between directions
            np.testing.assert_array_equal(T_id[:, l, :, l], np.eye(4 * n_nodes))
            np.testing.assert_array_equal(T_c[:, l, :, 1 - l], 0)
            np.testing.assert_array_equal(
                T_c[:, l, :, l], np.kron(np.eye(4), np.ones((n_nodes, 1)))
            )


@pytest.mark.parametrize("flag_index, flag", [(0, "shear"), (1, "normal")])
def test_derived_constant_slip_matches_analytic(els_curved, obs_points, flag_index, flag):
    """Constant slip derived from quadratic nodes == Crouch & Starfield analytic kernel."""
    U, S, _ = bemcs.get_slip_kernels(
        *obs_points, els_curved, geometry="planestrain", mu=MU, nu=NU, basis="constant"
    )
    sxx, syy, sxy, ux, uy = ref.constant_slip_kernel_planestrain(
        *obs_points, els_curved, MU, NU, flag
    )
    l = flag_index
    for got, want in ((U[:, :, 0, l], ux), (U[:, :, 1, l], uy), (S[:, :, 0, l], sxx),
                      (S[:, :, 1, l], syy), (S[:, :, 2, l], sxy)):
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)


def _slip_at_nodes_antiplane(els, T):
    """Slip and slip gradient at the 3 nodes of each element for each column of T."""
    m_slip, m_grad = core.get_matrices_slip_slip_gradient_antiplane(els)
    return m_slip @ T, m_grad @ T  # rows: 3 * element + node


def test_antiplane_hat_is_smooth_unit_bump(els_curved):
    T = GF.hat_coefficients(els_curved, CONN, geometry="antiplane")
    assert T.shape == (12, 2)
    slip, grad = _slip_at_nodes_antiplane(els_curved, T)
    for p, (a, b, c) in enumerate(CONN):
        assert slip[3 * b + 1, p] == pytest.approx(1.0)  # centre of middle element
        for e in (a, c):  # open ends: zero slip and gradient
            end = 3 * e if e == a else 3 * e + 2
            assert slip[end, p] == pytest.approx(0, abs=1e-12)
            assert grad[end, p] == pytest.approx(0, abs=1e-12)
        for left, right in ((a, b), (b, c)):  # internal nodes: continuous slip and gradient
            assert slip[3 * left + 2, p] == pytest.approx(slip[3 * right, p], abs=1e-12)
            assert grad[3 * left + 2, p] == pytest.approx(grad[3 * right, p], abs=1e-12)
        outside = [e for e in range(4) if e not in (a, b, c)]
        for e in outside:
            np.testing.assert_array_equal(T[3 * e : 3 * e + 3, p], 0)


def test_planestrain_hat_is_smooth_unit_bump(els_curved):
    T = GF.hat_coefficients(els_curved, CONN, geometry="planestrain")
    assert T.shape == (12, 2, 2, 2)
    m_slip, m_grad = core.get_matrices_slip_slip_gradient(els_curved, reference="global")
    for p, (a, b, c) in enumerate(CONN):
        for m in (0, 1):  # unit shear patch, unit normal patch
            # back to bemcs.bemcs's per-element block order [t0 t1 t2 n0 n1 n2]
            blocks = T[:, :, p, m].reshape(4, 3, 2).transpose(0, 2, 1).reshape(-1)
            slip = (m_slip @ blocks).reshape(-1, 2)  # global (x, y) slip per node
            grad = (m_grad @ blocks).reshape(-1, 2)
            unit = np.array(
                [els_curved.x_shears[b], els_curved.y_shears[b]]
                if m == 0
                else [els_curved.x_normals[b], els_curved.y_normals[b]]
            )
            np.testing.assert_allclose(slip[3 * b + 1], unit, atol=1e-12)
            for e, end in ((a, 3 * a), (c, 3 * c + 2)):
                np.testing.assert_allclose(slip[end], 0, atol=1e-12)
                np.testing.assert_allclose(grad[end], 0, atol=1e-12)
            for left, right in ((a, b), (b, c)):
                np.testing.assert_allclose(slip[3 * left + 2], slip[3 * right], atol=1e-12)
                np.testing.assert_allclose(grad[3 * left + 2], grad[3 * right], atol=1e-12)


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_trapezoid_force_is_continuous_and_tapers(els_curved, geometry):
    """Force at element ends: start = ramp-1 coefficient, end = ramp-0 coefficient.
    A trapezoid is zero at the open ends, continuous (in the global frame) at the
    internal nodes, and equal to 1 at both ends of its middle element."""
    T = GF.trapezoid_coefficients(els_curved, CONN, geometry=geometry)
    for p, (a, b, c) in enumerate(CONN):
        for m in ((0,) if geometry == "antiplane" else (0, 1)):
            def force(e, end):  # end 0 = start node (x1), 1 = end node (x2)
                row = 2 * e + (1 if end == 0 else 0)
                if geometry == "antiplane":
                    return np.array([T[row, p]])
                return els_curved.rot_mats[e] @ T[row, :, p, m]  # local -> global

            np.testing.assert_allclose(force(a, 0), 0, atol=1e-12)
            np.testing.assert_allclose(force(c, 1), 0, atol=1e-12)
            np.testing.assert_allclose(force(a, 1), force(b, 0), atol=1e-12)
            np.testing.assert_allclose(force(b, 1), force(c, 0), atol=1e-12)
            if geometry == "antiplane":
                np.testing.assert_allclose([force(b, 0), force(b, 1)], 1)
            else:  # unit force along t (m=0) or n (m=1) of the middle element
                local = els_curved.rot_mats[b].T @ force(b, 0)
                np.testing.assert_allclose(local, np.eye(2)[m], atol=1e-12)


# ============================================================================
# level 2: one source type
# ============================================================================


@pytest.mark.parametrize("kind", ["slip", "force"])
@pytest.mark.parametrize("missing", ["geometry", "mu"])
def test_level2_required_keywords(els_curved, obs_points, kind, missing):
    kw = dict(geometry="antiplane", mu=1.0)
    del kw[missing]
    with pytest.raises(TypeError):
        LEVEL2[kind](*obs_points, els_curved, **kw)


@pytest.mark.parametrize("kind", ["slip", "force"])
def test_level2_nu_required_for_planestrain(els_curved, obs_points, kind):
    with pytest.raises(TypeError, match="nu"):
        LEVEL2[kind](*obs_points, els_curved, geometry="planestrain", mu=1.0)
    LEVEL2[kind](*obs_points, els_curved, geometry="antiplane", mu=1.0)  # fine


@pytest.mark.parametrize("kind", ["slip", "force"])
def test_level2_geometry_and_mu_keyword_only(els_curved, obs_points, kind):
    with pytest.raises(TypeError):
        LEVEL2[kind](*obs_points, els_curved, "antiplane", 1.0)


@pytest.mark.parametrize(
    "kind, kwargs, match",
    [
        ("slip", dict(geometry="planestress"), "geometry"),
        ("slip", dict(mu=-1.0), "mu"),
        ("slip", dict(geometry="planestrain", nu=0.5), "nu"),
        ("slip", dict(basis="linear"), "not available"),
        ("force", dict(basis="quadratic"), "not available"),
        ("force", dict(basis="hat"), "not available"),
        ("slip", dict(basis="hat"), "needs connectivity"),
        ("force", dict(basis="trapezoid"), "needs connectivity"),
        ("slip", dict(connectivity=CONN), "only used with patch bases"),
        ("force", dict(basis="trapezoid", connectivity=np.array([[0, 1]])), r"\(n_patches, 3\)"),
        ("force", dict(basis="trapezoid", connectivity=np.array([[0.0, 1, 2]])), "int"),
        ("slip", dict(basis="hat", connectivity=np.array([[1, 2, 7]])), "out of range"),
    ],
)
def test_level2_invalid_arguments(els_curved, obs_points, kind, kwargs, match):
    kw = dict(geometry="antiplane", mu=1.0, nu=0.25)
    kw.update(kwargs)
    with pytest.raises(ValueError, match=match):
        LEVEL2[kind](*obs_points, els_curved, **kw)


def test_hat_patch_must_be_a_chain(els_curved, obs_points):
    with pytest.raises(ValueError, match="chain"):
        bemcs.get_slip_kernels(
            *obs_points, els_curved, geometry="antiplane", mu=1.0, basis="hat",
            connectivity=np.array([[0, 1, 3]]),
        )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_level2_shapes_and_info(els_curved, obs_points, case):
    geometry, kind, basis, n_src = case
    U, S, info = _level2(els_curved, obs_points, geometry, kind, basis)
    n_obs = obs_points[0].size
    if geometry == "planestrain":
        assert U.shape == (n_obs, n_src, 2, 2) and S.shape == (n_obs, n_src, 3, 2)
        assert info.directions == ("t", "n") and info.nu == NU
    else:
        assert U.shape == (n_obs, n_src, 1) and S.shape == (n_obs, n_src, 2)
        assert info.directions == () and info.nu is None
    assert (info.u_shape, info.s_shape) == (U.shape, S.shape)
    assert info.basis == {kind: basis} and info.backend == "numpy" and info.mu == MU
    assert np.all(info.src_label == kind[0])
    if basis in ("quadratic", "linear"):
        per = n_src // 4
        np.testing.assert_array_equal(info.src_index, np.repeat(np.arange(4), per))
        np.testing.assert_array_equal(info.src_node, np.tile(np.arange(per), 4))
    else:
        np.testing.assert_array_equal(info.src_index, np.arange(n_src))
        assert np.all(info.src_node == -1)
    assert np.isfinite(U).all() and np.isfinite(S).all()
    assert geometry in str(info)


def test_level2_defaults(els_curved, obs_points):
    for geometry in GEOMETRIES:
        for kind, default in (("slip", "quadratic"), ("force", "linear")):
            *_, info = LEVEL2[kind](*obs_points, els_curved, geometry=geometry, mu=1.0, nu=NU)
            assert info.basis == {kind: default}


@pytest.mark.parametrize("kind", ["slip", "force"])
def test_observation_points_are_flattened(els_curved, obs_points, kind):
    xo, yo = obs_points
    a = LEVEL2[kind](xo, yo, els_curved, geometry="antiplane", mu=1.0)
    b = LEVEL2[kind](xo[:, None], yo[:, None], els_curved, geometry="antiplane", mu=1.0)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_constant_and_patch_bases_are_combinations_of_element_bases(els_curved, obs_points, case):
    geometry, kind, basis, _ = case
    elem = "quadratic" if kind == "slip" else "linear"
    Ue, Se, _ = _level2(els_curved, obs_points, geometry, kind, elem)
    U, S, _ = _level2(els_curved, obs_points, geometry, kind, basis)
    conn = CONN if basis in ("hat", "trapezoid") else None
    T = GF.basis_transform(els_curved, kind=kind, basis=basis, geometry=geometry, connectivity=conn)
    sub = "osc,sb->obc" if geometry == "antiplane" else "oscl,slbm->obcm"
    np.testing.assert_allclose(U, np.einsum(sub, Ue, T), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(S, np.einsum(sub, Se, T), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize("kind", ["slip", "force"])
def test_displacement_scales_with_one_over_mu(els_curved, obs_points, geometry, kind):
    """Slip: stress ~ mu, displacement independent of mu. Force: the reverse."""
    U1, S1, _ = LEVEL2[kind](*obs_points, els_curved, geometry=geometry, mu=1.0, nu=NU)
    U2, S2, _ = LEVEL2[kind](*obs_points, els_curved, geometry=geometry, mu=2.5, nu=NU)
    if kind == "slip":
        np.testing.assert_allclose(U2, U1, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(S2, 2.5 * S1, rtol=1e-12, atol=1e-14)
    else:
        np.testing.assert_allclose(U2, U1 / 2.5, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(S2, S1, rtol=1e-12, atol=1e-14)


@pytest.mark.numba
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_numba_backend_matches_numpy(els_curved, obs_points, case):
    geometry, kind, basis, _ = case
    Un, Sn, info_n = _level2(els_curved, obs_points, geometry, kind, basis)
    Uj, Sj, info_j = _level2(els_curved, obs_points, geometry, kind, basis, use_numba=True)
    assert (info_n.backend, info_j.backend) == ("numpy", "numba")
    np.testing.assert_allclose(Uj, Un, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(Sj, Sn, rtol=1e-10, atol=1e-12)


# ============================================================================
# level 1: mesh
# ============================================================================


@pytest.mark.parametrize(
    "args, kwargs, match",
    [
        (("dislocation",), {}, "labels"),
        ((["s", "x", "s", "s"],), {}, "'s' or 'f'"),
        ((["s", "f"],), {}, "shape"),
        (("slip",), dict(basis={"slip": "linear"}), "not available"),
        (("slip",), dict(basis={"slp": "hat"}), "keys"),
        (("slip",), dict(basis={"slip": "hat"}), "needs connectivity"),
        (("slip",), dict(connectivity={"slip": CONN}), "only used with patch bases"),
        (("slip",), dict(basis={"force": "trapezoid"}), "no element is labelled"),
        (
            (["s", "s", "s", "f"],),
            dict(basis={"slip": "hat"}, connectivity={"slip": np.array([[1, 2, 3]])}),
            "not labelled",
        ),
    ],
)
def test_mesh_validation(els_curved, args, kwargs, match):
    with pytest.raises(ValueError, match=match):
        bemcs.Mesh(els_curved, *args, **kwargs)


def test_mesh_defaults_and_repr(els_curved):
    mesh = bemcs.Mesh(els_curved, ["s", "f", "s", "f"])
    assert mesh.basis == {"slip": "quadratic", "force": "linear"}
    assert mesh.kinds == ["slip", "force"]
    np.testing.assert_array_equal(mesh.elements("slip"), [0, 2])
    assert "slip: 2 elements" in repr(mesh)


def test_get_kernels_requires_mesh_and_keywords(els_curved, obs_points):
    mesh = bemcs.Mesh(els_curved, "slip")
    with pytest.raises(TypeError, match="Mesh"):
        bemcs.get_kernels(*obs_points, els_curved, geometry="antiplane", mu=1.0)
    with pytest.raises(TypeError):
        bemcs.get_kernels(*obs_points, mesh, mu=1.0)
    with pytest.raises(TypeError):
        bemcs.get_kernels(*obs_points, mesh, geometry="antiplane")
    with pytest.raises(TypeError, match="nu"):
        bemcs.get_kernels(*obs_points, mesh, geometry="planestrain", mu=1.0)


@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize("kind", ["slip", "force"])
def test_single_kind_mesh_equals_level2(els_curved, obs_points, geometry, kind):
    U, S, info = bemcs.get_kernels(
        *obs_points, bemcs.Mesh(els_curved, kind), geometry=geometry, mu=MU, nu=NU
    )
    U2, S2, info2 = LEVEL2[kind](*obs_points, els_curved, geometry=geometry, mu=MU, nu=NU)
    np.testing.assert_array_equal(U, U2)
    np.testing.assert_array_equal(S, S2)
    np.testing.assert_array_equal(info.src_index, info2.src_index)


@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize(
    "basis",
    [
        {},
        {"slip": "constant", "force": "constant"},
    ],
    ids=["element", "constant"],
)
def test_mixed_mesh_is_slip_block_then_force_block(els_curved, obs_points, geometry, basis):
    mesh = bemcs.Mesh(els_curved, ["s", "f", "s", "f"], basis=basis)
    U, S, info = bemcs.get_kernels(*obs_points, mesh, geometry=geometry, mu=MU, nu=NU)
    Us, Ss, info_s = bemcs.get_slip_kernels(
        *obs_points, _subset(els_curved, [0, 2]), geometry=geometry, mu=MU, nu=NU,
        basis=mesh.basis["slip"],
    )
    Uf, Sf, info_f = bemcs.get_force_kernels(
        *obs_points, _subset(els_curved, [1, 3]), geometry=geometry, mu=MU, nu=NU,
        basis=mesh.basis["force"],
    )
    ns = Us.shape[1]
    assert list(info.src_label) == ["s"] * ns + ["f"] * Uf.shape[1]
    np.testing.assert_array_equal(
        info.src_index, np.r_[np.array([0, 2])[info_s.src_index], np.array([1, 3])[info_f.src_index]]
    )
    np.testing.assert_array_equal(info.src_node, np.r_[info_s.src_node, info_f.src_node])
    np.testing.assert_allclose(U[:, :ns], Us, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(U[:, ns:], Uf, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(S[:, :ns], Ss, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(S[:, ns:], Sf, rtol=1e-12, atol=1e-14)
    assert info.basis == mesh.basis


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_mesh_with_patches_uses_global_connectivity(els_curved, obs_points, geometry):
    """Hat slip on elements 1-3 (global indices) plus linear force on element 0."""
    mesh = bemcs.Mesh(
        els_curved, ["f", "s", "s", "s"], basis={"slip": "hat"},
        connectivity={"slip": np.array([[1, 2, 3]])},
    )
    U, S, info = bemcs.get_kernels(*obs_points, mesh, geometry=geometry, mu=MU, nu=NU)
    assert list(info.src_label) == ["s", "f", "f"]
    np.testing.assert_array_equal(info.src_index, [0, 0, 0])  # patch row 0; element 0
    np.testing.assert_array_equal(info.src_node, [-1, 0, 1])
    Uh, Sh, _ = bemcs.get_slip_kernels(
        *obs_points, _subset(els_curved, [1, 2, 3]), geometry=geometry, mu=MU, nu=NU,
        basis="hat", connectivity=np.array([[0, 1, 2]]),
    )
    np.testing.assert_allclose(U[:, :1], Uh, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(S[:, :1], Sh, rtol=1e-10, atol=1e-12)


# ============================================================================
# temporary: agreement with the old per-case functions (remove in stage 3)
# ============================================================================

_JIT = pytest.mark.numba  # the old function calls a numba-jitted kernel internally
OLD_CASES = [
    ("antiplane", "slip", "quadratic"),
    ("antiplane", "force", "linear"),
    pytest.param(("antiplane", "force", "trapezoid"), marks=_JIT, id="antiplane-force-trapezoid"),
    ("planestrain", "slip", "quadratic"),
    ("planestrain", "slip", "hat"),
    ("planestrain", "force", "linear"),
    pytest.param(("planestrain", "force", "trapezoid"), marks=_JIT, id="planestrain-force-trapezoid"),
]


def _old(els, obs, geometry, kind, basis):
    """Old kernels as a list of (selector into new U/S, old array)."""
    xo, yo = obs
    n = xo.size
    arr = (els.x_centers, els.y_centers, els.half_lengths, els.rot_mats, els.rot_mats_inv)
    if (geometry, kind, basis) == ("antiplane", "slip", "quadratic"):
        sx, sy, u = core.get_displacement_stress_kernel_slip_antiplane(xo, yo, els, mu=MU)
        return [(("U", 0), u), (("S", 0), sx), (("S", 1), sy)]
    if (geometry, kind, basis) == ("antiplane", "force", "linear"):
        sx, sy, u = core.get_displacement_stress_kernel_force_antiplane(xo, yo, els, mu=MU)
        r = lambda K: K.transpose(0, 2, 1).reshape(n, -1)  # noqa: E731
        return [(("U", 0), r(u)), (("S", 0), r(sx)), (("S", 1), r(sy))]
    if (geometry, kind, basis) == ("antiplane", "force", "trapezoid"):
        # the old function ignores mu (issue 14), so compare at mu = 1
        sx, sy, u = GF.get_kernels_trapezoidalforce_antiplane(xo, yo, els, CONN)
        return [(("U", 0), u), (("S", 0), sx), (("S", 1), sy)]
    if (geometry, kind, basis) == ("planestrain", "slip", "quadratic"):
        out = []
        for l, flag in enumerate(("shear", "normal")):
            sxx, syy, sxy, ux, uy = core.get_displacement_stress_kernel(xo, yo, els, MU, NU, flag)
            out += [(("U", 0, l), ux), (("U", 1, l), uy)]
            out += [(("S", 0, l), sxx), (("S", 1, l), syy), (("S", 2, l), sxy)]
        return out
    if basis in ("hat", "trapezoid"):
        fn = (
            GF.get_kernels_quadratichatslip_planestrain
            if basis == "hat"
            else GF.get_kernels_trapezoidalforce_planestrain
        )
        ux, uy, sxx, syy, sxy = fn(xo, yo, els, CONN, MU, NU)
        return [(("U", 0, None), ux), (("U", 1, None), uy), (("S", 0, None), sxx),
                (("S", 1, None), syy), (("S", 2, None), sxy)]
    K = GF.get_displacement_stress_kernel_force_planestrain.py_func(xo, yo, *arr, MU, NU)
    out = []
    for c, name in enumerate(("U0", "U1", "S0", "S1", "S2")):
        r = K[c].transpose(0, 3, 2, 1).reshape(n, -1, 2)
        out += [((name[0], int(name[1]), l), r[:, :, l]) for l in (0, 1)]
    return out


@pytest.mark.parametrize(
    "case", OLD_CASES, ids=lambda c: "-".join(c) if isinstance(c, tuple) else None
)
def test_matches_old_function(els_curved, obs_points, case):
    geometry, kind, basis = case
    mu = 1.0 if (geometry, basis) == ("antiplane", "trapezoid") else MU
    U, S, _ = _level2(els_curved, obs_points, geometry, kind, basis, mu=mu)
    n = obs_points[0].size
    for sel, old in _old(els_curved, obs_points, geometry, kind, basis):
        arr = {"U": U, "S": S}[sel[0]]
        if len(sel) == 2:
            new = arr[:, :, sel[1]]
        elif sel[2] is None:  # interleaved patch columns [t0, n0, t1, n1, ...]
            new = arr[:, :, sel[1], :].reshape(n, -1)
        else:
            new = arr[:, :, sel[1], sel[2]]
        np.testing.assert_allclose(new, old, rtol=1e-10, atol=1e-12, err_msg=str(sel))
