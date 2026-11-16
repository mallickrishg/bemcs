"""Tests for the three-level kernel API (tasks.md section 7):

level 3  element_kernels_slip/force, basis_transform, hat/trapezoid coefficients
level 2  get_slip_kernels, get_force_kernels
level 1  Mesh, get_kernels

Comparisons with independent references (analytic constant slip, point-force
quadrature) live in test_force_kernels_reference.py and here; near-field and
precision checks in test_near_field.py and test_kernel_precision.py.
"""

import numpy as np
import pytest

import bemcs
import bemcs.bemAssembly as GF
import reference_kernels as ref

MU, NU = 1.3, 0.3
CONN = np.array([[0, 1, 2], [1, 2, 3]])  # hats: neighbours share two elements
TRAP_CONN = np.array([[0, 1, 2]])  # trapezoids: neighbours may share at most one element
GEOMETRIES = ["antiplane", "planestrain"]
LEVEL2 = {"slip": bemcs.get_slip_kernels, "force": bemcs.get_force_kernels}

# (kind, basis, n_src on the 4-element test mesh)
BASES = [
    ("slip", "quadratic", 12),
    ("slip", "constant", 4),
    ("slip", "hat", 2),
    ("force", "linear", 8),
    ("force", "constant", 4),
    ("force", "trapezoid", 1),
]
CASES = [(g, k, b, n) for g in GEOMETRIES for (k, b, n) in BASES]
CASE_IDS = ["-".join(c[:3]) for c in CASES]


def _level2(els, obs, geometry, kind, basis, mu=MU, **kw):
    conn = {"hat": CONN, "trapezoid": TRAP_CONN}.get(basis)
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
    m_slip, m_grad = GF.slip_node_matrices(els, geometry="antiplane")
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
    m_slip, m_grad = GF.slip_node_matrices(els_curved, geometry="planestrain", reference="global")
    for p, (a, b, c) in enumerate(CONN):
        for m in (0, 1):  # unit shear patch, unit normal patch
            coefs = T[:, :, p, m].reshape(-1)  # interleaved [t0, n0, t1, n1, ...]
            slip = (m_slip @ coefs).reshape(-1, 2)  # global (x, y) slip per node
            grad = (m_grad @ coefs).reshape(-1, 2)
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
    T = GF.trapezoid_coefficients(els_curved, TRAP_CONN, geometry=geometry)
    for p, (a, b, c) in enumerate(TRAP_CONN):
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
    default = "numba" if GF._check_numba(None) else "numpy"  # numba when installed
    assert info.basis == {kind: basis} and info.backend == default and info.mu == MU
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
    conn = {"hat": CONN, "trapezoid": TRAP_CONN}.get(basis)
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
    Un, Sn, info_n = _level2(els_curved, obs_points, geometry, kind, basis, use_numba=False)
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


@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize("kind", ["slip", "force"])
def test_points_on_an_element_give_the_two_sided_average(els_curved, kind, geometry):
    """On an element (local y at rounding level) stresses and displacements are
    the mean of the values just above and below the element."""
    e = 1
    t = np.array([-0.7, -0.2, 0.0, 0.35, 0.8]) * els_curved.half_lengths[e]
    R = els_curved.rot_mats[e]
    x_on = els_curved.x_centers[e] + R[0, 0] * t
    y_on = els_curved.y_centers[e] + R[1, 0] * t
    nx, ny = els_curved.x_normals[e], els_curved.y_normals[e]
    kw = dict(geometry=geometry, mu=MU, nu=NU)
    U0, S0, _ = LEVEL2[kind](x_on, y_on, els_curved, **kw)
    Up, Sp, _ = LEVEL2[kind](x_on + 1e-9 * nx, y_on + 1e-9 * ny, els_curved, **kw)
    Um, Sm, _ = LEVEL2[kind](x_on - 1e-9 * nx, y_on - 1e-9 * ny, els_curved, **kw)
    assert np.abs(Sp - Sm).max() > 0.1  # there is a jump to average over
    np.testing.assert_allclose(S0, 0.5 * (Sp + Sm), atol=1e-6)
    np.testing.assert_allclose(U0, 0.5 * (Up + Um), atol=1e-6)


def test_trapezoid_patches_may_share_at_most_one_element(els_curved, obs_points):
    """Neighbouring trapezoids share one element ([0,1,2], [2,3,4], ...), unlike hats (two)."""
    with pytest.raises(ValueError, match="share at most one element"):
        GF.trapezoid_coefficients(els_curved, CONN, geometry="planestrain")
    with pytest.raises(ValueError, match="share at most one element"):
        bemcs.get_force_kernels(*obs_points, els_curved, geometry="antiplane", mu=1.0,
                                basis="trapezoid", connectivity=CONN)
    hats = GF.hat_coefficients(els_curved, CONN, geometry="antiplane")  # fine for hats
    assert hats.shape == (12, 2)


def test_default_trapezoids_share_exactly_one_element():
    from test_kernel_precision import _chain

    els = _chain(0.0)  # 5 elements
    conn = GF.trapezoid_connectivity(els)
    np.testing.assert_array_equal(conn, [[0, 1, 2], [2, 3, 4]])
    for a, b in zip(conn[:-1], conn[1:]):
        assert len(set(a) & set(b)) == 1


def _line_els(segments):
    els = bemcs.initialize_els()
    els.x1 = np.array([a[0] for a, _ in segments], float)
    els.y1 = np.array([a[1] for a, _ in segments], float)
    els.x2 = np.array([b[0] for _, b in segments], float)
    els.y2 = np.array([b[1] for _, b in segments], float)
    bemcs.standardize_els_geometry(els, reorder=False)
    return els


def _straight(n, x0=0.0, y=0.0):
    return [((x0 + i, y), (x0 + i + 1, y)) for i in range(n)]


def test_trapezoids_restart_at_a_break_symmetrically():
    """8 elements on [0, 8], break at x = 4: collocation (middle) elements 2 and 5, 1.5
    elements from the break on both sides; the two trapezoids there share nothing."""
    els = _line_els(_straight(8))
    conn = GF.trapezoid_connectivity(els, breaks=[[4.0, 0.0]])
    np.testing.assert_array_equal(conn, [[1, 2, 3], [4, 5, 6]])
    GF._check_trapezoid_overlap(conn)
    # longer: every other element counted from the break on each side
    els = _line_els(_straight(12))
    conn = GF.trapezoid_connectivity(els, breaks=[[6.0, 0.0]])
    np.testing.assert_array_equal(conn[:, 1], [2, 4, 7, 9])
    centres = els.x_centers[conn[:, 1]]
    np.testing.assert_allclose(np.sort(6 - centres[centres < 6]), np.sort(centres[centres > 6] - 6))  # 1.5, 3.5
    # without breaks: the previous default, through x = 6
    np.testing.assert_array_equal(GF.trapezoid_connectivity(els)[:, 1], [1, 3, 5, 7, 9])


def test_trapezoids_restart_at_two_breaks_and_on_closed_loops():
    els = _line_els(_straight(11))
    conn = GF.trapezoid_connectivity(els, breaks=[[3.0, 0.0], [8.0, 0.0]])
    GF._check_trapezoid_overlap(conn)
    mids = conn[:, 1].tolist()
    assert 1 in mids and 2 not in mids  # left segment [0, 1, 2], counted from the break at 3
    assert 4 in mids and 9 in mids  # both sides of the break at 8 flanked 1.5 elements away
    # a closed square loop of 8 elements with a break at one corner
    sq = [((0, 0), (1, 0)), ((1, 0), (2, 0)), ((2, 0), (2, 1)), ((2, 1), (2, 2)),
          ((2, 2), (1, 2)), ((1, 2), (0, 2)), ((0, 2), (0, 1)), ((0, 1), (0, 0))]
    els = _line_els(sq)
    conn = GF.trapezoid_connectivity(els, breaks=[[0.0, 0.0]])
    GF._check_trapezoid_overlap(conn)
    # 8 elements between the break and itself: an even segment can be flanked at 1.5 elements on
    # one side of the break only (the other gets 2.5)
    d = np.hypot(els.x_centers[conn[:, 1]], els.y_centers[conn[:, 1]])
    assert np.sort(d)[0] == pytest.approx(1.5) and np.sort(d)[1] <= 2.55


def test_mesh_restarts_trapezoids_at_fault_crossings():
    """A fault crossing an interface (no shared node) at an interface node."""
    interface = _straight(8, x0=-4.0, y=-1.0)
    fault = [((0.0, 0.0), (0.0, -0.75)), ((0.0, -0.75), (0.0, -1.5)), ((0.0, -1.5), (0.0, -2.25))]
    els = _line_els(fault + interface)
    labels = ["s"] * 3 + ["f"] * 8
    np.testing.assert_allclose(GF.fault_crossings(els, labels), [[0.0, -1.0]])
    mesh = bemcs.Mesh(els, labels=labels, basis={"slip": "hat", "force": "trapezoid"})
    np.testing.assert_array_equal(mesh.connectivity["force"], 3 + np.array([[1, 2, 3], [4, 5, 6]]))
    # no crossing: the default stays None (stride 2 from one end)
    fault_off = [((5.0, 0.0), (5.0, -0.75)), ((5.0, -0.75), (5.0, -1.5)), ((5.0, -1.5), (5.0, -2.25))]
    mesh = bemcs.Mesh(_line_els(fault_off + interface), labels=labels,
                      basis={"slip": "hat", "force": "trapezoid"})
    assert len(mesh.fault_crossings) == 0 and mesh.connectivity["force"] is None
    # an explicit connectivity is kept as given
    given = 3 + np.array([[0, 1, 2], [2, 3, 4], [4, 5, 6]])
    mesh = bemcs.Mesh(els, labels=labels, basis={"slip": "hat", "force": "trapezoid"},
                      connectivity={"force": given})
    np.testing.assert_array_equal(mesh.connectivity["force"], given)
