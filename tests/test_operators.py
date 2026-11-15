"""Node operators, continuity rows, tractions and values along elements
(stage 2, tasks.md 7.4), in the interleaved column convention of the kernels.

The old block-ordered functions (columns [s0 s1 s2 n0 n1 n2] per element) are
used as references through a column permutation; those comparisons go away
with the old functions in stage 3.
"""

import numpy as np
import pytest

import bemcs
import bemcs.bemAssembly as GF
import bemcs.bemcs as core

CONN = np.array([[0, 1, 2], [1, 2, 3]])


def _els(x1, y1, x2, y2):
    els = bemcs.initialize_els()
    els.x1, els.y1 = np.array(x1, float), np.array(y1, float)
    els.x2, els.y2 = np.array(x2, float), np.array(y2, float)
    bemcs.standardize_els_geometry(els, reorder=False)
    return els


@pytest.fixture(scope="module")
def els_triple():
    """Three elements meeting at the origin, plus one attached to the first."""
    return _els(
        [-2.0, -1.0, 0.0, 0.0],
        [0.3, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 1.0, 0.6],
        [0.0, 0.0, 0.1, -0.8],
    )


def _old_to_new_columns(n_els):
    """perm[new_col] = old_col for plane-strain quadratic slip."""
    perm = np.empty(6 * n_els, dtype=int)
    for e in range(n_els):
        for node in range(3):
            for l in range(2):
                perm[2 * (3 * e + node) + l] = 6 * e + 3 * l + node
    return perm


def _same_rows_up_to_sign(A, B, tol=1e-12):
    assert A.shape == B.shape
    for a, b in zip(A, B):
        if not (np.allclose(a, b, atol=tol) or np.allclose(a, -b, atol=tol)):
            raise AssertionError(f"row mismatch:\n{a}\n{b}")


# ----------------------------------------------------------------------------
# node matrices
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("reference", ["global", "local"])
def test_slip_node_matrices_planestrain_match_old(els_curved, reference):
    new = GF.slip_node_matrices(els_curved, geometry="planestrain", reference=reference)
    old = core.get_matrices_slip_slip_gradient(els_curved, reference=reference)
    perm = _old_to_new_columns(4)
    for N, O in zip(new, old):
        np.testing.assert_allclose(N, O[:, perm], atol=1e-14)


def test_slip_node_matrices_antiplane_match_old(els_curved):
    new = GF.slip_node_matrices(els_curved, geometry="antiplane")
    old = core.get_matrices_slip_slip_gradient_antiplane(els_curved)
    for N, O in zip(new, old):
        np.testing.assert_allclose(N, O, atol=1e-14)


@pytest.mark.parametrize("geometry", ["antiplane", "planestrain"])
def test_force_node_matrix_is_ramp_values(els_curved, geometry):
    M = GF.force_node_matrix(els_curved, geometry=geometry, reference="local")
    pe = 1 if geometry == "antiplane" else 2
    for e in range(4):
        for k, ramp in ((0, 1), (1, 0)):  # start node <- ramp 1, end node <- ramp 0
            for c in range(pe):
                row = M[pe * (2 * e + k) + c]
                assert row[pe * (2 * e + ramp) + c] == 1.0
                assert row.sum() == 1.0


# ----------------------------------------------------------------------------
# continuity rows vs construct_smoothoperator
# ----------------------------------------------------------------------------


def _old_with_clamped_tips(els, geometry):
    """construct_smoothoperator rows, with a slip-gradient row added after the slip
    rows of every open node (clamped tips, tasks.md 7.4)."""
    labels = core.label_nodes(els, verbose=False)
    pe = 1 if geometry == "antiplane" else 2
    if geometry == "planestrain":
        o, i, t = core.construct_smoothoperator(els, *labels)
        _, grad = core.get_matrices_slip_slip_gradient(els)
    else:
        o, i, t = core.construct_smoothoperator_antiplane(els, *labels)
        _, grad = core.get_matrices_slip_slip_gradient_antiplane(els)
    open_rows = []
    for k, node in enumerate(labels[0]):
        n = int(np.abs(node).item())
        open_rows += list(o[pe * k : pe * (k + 1)]) + [grad[pe * n + c] for c in range(pe)]
    return np.vstack(open_rows + [i, t])


@pytest.mark.parametrize("which", ["curved", "triple"])
@pytest.mark.parametrize("geometry", ["antiplane", "planestrain"])
def test_slip_continuity_matches_construct_smoothoperator(els_curved, els_triple, which, geometry):
    els = els_curved if which == "curved" else els_triple
    n = len(els.x1)
    A, info = GF.continuity_rows(els, kind="slip", geometry=geometry)
    old = _old_with_clamped_tips(els, geometry)
    if geometry == "planestrain":
        old = old[:, _old_to_new_columns(n)]
    _same_rows_up_to_sign(A, old)
    assert set(info["node_type"]) <= {"open", "overlap", "triple"}
    if which == "triple":
        pe = 1 if geometry == "antiplane" else 2
        assert (info["node_type"] == "triple").sum() == 3 * pe
    # one BC row per component on every non-tip element closes the system
    pe = 1 if geometry == "antiplane" else 2
    n_tip = len(GF.tip_elements(els))
    assert pe * (n - n_tip) + len(A) == 3 * pe * n


def test_tip_elements(els_curved, els_triple):
    np.testing.assert_array_equal(GF.tip_elements(els_curved), [0, 3])
    np.testing.assert_array_equal(GF.tip_elements(els_triple), [0, 2, 3])
    lone = _els([0.0], [0.0], [1.0], [0.0])
    with pytest.raises(ValueError, match="both end nodes open"):
        GF.tip_elements(lone)
    mesh = bemcs.Mesh(els_curved, ["s", "s", "s", "f"])
    np.testing.assert_array_equal(mesh.tip_elements("slip"), [0, 2])  # node 2-3 is open for slip
    np.testing.assert_array_equal(mesh.bc_elements("slip"), [1])


def test_force_continuity_rejects_triple_junction(els_triple):
    with pytest.raises(ValueError, match="triple"):
        GF.continuity_rows(els_triple, kind="force", geometry="planestrain")


@pytest.mark.parametrize("geometry", ["antiplane", "planestrain"])
def test_hats_and_trapezoids_satisfy_continuity(els_curved, geometry):
    """The patch bases lie in the null space of the continuity rows: A T = 0."""
    for kind, fn in (("slip", GF.hat_coefficients), ("force", GF.trapezoid_coefficients)):
        A, _ = GF.continuity_rows(els_curved, kind=kind, geometry=geometry)
        T = fn(els_curved, CONN, geometry=geometry)
        T = T if geometry == "antiplane" else T.reshape(T.shape[0] * 2, -1)
        np.testing.assert_allclose(A @ T, 0, atol=1e-12)


# ----------------------------------------------------------------------------
# mesh-level continuity operator
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("geometry", ["antiplane", "planestrain"])
def test_mesh_continuity_operator_blocks(els_curved, geometry):
    pe = 1 if geometry == "antiplane" else 2
    mesh = bemcs.Mesh(els_curved, ["s", "s", "f", "f"])
    A, info = mesh.continuity_operator(geometry=geometry)
    assert A.shape[1] == mesh.n_columns(geometry=geometry) == pe * (6 + 4)
    slip_rows = info["kind"] == "s"
    # slip rows act only on slip columns, force rows only on force columns
    assert not A[slip_rows][:, mesh.columns("force", geometry=geometry)].any()
    assert not A[~slip_rows][:, mesh.columns("slip", geometry=geometry)].any()
    # the node shared by slip element 1 and force element 2 is open for both
    shared = (np.isclose(info["x"], els_curved.x2[1])) & (np.isclose(info["y"], els_curved.y2[1]))
    assert set(info["node_type"][shared]) == {"open"}
    assert set(info["element"][shared & slip_rows]) == {1}
    assert set(info["element"][shared & ~slip_rows]) == {2}


def test_mesh_continuity_operator_skips_patch_and_constant_bases(els_curved):
    mesh = bemcs.Mesh(
        els_curved, "slip", basis={"slip": "hat"}, connectivity={"slip": CONN}
    )
    A, _ = mesh.continuity_operator(geometry="planestrain")
    assert A.shape == (0, mesh.n_columns(geometry="planestrain"))
    mesh = bemcs.Mesh(els_curved, "force", basis={"force": "constant"})
    assert mesh.continuity_operator(geometry="antiplane")[0].shape == (0, 4)


# ----------------------------------------------------------------------------
# tractions and values along elements
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["global", "local"])
def test_traction_kernels_match_old(els_curved, flag):
    xo, yo = els_curved.x_centers, els_curved.y_centers + 0.05
    _, S, _ = bemcs.get_slip_kernels(xo, yo, els_curved, geometry="planestrain", mu=1.0, nu=0.25)
    if flag == "global":
        t = GF.traction_kernels(S, els_curved.x_normals, els_curved.y_normals)
    else:
        t = GF.traction_kernels(
            S, els_curved.x_normals, els_curved.y_normals, els_curved.x_shears, els_curved.y_shears
        )
    for l, f in enumerate(("shear", "normal")):
        K = core.get_displacement_stress_kernel(xo, yo, els_curved, 1.0, 0.25, f)
        old = core.get_traction_kernels(els_curved, K, flag=flag)
        for c in range(2):
            np.testing.assert_allclose(t[:, :, c, l], old[c], rtol=1e-9, atol=1e-11)


def test_traction_kernels_antiplane_match_old(els_curved):
    xo, yo = els_curved.x_centers, els_curved.y_centers + 0.05
    _, S, _ = bemcs.get_slip_kernels(xo, yo, els_curved, geometry="antiplane", mu=1.0)
    t = GF.traction_kernels(S, els_curved.x_normals, els_curved.y_normals)
    old = core.get_traction_kernels_antiplane(els_curved, (S[:, :, 0], S[:, :, 1]))
    np.testing.assert_allclose(t[:, :, 0], old, atol=1e-14)


def test_values_on_elements_match_get_slipvector_on_fault(els_curved):
    rng = np.random.default_rng(3)
    c_new = rng.normal(size=24)
    c_old = np.empty(24)
    c_old[_old_to_new_columns(4)] = c_new
    x, y, v = GF.values_on_elements(els_curved, c_new, kind="slip", geometry="planestrain", n_eval=5)
    xo, yo, sx, sy = core.get_slipvector_on_fault(els_curved, c_old, 5)
    np.testing.assert_allclose(x, xo, atol=1e-14)
    np.testing.assert_allclose(y, yo, atol=1e-14)
    np.testing.assert_allclose(v, np.c_[sx, sy], atol=1e-13)


def test_values_on_elements_antiplane_match_old(els_curved):
    c = np.random.default_rng(4).normal(size=12)
    x, y, v = GF.values_on_elements(els_curved, c, kind="slip", geometry="antiplane", n_eval=5)
    xo, yo, s = core.get_slipvector_on_fault_antiplane(els_curved, c, 5)
    np.testing.assert_allclose(v, s, atol=1e-14)
    np.testing.assert_allclose(x, xo, atol=1e-14)


def test_force_values_are_linear_ramps(els_curved):
    c = np.zeros(8)
    c[2 * 1 + 0] = 1.0  # element 1, ramp 0 = (w+s)/2w: 0 at start, 1 at end
    _, _, v = GF.values_on_elements(els_curved, c, kind="force", geometry="antiplane", n_eval=4)
    np.testing.assert_allclose(v[4:8], [0.125, 0.375, 0.625, 0.875])
    assert not v[:4].any() and not v[8:].any()


@pytest.mark.parametrize("geometry", ["antiplane", "planestrain"])
def test_mesh_values_through_patch_basis(els_curved, geometry):
    """A unit hat amplitude gives unit slip at the centre of its middle element."""
    mesh = bemcs.Mesh(els_curved, "slip", basis={"slip": "hat"}, connectivity={"slip": CONN})
    x = np.zeros(mesh.n_columns(geometry=geometry))
    x[0] = 1.0  # patch 0 (middle element 1), direction t
    _, _, v = mesh.values_on_elements(x, "slip", geometry=geometry, n_eval=3, frame="local")
    centre = 3 * 1 + 1
    np.testing.assert_allclose(v[centre], 1.0 if geometry == "antiplane" else [1.0, 0.0], atol=1e-12)
