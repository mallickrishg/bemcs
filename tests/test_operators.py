"""Node operators, continuity rows, tractions and values along elements
(tasks.md 7.4), in the interleaved column convention of the kernels.

The continuity rows are checked against the frozen pre-revamp operators in
tests/reference_operators.py (block column order [s0 s1 s2 n0 n1 n2] per
element), through a column permutation.
"""

import numpy as np
import pytest

import bemcs
import bemcs.bemAssembly as GF
import bemcs.bemcs as core
import reference_operators as refops

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
    old = refops.get_matrices_slip_slip_gradient(els_curved, reference=reference)
    perm = _old_to_new_columns(4)
    for N, O in zip(new, old):
        np.testing.assert_allclose(N, O[:, perm], atol=1e-14)


def test_slip_node_matrices_antiplane_match_old(els_curved):
    new = GF.slip_node_matrices(els_curved, geometry="antiplane")
    old = refops.get_matrices_slip_slip_gradient_antiplane(els_curved)
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
        o, i, t = refops.construct_smoothoperator(els, *labels)
        _, grad = refops.get_matrices_slip_slip_gradient(els)
    else:
        o, i, t = refops.construct_smoothoperator_antiplane(els, *labels)
        _, grad = refops.get_matrices_slip_slip_gradient_antiplane(els)
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
    mesh = bemcs.Mesh(els_curved, ["s", "s", "f", "f"], basis={"force": "constant"})
    A, info = mesh.continuity_operator(geometry=geometry)
    assert A.shape[1] == mesh.n_columns(geometry=geometry) == pe * (6 + 2)
    assert set(info["kind"]) == {"s"}  # constant force: no continuity rows
    assert not A[:, mesh.columns("force", geometry=geometry)].any()
    # the node shared by slip element 1 and force element 2 is open (clamped) for slip
    shared = np.isclose(info["x"], els_curved.x2[1]) & np.isclose(info["y"], els_curved.y2[1])
    assert set(info["node_type"][shared]) == {"open"}
    assert set(info["quantity"][shared]) == {"slip", "gradient"}
    assert set(info["element"][shared]) == {1}


def test_mesh_continuity_operator_rejects_linear_force(els_curved):
    """Force sources are solved with trapezoids only (decided 2026-11-16)."""
    mesh = bemcs.Mesh(els_curved, ["s", "s", "f", "f"])  # default force basis: linear
    with pytest.raises(ValueError, match="trapezoid"):
        mesh.continuity_operator(geometry="planestrain")


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


@pytest.mark.parametrize("local", [False, True])
def test_traction_kernels_planestrain(els_curved, local):
    """t = sigma . n per source coefficient; optionally projected on (shear, normal)."""
    xo, yo = els_curved.x_centers, els_curved.y_centers + 0.05
    _, S, _ = bemcs.get_slip_kernels(xo, yo, els_curved, geometry="planestrain", mu=1.0, nu=0.25)
    nx, ny = els_curved.x_normals, els_curved.y_normals
    sx, sy = els_curved.x_shears, els_curved.y_shears
    t = GF.traction_kernels(S, nx, ny, sx, sy) if local else GF.traction_kernels(S, nx, ny)
    assert t.shape == (4, 12, 2, 2)
    for i in range(4):
        for j in range(12):
            for l in range(2):
                sig = np.array([[S[i, j, 0, l], S[i, j, 2, l]], [S[i, j, 2, l], S[i, j, 1, l]]])
                tv = sig @ np.array([nx[i], ny[i]])
                if local:
                    tv = np.array([tv @ [sx[i], sy[i]], tv @ [nx[i], ny[i]]])
                np.testing.assert_allclose(t[i, j, :, l], tv, atol=1e-14)


def test_traction_kernels_antiplane(els_curved):
    xo, yo = els_curved.x_centers, els_curved.y_centers + 0.05
    _, S, _ = bemcs.get_slip_kernels(xo, yo, els_curved, geometry="antiplane", mu=1.0)
    t = GF.traction_kernels(S, els_curved.x_normals, els_curved.y_normals)
    expected = S[:, :, 0] * els_curved.x_normals[:, None] + S[:, :, 1] * els_curved.y_normals[:, None]
    np.testing.assert_allclose(t[:, :, 0], expected, atol=1e-15)


@pytest.mark.parametrize("geometry", ["antiplane", "planestrain"])
def test_values_on_elements_quadratic_slip(els_curved, geometry):
    """Slip along each element = quadratic Lagrange interpolation of its 3
    coefficients, which are slip values at s = -2w/3, 0, +2w/3 (local frame),
    rotated to (x, y) in plane strain."""
    pe = 1 if geometry == "antiplane" else 2
    c = np.random.default_rng(3).normal(size=12 * pe)
    n_eval = 5
    x, y, v = GF.values_on_elements(els_curved, c, kind="slip", geometry=geometry, n_eval=n_eval)
    t = (np.arange(n_eval) + 0.5) / n_eval * 2 - 1  # sub-interval midpoints, s / w
    nodes = np.array([-2.0, 0.0, 2.0]) / 3  # interpolation points, as s / w

    def lagrange_k(k):
        others = [m for m in range(3) if m != k]
        return np.prod([(t - nodes[m]) / (nodes[k] - nodes[m]) for m in others], axis=0)

    lagrange = np.stack([lagrange_k(k) for k in range(3)], axis=1)
    for e in range(4):
        R = els_curved.rot_mats[e]
        w = els_curved.half_lengths[e]
        rows = slice(e * n_eval, (e + 1) * n_eval)
        np.testing.assert_allclose(x[rows], els_curved.x_centers[e] + R[0, 0] * t * w, atol=1e-14)
        np.testing.assert_allclose(y[rows], els_curved.y_centers[e] + R[1, 0] * t * w, atol=1e-14)
        local = lagrange @ c.reshape(4, 3, pe)[e]  # (n_eval, pe)
        if geometry == "antiplane":
            np.testing.assert_allclose(v[rows], local[:, 0], atol=1e-14)
        else:
            unit = np.array([[els_curved.x_shears[e], els_curved.x_normals[e]],
                             [els_curved.y_shears[e], els_curved.y_normals[e]]])
            np.testing.assert_allclose(v[rows], local @ unit.T, atol=1e-14)
            _, _, v_loc = GF.values_on_elements(
                els_curved, c, kind="slip", geometry=geometry, n_eval=n_eval, frame="local"
            )
            np.testing.assert_allclose(v_loc[rows], local, atol=1e-14)


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
