"""One-ring hats (tasks.md 7.4, T2): one hat per non-tip element, built from the
mesh topology, including triple junctions."""

import numpy as np
import pytest

import bemcs
import bemcs.bemAssembly as GF
from meshes import kinked_chain


def _els(segments):
    els = bemcs.initialize_els()
    (els.x1, els.y1), (els.x2, els.y2) = [np.array(v, float).T for v in zip(*segments)]
    bemcs.standardize_els_geometry(els, reorder=False)
    return els


def _branch(start, end, n):
    """n elements along a straight branch from start to end."""
    pts = [np.array(start) + (np.array(end) - np.array(start)) * k / n for k in range(n + 1)]
    return [(tuple(a), tuple(b)) for a, b in zip(pts[:-1], pts[1:])]


MESHES = {
    "straight": lambda: kinked_chain(0.0),
    "kink30": lambda: kinked_chain(30.0),
    # main fault through J = (0, 0) plus a splay; orientations: 1 in, 2 out
    "splay_1in_2out": lambda: _els(
        _branch((-2, 0), (0, 0), 2) + _branch((0, 0), (2, 0), 2) + _branch((0, 0), (1.2, -1.6), 2)
    ),
    # 2 in, 1 out, with a non-symmetric junction angle
    "Y_2in_1out": lambda: _els(
        _branch((-1.6, 1.2), (0, 0), 2) + _branch((-1.9, -0.6), (0, 0), 2) + _branch((0, 0), (2, 0.3), 2)
    ),
}
GEOMETRIES = ["antiplane", "planestrain"]


def _flat(T, geometry):
    return T if geometry == "antiplane" else T.reshape(T.shape[0] * 2, -1)


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_equal_to_three_element_hats_on_a_straight_fault(geometry):
    els = kinked_chain(0.0)  # 5 elements: non-tip elements 1, 2, 3
    T1 = GF.hat_coefficients(els, geometry=geometry)
    T3 = GF.hat_coefficients(els, np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]]), geometry=geometry)
    np.testing.assert_array_equal(GF.hat_elements(els), [1, 2, 3])
    np.testing.assert_allclose(T1, T3, atol=1e-13)


@pytest.mark.parametrize("name", MESHES)
@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_hats_span_exactly_the_continuity_null_space(name, geometry):
    els = MESHES[name]()
    pe = 1 if geometry == "antiplane" else 2
    A, _ = GF.continuity_rows(els, kind="slip", geometry=geometry)
    T = _flat(GF.hat_coefficients(els, geometry=geometry), geometry)
    n_hats = len(GF.hat_elements(els))
    assert T.shape[1] == pe * n_hats
    null_dim = A.shape[1] - np.linalg.matrix_rank(A)
    assert pe * n_hats == null_dim
    np.testing.assert_allclose(A @ T, 0, atol=1e-12)
    assert np.linalg.matrix_rank(T) == pe * n_hats


@pytest.mark.parametrize("name", MESHES)
@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_each_hat_has_unit_slip_at_its_own_centre(name, geometry):
    els = MESHES[name]()
    pe = 1 if geometry == "antiplane" else 2
    T = _flat(GF.hat_coefficients(els, geometry=geometry), geometry)
    m_local, _ = GF.slip_node_matrices(els, geometry=geometry, reference="local")
    for p, e in enumerate(GF.hat_elements(els)):
        centre = m_local[pe * (3 * e + 1) : pe * (3 * e + 1) + pe]
        np.testing.assert_allclose(centre @ T[:, pe * p : pe * (p + 1)], np.eye(pe), atol=1e-12)


@pytest.mark.parametrize("name", MESHES)
@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_reduced_solve_equals_general_solve(name, geometry):
    """Prescribed slip at every non-tip element centre: hats (2(N - N_tip) unknowns)
    and the general form (element DOFs + continuity rows) give the same slip."""
    els = MESHES[name]()
    pe = 1 if geometry == "antiplane" else 2
    centres = GF.hat_elements(els)
    m_local, _ = GF.slip_node_matrices(els, geometry=geometry, reference="local")
    B = np.vstack([m_local[pe * (3 * e + 1) : pe * (3 * e + 1) + pe] for e in centres])
    target = np.random.default_rng(len(name)).normal(size=pe * len(centres))
    A, _ = GF.continuity_rows(els, kind="slip", geometry=geometry)
    c_general = np.linalg.solve(np.vstack((B, A)), np.r_[target, np.zeros(len(A))])
    T = _flat(GF.hat_coefficients(els, geometry=geometry), geometry)
    a = np.linalg.solve(B @ T, target)
    np.testing.assert_allclose(T @ a, c_general, atol=1e-11)


def test_mesh_uses_one_ring_hats_without_connectivity():
    els = MESHES["splay_1in_2out"]()  # 6 slip elements, tips 0, 3, 5
    mesh = bemcs.Mesh(els, "slip", basis={"slip": "hat"})
    assert mesh.n_src("slip") == 3
    xo, yo = np.array([0.3, -0.7]), np.array([0.9, -0.4])
    U, S, info = bemcs.get_kernels(xo, yo, mesh, geometry="planestrain", mu=1.0, nu=0.25)
    assert S.shape == (2, 3, 3, 2)
    np.testing.assert_array_equal(info.src_index, GF.hat_elements(els))
    Ue, Se, _ = bemcs.get_slip_kernels(xo, yo, els, geometry="planestrain", mu=1.0, nu=0.25)
    T = GF.hat_coefficients(els, geometry="planestrain")
    np.testing.assert_allclose(S, np.einsum("oscl,slbm->obcm", Se, T), atol=1e-12)


def test_splay_hat_next_to_the_junction_crosses_it():
    """A splay element's hat puts slip on all three branches at the junction."""
    els = MESHES["splay_1in_2out"]()
    T = GF.hat_coefficients(els, geometry="antiplane")
    p = list(GF.hat_elements(els)).index(4)  # splay element touching J
    touched = sorted({k // 3 for k in np.flatnonzero(np.abs(T[:, p]) > 1e-12)})
    assert touched == [1, 2, 4, 5]  # its far neighbour and both main-fault elements at J
