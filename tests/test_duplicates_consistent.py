"""While bemcs.bemcs and bemcs.bemAssembly both define the same function names
(see KNOWN_DUPLICATES in test_api.py), the two versions must agree numerically.
These tests are what makes it safe to delete one copy of each pair."""

import numpy as np
import pytest

import bemcs
import bemcs.bemAssembly as GF


def _array_args(els):
    return (
        els.x_centers,
        els.y_centers,
        els.half_lengths,
        els.rot_mats,
        els.rot_mats_inv,
    )


def test_slip_functions_agree():
    x = np.linspace(-1, 1, 7)
    for a in (0.5, 1.0, 2.3):
        np.testing.assert_allclose(
            bemcs.slip_functions(x, a), GF.slip_functions(x, a, basis="quadratic")
        )


def test_slipgradient_functions_agree():
    x = np.linspace(-1, 1, 7)
    for a in (0.5, 1.0, 2.3):
        np.testing.assert_allclose(
            bemcs.slipgradient_functions(x, a),
            GF.slipgradient_functions(x, a, basis="quadratic"),
        )


def test_get_matrices_slip_slip_gradient_agree(els_curved):
    A = bemcs.get_matrices_slip_slip_gradient(els_curved, flag="node")
    B = GF.get_matrices_slip_slip_gradient(els_curved, flag="node", basis="quadratic")
    assert len(A) == len(B) == 2
    for p, q in zip(A, B):
        np.testing.assert_allclose(p, q)


@pytest.mark.parametrize(
    "name",
    [
        "get_displacement_stress_kernel_slip_antiplane",
        "get_displacement_stress_kernel_force_antiplane",
    ],
)
def test_antiplane_kernels_agree(name, els_curved, obs_points):
    xo, yo = obs_points
    A = getattr(bemcs, name)(xo, yo, els_curved, mu=1.0)
    B = getattr(GF, name).py_func(xo, yo, *_array_args(els_curved), mu=1.0)  # no JIT
    assert len(A) == len(B) == 3
    for p, q in zip(A, B):
        np.testing.assert_allclose(p, q, rtol=1e-8, atol=1e-10)
