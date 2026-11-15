"""Force kernels against an independent reference: Gauss-Legendre integration of
the point-force solution along the element (Kelvin in plane strain, the 2-D
Laplace Green's function in antiplane).

Conventions checked here: plane-strain force DOFs are in the element-local
frame (direction 0 = along the element, 1 = normal); linear ramp 0 is (w+s)/2w
and ramp 1 is (w-s)/2w for s in [-w, w]; constant force is their sum.
"""

import numpy as np
import pytest

import bemcs
import reference_kernels as ref

MU, NU = 1.3, 0.3
XO = np.array([-1.0, 0.5, 2.5, 1.0, 3.0])
YO = np.array([0.7, -0.8, 1.5, 2.0, -1.0])
ANGLES = [0.0, np.pi / 2, np.pi / 6, -0.7]


def _element(angle):
    els = bemcs.initialize_els()
    els.x1, els.y1 = np.array([0.0]), np.array([0.0])
    els.x2 = np.array([2 * np.cos(angle)])
    els.y2 = np.array([2 * np.sin(angle)])
    bemcs.standardize_els_geometry(els, reorder=False)
    return els


SHAPES = {"linear": ref.RAMPS, "constant": [ref.CONSTANT]}


@pytest.mark.parametrize("angle", ANGLES)
@pytest.mark.parametrize("basis", ["linear", "constant"])
@pytest.mark.parametrize("use_numba", [False, pytest.param(True, marks=pytest.mark.numba)])
def test_planestrain_force_matches_kelvin(angle, basis, use_numba):
    els = _element(angle)
    U, S, info = bemcs.get_force_kernels(
        XO, YO, els, geometry="planestrain", mu=MU, nu=NU, basis=basis, use_numba=use_numba
    )
    for j, shape in enumerate(SHAPES[basis]):
        for l in range(2):  # direction t, n
            want = ref.kelvin_line_force_planestrain(els, 0, XO, YO, np.eye(2)[l], shape, MU, NU)
            got = np.concatenate([U[:, j, :, l].T, S[:, j, :, l].T])
            np.testing.assert_allclose(got, want, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("angle", ANGLES)
@pytest.mark.parametrize("basis", ["linear", "constant"])
@pytest.mark.parametrize("use_numba", [False, pytest.param(True, marks=pytest.mark.numba)])
def test_antiplane_force_matches_laplace(angle, basis, use_numba):
    els = _element(angle)
    U, S, info = bemcs.get_force_kernels(
        XO, YO, els, geometry="antiplane", mu=MU, basis=basis, use_numba=use_numba
    )
    for j, shape in enumerate(SHAPES[basis]):
        want = ref.laplacian_line_force_antiplane(els, 0, XO, YO, shape, MU)
        got = np.concatenate([U[:, j, :].T, S[:, j, :].T])
        np.testing.assert_allclose(got, want, rtol=1e-7, atol=1e-9)
