"""Check plane-strain linear-force kernels against an independent reference:
numerical integration of the Kelvin point-source solution along the element.

Conventions established by these tests: force DOFs are in the element-local
frame (index 0 = along the element, 1 = normal), and basis functions 0 and 1 are
the linear ramps (w+s)/2w and (w-s)/2w for s in [-w, w] (local coordinate).
"""

import numpy as np
import pytest

import bemcs
import bemcs.bemAssembly as GF
import bemcs.bemcs as core

MU, NU = 1.0, 0.3


def _element(angle):
    els = bemcs.initialize_els()
    els.x1, els.y1 = np.array([0.0]), np.array([0.0])
    els.x2 = np.array([2 * np.cos(angle)])
    els.y2 = np.array([2 * np.sin(angle)])
    bemcs.standardize_els_geometry(els, reorder=False)
    return els


def _kelvin_reference(els, xo, yo, f_local, ramp):
    """ux, uy, sxx, syy, sxy at (xo, yo) for one local-frame force with a ramp basis."""
    R = els.rot_mats[0]
    w = els.half_lengths[0]
    f_global = R @ f_local
    gl, gw = np.polynomial.legendre.leggauss(80)
    out = np.zeros((5, len(xo)))
    for s, wt in zip(gl * w, gw * w):
        xs = els.x_centers[0] + R[0, 0] * s
        ys = els.y_centers[0] + R[1, 0] * s
        b = ramp(s, w) * wt
        u = core.kelvin_point_source_disp(xo, yo, xs, ys, *f_global, MU, NU)
        t = core.kelvin_point_source_stress(xo, yo, xs, ys, *f_global, MU, NU)
        out += b * np.array([u[0], u[1], t[0], t[1], t[2]], dtype=float)
    return out


# basis 0 ramps up towards the element's end node, basis 1 towards its start node
RAMPS = [lambda s, w: (w + s) / (2 * w), lambda s, w: (w - s) / (2 * w)]


@pytest.fixture(scope="module", params=[0.0, np.pi / 2, np.pi / 6, -0.7])
def element_and_numba_kernel(request):
    els = _element(request.param)
    xo = np.array([-1.0, 0.5, 2.5, 1.0, 3.0])
    yo = np.array([0.7, -0.8, 1.5, 2.0, -1.0])
    K = GF.get_displacement_stress_kernel_force_planestrain(
        xo,
        yo,
        els.x_centers,
        els.y_centers,
        els.half_lengths,
        els.rot_mats,
        els.rot_mats_inv,
        mu=MU,
        nu=NU,
    )
    return els, xo, yo, K


@pytest.mark.parametrize("force_dir", [0, 1])
@pytest.mark.parametrize("basis", [0, 1])
def test_bemAssembly_planestrain_force_kernel_matches_kelvin(
    element_and_numba_kernel, force_dir, basis
):
    els, xo, yo, K = element_and_numba_kernel
    ref = _kelvin_reference(els, xo, yo, np.eye(2)[force_dir], RAMPS[basis])
    for k in range(5):
        np.testing.assert_allclose(
            K[k][:, force_dir, basis, 0], ref[k], rtol=1e-7, atol=1e-9
        )


@pytest.mark.xfail(
    strict=True,
    reason="bemcs.bemcs.get_displacement_stress_kernel_force_planestrain applies "
    "rot_mats_inv to displacements and mis-rotates stresses: wrong for any "
    "non-horizontal element (errors up to ~0.16 vs the Kelvin reference).",
)
def test_bemcs_planestrain_force_kernel_matches_kelvin_rotated():
    els = _element(np.pi / 6)
    xo = np.array([-1.0, 0.5, 2.5, 1.0, 3.0])
    yo = np.array([0.7, -0.8, 1.5, 2.0, -1.0])
    K = core.get_displacement_stress_kernel_force_planestrain(
        xo, yo, els, mu=MU, nu=NU
    )
    ref = _kelvin_reference(els, xo, yo, np.eye(2)[0], RAMPS[0])
    err = max(
        min(np.abs(K[k][:, 0, b, 0] - ref[k]).max() for b in (0, 1)) for k in range(5)
    )
    assert err < 1e-7
