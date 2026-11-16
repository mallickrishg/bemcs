"""Plane-strain compliance used for heterogeneous media (tasks.md issue 1).

The heterogeneous solver itself (`bemcs.solve` with 'h' rows) is tested in
test_solve.py: exact antiplane image solution, continuity of the true traction
across a plane-strain interface, and independence of the sampled side.
"""

import numpy as np

import bemcs.bemAssembly as GF

MU, NU = 1.3, 0.27


def test_compliance_inverts_plane_strain_hooke():
    rng = np.random.default_rng(0)
    exx, eyy, exy = rng.normal(size=(3, 5))
    lam = 2 * MU * NU / (1 - 2 * NU)
    sxx = 2 * MU * exx + lam * (exx + eyy)
    syy = 2 * MU * eyy + lam * (exx + eyy)
    sxy = 2 * MU * exy
    got = GF.plane_strain_compliance(sxx, syy, sxy, MU, NU)
    np.testing.assert_allclose(got, (exx, eyy, exy), atol=1e-14)
    # the old nu/(1+nu) (plane-stress) form differs: issue 1
    nup = NU / (1 + NU)
    old_exx = ((1 - nup) * sxx - nup * syy) / (2 * MU)
    assert np.abs(old_exx - exx).max() > 1e-3
