"""Exact solutions used as independent references (no package code).

Circular elastic inhomogeneity in plane strain under a uniform remote stress
(Muskhelishvili, Some Basic Problems of the Mathematical Theory of Elasticity, §56).
Kolosov-Muskhelishvili potentials, kappa = 3 - 4 nu (plane strain):

    sxx + syy           = 4 Re phi'(z)
    syy - sxx + 2i sxy  = 2 (conj(z) phi''(z) + psi'(z))
    2 mu (ux + i uy)    = kappa phi(z) - z conj(phi'(z)) - conj(psi(z))
    traction resultant  F = phi(z) + z conj(phi'(z)) + conj(psi(z))   (dF ~ traction ds)

Matrix (|z| > a): phi = G z + b1 a^2 / z,  psi = G2 z + c1 a^2 / z + c3 a^4 / z^3,
with G = (sxx + syy)/4, G2 = (syy - sxx)/2 + i sxy from the remote stress.
Inclusion (|z| < a): phi = A z, psi = B z (uniform stress).
The five complex constants follow from continuity of displacement and of F around
the circle. They are found here by imposing both at many boundary points (a small
real-linear least-squares system); the ansatz is exact, which the residual confirms.
"""

import numpy as np


def _potentials(z, a, p, inside):
    """phi, phi', phi'', psi, psi' for parameters p = (G, G2, b1, c1, c3, A, B)."""
    G, G2, b1, c1, c3, A, B = p
    if inside:
        return A * z, A + 0 * z, 0 * z, B * z, B + 0 * z
    return (G * z + b1 * a**2 / z, G - b1 * a**2 / z**2, 2 * b1 * a**2 / z**3,
            G2 * z + c1 * a**2 / z + c3 * a**4 / z**3, G2 - c1 * a**2 / z**2 - 3 * c3 * a**4 / z**4)


def _disp_and_resultant(z, a, p, inside, mu, kappa):
    phi, dphi, _, psi, _ = _potentials(z, a, p, inside)
    u = (kappa * phi - z * np.conj(dphi) - np.conj(psi)) / (2 * mu)
    F = phi + z * np.conj(dphi) + np.conj(psi)
    return u, F


def circular_inclusion_constants(a, stress, mu_m, mu_i, nu, n_points=64):
    """Solve for (b1, c1, c3, A, B); returns the parameter tuple and the relative
    residual of the continuity conditions (~1e-15 if the ansatz is exact)."""
    sxx, syy, sxy = stress
    G, G2 = (sxx + syy) / 4.0, (syy - sxx) / 2.0 + 1j * sxy
    kappa = 3 - 4 * nu
    z = a * np.exp(1j * np.linspace(0, 2 * np.pi, n_points, endpoint=False))

    def residual(q):
        b1, c1, c3, A, B = q[0::2] + 1j * q[1::2]
        p = (G, G2, b1, c1, c3, A, B)
        uo, Fo = _disp_and_resultant(z, a, p, False, mu_m, kappa)
        ui, Fi = _disp_and_resultant(z, a, p, True, mu_i, kappa)
        r = np.r_[uo - ui, Fo - Fi]
        return np.r_[r.real, r.imag]

    r0 = residual(np.zeros(10))  # residuals are affine in the 10 real unknowns
    M = np.stack([residual(np.eye(10)[k]) - r0 for k in range(10)], axis=1)
    q, *_ = np.linalg.lstsq(M, -r0, rcond=None)
    rel = np.abs(M @ q + r0).max() / max(np.abs(r0).max(), 1e-300)
    b1, c1, c3, A, B = q[0::2] + 1j * q[1::2]
    return (G, G2, b1, c1, c3, A, B), rel


def circular_inclusion_planestrain(x, y, a, stress, mu_m, mu_i, nu):
    """Exact total fields (ux, uy, sxx, syy, sxy) of a circular inclusion (radius a,
    centred at the origin, shear modulus mu_i) in an infinite matrix (mu_m), both with
    Poisson's ratio nu, under the remote stress (sxx, syy, sxy). True stresses."""
    p, _ = circular_inclusion_constants(a, stress, mu_m, mu_i, nu)
    kappa = 3 - 4 * nu
    z = np.asarray(x, float) + 1j * np.asarray(y, float)
    inside = np.abs(z) < a
    out = np.zeros((5,) + z.shape)
    for mask, ins, mu in ((inside, True, mu_i), (~inside, False, mu_m)):
        zz = z[mask]
        phi, dphi, d2phi, psi, dpsi = _potentials(zz, a, p, ins)
        u = (kappa * phi - zz * np.conj(dphi) - np.conj(psi)) / (2 * mu)
        s_sum = 4 * dphi.real
        s_dif = 2 * (np.conj(zz) * d2phi + dpsi)  # syy - sxx + 2i sxy
        sxx = 0.5 * (s_sum - s_dif.real)
        syy = 0.5 * (s_sum + s_dif.real)
        out[0][mask], out[1][mask] = u.real, u.imag
        out[2][mask], out[3][mask], out[4][mask] = sxx, syy, 0.5 * s_dif.imag
    return out


def inclusion_stress_closed_form(stress, mu_m, mu_i, nu):
    """Uniform stress inside the inclusion (Eshelby / Muskhelishvili closed form):
    mean stress scaled by (kappa+1) mu_i / (2 mu_i + mu_m (kappa-1)), deviatoric part
    by mu_i (kappa+1) / (mu_m + kappa mu_i)."""
    sxx, syy, sxy = stress
    k = 3 - 4 * nu
    p, d, t = 0.5 * (sxx + syy), 0.5 * (sxx - syy), sxy
    fp = (k + 1) * mu_i / (2 * mu_i + mu_m * (k - 1))
    fd = mu_i * (k + 1) / (mu_m + k * mu_i)
    return np.array([fp * p + fd * d, fp * p - fd * d, fd * t])
