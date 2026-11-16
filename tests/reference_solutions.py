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


# ---------------------------------------------------------------------------
# Edge dislocation near a bonded planar bimaterial interface (Dundurs 1969), plane strain
# ---------------------------------------------------------------------------
# Interface y = 0; material 1 (mu1) below, material 2 (mu2) above; same nu. A dislocation
# with Burgers vector b = bx + i by at z0 = x0 + i y0, y0 < 0 (in material 1).
# Material 1: phi = A log(z - z0) + images at the mirror point z0* (in material 2),
#             psi = conj(A) log(z - z0) - A conj(z0) / (z - z0) + images;
# material 2: transmitted terms with singularities at z0 (in material 1).
# The homogeneous part is the infinite-medium dislocation, A = -i mu1 b / (pi (kappa + 1)).
# The image and transmitted coefficients follow from continuity of displacement and of the
# traction resultant along y = 0, imposed at many points (real-linear least squares); the
# ansatz is complete if the residual is at round-off. Log cuts point away from the interface.

_IMG = ("p_log", "p_1", "p_2", "s_log", "s_1", "s_2", "s_3")  # phi: log, (z-p)^-1, ^-2; psi: log, ^-1, ^-2, ^-3


def _clog(w, up):
    """log w with its branch cut along the +i (up=True) or -i direction from 0."""
    rot = 1j if up else -1j
    return np.log(np.abs(w)) + 1j * (np.angle(w * rot) - np.angle(rot))


def _terms(z, p, up):
    """Basis functions (value, derivative, second derivative) at z for singularities at p."""
    w = z - p
    L = _clog(w, up)
    return {
        "log": (L, 1 / w, -1 / w**2),
        1: (1 / w, -1 / w**2, 2 / w**3),
        2: (1 / w**2, -2 / w**3, 6 / w**4),
        3: (1 / w**3, -3 / w**4, 12 / w**5),
    }


def _bimat_potentials(z, z0, A, coef, upper):
    """phi, phi', phi'', psi, psi' in material 1 (upper=False) or 2 (upper=True)."""
    zb0 = np.conj(z0)
    if not upper:
        t0 = _terms(z, z0, up=False)  # the dislocation, cut downward
        ti = _terms(z, zb0, up=True)  # images in material 2, cut upward
        phi = [A * t0["log"][k] for k in range(3)]
        psi = [np.conj(A) * t0["log"][k] - A * zb0 * t0[1][k] for k in range(2)]
        c = coef[0]
    else:
        ti = _terms(z, z0, up=False)  # transmitted, singular at z0 (material 1), cut downward
        phi = [0 * z] * 3
        psi = [0 * z] * 2
        c = coef[1]
    for k in range(3):
        phi[k] = phi[k] + c["p_log"] * ti["log"][k] + c["p_1"] * ti[1][k] + c["p_2"] * ti[2][k]
    for k in range(2):
        psi[k] = psi[k] + c["s_log"] * ti["log"][k] + c["s_1"] * ti[1][k] + c["s_2"] * ti[2][k] + c["s_3"] * ti[3][k]
    if upper:  # constants (rigid translation / resultant)
        phi[0] = phi[0] + c["p_0"]
        psi[0] = psi[0] + c["s_0"]
    return phi[0], phi[1], phi[2], psi[0], psi[1]


def _unpack(q):
    v = q[0::2] + 1j * q[1::2]
    c1 = dict(zip(_IMG, v[:7]))
    c2 = dict(zip(_IMG + ("p_0", "s_0"), v[7:16]))
    return (c1, c2)


def edge_dislocation_bimaterial_constants(z0, b, mu1, mu2, nu, n_points=200):
    """Image/transmitted coefficients and the relative continuity residual."""
    kappa = 3 - 4 * nu
    A = -1j * mu1 * b / (np.pi * (kappa + 1))
    d = -z0.imag
    xs = z0.real + d * np.r_[-np.geomspace(1e3, 1e-2, n_points // 2), np.geomspace(1e-2, 1e3, n_points // 2)]
    z = xs + 0j

    def fields(q):
        c = _unpack(q)
        out = []
        for upper, mu in ((False, mu1), (True, mu2)):
            phi, dphi, _, psi, _ = _bimat_potentials(z, z0, A, c, upper)
            u = (kappa * phi - z * np.conj(dphi) - np.conj(psi)) / (2 * mu)
            F = phi + z * np.conj(dphi) + np.conj(psi)
            out.append((u, F))
        (u1, F1), (u2, F2) = out
        r = np.r_[u1 - u2, F1 - F2]
        return np.r_[r.real, r.imag]

    n_q = 2 * (7 + 9)
    r0 = fields(np.zeros(n_q))
    M = np.stack([fields(np.eye(n_q)[k]) - r0 for k in range(n_q)], axis=1)
    q, *_ = np.linalg.lstsq(M, -r0, rcond=None)
    rel = np.abs(M @ q + r0).max() / np.abs(r0).max()
    return A, _unpack(q), rel


def edge_dislocation_bimaterial(x, y, x0, y0, b, mu1, mu2, nu):
    """True stresses (sxx, syy, sxy) of the dislocation at (x0, y0 < 0) with Burgers vector
    b = (bx, by), interface y = 0, mu1 below and mu2 above. Also returns the residual."""
    z0 = complex(x0, y0)
    bb = complex(*b)
    A, c, rel = edge_dislocation_bimaterial_constants(z0, bb, mu1, mu2, nu)
    z = np.asarray(x, float) + 1j * np.asarray(y, float)
    out = np.zeros((3,) + z.shape)
    for upper in (False, True):
        mask = (z.imag > 0) if upper else (z.imag <= 0)
        zz = z[mask]
        phi, dphi, d2phi, psi, dpsi = _bimat_potentials(zz, z0, A, c, upper)
        s_sum = 4 * dphi.real
        s_dif = 2 * (np.conj(zz) * d2phi + dpsi)
        out[0][mask] = 0.5 * (s_sum - s_dif.real)
        out[1][mask] = 0.5 * (s_sum + s_dif.real)
        out[2][mask] = 0.5 * s_dif.imag
    return out, rel


def edge_dislocation_bimaterial_displacement(x, y, x0, y0, b, mu1, mu2, nu):
    """Displacement (ux, uy) with the dislocation's cut running straight down from (x0, y0)."""
    z0 = complex(x0, y0)
    A, c, _ = edge_dislocation_bimaterial_constants(z0, complex(*b), mu1, mu2, nu)
    kappa = 3 - 4 * nu
    z = np.asarray(x, float) + 1j * np.asarray(y, float)
    u = np.zeros(z.shape, dtype=complex)
    for upper, mu in ((False, mu1), (True, mu2)):
        mask = (z.imag > 0) if upper else (z.imag <= 0)
        phi, dphi, _, psi, _ = _bimat_potentials(z[mask], z0, A, c, upper)
        u[mask] = (kappa * phi - z[mask] * np.conj(dphi) - np.conj(psi)) / (2 * mu)
    return np.array([u.real, u.imag])
