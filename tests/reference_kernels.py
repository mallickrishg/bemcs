"""Independent reference solutions used only by the tests.

These are not part of the package: the package derives constant bases from the
quadratic-slip and linear-force kernels, and these references check that the
derivation and the higher-order kernels are right (tasks.md section 7.1).
"""

import numpy as np

import bemcs.bemcs as core


# ---------------------------------------------------------------------------
# analytic constant-slip kernels (Crouch & Starfield), plane strain
# ---------------------------------------------------------------------------


def constant_slip_kernel_planestrain(x_obs, y_obs, els, mu, nu, flag):
    """Function to calculate displacement and stress kernels at a numpy array of locations [x_obs,y_obs]

    flag can either be "shear" or "normal" for kernels resulting shear slip or tensile slip

    kernels returned are u_x, u_y, stress_xx, stress_yy, stress_xy
    """
    n_obs = len(x_obs)
    n_els = len(els.x1)
    kernel_ux = np.zeros((n_obs, n_els))
    kernel_uy = np.zeros((n_obs, n_els))
    kernel_sxx = np.zeros((n_obs, n_els))
    kernel_syy = np.zeros((n_obs, n_els))
    kernel_sxy = np.zeros((n_obs, n_els))

    # check for which slip component kernels the user wants
    if flag == "shear":
        flag_strike_slip = 1.0
        flag_tensile_slip = 0.0
    elif flag == "normal":
        flag_strike_slip = 0.0
        flag_tensile_slip = 1.0
    else:
        raise ValueError(
            "shear/tensile flag must be 'shear' or 'normal', no other values allowed"
        )

    for i in range(n_els):
        # Center observation locations (no translation needed)
        x_trans = x_obs - els.x_centers[i]
        y_trans = y_obs - els.y_centers[i]

        # Rotate observations such that fault element is horizontal
        rotated_coordinates = els.rot_mats_inv[i, :, :] @ np.vstack(
            (x_trans.T, y_trans.T)
        )
        x_rot = rotated_coordinates[0, :].T + els.x_centers[i]
        y_rot = rotated_coordinates[1, :].T + els.y_centers[i]

        slip_vector = np.array([1.0])
        strike_slip = slip_vector * flag_strike_slip
        tensile_slip = slip_vector * flag_tensile_slip

        # Calculate displacements and stresses for current element
        (
            displacement_local,
            stress_local,
        ) = core.displacements_stresses_constant_no_rotation(
            x_rot,
            y_rot,
            els.half_lengths[i],
            mu,
            nu,
            strike_slip,
            tensile_slip,
            els.x_centers[i],
            els.y_centers[i],
        )
        displacement_eval, stress_eval = core.rotate_displacement_stress(
            displacement_local, stress_local, els.rot_mats_inv[i, :, :]
        )
        # index = 3 * i
        kernel_sxx[:, i] = stress_eval[0, :]
        kernel_syy[:, i] = stress_eval[1, :]
        kernel_sxy[:, i] = stress_eval[2, :]
        kernel_ux[:, i] = displacement_eval[0, :]
        kernel_uy[:, i] = displacement_eval[1, :]
    return kernel_sxx, kernel_syy, kernel_sxy, kernel_ux, kernel_uy


# ---------------------------------------------------------------------------
# point-force quadrature along an element
# ---------------------------------------------------------------------------

# linear force ramps on s in [-w, w]: ramp 0 = (w+s)/2w, ramp 1 = (w-s)/2w
RAMPS = [lambda s, w: (w + s) / (2 * w), lambda s, w: (w - s) / (2 * w)]
CONSTANT = lambda s, w: np.ones_like(s)  # noqa: E731


def _quadrature(els, e, n_points=80):
    R = els.rot_mats[e]
    w = els.half_lengths[e]
    gl, gw = np.polynomial.legendre.leggauss(n_points)
    s = gl * w
    xs = els.x_centers[e] + R[0, 0] * s
    ys = els.y_centers[e] + R[1, 0] * s
    return s, gw * w, xs, ys, w, R


def kelvin_line_force_planestrain(els, e, xo, yo, f_local, shape, mu, nu):
    """ux, uy, sxx, syy, sxy of a local-frame force with profile `shape` on element e."""
    s, wts, xs, ys, w, R = _quadrature(els, e)
    f_global = R @ np.asarray(f_local, dtype=float)
    out = np.zeros((5, len(xo)))
    for si, wt, x0, y0 in zip(s, wts, xs, ys):
        b = shape(si, w) * wt
        u = core.kelvin_point_source_disp(xo, yo, x0, y0, *f_global, mu, nu)
        t = core.kelvin_point_source_stress(xo, yo, x0, y0, *f_global, mu, nu)
        out += b * np.array([u[0], u[1], t[0], t[1], t[2]], dtype=float)
    return out


def laplacian_line_force_antiplane(els, e, xo, yo, shape, mu):
    """uz, sxz, syz of a unit antiplane force with profile `shape` on element e."""
    s, wts, xs, ys, w, _ = _quadrature(els, e)
    out = np.zeros((3, len(xo)))
    for si, wt, x0, y0 in zip(s, wts, xs, ys):
        b = shape(si, w) * wt
        out += b * np.array(
            [
                core.laplacian_pointforce_disp(xo, yo, x0, y0, 1.0, mu),
                core.laplacian_pointforce_sx(xo, yo, x0, y0, 1.0),
                core.laplacian_pointforce_sy(xo, yo, x0, y0, 1.0),
            ]
        )
    return out
