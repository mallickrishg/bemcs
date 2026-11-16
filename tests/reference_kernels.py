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


# ---------------------------------------------------------------------------
# Crouch & Starfield constant-slip element (moved from bemcs.bemcs, no longer in the package)
# ---------------------------------------------------------------------------


def constant_kernel(x, y, a, nu):
    """From Starfield and Crouch, pages 49 and 82"""
    f = np.zeros((7, x.size))

    f[0, :] = (
        -1
        / (4 * np.pi * (1 - nu))
        * (
            y * (np.arctan2(y, (x - a)) - np.arctan2(y, (x + a)))
            - (x - a) * np.log(np.sqrt((x - a) ** 2 + y**2))
            + (x + a) * np.log(np.sqrt((x + a) ** 2 + y**2))
        )
    )

    f[1, :] = (
        -1
        / (4 * np.pi * (1 - nu))
        * ((np.arctan2(y, (x - a)) - np.arctan2(y, (x + a))))
    )

    f[2, :] = (
        1
        / (4 * np.pi * (1 - nu))
        * (np.log(np.sqrt((x - a) ** 2 + y**2)) - np.log(np.sqrt((x + a) ** 2 + y**2)))
    )

    f[3, :] = (
        1
        / (4 * np.pi * (1 - nu))
        * (y / ((x - a) ** 2 + y**2) - y / ((x + a) ** 2 + y**2))
    )

    f[4, :] = (
        1
        / (4 * np.pi * (1 - nu))
        * ((x - a) / ((x - a) ** 2 + y**2) - (x + a) / ((x + a) ** 2 + y**2))
    )

    f[5, :] = (
        1
        / (4 * np.pi * (1 - nu))
        * (
            ((x - a) ** 2 - y**2) / ((x - a) ** 2 + y**2) ** 2
            - ((x + a) ** 2 - y**2) / ((x + a) ** 2 + y**2) ** 2
        )
    )

    f[6, :] = (
        2
        * y
        / (4 * np.pi * (1 - nu))
        * ((x - a) / ((x - a) ** 2 + y**2) ** 2 - (x + a) / ((x + a) ** 2 + y**2) ** 2)
    )
    return f


def f_slip_to_displacement_stress(x_component, y_component, f, y, mu, nu):
    """This is the generalization from Starfield and Crouch"""
    displacement = np.zeros((2, y.size))
    stress = np.zeros((3, y.size))

    # The sign change here is to:
    # 1 - Ensure consistenty with Okada convention
    # 2 - For a horizontal/flat fault make the upper half move in the +x direction
    x_component = -1 * x_component
    y_component = -1 * y_component

    displacement[0, :] = x_component * (
        2 * (1 - nu) * f[1, :] - y * f[4, :]
    ) + y_component * (-1 * (1 - 2 * nu) * f[2, :] - y * f[3, :])

    displacement[1, :] = x_component * (
        (1 - 2 * nu) * f[2, :] - y * f[3, :]
    ) + y_component * (
        2 * (1 - nu) * f[1, :] - y * -f[4, :]
    )  # Note the negative sign in front f[4, :] because f[4, :] = f,xx = -f,yy

    stress[0, :] = 2 * x_component * mu * (
        2 * f[3, :] + y * f[5, :]
    ) + 2 * y_component * mu * (-f[4, :] + y * f[6, :])

    stress[1, :] = 2 * x_component * mu * (-y * f[5, :]) + 2 * y_component * mu * (
        -f[4, :] - y * f[6, :]
    )

    stress[2, :] = 2 * x_component * mu * (
        -f[4, :] + y * f[6, :]
    ) + 2 * y_component * mu * (-y * f[5, :])

    return displacement, stress


def displacements_stresses_constant_no_rotation(
    x,
    y,
    a,
    mu,
    nu,
    x_component,
    y_component,
    x_center,
    y_center,
):
    """Calculate displacements and stresses for constant slip elements"""
    # Translate into local coordinate system
    x = x - x_center
    y = y - y_center
    f = constant_kernel(x, y, a, nu)
    displacement, stress = f_slip_to_displacement_stress(
        x_component, y_component, f, y, mu, nu
    )
    return displacement, stress


def rotate_displacement_stress(displacement, stress, inverse_rotation_matrix):
    """Rotate displacements stresses from local to global reference frame"""
    displacement = np.matmul(displacement.T, inverse_rotation_matrix).T
    for i in range(0, stress.shape[1]):
        stress_tensor = np.array(
            [[stress[0, i], stress[2, i]], [stress[2, i], stress[1, i]]]
        )
        stress_tensor_global = (
            inverse_rotation_matrix.T @ stress_tensor @ inverse_rotation_matrix
        )
        stress[0, i] = stress_tensor_global[0, 0]
        stress[1, i] = stress_tensor_global[1, 1]
        stress[2, i] = stress_tensor_global[0, 1]
    return displacement, stress



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
        ) = displacements_stresses_constant_no_rotation(
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
        displacement_eval, stress_eval = rotate_displacement_stress(
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
