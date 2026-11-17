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


# ---------------------------------------------------------------------------
# point dislocations, and quadratic slip on an element by dislocation quadrature
# ---------------------------------------------------------------------------
# Textbook solutions (Hirth & Lothe, Theory of Dislocations, ch. 3), independent of
# the package. Dislocation at the origin with the cut along the negative x-axis:
# u(x, 0+) - u(x, 0-) = b for x < 0, continuous for x > 0.


def _edge_bx(x, y, theta, mu, nu):
    """Unit edge dislocation, Burgers vector along +x; theta = polar angle (cut on -x)."""
    r2 = x**2 + y**2
    c = 1.0 / (2 * np.pi)
    k = 1.0 / (4 * (1 - nu))
    ux = c * (theta + 2 * k * x * y / r2)
    uy = -c * ((1 - 2 * nu) * k * np.log(r2) + k * (x**2 - y**2) / r2)
    d = mu / (2 * np.pi * (1 - nu))
    sxx = -d * y * (3 * x**2 + y**2) / r2**2
    syy = d * y * (x**2 - y**2) / r2**2
    sxy = d * x * (x**2 - y**2) / r2**2
    return np.array([ux, uy, sxx, syy, sxy])


def edge_dislocation_planestrain(x, y, bx, by, mu, nu):
    """ux, uy, sxx, syy, sxy of an edge dislocation (Burgers (bx, by)) at the origin,
    cut along the negative x-axis. The by part is the bx solution rotated by +90
    degrees, with its angle shifted so that the cut is also on the negative x-axis
    (the shift is a rigid translation)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    theta = np.arctan2(y, x)
    out = bx * _edge_bx(x, y, theta, mu, nu)
    # rotated frame: x' = R^T X = (y, -x); fields back with u = R u', sigma = R sigma' R^T
    p = _edge_bx(y, -x, theta - np.pi / 2, mu, nu)
    out += by * np.array([-p[1], p[0], p[3], p[2], -p[4]])
    return out


def screw_dislocation_antiplane(x, y, b, mu):
    """uz, sxz, syz of a screw dislocation at the origin, cut along the negative x-axis."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r2 = x**2 + y**2
    return b * np.array([np.arctan2(y, x) / (2 * np.pi), -mu * y / (2 * np.pi * r2),
                         mu * x / (2 * np.pi * r2)])


# Lagrange basis of the quadratic slip element: node k is slip at s = (-2/3, 0, 2/3) a
_QNODES = np.array([-2.0, 0.0, 2.0]) / 3


def quadratic_lagrange(k, s, a):
    """Value and derivative (d/ds) of the quadratic basis function of node k."""
    z = np.asarray(s, float) / a
    others = [m for m in range(3) if m != k]
    zk, z1, z2 = _QNODES[k], _QNODES[others[0]], _QNODES[others[1]]
    den = (zk - z1) * (zk - z2)
    return (z - z1) * (z - z2) / den, ((z - z1) + (z - z2)) / den / a


def quadratic_slip_by_dislocations(els, e, xo, yo, node, direction, geometry, mu, nu=None,
                                   panels=64, order=16):
    """Field of unit quadratic slip (basis function `node`, direction 0 = t, 1 = n in
    plane strain) on element e, as a superposition of point dislocations.

    A displacement jump s(xi) on [-a, a] equals dislocations of strength s(a) at a and
    -s(-a) at -a, plus a density -s'(xi) in between (each dislocation's cut runs along
    -x from it). The jump here is u(+n side) - u(-n side) = s. The density integral
    uses composite Gauss-Legendre (`panels` x `order` points); observation points must
    be off the element.

    Returns plane strain (ux, uy, sxx, syy, sxy) or antiplane (uz, sxz, syz), global."""
    a = els.half_lengths[e]
    R = np.asarray(els.rot_mats[e])
    dx, dy = np.asarray(xo, float) - els.x_centers[e], np.asarray(yo, float) - els.y_centers[e]
    xl, yl = R[0, 0] * dx + R[1, 0] * dy, R[0, 1] * dx + R[1, 1] * dy  # local frame
    if geometry == "planestrain":
        b = np.eye(2)[direction]
        point = lambda x, s: edge_dislocation_planestrain(x - s, yl, *b, mu, nu)  # noqa: E731
    else:
        point = lambda x, s: screw_dislocation_antiplane(x - s, yl, 1.0, mu)  # noqa: E731
    phi = lambda s: quadratic_lagrange(node, s, a)  # noqa: E731
    out = phi(a)[0] * point(xl, a) - phi(-a)[0] * point(xl, -a)
    g, w = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(-a, a, panels + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = 0.5 * (hi + lo) + 0.5 * (hi - lo) * g
        for si, wi in zip(s, 0.5 * (hi - lo) * w):
            out = out - wi * phi(si)[1] * point(xl, si)
    if geometry == "antiplane":  # uz is a scalar; (sxz, syz) rotates as a vector
        return np.array([out[0], *np.einsum("ij,j...->i...", R, out[1:])])
    ux, uy, sxx, syy, sxy = out
    S = np.array([[sxx, sxy], [sxy, syy]])
    u = np.einsum("ij,j...->i...", R, np.array([ux, uy]))
    Sg = np.einsum("ij,jk...,lk->il...", R, S, R)
    return np.array([u[0], u[1], Sg[0, 0], Sg[1, 1], Sg[0, 1]])


# ---------------------------------------------------------------------------
# arbitrary-precision versions of the generated kernel formulas (mpmath)
# ---------------------------------------------------------------------------


class _MpNumpy:
    import mpmath as _mp

    pi = _mp.pi
    log = staticmethod(_mp.log)
    arctan = staticmethod(_mp.atan)
    sign = staticmethod(_mp.sign)


def mp_version(jitted):
    """The same formula as a jitted bemAssembly kernel (`GF._quadratic_slip_f`,
    `GF._linear_force_g`), evaluated with mpmath (set mpmath.mp.dps in the caller)."""
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(jitted.py_func))
    src = "\n".join(l for l in src.splitlines() if not l.startswith("@"))
    ns = {"np": _MpNumpy}
    exec(src, ns)
    return ns[jitted.py_func.__name__]
