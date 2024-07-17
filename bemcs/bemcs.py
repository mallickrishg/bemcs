import addict
import matplotlib
import quadpy
import scipy
import numpy as np
import matplotlib.pyplot as plt


def plot_els_geometry(els):
    """Plot element geometry"""
    plt.figure()
    for i in range(len(els.x1)):
        plt.plot(
            [els.x1[i], els.x2[i]],
            [els.y1[i], els.y2[i]],
            "-",
            color="r",
            linewidth=0.5,
        )
        plt.plot(
            [els.x1[i], els.x2[i]],
            [els.y1[i], els.y2[i]],
            "r.",
            markersize=1,
            linewidth=0.5,
        )

    # Plot unit normal & shear vectors
    plt.quiver(
        els.x_centers,
        els.y_centers,
        els.x_normals,
        els.y_normals,
        units="width",
        color="gray",
        width=0.002,
    )
    plt.quiver(
        els.x_centers,
        els.y_centers,
        els.x_shears,
        els.y_shears,
        units="width",
        color="green",
        width=0.002,
    )

    for i in range(len(els.x1)):
        plt.text(
            els.x_centers[i],
            els.y_centers[i],
            str(i),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=8,
        )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("element geometry and normals")
    plt.gca().set_aspect("equal")
    plt.show(block=False)


def discretized_line(x_start, y_start, x_end, y_end, n_elements):
    """Create geometry of discretized line"""
    n_pts = n_elements + 1
    x = np.linspace(x_start, x_end, n_pts)
    y = np.linspace(y_start, y_end, n_pts)
    x1 = x[:-1]
    y1 = y[:-1]
    x2 = x[1:]
    y2 = y[1:]
    return x1, y1, x2, y2


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


def quadratic_kernel_farfield(x, y, a, nu):
    """Kernels with quadratic shape functions
    f has dimensions of (f=7, shapefunctions=3, n_obs)

    Classic form of:
    arctan_x_minus_a = np.arctan((a - x) / y)
    arctan_x_plus_a = np.arctan((a + x) / y)

    but we have replaced these with the equaivalaent terms below to
    avoid singularities at y = 0.  Singularities at x = +/- a still exist
    """

    arctan_x_minus_a = np.pi / 2 * np.sign(y / (a - x)) - np.arctan(y / (a - x))
    arctan_x_plus_a = np.pi / 2 * np.sign(y / (a + x)) - np.arctan(y / (a + x))

    f = np.zeros((7, 3, x.size))

    # f0
    f[0, 0, :] = (
        -1
        / 64
        * (
            6 * y**3 * (arctan_x_plus_a + arctan_x_minus_a)
            - 6 * a**3 * np.log(a**2 + 2 * a * x + x**2 + y**2)
            - 8 * a**3
            - 12 * a**2 * x
            + 12 * a * x**2
            - 12 * a * y**2
            + 6
            * (
                (2 * a * x - 3 * x**2) * arctan_x_plus_a
                + (2 * a * x - 3 * x**2) * arctan_x_minus_a
            )
            * y
            + 3
            * (a * x**2 - x**3 - (a - 3 * x) * y**2)
            * np.log(abs(a**2 + 2 * a * x + x**2 + y**2))
            - 3
            * (a * x**2 - x**3 - (a - 3 * x) * y**2)
            * np.log(abs(a**2 - 2 * a * x + x**2 + y**2))
        )
        / (np.pi * a**2 * nu - np.pi * a**2)
    )

    f[0, 1, :] = (
        1
        / 32
        * (
            6 * y**3 * (arctan_x_plus_a + arctan_x_minus_a)
            + a**3 * np.log(a**2 + 2 * a * x + x**2 + y**2)
            + a**3 * np.log(a**2 - 2 * a * x + x**2 + y**2)
            - 8 * a**3
            + 12 * a * x**2
            - 12 * a * y**2
            + 2
            * (
                (4 * a**2 - 9 * x**2) * arctan_x_plus_a
                + (4 * a**2 - 9 * x**2) * arctan_x_minus_a
            )
            * y
            + (4 * a**2 * x - 3 * x**3 + 9 * x * y**2)
            * np.log(abs(a**2 + 2 * a * x + x**2 + y**2))
            - (4 * a**2 * x - 3 * x**3 + 9 * x * y**2)
            * np.log(abs(a**2 - 2 * a * x + x**2 + y**2))
        )
        / (np.pi * a**2 * nu - np.pi * a**2)
    )

    f[0, 2, :] = (
        -1
        / 64
        * (
            6 * y**3 * (arctan_x_plus_a + arctan_x_minus_a)
            - 6 * a**3 * np.log(a**2 - 2 * a * x + x**2 + y**2)
            - 8 * a**3
            + 12 * a**2 * x
            + 12 * a * x**2
            - 12 * a * y**2
            - 6
            * (
                (2 * a * x + 3 * x**2) * arctan_x_plus_a
                + (2 * a * x + 3 * x**2) * arctan_x_minus_a
            )
            * y
            - 3
            * (a * x**2 + x**3 - (a + 3 * x) * y**2)
            * np.log(abs(a**2 + 2 * a * x + x**2 + y**2))
            + 3
            * (a * x**2 + x**3 - (a + 3 * x) * y**2)
            * np.log(abs(a**2 - 2 * a * x + x**2 + y**2))
        )
        / (np.pi * a**2 * nu - np.pi * a**2)
    )

    # f1
    f[1, 0, :] = (
        -3
        / 32
        * (
            3 * y**2 * (arctan_x_plus_a + arctan_x_minus_a)
            - (a - 3 * x) * y * np.log(abs(a**2 + 2 * a * x + x**2 + y**2))
            + (a - 3 * x) * y * np.log(abs(a**2 - 2 * a * x + x**2 + y**2))
            - 6 * a * y
            + (2 * a * x - 3 * x**2) * arctan_x_plus_a
            + (2 * a * x - 3 * x**2) * arctan_x_minus_a
        )
        / (np.pi * a**2 * nu - np.pi * a**2)
    )

    f[1, 1, :] = (
        1
        / 16
        * (
            9 * y**2 * (arctan_x_plus_a + arctan_x_minus_a)
            + 9 * x * y * np.log(abs(a**2 + 2 * a * x + x**2 + y**2))
            - 9 * x * y * np.log(abs(a**2 - 2 * a * x + x**2 + y**2))
            - 18 * a * y
            + (4 * a**2 - 9 * x**2) * arctan_x_plus_a
            + (4 * a**2 - 9 * x**2) * arctan_x_minus_a
        )
        / (np.pi * a**2 * nu - np.pi * a**2)
    )

    f[1, 2, :] = (
        -3
        / 32
        * (
            3 * y**2 * (arctan_x_plus_a + arctan_x_minus_a)
            + (a + 3 * x) * y * np.log(abs(a**2 + 2 * a * x + x**2 + y**2))
            - (a + 3 * x) * y * np.log(abs(a**2 - 2 * a * x + x**2 + y**2))
            - 6 * a * y
            - (2 * a * x + 3 * x**2) * arctan_x_plus_a
            - (2 * a * x + 3 * x**2) * arctan_x_minus_a
        )
        / (np.pi * a**2 * nu - np.pi * a**2)
    )

    # f2
    f[2, 0, :] = (
        3
        / 64
        * (
            8 * a**2
            - 12 * a * x
            - 4 * ((a - 3 * x) * arctan_x_plus_a + (a - 3 * x) * arctan_x_minus_a) * y
            - (2 * a * x - 3 * x**2 + 3 * y**2)
            * np.log(abs(a**2 + 2 * a * x + x**2 + y**2))
            + (2 * a * x - 3 * x**2 + 3 * y**2)
            * np.log(abs(a**2 - 2 * a * x + x**2 + y**2))
        )
        / (np.pi * a**2 * nu - np.pi * a**2)
    )

    f[2, 1, :] = (
        1
        / 32
        * (
            36 * a * x
            - 36 * (x * arctan_x_plus_a + x * arctan_x_minus_a) * y
            + (4 * a**2 - 9 * x**2 + 9 * y**2)
            * np.log(abs(a**2 + 2 * a * x + x**2 + y**2))
            - (4 * a**2 - 9 * x**2 + 9 * y**2)
            * np.log(abs(a**2 - 2 * a * x + x**2 + y**2))
        )
        / (np.pi * a**2 * nu - np.pi * a**2)
    )

    f[2, 2, :] = (
        -3
        / 64
        * (
            8 * a**2
            + 12 * a * x
            - 4 * ((a + 3 * x) * arctan_x_plus_a + (a + 3 * x) * arctan_x_minus_a) * y
            - (2 * a * x + 3 * x**2 - 3 * y**2)
            * np.log(abs(a**2 + 2 * a * x + x**2 + y**2))
            + (2 * a * x + 3 * x**2 - 3 * y**2)
            * np.log(abs(a**2 - 2 * a * x + x**2 + y**2))
        )
        / (np.pi * a**2 * nu - np.pi * a**2)
    )

    # f3
    f[3, 0, :] = (
        3
        / 32
        * (
            4 * a**2 * y**3
            - 2
            * ((a - 3 * x) * arctan_x_plus_a + (a - 3 * x) * arctan_x_minus_a)
            * y**4
            - 4
            * (
                (a**3 - 3 * a**2 * x + a * x**2 - 3 * x**3) * arctan_x_plus_a
                + (a**3 - 3 * a**2 * x + a * x**2 - 3 * x**3) * arctan_x_minus_a
            )
            * y**2
            + 4 * (a**4 - 3 * a**3 * x + a**2 * x**2) * y
            - 2
            * (
                a**5
                - 3 * a**4 * x
                - 2 * a**3 * x**2
                + 6 * a**2 * x**3
                + a * x**4
                - 3 * x**5
            )
            * arctan_x_plus_a
            - 2
            * (
                a**5
                - 3 * a**4 * x
                - 2 * a**3 * x**2
                + 6 * a**2 * x**3
                + a * x**4
                - 3 * x**5
            )
            * arctan_x_minus_a
            - 3
            * (y**5 + 2 * (a**2 + x**2) * y**3 + (a**4 - 2 * a**2 * x**2 + x**4) * y)
            * np.log(abs(a**2 + 2 * a * x + x**2 + y**2))
            + 3
            * (y**5 + 2 * (a**2 + x**2) * y**3 + (a**4 - 2 * a**2 * x**2 + x**4) * y)
            * np.log(abs(a**2 - 2 * a * x + x**2 + y**2))
        )
        / (
            np.pi * a**6 * nu
            - np.pi * a**6
            + (np.pi * a**2 * nu - np.pi * a**2) * x**4
            + (np.pi * a**2 * nu - np.pi * a**2) * y**4
            - 2 * (np.pi * a**4 * nu - np.pi * a**4) * x**2
            + 2
            * (
                np.pi * a**4 * nu
                - np.pi * a**4
                + (np.pi * a**2 * nu - np.pi * a**2) * x**2
            )
            * y**2
        )
    )

    f[3, 1, :] = (
        1
        / 16
        * (
            20 * a**3 * x * y
            - 18 * (x * arctan_x_plus_a + x * arctan_x_minus_a) * y**4
            - 36
            * (
                (a**2 * x + x**3) * arctan_x_plus_a
                + (a**2 * x + x**3) * arctan_x_minus_a
            )
            * y**2
            - 18 * (a**4 * x - 2 * a**2 * x**3 + x**5) * arctan_x_plus_a
            - 18 * (a**4 * x - 2 * a**2 * x**3 + x**5) * arctan_x_minus_a
            + 9
            * (y**5 + 2 * (a**2 + x**2) * y**3 + (a**4 - 2 * a**2 * x**2 + x**4) * y)
            * np.log(abs(a**2 + 2 * a * x + x**2 + y**2))
            - 9
            * (y**5 + 2 * (a**2 + x**2) * y**3 + (a**4 - 2 * a**2 * x**2 + x**4) * y)
            * np.log(abs(a**2 - 2 * a * x + x**2 + y**2))
        )
        / (
            np.pi * a**6 * nu
            - np.pi * a**6
            + (np.pi * a**2 * nu - np.pi * a**2) * x**4
            + (np.pi * a**2 * nu - np.pi * a**2) * y**4
            - 2 * (np.pi * a**4 * nu - np.pi * a**4) * x**2
            + 2
            * (
                np.pi * a**4 * nu
                - np.pi * a**4
                + (np.pi * a**2 * nu - np.pi * a**2) * x**2
            )
            * y**2
        )
    )

    f[3, 2, :] = (
        -3
        / 32
        * (
            4 * a**2 * y**3
            - 2
            * ((a + 3 * x) * arctan_x_plus_a + (a + 3 * x) * arctan_x_minus_a)
            * y**4
            - 4
            * (
                (a**3 + 3 * a**2 * x + a * x**2 + 3 * x**3) * arctan_x_plus_a
                + (a**3 + 3 * a**2 * x + a * x**2 + 3 * x**3) * arctan_x_minus_a
            )
            * y**2
            + 4 * (a**4 + 3 * a**3 * x + a**2 * x**2) * y
            - 2
            * (
                a**5
                + 3 * a**4 * x
                - 2 * a**3 * x**2
                - 6 * a**2 * x**3
                + a * x**4
                + 3 * x**5
            )
            * arctan_x_plus_a
            - 2
            * (
                a**5
                + 3 * a**4 * x
                - 2 * a**3 * x**2
                - 6 * a**2 * x**3
                + a * x**4
                + 3 * x**5
            )
            * arctan_x_minus_a
            + 3
            * (y**5 + 2 * (a**2 + x**2) * y**3 + (a**4 - 2 * a**2 * x**2 + x**4) * y)
            * np.log(abs(a**2 + 2 * a * x + x**2 + y**2))
            - 3
            * (y**5 + 2 * (a**2 + x**2) * y**3 + (a**4 - 2 * a**2 * x**2 + x**4) * y)
            * np.log(abs(a**2 - 2 * a * x + x**2 + y**2))
        )
        / (
            np.pi * a**6 * nu
            - np.pi * a**6
            + (np.pi * a**2 * nu - np.pi * a**2) * x**4
            + (np.pi * a**2 * nu - np.pi * a**2) * y**4
            - 2 * (np.pi * a**4 * nu - np.pi * a**4) * x**2
            + 2
            * (
                np.pi * a**4 * nu
                - np.pi * a**4
                + (np.pi * a**2 * nu - np.pi * a**2) * x**2
            )
            * y**2
        )
    )

    # f4
    f[4, 0, :] = (
        3
        / 32
        * (
            6 * y**5 * (arctan_x_plus_a + arctan_x_minus_a)
            - 6 * a**5
            - 4 * a**4 * x
            + 18 * a**3 * x**2
            + 4 * a**2 * x**3
            - 12 * a * x**4
            - 12 * a * y**4
            + 12
            * ((a**2 + x**2) * arctan_x_plus_a + (a**2 + x**2) * arctan_x_minus_a)
            * y**3
            - 2 * (9 * a**3 - 2 * a**2 * x + 12 * a * x**2) * y**2
            + 6
            * (
                (a**4 - 2 * a**2 * x**2 + x**4) * arctan_x_plus_a
                + (a**4 - 2 * a**2 * x**2 + x**4) * arctan_x_minus_a
            )
            * y
            - (
                a**5
                - 3 * a**4 * x
                - 2 * a**3 * x**2
                + 6 * a**2 * x**3
                + a * x**4
                - 3 * x**5
                + (a - 3 * x) * y**4
                + 2 * (a**3 - 3 * a**2 * x + a * x**2 - 3 * x**3) * y**2
            )
            * np.log(abs(a**2 + 2 * a * x + x**2 + y**2))
            + (
                a**5
                - 3 * a**4 * x
                - 2 * a**3 * x**2
                + 6 * a**2 * x**3
                + a * x**4
                - 3 * x**5
                + (a - 3 * x) * y**4
                + 2 * (a**3 - 3 * a**2 * x + a * x**2 - 3 * x**3) * y**2
            )
            * np.log(abs(a**2 - 2 * a * x + x**2 + y**2))
        )
        / (
            np.pi * a**6 * nu
            - np.pi * a**6
            + (np.pi * a**2 * nu - np.pi * a**2) * x**4
            + (np.pi * a**2 * nu - np.pi * a**2) * y**4
            - 2 * (np.pi * a**4 * nu - np.pi * a**4) * x**2
            + 2
            * (
                np.pi * a**4 * nu
                - np.pi * a**4
                + (np.pi * a**2 * nu - np.pi * a**2) * x**2
            )
            * y**2
        )
    )

    f[4, 1, :] = (
        -1
        / 16
        * (
            18 * y**5 * (arctan_x_plus_a + arctan_x_minus_a)
            - 26 * a**5
            + 62 * a**3 * x**2
            - 36 * a * x**4
            - 36 * a * y**4
            + 36
            * ((a**2 + x**2) * arctan_x_plus_a + (a**2 + x**2) * arctan_x_minus_a)
            * y**3
            - 2 * (31 * a**3 + 36 * a * x**2) * y**2
            + 18
            * (
                (a**4 - 2 * a**2 * x**2 + x**4) * arctan_x_plus_a
                + (a**4 - 2 * a**2 * x**2 + x**4) * arctan_x_minus_a
            )
            * y
            + 9
            * (
                a**4 * x
                - 2 * a**2 * x**3
                + x**5
                + x * y**4
                + 2 * (a**2 * x + x**3) * y**2
            )
            * np.log(abs(a**2 + 2 * a * x + x**2 + y**2))
            - 9
            * (
                a**4 * x
                - 2 * a**2 * x**3
                + x**5
                + x * y**4
                + 2 * (a**2 * x + x**3) * y**2
            )
            * np.log(abs(a**2 - 2 * a * x + x**2 + y**2))
        )
        / (
            np.pi * a**6 * nu
            - np.pi * a**6
            + (np.pi * a**2 * nu - np.pi * a**2) * x**4
            + (np.pi * a**2 * nu - np.pi * a**2) * y**4
            - 2 * (np.pi * a**4 * nu - np.pi * a**4) * x**2
            + 2
            * (
                np.pi * a**4 * nu
                - np.pi * a**4
                + (np.pi * a**2 * nu - np.pi * a**2) * x**2
            )
            * y**2
        )
    )

    f[4, 2, :] = (
        3
        / 32
        * (
            6 * y**5 * (arctan_x_plus_a + arctan_x_minus_a)
            - 6 * a**5
            + 4 * a**4 * x
            + 18 * a**3 * x**2
            - 4 * a**2 * x**3
            - 12 * a * x**4
            - 12 * a * y**4
            + 12
            * ((a**2 + x**2) * arctan_x_plus_a + (a**2 + x**2) * arctan_x_minus_a)
            * y**3
            - 2 * (9 * a**3 + 2 * a**2 * x + 12 * a * x**2) * y**2
            + 6
            * (
                (a**4 - 2 * a**2 * x**2 + x**4) * arctan_x_plus_a
                + (a**4 - 2 * a**2 * x**2 + x**4) * arctan_x_minus_a
            )
            * y
            + (
                a**5
                + 3 * a**4 * x
                - 2 * a**3 * x**2
                - 6 * a**2 * x**3
                + a * x**4
                + 3 * x**5
                + (a + 3 * x) * y**4
                + 2 * (a**3 + 3 * a**2 * x + a * x**2 + 3 * x**3) * y**2
            )
            * np.log(abs(a**2 + 2 * a * x + x**2 + y**2))
            - (
                a**5
                + 3 * a**4 * x
                - 2 * a**3 * x**2
                - 6 * a**2 * x**3
                + a * x**4
                + 3 * x**5
                + (a + 3 * x) * y**4
                + 2 * (a**3 + 3 * a**2 * x + a * x**2 + 3 * x**3) * y**2
            )
            * np.log(abs(a**2 - 2 * a * x + x**2 + y**2))
        )
        / (
            np.pi * a**6 * nu
            - np.pi * a**6
            + (np.pi * a**2 * nu - np.pi * a**2) * x**4
            + (np.pi * a**2 * nu - np.pi * a**2) * y**4
            - 2 * (np.pi * a**4 * nu - np.pi * a**4) * x**2
            + 2
            * (
                np.pi * a**4 * nu
                - np.pi * a**4
                + (np.pi * a**2 * nu - np.pi * a**2) * x**2
            )
            * y**2
        )
    )

    # f5
    f[5, 0, :] = (
        3
        / 32
        * (
            8 * a**8
            - 24 * a**7 * x
            - 16 * a**6 * x**2
            + 60 * a**5 * x**3
            + 8 * a**4 * x**4
            - 48 * a**3 * x**5
            + 12 * a * x**7
            + 12 * a * x * y**6
            + 4 * (2 * a**4 + 12 * a**3 * x + 9 * a * x**3) * y**4
            + 4 * (4 * a**6 + 3 * a**5 * x - 12 * a**4 * x**2 + 9 * a * x**5) * y**2
            - 3
            * (
                a**8
                - 4 * a**6 * x**2
                + 6 * a**4 * x**4
                - 4 * a**2 * x**6
                + x**8
                + y**8
                + 4 * (a**2 + x**2) * y**6
                + 2 * (3 * a**4 + 2 * a**2 * x**2 + 3 * x**4) * y**4
                + 4 * (a**6 - a**4 * x**2 - a**2 * x**4 + x**6) * y**2
            )
            * np.log(abs(a**2 + 2 * a * x + x**2 + y**2))
            + 3
            * (
                a**8
                - 4 * a**6 * x**2
                + 6 * a**4 * x**4
                - 4 * a**2 * x**6
                + x**8
                + y**8
                + 4 * (a**2 + x**2) * y**6
                + 2 * (3 * a**4 + 2 * a**2 * x**2 + 3 * x**4) * y**4
                + 4 * (a**6 - a**4 * x**2 - a**2 * x**4 + x**6) * y**2
            )
            * np.log(abs(a**2 - 2 * a * x + x**2 + y**2))
        )
        / (
            np.pi * a**10 * nu
            - np.pi * a**10
            + (np.pi * a**2 * nu - np.pi * a**2) * x**8
            + (np.pi * a**2 * nu - np.pi * a**2) * y**8
            - 4 * (np.pi * a**4 * nu - np.pi * a**4) * x**6
            + 4
            * (
                np.pi * a**4 * nu
                - np.pi * a**4
                + (np.pi * a**2 * nu - np.pi * a**2) * x**2
            )
            * y**6
            + 6 * (np.pi * a**6 * nu - np.pi * a**6) * x**4
            + 2
            * (
                3 * np.pi * a**6 * nu
                - 3 * np.pi * a**6
                + 3 * (np.pi * a**2 * nu - np.pi * a**2) * x**4
                + 2 * (np.pi * a**4 * nu - np.pi * a**4) * x**2
            )
            * y**4
            - 4 * (np.pi * a**8 * nu - np.pi * a**8) * x**2
            + 4
            * (
                np.pi * a**8 * nu
                - np.pi * a**8
                + (np.pi * a**2 * nu - np.pi * a**2) * x**6
                - (np.pi * a**4 * nu - np.pi * a**4) * x**4
                - (np.pi * a**6 * nu - np.pi * a**6) * x**2
            )
            * y**2
        )
    )

    f[5, 1, :] = (
        1
        / 16
        * (
            56 * a**7 * x
            - 148 * a**5 * x**3
            + 128 * a**3 * x**5
            - 36 * a * x**7
            - 36 * a * x * y**6
            - 12 * (8 * a**3 * x + 9 * a * x**3) * y**4
            - 4 * (a**5 * x - 8 * a**3 * x**3 + 27 * a * x**5) * y**2
            + 9
            * (
                a**8
                - 4 * a**6 * x**2
                + 6 * a**4 * x**4
                - 4 * a**2 * x**6
                + x**8
                + y**8
                + 4 * (a**2 + x**2) * y**6
                + 2 * (3 * a**4 + 2 * a**2 * x**2 + 3 * x**4) * y**4
                + 4 * (a**6 - a**4 * x**2 - a**2 * x**4 + x**6) * y**2
            )
            * np.log(abs(a**2 + 2 * a * x + x**2 + y**2))
            - 9
            * (
                a**8
                - 4 * a**6 * x**2
                + 6 * a**4 * x**4
                - 4 * a**2 * x**6
                + x**8
                + y**8
                + 4 * (a**2 + x**2) * y**6
                + 2 * (3 * a**4 + 2 * a**2 * x**2 + 3 * x**4) * y**4
                + 4 * (a**6 - a**4 * x**2 - a**2 * x**4 + x**6) * y**2
            )
            * np.log(abs(a**2 - 2 * a * x + x**2 + y**2))
        )
        / (
            np.pi * a**10 * nu
            - np.pi * a**10
            + (np.pi * a**2 * nu - np.pi * a**2) * x**8
            + (np.pi * a**2 * nu - np.pi * a**2) * y**8
            - 4 * (np.pi * a**4 * nu - np.pi * a**4) * x**6
            + 4
            * (
                np.pi * a**4 * nu
                - np.pi * a**4
                + (np.pi * a**2 * nu - np.pi * a**2) * x**2
            )
            * y**6
            + 6 * (np.pi * a**6 * nu - np.pi * a**6) * x**4
            + 2
            * (
                3 * np.pi * a**6 * nu
                - 3 * np.pi * a**6
                + 3 * (np.pi * a**2 * nu - np.pi * a**2) * x**4
                + 2 * (np.pi * a**4 * nu - np.pi * a**4) * x**2
            )
            * y**4
            - 4 * (np.pi * a**8 * nu - np.pi * a**8) * x**2
            + 4
            * (
                np.pi * a**8 * nu
                - np.pi * a**8
                + (np.pi * a**2 * nu - np.pi * a**2) * x**6
                - (np.pi * a**4 * nu - np.pi * a**4) * x**4
                - (np.pi * a**6 * nu - np.pi * a**6) * x**2
            )
            * y**2
        )
    )

    f[5, 2, :] = (
        -3
        / 32
        * (
            8 * a**8
            + 24 * a**7 * x
            - 16 * a**6 * x**2
            - 60 * a**5 * x**3
            + 8 * a**4 * x**4
            + 48 * a**3 * x**5
            - 12 * a * x**7
            - 12 * a * x * y**6
            + 4 * (2 * a**4 - 12 * a**3 * x - 9 * a * x**3) * y**4
            + 4 * (4 * a**6 - 3 * a**5 * x - 12 * a**4 * x**2 - 9 * a * x**5) * y**2
            + 3
            * (
                a**8
                - 4 * a**6 * x**2
                + 6 * a**4 * x**4
                - 4 * a**2 * x**6
                + x**8
                + y**8
                + 4 * (a**2 + x**2) * y**6
                + 2 * (3 * a**4 + 2 * a**2 * x**2 + 3 * x**4) * y**4
                + 4 * (a**6 - a**4 * x**2 - a**2 * x**4 + x**6) * y**2
            )
            * np.log(abs(a**2 + 2 * a * x + x**2 + y**2))
            - 3
            * (
                a**8
                - 4 * a**6 * x**2
                + 6 * a**4 * x**4
                - 4 * a**2 * x**6
                + x**8
                + y**8
                + 4 * (a**2 + x**2) * y**6
                + 2 * (3 * a**4 + 2 * a**2 * x**2 + 3 * x**4) * y**4
                + 4 * (a**6 - a**4 * x**2 - a**2 * x**4 + x**6) * y**2
            )
            * np.log(abs(a**2 - 2 * a * x + x**2 + y**2))
        )
        / (
            np.pi * a**10 * nu
            - np.pi * a**10
            + (np.pi * a**2 * nu - np.pi * a**2) * x**8
            + (np.pi * a**2 * nu - np.pi * a**2) * y**8
            - 4 * (np.pi * a**4 * nu - np.pi * a**4) * x**6
            + 4
            * (
                np.pi * a**4 * nu
                - np.pi * a**4
                + (np.pi * a**2 * nu - np.pi * a**2) * x**2
            )
            * y**6
            + 6 * (np.pi * a**6 * nu - np.pi * a**6) * x**4
            + 2
            * (
                3 * np.pi * a**6 * nu
                - 3 * np.pi * a**6
                + 3 * (np.pi * a**2 * nu - np.pi * a**2) * x**4
                + 2 * (np.pi * a**4 * nu - np.pi * a**4) * x**2
            )
            * y**4
            - 4 * (np.pi * a**8 * nu - np.pi * a**8) * x**2
            + 4
            * (
                np.pi * a**8 * nu
                - np.pi * a**8
                + (np.pi * a**2 * nu - np.pi * a**2) * x**6
                - (np.pi * a**4 * nu - np.pi * a**4) * x**4
                - (np.pi * a**6 * nu - np.pi * a**6) * x**2
            )
            * y**2
        )
    )

    # f6
    f[6, 0, :] = (
        -3
        / 16
        * (
            3 * y**8 * (arctan_x_plus_a + arctan_x_minus_a)
            - 6 * a * y**7
            + 12
            * ((a**2 + x**2) * arctan_x_plus_a + (a**2 + x**2) * arctan_x_minus_a)
            * y**6
            - 6 * (4 * a**3 + 3 * a * x**2) * y**5
            + 6
            * (
                (3 * a**4 + 2 * a**2 * x**2 + 3 * x**4) * arctan_x_plus_a
                + (3 * a**4 + 2 * a**2 * x**2 + 3 * x**4) * arctan_x_minus_a
            )
            * y**4
            - 2 * (15 * a**5 - 8 * a**4 * x + 9 * a * x**4) * y**3
            + 12
            * (
                (a**6 - a**4 * x**2 - a**2 * x**4 + x**6) * arctan_x_plus_a
                + (a**6 - a**4 * x**2 - a**2 * x**4 + x**6) * arctan_x_minus_a
            )
            * y**2
            - 2
            * (
                6 * a**7
                - 8 * a**6 * x
                + 3 * a**5 * x**2
                + 8 * a**4 * x**3
                - 12 * a**3 * x**4
                + 3 * a * x**6
            )
            * y
            + 3
            * (a**8 - 4 * a**6 * x**2 + 6 * a**4 * x**4 - 4 * a**2 * x**6 + x**8)
            * arctan_x_plus_a
            + 3
            * (a**8 - 4 * a**6 * x**2 + 6 * a**4 * x**4 - 4 * a**2 * x**6 + x**8)
            * arctan_x_minus_a
        )
        / (
            np.pi * a**10 * nu
            - np.pi * a**10
            + (np.pi * a**2 * nu - np.pi * a**2) * x**8
            + (np.pi * a**2 * nu - np.pi * a**2) * y**8
            - 4 * (np.pi * a**4 * nu - np.pi * a**4) * x**6
            + 4
            * (
                np.pi * a**4 * nu
                - np.pi * a**4
                + (np.pi * a**2 * nu - np.pi * a**2) * x**2
            )
            * y**6
            + 6 * (np.pi * a**6 * nu - np.pi * a**6) * x**4
            + 2
            * (
                3 * np.pi * a**6 * nu
                - 3 * np.pi * a**6
                + 3 * (np.pi * a**2 * nu - np.pi * a**2) * x**4
                + 2 * (np.pi * a**4 * nu - np.pi * a**4) * x**2
            )
            * y**4
            - 4 * (np.pi * a**8 * nu - np.pi * a**8) * x**2
            + 4
            * (
                np.pi * a**8 * nu
                - np.pi * a**8
                + (np.pi * a**2 * nu - np.pi * a**2) * x**6
                - (np.pi * a**4 * nu - np.pi * a**4) * x**4
                - (np.pi * a**6 * nu - np.pi * a**6) * x**2
            )
            * y**2
        )
    )

    f[6, 1, :] = (
        1
        / 8
        * (
            9 * y**8 * (arctan_x_plus_a + arctan_x_minus_a)
            - 18 * a * y**7
            + 36
            * ((a**2 + x**2) * arctan_x_plus_a + (a**2 + x**2) * arctan_x_minus_a)
            * y**6
            - 2 * (32 * a**3 + 27 * a * x**2) * y**5
            + 18
            * (
                (3 * a**4 + 2 * a**2 * x**2 + 3 * x**4) * arctan_x_plus_a
                + (3 * a**4 + 2 * a**2 * x**2 + 3 * x**4) * arctan_x_minus_a
            )
            * y**4
            - 2 * (37 * a**5 + 8 * a**3 * x**2 + 27 * a * x**4) * y**3
            + 36
            * (
                (a**6 - a**4 * x**2 - a**2 * x**4 + x**6) * arctan_x_plus_a
                + (a**6 - a**4 * x**2 - a**2 * x**4 + x**6) * arctan_x_minus_a
            )
            * y**2
            - 2 * (14 * a**7 + a**5 * x**2 - 24 * a**3 * x**4 + 9 * a * x**6) * y
            + 9
            * (a**8 - 4 * a**6 * x**2 + 6 * a**4 * x**4 - 4 * a**2 * x**6 + x**8)
            * arctan_x_plus_a
            + 9
            * (a**8 - 4 * a**6 * x**2 + 6 * a**4 * x**4 - 4 * a**2 * x**6 + x**8)
            * arctan_x_minus_a
        )
        / (
            np.pi * a**10 * nu
            - np.pi * a**10
            + (np.pi * a**2 * nu - np.pi * a**2) * x**8
            + (np.pi * a**2 * nu - np.pi * a**2) * y**8
            - 4 * (np.pi * a**4 * nu - np.pi * a**4) * x**6
            + 4
            * (
                np.pi * a**4 * nu
                - np.pi * a**4
                + (np.pi * a**2 * nu - np.pi * a**2) * x**2
            )
            * y**6
            + 6 * (np.pi * a**6 * nu - np.pi * a**6) * x**4
            + 2
            * (
                3 * np.pi * a**6 * nu
                - 3 * np.pi * a**6
                + 3 * (np.pi * a**2 * nu - np.pi * a**2) * x**4
                + 2 * (np.pi * a**4 * nu - np.pi * a**4) * x**2
            )
            * y**4
            - 4 * (np.pi * a**8 * nu - np.pi * a**8) * x**2
            + 4
            * (
                np.pi * a**8 * nu
                - np.pi * a**8
                + (np.pi * a**2 * nu - np.pi * a**2) * x**6
                - (np.pi * a**4 * nu - np.pi * a**4) * x**4
                - (np.pi * a**6 * nu - np.pi * a**6) * x**2
            )
            * y**2
        )
    )

    f[6, 2, :] = (
        -3
        / 16
        * (
            3 * y**8 * (arctan_x_plus_a + arctan_x_minus_a)
            - 6 * a * y**7
            + 12
            * ((a**2 + x**2) * arctan_x_plus_a + (a**2 + x**2) * arctan_x_minus_a)
            * y**6
            - 6 * (4 * a**3 + 3 * a * x**2) * y**5
            + 6
            * (
                (3 * a**4 + 2 * a**2 * x**2 + 3 * x**4) * arctan_x_plus_a
                + (3 * a**4 + 2 * a**2 * x**2 + 3 * x**4) * arctan_x_minus_a
            )
            * y**4
            - 2 * (15 * a**5 + 8 * a**4 * x + 9 * a * x**4) * y**3
            + 12
            * (
                (a**6 - a**4 * x**2 - a**2 * x**4 + x**6) * arctan_x_plus_a
                + (a**6 - a**4 * x**2 - a**2 * x**4 + x**6) * arctan_x_minus_a
            )
            * y**2
            - 2
            * (
                6 * a**7
                + 8 * a**6 * x
                + 3 * a**5 * x**2
                - 8 * a**4 * x**3
                - 12 * a**3 * x**4
                + 3 * a * x**6
            )
            * y
            + 3
            * (a**8 - 4 * a**6 * x**2 + 6 * a**4 * x**4 - 4 * a**2 * x**6 + x**8)
            * arctan_x_plus_a
            + 3
            * (a**8 - 4 * a**6 * x**2 + 6 * a**4 * x**4 - 4 * a**2 * x**6 + x**8)
            * arctan_x_minus_a
        )
        / (
            np.pi * a**10 * nu
            - np.pi * a**10
            + (np.pi * a**2 * nu - np.pi * a**2) * x**8
            + (np.pi * a**2 * nu - np.pi * a**2) * y**8
            - 4 * (np.pi * a**4 * nu - np.pi * a**4) * x**6
            + 4
            * (
                np.pi * a**4 * nu
                - np.pi * a**4
                + (np.pi * a**2 * nu - np.pi * a**2) * x**2
            )
            * y**6
            + 6 * (np.pi * a**6 * nu - np.pi * a**6) * x**4
            + 2
            * (
                3 * np.pi * a**6 * nu
                - 3 * np.pi * a**6
                + 3 * (np.pi * a**2 * nu - np.pi * a**2) * x**4
                + 2 * (np.pi * a**4 * nu - np.pi * a**4) * x**2
            )
            * y**4
            - 4 * (np.pi * a**8 * nu - np.pi * a**8) * x**2
            + 4
            * (
                np.pi * a**8 * nu
                - np.pi * a**8
                + (np.pi * a**2 * nu - np.pi * a**2) * x**6
                - (np.pi * a**4 * nu - np.pi * a**4) * x**4
                - (np.pi * a**6 * nu - np.pi * a**6) * x**2
            )
            * y**2
        )
    )
    return f


def quadratic_kernel_coincident(a, nu):
    """Kernels for coincident integrals
    f, shape_function_idx, node_idx"""
    f = np.zeros((7, 3, 3))

    # f0
    f[0, 0, 0] = (
        -5 / 144 * a * np.log(25 / 9 * a**2) / (np.pi - np.pi * nu)
        - 17 / 288 * a * np.log(1 / 9 * a**2) / (np.pi - np.pi * nu)
        + 1 / 12 * a / (np.pi - np.pi * nu)
    )
    f[0, 1, 0] = (
        -25 / 288 * a * np.log(25 / 9 * a**2) / (np.pi - np.pi * nu)
        + 7 / 288 * a * np.log(1 / 9 * a**2) / (np.pi - np.pi * nu)
        + 1 / 12 * a / (np.pi - np.pi * nu)
    )
    f[0, 2, 0] = (
        -25 / 288 * a * np.log(25 / 9 * a**2) / (np.pi - np.pi * nu)
        - 1 / 144 * a * np.log(1 / 9 * a**2) / (np.pi - np.pi * nu)
        - 1 / 6 * a / (np.pi - np.pi * nu)
    )
    f[0, 0, 1] = -3 / 16 * a * np.log(a) / (np.pi - np.pi * nu) - 1 / 8 * a / (
        np.pi - np.pi * nu
    )
    f[0, 1, 1] = -1 / 8 * a * np.log(a) / (np.pi - np.pi * nu) + 1 / 4 * a / (
        np.pi - np.pi * nu
    )
    f[0, 2, 1] = -3 / 16 * a * np.log(a) / (np.pi - np.pi * nu) - 1 / 8 * a / (
        np.pi - np.pi * nu
    )
    f[0, 0, 2] = (
        -25 / 288 * a * np.log(25 / 9 * a**2) / (np.pi - np.pi * nu)
        - 1 / 144 * a * np.log(1 / 9 * a**2) / (np.pi - np.pi * nu)
        - 1 / 6 * a / (np.pi - np.pi * nu)
    )
    f[0, 1, 2] = (
        -25 / 288 * a * np.log(25 / 9 * a**2) / (np.pi - np.pi * nu)
        + 7 / 288 * a * np.log(1 / 9 * a**2) / (np.pi - np.pi * nu)
        + 1 / 12 * a / (np.pi - np.pi * nu)
    )
    f[0, 2, 2] = (
        -5 / 144 * a * np.log(25 / 9 * a**2) / (np.pi - np.pi * nu)
        - 17 / 288 * a * np.log(1 / 9 * a**2) / (np.pi - np.pi * nu)
        + 1 / 12 * a / (np.pi - np.pi * nu)
    )

    # f1
    f[1, 0, 0] = 1 / 4 / (nu - 1)
    f[1, 1, 0] = 0
    f[1, 2, 0] = 0
    f[1, 0, 1] = 0
    f[1, 1, 1] = 1 / 4 / (nu - 1)
    f[1, 2, 1] = 0
    f[1, 0, 2] = 0
    f[1, 1, 2] = 0
    f[1, 2, 2] = 1 / 4 / (nu - 1)

    # f2
    f[2, 0, 0] = (
        1 / 8 * np.log(25 / 9 * a**2) / (np.pi - np.pi * nu)
        - 1 / 8 * np.log(1 / 9 * a**2) / (np.pi - np.pi * nu)
        - 3 / 4 / (np.pi - np.pi * nu)
    )
    f[2, 1, 0] = 3 / 4 / (np.pi - np.pi * nu)
    f[2, 2, 0] = 0
    f[2, 0, 1] = -3 / 8 / (np.pi - np.pi * nu)
    f[2, 1, 1] = 0
    f[2, 2, 1] = 3 / 8 / (np.pi - np.pi * nu)
    f[2, 0, 2] = 0
    f[2, 1, 2] = -3 / 4 / (np.pi - np.pi * nu)
    f[2, 2, 2] = (
        -1 / 8 * np.log(25 / 9 * a**2) / (np.pi - np.pi * nu)
        + 1 / 8 * np.log(1 / 9 * a**2) / (np.pi - np.pi * nu)
        + 3 / 4 / (np.pi - np.pi * nu)
    )

    # f3
    f[3, 0, 0] = -9 / 16 / (a * nu - a)
    f[3, 1, 0] = 3 / 4 / (a * nu - a)
    f[3, 2, 0] = -3 / 16 / (a * nu - a)
    f[3, 0, 1] = -3 / 16 / (a * nu - a)
    f[3, 1, 1] = 0
    f[3, 2, 1] = 3 / 16 / (a * nu - a)
    f[3, 0, 2] = 3 / 16 / (a * nu - a)
    f[3, 1, 2] = -3 / 4 / (a * nu - a)
    f[3, 2, 2] = 9 / 16 / (a * nu - a)

    # f4
    f[4, 0, 0] = (
        9 / 32 * np.log(25 / 9 * a**2) / (np.pi * a * nu - np.pi * a)
        - 9 / 32 * np.log(1 / 9 * a**2) / (np.pi * a * nu - np.pi * a)
        + 27 / 80 / (np.pi * a * nu - np.pi * a)
    )
    f[4, 1, 0] = (
        -3 / 8 * np.log(25 / 9 * a**2) / (np.pi * a * nu - np.pi * a)
        + 3 / 8 * np.log(1 / 9 * a**2) / (np.pi * a * nu - np.pi * a)
        + 9 / 8 / (np.pi * a * nu - np.pi * a)
    )
    f[4, 2, 0] = (
        3 / 32 * np.log(25 / 9 * a**2) / (np.pi * a * nu - np.pi * a)
        - 3 / 32 * np.log(1 / 9 * a**2) / (np.pi * a * nu - np.pi * a)
        - 9 / 16 / (np.pi * a * nu - np.pi * a)
    )
    f[4, 0, 1] = -9 / 16 / (np.pi * a * nu - np.pi * a)
    f[4, 1, 1] = 13 / 8 / (np.pi * a * nu - np.pi * a)
    f[4, 2, 1] = -9 / 16 / (np.pi * a * nu - np.pi * a)
    f[4, 0, 2] = (
        3 / 32 * np.log(25 / 9 * a**2) / (np.pi * a * nu - np.pi * a)
        - 3 / 32 * np.log(1 / 9 * a**2) / (np.pi * a * nu - np.pi * a)
        - 9 / 16 / (np.pi * a * nu - np.pi * a)
    )
    f[4, 1, 2] = (
        -3 / 8 * np.log(25 / 9 * a**2) / (np.pi * a * nu - np.pi * a)
        + 3 / 8 * np.log(1 / 9 * a**2) / (np.pi * a * nu - np.pi * a)
        + 9 / 8 / (np.pi * a * nu - np.pi * a)
    )
    f[4, 2, 2] = (
        9 / 32 * np.log(25 / 9 * a**2) / (np.pi * a * nu - np.pi * a)
        - 9 / 32 * np.log(1 / 9 * a**2) / (np.pi * a * nu - np.pi * a)
        + 27 / 80 / (np.pi * a * nu - np.pi * a)
    )

    # f5
    f[5, 0, 0] = (
        9 / 32 * np.log(25 / 9 * a**2) / (np.pi * a**2 * nu - np.pi * a**2)
        - 9 / 32 * np.log(1 / 9 * a**2) / (np.pi * a**2 * nu - np.pi * a**2)
        + 621 / 100 / (np.pi * a**2 * nu - np.pi * a**2)
    )
    f[5, 1, 0] = (
        -9 / 16 * np.log(25 / 9 * a**2) / (np.pi * a**2 * nu - np.pi * a**2)
        + 9 / 16 * np.log(1 / 9 * a**2) / (np.pi * a**2 * nu - np.pi * a**2)
        - 27 / 5 / (np.pi * a**2 * nu - np.pi * a**2)
    )
    f[5, 2, 0] = (
        9 / 32 * np.log(25 / 9 * a**2) / (np.pi * a**2 * nu - np.pi * a**2)
        - 9 / 32 * np.log(1 / 9 * a**2) / (np.pi * a**2 * nu - np.pi * a**2)
        + 27 / 20 / (np.pi * a**2 * nu - np.pi * a**2)
    )
    f[5, 0, 1] = 3 / 4 / (np.pi * a**2 * nu - np.pi * a**2)
    f[5, 1, 1] = 0
    f[5, 2, 1] = -3 / 4 / (np.pi * a**2 * nu - np.pi * a**2)
    f[5, 0, 2] = (
        -9 / 32 * np.log(25 / 9 * a**2) / (np.pi * a**2 * nu - np.pi * a**2)
        + 9 / 32 * np.log(1 / 9 * a**2) / (np.pi * a**2 * nu - np.pi * a**2)
        - 27 / 20 / (np.pi * a**2 * nu - np.pi * a**2)
    )
    f[5, 1, 2] = (
        9 / 16 * np.log(25 / 9 * a**2) / (np.pi * a**2 * nu - np.pi * a**2)
        - 9 / 16 * np.log(1 / 9 * a**2) / (np.pi * a**2 * nu - np.pi * a**2)
        + 27 / 5 / (np.pi * a**2 * nu - np.pi * a**2)
    )
    f[5, 2, 2] = (
        -9 / 32 * np.log(25 / 9 * a**2) / (np.pi * a**2 * nu - np.pi * a**2)
        + 9 / 32 * np.log(1 / 9 * a**2) / (np.pi * a**2 * nu - np.pi * a**2)
        - 621 / 100 / (np.pi * a**2 * nu - np.pi * a**2)
    )

    # f6
    f[6, 0, 0] = -9 / 16 / (a**2 * nu - a**2)
    f[6, 1, 0] = 9 / 8 / (a**2 * nu - a**2)
    f[6, 2, 0] = -9 / 16 / (a**2 * nu - a**2)
    f[6, 0, 1] = -9 / 16 / (a**2 * nu - a**2)
    f[6, 1, 1] = 9 / 8 / (a**2 * nu - a**2)
    f[6, 2, 1] = -9 / 16 / (a**2 * nu - a**2)
    f[6, 0, 2] = -9 / 16 / (a**2 * nu - a**2)
    f[6, 1, 2] = 9 / 8 / (a**2 * nu - a**2)
    f[6, 2, 2] = -9 / 16 / (a**2 * nu - a**2)
    return f


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


def displacements_stresses_quadratic_no_rotation(
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
    """This function implements variable slip on a quadratic element"""
    displacement_all = np.zeros((2, x.size))
    stress_all = np.zeros((3, x.size))

    # Rotate and translate into local coordinate system
    x = x - x_center
    y = y - y_center
    global_components = np.vstack((x_component, y_component)).T
    f_all = quadratic_kernel_farfield(x, y, a, nu)
    for i in range(0, 3):
        f = f_all[:, i, :]
        displacement, stress = f_slip_to_displacement_stress(
            global_components[i, 0], global_components[i, 1], f, y, mu, nu
        )
        # Multiply by coefficient for current shape function and sum
        displacement_all += displacement
        stress_all += stress
    return displacement_all, stress_all


def displacements_stresses_quadratic_slip_no_rotation_antiplane(
    xo,
    yo,
    w,
    mu,
    x_center=0,
    y_center=0,
):
    """compute displacement and stress kernels for a quadratically varying slip on a horizontal source element (-w <= x <= w, y = 0)

    INPUTS

    xo,yo - observation locations provided as individual vectors [Nobs x 1]

    x_center,y_center - source element center location (scalars)

    w - source element half-length

    mu - Shear modulus

    OUTPUTS

    Disp - displacement kernels [Nobs x 3 basis functions]

    Stress - 3-d stress_kernels     [Nobs x (sx or sy) x 3 basis functions]"""

    x = xo - x_center
    y = yo - y_center
    Nobs = len(x[:, 0])

    u1 = (
        (3 / 16)
        * w ** (-2)
        * np.pi ** (-1)
        * (
            6 * w * y
            + ((-2) * w * x + 3 * (x + (-1) * y) * (x + y))
            * np.arctan((w + (-1) * x) * y ** (-1))
            + ((-2) * w * x + 3 * (x + (-1) * y) * (x + y))
            * np.arctan((w + x) * y ** (-1))
            + (w + (-3) * x)
            * y
            * ((-1) * np.log((w + (-1) * x) ** 2 + y**2) + np.log((w + x) ** 2 + y**2))
        )
    )

    u2 = (
        (1 / 8)
        * w ** (-2)
        * np.pi ** (-1)
        * (
            (-18) * w * y
            + (4 * w**2 + 9 * y**2) * np.arctan((w + (-1) * x) * y ** (-1))
            + 9 * x**2 * np.arctan(((-1) * w + x) * y ** (-1))
            + (4 * w**2 + (-9) * x**2 + 9 * y**2) * np.arctan((w + x) * y ** (-1))
            + 9
            * x
            * y
            * ((-1) * np.log((w + (-1) * x) ** 2 + y**2) + np.log((w + x) ** 2 + y**2))
        )
    )

    u3 = (
        (3 / 16)
        * w ** (-2)
        * np.pi ** (-1)
        * (
            6 * w * y
            + (2 * w * x + 3 * (x + (-1) * y) * (x + y))
            * np.arctan((w + (-1) * x) * y ** (-1))
            + (2 * w * x + 3 * (x + (-1) * y) * (x + y))
            * np.arctan((w + x) * y ** (-1))
            + (w + 3 * x)
            * y
            * (np.log((w + (-1) * x) ** 2 + y**2) + (-1) * np.log((w + x) ** 2 + y**2))
        )
    )

    ex1 = (
        (3 / 16)
        * w ** (-2)
        * np.pi ** (-1)
        * (
            (-2)
            * (w + (-3) * x)
            * (np.arctan((w + (-1) * x) * y ** (-1)) + np.arctan((w + x) * y ** (-1)))
            + y
            * (
                w**2
                * (
                    (-1) * ((w + (-1) * x) ** 2 + y**2) ** (-1)
                    + 5 * ((w + x) ** 2 + y**2) ** (-1)
                )
                + 3 * np.log((w + (-1) * x) ** 2 + y**2)
                + (-3) * np.log((w + x) ** 2 + y**2)
            )
        )
    )

    ex2 = (
        (1 / 8)
        * w ** (-2)
        * np.pi ** (-1)
        * (
            (-18)
            * x
            * (np.arctan((w + (-1) * x) * y ** (-1)) + np.arctan((w + x) * y ** (-1)))
            + y
            * (
                20
                * w**3
                * x
                * (
                    (w**4 + 2 * w**2 * ((-1) * x**2 + y**2) + (x**2 + y**2) ** 2)
                    ** (-1)
                )
                + (-9) * np.log((w + (-1) * x) ** 2 + y**2)
                + 9 * np.log((w + x) ** 2 + y**2)
            )
        )
    )

    ex3 = (
        (3 / 16)
        * w ** (-2)
        * np.pi ** (-1)
        * (
            2
            * (w + 3 * x)
            * (np.arctan((w + (-1) * x) * y ** (-1)) + np.arctan((w + x) * y ** (-1)))
            + y
            * (
                w**2
                * (
                    (-5) * ((w + (-1) * x) ** 2 + y**2) ** (-1)
                    + ((w + x) ** 2 + y**2) ** (-1)
                )
                + 3 * np.log((w + (-1) * x) ** 2 + y**2)
                + (-3) * np.log((w + x) ** 2 + y**2)
            )
        )
    )

    ey1 = (
        (3 / 16)
        * w ** (-2)
        * np.pi ** (-1)
        * (
            w
            * (
                12
                + w * ((-1) * w + x) * ((w + (-1) * x) ** 2 + y**2) ** (-1)
                + (-5) * w * (w + x) * ((w + x) ** 2 + y**2) ** (-1)
            )
            + (-6)
            * y
            * (np.arctan((w + (-1) * x) * y ** (-1)) + np.arctan((w + x) * y ** (-1)))
            + (-1)
            * (w + (-3) * x)
            * (np.log((w + (-1) * x) ** 2 + y**2) + (-1) * np.log((w + x) ** 2 + y**2))
        )
    )

    ey2 = (
        (-1 / 8)
        * w ** (-2)
        * np.pi ** (-1)
        * (
            w
            * (
                36
                + 5 * w * ((-1) * w + x) * ((w + (-1) * x) ** 2 + y**2) ** (-1)
                + (-5) * w * (w + x) * ((w + x) ** 2 + y**2) ** (-1)
            )
            + (-18)
            * y
            * (np.arctan((w + (-1) * x) * y ** (-1)) + np.arctan((w + x) * y ** (-1)))
            + 9 * x * np.log((w + (-1) * x) ** 2 + y**2)
            + (-9) * x * np.log((w + x) ** 2 + y**2)
        )
    )

    ey3 = (
        (3 / 16)
        * w ** (-2)
        * np.pi ** (-1)
        * (
            w
            * (
                12
                + 5 * w * ((-1) * w + x) * ((w + (-1) * x) ** 2 + y**2) ** (-1)
                + (-1) * w * (w + x) * ((w + x) ** 2 + y**2) ** (-1)
            )
            + (-6)
            * y
            * (np.arctan((w + (-1) * x) * y ** (-1)) + np.arctan((w + x) * y ** (-1)))
            + (w + 3 * x)
            * (np.log((w + (-1) * x) ** 2 + y**2) + (-1) * np.log((w + x) ** 2 + y**2))
        )
    )

    sx = 2 * mu * np.hstack([ex1, ex2, ex3])
    sy = 2 * mu * np.hstack([ey1, ey2, ey3])

    # Create a 2D numpy array for displacements
    # Disp_kernels - [Nobs x 3 basis functions]
    Disp = np.hstack([u1, u2, u3])

    # Create a 3D numpy array for Stress
    # Stress_kernels - [Nobs x (sx or sy) x 3 basis functions]
    Stress = np.zeros((Nobs, 2, 3))

    # Assign values to the stress kernels
    Stress[:, 0, :] = sx
    Stress[:, 1, :] = sy

    return Disp, Stress


def displacements_stresses_linear_force_no_rotation_antiplane(
    xo, yo, w, mu, x_center, y_center
):
    """Compute displacement and stress kernels for a linearly varying force on a
    horizontal source element (-w <= x <= w, y = 0)

    INPUTS

    xo, yo - observation locations provided as individual vectors [Nobs x 1]

    x_center, y_center - source element center location (scalars)

    w - source element half-length

    mu - Elastic parameters

    OUTPUTS

    Disp - 2-d displacement kernels [Nobs x 2 basis functions]

    Stress - 3-d stress_kernels     [Nobs x (sx or sy) x 2 basis functions]"""

    n_obs = len(xo)
    x = xo - x_center
    y = yo - y_center

    u_1 = (
        (1 / 16)
        * np.pi ** (-1)
        * w ** (-1)
        * (
            (-4) * w * (2 * w + x)
            + 4 * (w + x) * y * np.arctan((w - x) / y)
            + 4 * (w + x) * y * np.arctan((w + x) / y)
            + ((w + (-1) * x) * (3 * w + x) + y**2) * np.log((w + (-1) * x) ** 2 + y**2)
            + (w + x + (-1) * y) * (w + x + y) * np.log((w + x) ** 2 + y**2)
        )
    )

    u_2 = (
        (1 / 16)
        * np.pi ** (-1)
        * w ** (-1)
        * (
            4 * w * ((-2) * w + x)
            + 4 * (w + (-1) * x) * y * np.arctan((w - x) / y)
            + 4 * (w + (-1) * x) * y * np.arctan((w + x) / y)
            + (w + (-1) * x + (-1) * y)
            * (w + (-1) * x + y)
            * np.log((w + (-1) * x) ** 2 + y**2)
            + ((3 * w + (-1) * x) * (w + x) + y**2) * np.log((w + x) ** 2 + y**2)
        )
    )

    ex_1 = (
        (1 / 8)
        * np.pi ** (-1)
        * w ** (-1)
        * (
            (-4) * w
            + 2 * y * (np.arctan((w - x) / y) + np.arctan((w + x) / y))
            + (-1) * (w + x) * np.log((w + (-1) * x) ** 2 + y**2)
            + (w + x) * np.log((w + x) ** 2 + y**2)
        )
    )

    ex_2 = (
        (-1 / 8)
        * np.pi ** (-1)
        * w ** (-1)
        * (
            (-4) * w
            + 2 * y * (np.arctan((w - x) / y) + np.arctan((w + x) / y))
            + (w + (-1) * x)
            * (np.log((w + (-1) * x) ** 2 + y**2) + (-1) * np.log((w + x) ** 2 + y**2))
        )
    )

    ey_1 = (
        (1 / 8)
        * np.pi ** (-1)
        * w ** (-1)
        * (
            2 * (w + x) * (np.arctan((w - x) / y) + np.arctan((w + x) / y))
            + y
            * (np.log((w + (-1) * x) ** 2 + y**2) + (-1) * np.log((w + x) ** 2 + y**2))
        )
    )

    ey_2 = (
        (1 / 8)
        * np.pi ** (-1)
        * w ** (-1)
        * (
            2 * (w + (-1) * x) * (np.arctan((w - x) / y) + np.arctan((w + x) / y))
            + y
            * ((-1) * np.log((w + (-1) * x) ** 2 + y**2) + np.log((w + x) ** 2 + y**2))
        )
    )

    # Store displacement kernels (2-d matrix), [Nobs x 2 basis functions]
    Disp = np.hstack((u_1, u_2))

    # Store stress kernels (3-d matrix), [Nobs x (sx or sy) x 2 basis functions]
    Stress = np.zeros((n_obs, 2, 2))
    Stress[:, 0, :] = np.hstack((ex_1, ex_2)) * 2 * mu
    Stress[:, 1, :] = np.hstack((ey_1, ey_2)) * 2 * mu

    return Disp, Stress


def displacements_stresses_linear_force_no_rotation_planestrain(
    x, y, xf, yf, w, nu, mu=1
):
    """
    Compute displacement and stress kernels for a linearly varying force on a horizontal source element.

    Parameters:
    x, y : array_like
        Observation locations provided as individual vectors [Nobs x 1]
    xf, yf : float
        Source element center location
    w : float
        Source element half-length
    nu, mu : float
        Elastic parameters (Poisson's ratio and Shear Modulus)

    Returns:
    Disp : ndarray
        Displacement kernels [Nobs x (ux or uy) x (fx or fy) x 2 basis functions]
    Stress, Strain : ndarray
        Stress and strain kernels [Nobs x (sxx, sxy, syy) x (fx or fy) x 2 basis functions]
    """

    Nobs = len(x)

    xo = x - xf
    yo = y - yf

    # Displacement kernels
    def ux_1(fx, fy, w):
        term1 = (
            (1 / 8)
            * fx
            * w ** (-1)
            * (w - xo)
            * (3 * w + xo)
            * mu ** (-1)
            * nu
            * (np.pi - np.pi * nu) ** (-1)
            * np.log((w - xo) ** 2 + yo**2)
        )
        term2 = (
            (1 / 32)
            * fx
            * np.pi ** (-1)
            * w ** (-1)
            * mu ** (-1)
            * (nu - 1) ** (-1)
            * (
                4 * w * (8 * w * (nu - 1) + xo * (-3 + 4 * nu))
                - 16 * (w + xo) * yo * (nu - 1) * np.arctan((w - xo) / yo)
                - 16 * (w + xo) * yo * (nu - 1) * np.arctan((w + xo) / yo)
                + (3 * (w - xo) * (3 * w + xo) + yo**2 * (5 - 4 * nu))
                * np.log((w - xo) ** 2 + yo**2)
                + (
                    3 * (w + xo) ** 2
                    - 5 * yo**2
                    - 4 * (w + xo - yo) * (w + xo + yo) * nu
                )
                * np.log((w + xo) ** 2 + yo**2)
            )
        )
        term3 = (
            (1 / 16)
            * fy
            * np.pi ** (-1)
            * w ** (-1)
            * yo
            * mu ** (-1)
            * (nu - 1) ** (-1)
            * (
                4 * w
                - 2 * yo * (np.arctan((w - xo) / yo) + np.arctan((w + xo) / yo))
                + (w + xo)
                * (np.log((w - xo) ** 2 + yo**2) - np.log((w + xo) ** 2 + yo**2))
            )
        )
        return (term1 + term2 + term3) / 2

    def ux_2(fx, fy, w):
        term1 = (
            (1 / 64)
            * fx
            * np.pi ** (-1)
            * w ** (-1)
            * (w - xo) ** 2
            * mu ** (-1)
            * (3 - 4 * nu)
            * (nu - 1) ** (-1)
            * np.log((w - xo) ** 2 + yo**2)
        )
        term2 = (
            (1 / 32)
            * np.pi ** (-1)
            * w ** (-1)
            * (w - xo)
            * mu ** (-1)
            * (nu - 1) ** (-1)
            * (
                (-8)
                * fx
                * yo
                * (nu - 1)
                * (np.arctan((w - xo) / yo) + np.arctan((w + xo) / yo))
                + fy * yo * np.log((w - xo) ** 2 + yo**2)
            )
        )
        term3 = (
            (1 / 64)
            * np.pi ** (-1)
            * w ** (-1)
            * mu ** (-1)
            * (nu - 1) ** (-1)
            * (
                4
                * w
                * ((-2) * fy * yo + fx * xo * (3 - 4 * nu) + 8 * fx * w * (nu - 1))
                + yo**2
                * (
                    4 * fy * (np.arctan((w - xo) / yo) + np.arctan((w + xo) / yo))
                    + fx * (-5 + 4 * nu) * np.log((w - xo) ** 2 + yo**2)
                )
                + (
                    2 * fy * (xo - w) * yo
                    + fx
                    * (
                        3 * (3 * w - xo) * (w + xo)
                        + 5 * yo**2
                        - 4 * (3 * w - xo) * (w + xo) * nu
                    )
                )
                * np.log((w + xo) ** 2 + yo**2)
            )
        )
        return term1 + term2 + term3

    def uy_1(fx, fy, w):
        term1 = (
            (1 / 32)
            * fx
            * np.pi ** (-1)
            * w ** (-1)
            * yo
            * mu ** (-1)
            * (nu - 1) ** (-1)
            * (
                4 * w
                - 2 * yo * (np.arctan((w - xo) / yo) + np.arctan((w + xo) / yo))
                + (w + xo)
                * (np.log((w - xo) ** 2 + yo**2) - np.log((w + xo) ** 2 + yo**2))
            )
        )
        term2 = (
            (1 / 64)
            * fy
            * np.pi ** (-1)
            * w ** (-1)
            * mu ** (-1)
            * (nu - 1) ** (-1)
            * (
                4 * w * (2 * w + xo) * (-3 + 4 * nu)
                + 8 * (w + xo) * yo * (nu * 2 - 1) * np.arctan((-w + xo) / yo)
                - 8 * (w + xo) * yo * (nu * 2 - 1) * np.arctan((w + xo) / yo)
                + (
                    w**2 * (9 - 12 * nu)
                    + yo**2 * (1 - 4 * nu)
                    + 2 * w * xo * (-3 + 4 * nu)
                    + xo**2 * (-3 + 4 * nu)
                )
                * np.log((w - xo) ** 2 + yo**2)
                - (
                    (
                        -3 * (w + xo) ** 2
                        + yo**2
                        + 4 * (w + xo - yo) * (w + xo + yo) * nu
                    )
                )
                * np.log((w + xo) ** 2 + yo**2)
            )
        )
        return term1 + term2

    def uy_2(fx, fy, w):
        term1_fx = fx * (
            (-1 / 32)
            * np.pi ** (-1)
            * w ** (-1)
            * yo
            * mu ** (-1)
            * (nu - 1) ** (-1)
            * (
                4 * w
                - 2 * yo * (np.arctan((w - xo) / yo) + np.arctan((w + xo) / yo))
                + (-w + xo) * np.log((w - xo) ** 2 + yo**2)
            )
            + (w - xo)
            * yo
            * (32 * np.pi * w * mu * (-1 + nu)) ** (-1)
            * np.log(w**2 + 2 * w * xo + xo**2 + yo**2)
        )

        term2_fy = fy * (
            (-1 / 8)
            * np.pi ** (-1)
            * w ** (-1)
            * (w - xo)
            * yo
            * mu ** (-1)
            * (nu - 1) ** (-1)
            * (2 * nu - 1)
            * (np.arctan((w - xo) / yo) + np.arctan((w + xo) / yo))
            + (1 / 64)
            * np.pi ** (-1)
            * w ** (-1)
            * (w - xo) ** 2
            * mu ** (-1)
            * (3 - 4 * nu)
            * (nu - 1) ** (-1)
            * np.log((w - xo) ** 2 + yo**2)
            + (1 / 64)
            * np.pi ** (-1)
            * w ** (-1)
            * mu ** (-1)
            * (nu - 1) ** (-1)
            * (
                4 * w * (2 * w - xo) * (4 * nu - 3)
                + yo**2 * (4 * nu - 1) * np.log((w - xo) ** 2 + yo**2)
                + (
                    3 * (3 * w - xo) * (w + xo)
                    + yo**2
                    - 4 * (3 * w - xo) * (w + xo) * nu
                    + yo**2 * (1 - 4 * nu)
                )
                * np.log((w + xo) ** 2 + yo**2)
            )
        )

        return term1_fx + term2_fy

    # Stress kernels
    def sxy_1(fx, fy, w):
        term1 = (
            (-1 / 8)
            * fx
            * np.pi ** (-1)
            * w ** (-1)
            * ((-1) + nu) ** (-1)
            * (
                2 * w * (w - xo) * yo * ((w - xo) ** 2 + yo**2) ** (-1)
                + 2 * (w + xo) * ((-1) + nu) * np.arctan((w - xo) / yo)
                + 2 * (w + xo) * ((-1) + nu) * np.arctan((w + xo) / yo)
                + (1 / 2)
                * yo
                * ((-3) + 2 * nu)
                * (np.log((w - xo) ** 2 + yo**2) - np.log((w + xo) ** 2 + yo**2))
            )
        )

        term2 = (
            (1 / 8)
            * fy
            * np.pi ** (-1)
            * w ** (-1)
            * ((-1) + nu) ** (-1)
            * (
                (-2) * w * (w - xo) ** 2 * ((w - xo) ** 2 + yo**2) ** (-1)
                + 4 * w * nu
                + 2 * yo * nu * np.arctan((-w + xo) / yo)
                + (-2) * yo * nu * np.arctan((w + xo) / yo)
                + (1 / 2)
                * (w + xo)
                * ((-1) + 2 * nu)
                * (np.log((w - xo) ** 2 + yo**2) - np.log((w + xo) ** 2 + yo**2))
            )
        )

        return term1 + term2

    def sxy_2(fx, fy, w):
        term1 = (
            (-1 / 16)
            * fx
            * np.pi ** (-1)
            * w ** (-1)
            * ((-1 + nu) ** (-1))
            * (
                4 * w * (w + xo) * yo * ((w + xo) ** 2 + yo**2) ** (-1)
                + 4 * (w - xo) * (-1 + nu) * np.arctan((w - xo) / yo)
                + 4 * (w - xo) * (-1 + nu) * np.arctan((w + xo) / yo)
                + (-1)
                * yo
                * ((-3) + 2 * nu)
                * (np.log((w - xo) ** 2 + yo**2) - np.log((w + xo) ** 2 + yo**2))
            )
        )

        term2 = (
            (-1 / 16)
            * fy
            * np.pi ** (-1)
            * w ** (-1)
            * ((-1 + nu) ** (-1))
            * (
                (-4) * w * (w + xo) ** 2 * ((w + xo) ** 2 + yo**2) ** (-1)
                + 8 * w * nu
                + (-4) * yo * nu * np.arctan((w - xo) / yo)
                + (-4) * yo * nu * np.arctan((w + xo) / yo)
                + (-1)
                * (w - xo)
                * ((-1) + 2 * nu)
                * (np.log((w - xo) ** 2 + yo**2) - np.log((w + xo) ** 2 + yo**2))
            )
        )

        return term1 + term2

    def sxx_1(fx, fy, w):
        term1_fx = (
            (1 / 8)
            * fx
            * np.pi ** (-1)
            * w ** (-1)
            * (-1 + nu) ** (-1)
            * (
                ((w - xo) ** 2 + yo**2) ** (-1)
                * ((-6) * w * (w - xo) ** 2 - 8 * w * yo**2)
                + 4 * w * nu
                - 2 * yo * (-2 + nu) * np.arctan((w - xo) / yo)
                - 2 * yo * (-2 + nu) * np.arctan((w + xo) / yo)
                + (1 / 2)
                * (w + xo)
                * (-3 + 2 * nu)
                * (np.log((w - xo) ** 2 + yo**2) - np.log((w + xo) ** 2 + yo**2))
            )
        )

        term2_fy = (
            (1 / 8)
            * fy
            * np.pi ** (-1)
            * w ** (-1)
            * (-1 + nu) ** (-1)
            * (
                2 * w * ((-w) + xo) * yo * ((w - xo) ** 2 + yo**2) ** (-1)
                + 2 * (w + xo) * nu * np.arctan((w - xo) / yo)
                + 2 * (w + xo) * nu * np.arctan((w + xo) / yo)
                + (1 / 2)
                * yo
                * (1 + 2 * nu)
                * (np.log((w - xo) ** 2 + yo**2) - np.log((w + xo) ** 2 + yo**2))
            )
        )

        return term1_fx + term2_fy

    def sxx_2(fx, fy, w):
        term1_fx = (
            (1 / 16)
            * fx
            * np.pi ** (-1)
            * w ** (-1)
            * (-1 + nu) ** (-1)
            * (
                4
                * w
                * ((w + xo) ** 2 + yo**2) ** (-1)
                * (3 * (w + xo) ** 2 + 4 * yo**2 - 2 * ((w + xo) ** 2 + yo**2) * nu)
                + 4 * yo * (-2 + nu) * np.arctan((w - xo) / yo)
                + 4 * yo * (-2 + nu) * np.arctan((w + xo) / yo)
                + (w - xo)
                * (-3 + 2 * nu)
                * (np.log((w - xo) ** 2 + yo**2) - np.log((w + xo) ** 2 + yo**2))
            )
        )

        term2_fy = (
            (1 / 16)
            * fy
            * np.pi ** (-1)
            * w ** (-1)
            * (-1 + nu) ** (-1)
            * (
                (-4) * w * (w + xo) * yo * ((w + xo) ** 2 + yo**2) ** (-1)
                + 4 * (w - xo) * nu * np.arctan((w - xo) / yo)
                + 4 * (w - xo) * nu * np.arctan((w + xo) / yo)
                + (-1)
                * yo
                * (1 + 2 * nu)
                * (np.log((w - xo) ** 2 + yo**2) - np.log((w + xo) ** 2 + yo**2))
            )
        )

        return term1_fx + term2_fy

    def syy_1(fx, fy, w):
        term1_fx = (
            (-1 / 16)
            * fx
            * np.pi ** (-1)
            * w ** (-1)
            * (-1 + nu) ** (-1)
            * (
                ((w - xo) ** 2 + yo**2) ** (-1)
                * ((-4) * w * (w - xo) ** 2 - 8 * w * yo**2)
                + 8 * w * nu
                - 4 * yo * (-1 + nu) * np.arctan((w - xo) / yo)
                - 4 * yo * (-1 + nu) * np.arctan((w + xo) / yo)
                + (w + xo)
                * (-1 + 2 * nu)
                * (np.log((w - xo) ** 2 + yo**2) - np.log((w + xo) ** 2 + yo**2))
            )
        )

        term2_fy = (
            (-1 / 8)
            * fy
            * np.pi ** (-1)
            * w ** (-1)
            * (-1 + nu) ** (-1)
            * (
                2 * w * ((-w) + xo) * yo * ((w - xo) ** 2 + yo**2) ** (-1)
                + 2 * (w + xo) * (-1 + nu) * np.arctan((w - xo) / yo)
                + 2 * (w + xo) * (-1 + nu) * np.arctan((w + xo) / yo)
                + (1 / 2)
                * yo
                * (-1 + 2 * nu)
                * (np.log((w - xo) ** 2 + yo**2) - np.log((w + xo) ** 2 + yo**2))
            )
        )

        return term1_fx + term2_fy

    def syy_2(fx, fy, w):
        term1_fx = (
            (-1 / 8)
            * fx
            * np.pi ** (-1)
            * w ** (-1)
            * (-1 + nu) ** (-1)
            * (
                2
                * w
                * ((w + xo) ** 2 + yo**2) ** (-1)
                * ((w + xo) ** 2 + 2 * yo**2 - 2 * ((w + xo) ** 2 + yo**2) * nu)
                + 2 * yo * (-1 + nu) * np.arctan((w - xo) / yo)
                + 2 * yo * (-1 + nu) * np.arctan((w + xo) / yo)
                + (1 / 2)
                * (w - xo)
                * (-1 + 2 * nu)
                * (np.log((w - xo) ** 2 + yo**2) - np.log((w + xo) ** 2 + yo**2))
            )
        )

        term2_fy = (
            (-1 / 8)
            * fy
            * np.pi ** (-1)
            * w ** (-1)
            * (-1 + nu) ** (-1)
            * (
                (-2) * w * (w + xo) * yo * ((w + xo) ** 2 + yo**2) ** (-1)
                + 2 * (w - xo) * (-1 + nu) * np.arctan((w - xo) / yo)
                + 2 * (w - xo) * (-1 + nu) * np.arctan((w + xo) / yo)
                + (-1 / 2)
                * yo
                * (-1 + 2 * nu)
                * (np.log((w - xo) ** 2 + yo**2) - np.log((w + xo) ** 2 + yo**2))
            )
        )

        return term1_fx + term2_fy

    exx_1 = (
        lambda fx, fy, w: 1
        / (2 * mu * (1 + nu))
        * (sxx_1(fx, fy, w) - nu * syy_1(fx, fy, w))
    )
    exx_2 = (
        lambda fx, fy, w: 1
        / (2 * mu * (1 + nu))
        * (sxx_2(fx, fy, w) - nu * syy_2(fx, fy, w))
    )

    eyy_1 = (
        lambda fx, fy, w: 1
        / (2 * mu * (1 + nu))
        * (syy_1(fx, fy, w) - nu * sxx_1(fx, fy, w))
    )
    eyy_2 = (
        lambda fx, fy, w: 1
        / (2 * mu * (1 + nu))
        * (syy_2(fx, fy, w) - nu * sxx_2(fx, fy, w))
    )

    exy_1 = lambda fx, fy, w: 1 / (2 * mu) * sxy_1(fx, fy, w)
    exy_2 = lambda fx, fy, w: 1 / (2 * mu) * sxy_2(fx, fy, w)

    Disp = np.zeros((Nobs, 2, 2, 2))
    Stress = np.zeros((Nobs, 3, 2, 2))
    Strain = np.zeros((Nobs, 3, 2, 2))

    # fx kernels
    Disp[:, 0, 0, :] = np.hstack([ux_1(1, 0, w), ux_2(1, 0, w)])
    Disp[:, 1, 0, :] = np.hstack([uy_1(1, 0, w), uy_2(1, 0, w)])
    Stress[:, 0, 0, :] = np.hstack([sxx_1(1, 0, w), sxx_2(1, 0, w)])
    Stress[:, 1, 0, :] = np.hstack([sxy_1(1, 0, w), sxy_2(1, 0, w)])
    Stress[:, 2, 0, :] = np.hstack([syy_1(1, 0, w), syy_2(1, 0, w)])
    Strain[:, 0, 0, :] = np.hstack([exx_1(1, 0, w), exx_2(1, 0, w)])
    Strain[:, 1, 0, :] = np.hstack([exy_1(1, 0, w), exy_2(1, 0, w)])
    Strain[:, 2, 0, :] = np.hstack([eyy_1(1, 0, w), eyy_2(1, 0, w)])

    # fy kernels
    Disp[:, 0, 1, :] = np.hstack([ux_1(0, 1, w), ux_2(0, 1, w)])
    Disp[:, 1, 1, :] = np.hstack([uy_1(0, 1, w), uy_2(0, 1, w)])
    Stress[:, 0, 1, :] = np.hstack([sxx_1(0, 1, w), sxx_2(0, 1, w)])
    Stress[:, 1, 1, :] = np.hstack([sxy_1(0, 1, w), sxy_2(0, 1, w)])
    Stress[:, 2, 1, :] = np.hstack([syy_1(0, 1, w), syy_2(0, 1, w)])
    Strain[:, 0, 1, :] = np.hstack([exx_1(0, 1, w), exx_2(0, 1, w)])
    Strain[:, 1, 1, :] = np.hstack([exy_1(0, 1, w), exy_2(0, 1, w)])
    Strain[:, 2, 1, :] = np.hstack([eyy_1(0, 1, w), eyy_2(0, 1, w)])

    return Disp, Stress, Strain


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


def get_quadratic_coefficients_for_linear_slip(
    a, node_coordinates, end_displacement_1, end_displacement_2
):
    """Get quadratic node coeficients/weights for the case of
        linear slip

        NOTE: We may want to generalize this so that it just works
        on an element dictionary and doesn't need a or
        node_coefficients.

    Args:
        a (_type_): element half width
        node_coordinates: "x" location of quadratic nodes
        end_displacement_1: displacement at endpoint 1
        end_displacement_2: displacement at endpoint 2

    Returns:
        quadratic_coefficients: 3n nd.array with 3 quadratic coefficients
    """
    center_displacment = (end_displacement_1 + end_displacement_2) / 2.0
    physical_displacements = np.array(
        [end_displacement_1, center_displacment, end_displacement_2]
    )
    quadratic_coefficients = phicoef(node_coordinates, physical_displacements, a)
    return quadratic_coefficients


# Slip functions
def slip_functions(x, a):
    """Get pre-multiplier (L) to quadratic coefficients (x) to compute slip (Lx = slip) at any point on the fault patch"""
    design_matrix = np.zeros((len(x), 3))
    f1 = (x / a) * (9 * (x / a) / 8 - 3 / 4)
    f2 = (1 - 3 * (x / a) / 2) * (1 + 3 * (x / a) / 2)
    f3 = (x / a) * (9 * (x / a) / 8 + 3 / 4)
    design_matrix[:, 0] = f1
    design_matrix[:, 1] = f2
    design_matrix[:, 2] = f3
    return design_matrix


def slip_functions_mean(x):
    """Get pre-multiplier (L) to quadratic coefficients (x) to compute average slip (Lx = mean_slip) over the fault patch"""
    design_matrix = np.zeros((len(x), 3))
    f1 = 3 / 8
    f2 = 1 / 4
    f3 = 3 / 8
    design_matrix[:, 0] = f1
    design_matrix[:, 1] = f2
    design_matrix[:, 2] = f3
    return design_matrix


# Slip gradient functions
def slipgradient_functions(x, a):
    """Get pre-multiplier (L) to quadratic coefficients (x) to compute slip-gradient (Lx = dslip/dx) at any point on the fault patch.

    Note that the slip gradient is only along the fault."""
    design_matrix = np.zeros((len(x), 3))
    df_1_dx = (9 * x) / (4 * a**2) - 3 / (4 * a)
    df_2_dx = -(9 * x) / (2 * a**2)
    df_3_dx = (9 * x) / (4 * a**2) + 3 / (4 * a)
    design_matrix[:, 0] = df_1_dx
    design_matrix[:, 1] = df_2_dx
    design_matrix[:, 2] = df_3_dx
    return design_matrix


# Compute 3qn coefficients for given slip
def phicoef(x, slip, a):
    """Get quadratic node coefficients for slip specified at the 3 nodes as an ordered set (x,slip)"""
    mat = slip_functions(x, a)
    return np.linalg.inv(mat) @ slip


# compute slip and slip gradients from 3qn coefficients
def get_slip_slipgradient(x, a, phi):
    """Get slip and slip gradient for a given fault patch at any point (x) within the fault
    from quadratic coefficients (phi)"""
    slip_mat = slip_functions(x, a)
    slipgradient_mat = slipgradient_functions(x, a)
    slip = slip_mat @ phi
    slipgradient = slipgradient_mat @ phi
    return slip, slipgradient


def get_matrices_slip_slip_gradient(els, flag="node", reference="global"):
    """Assemble design matrix in (x,y) coordinate system for 2 slip components (s,n) for a
    linear system of equations to calculate quadratic coefficients from applied boundary conditions for an ordered list of fault elements.

    flag = "node" : slip is applied at each node of a fault element
    flag = "mean" : slip is applied as a mean value over the entire fault element, not just at nodes

    reference = "global" : slip is applied in a global (x,y) coordinate system
    reference = "local"  : slip is applied in a local (s,n) coordinate system

    Unit vectors for each patch are used to premultiply the input matrices for the global reference frame
    [dx nx] [f1 f2 f3 0  0  0]
    [dy ny] [0  0  0  f1 f2 f3]"""

    stride = 6
    n_els = len(els.x1)
    mat_slip = np.zeros((stride * n_els, stride * n_els))
    mat_slip_gradient = np.zeros_like(mat_slip)

    for i in range(n_els):
        slip_mat_stack = np.zeros((stride, stride))
        slip_gradient_mat_stack = np.zeros_like(slip_mat_stack)
        unit_vec_mat = np.array(
            [
                [els.x_shears[i], els.x_normals[i]],
                [els.y_shears[i], els.y_normals[i]],
            ]
        )
        unit_vec_mat_stack = np.kron(np.eye(3), unit_vec_mat)
        x_obs = np.array([-els.half_lengths[i], 0.0, els.half_lengths[i]])

        if flag == "node":
            slip_mat = slip_functions(x_obs, els.half_lengths[i])
        elif flag == "mean":
            slip_mat = slip_functions_mean(x_obs)
        else:
            raise ValueError("Invalid flag. Use either 'node' or 'mean'.")

        slip_gradient_mat = slipgradient_functions(x_obs, els.half_lengths[i])
        slip_mat_stack[0::2, 0:3] = slip_mat
        slip_mat_stack[1::2, 3:] = slip_mat
        slip_gradient_mat_stack[0::2, 0:3] = slip_gradient_mat
        slip_gradient_mat_stack[1::2, 3:] = slip_gradient_mat

        if reference == "global":
            mat_slip[stride * i : stride * (i + 1), stride * i : stride * (i + 1)] = (
                unit_vec_mat_stack @ slip_mat_stack
            )
            mat_slip_gradient[
                stride * i : stride * (i + 1), stride * i : stride * (i + 1)
            ] = (unit_vec_mat_stack @ slip_gradient_mat_stack)
        elif reference == "local":
            mat_slip[stride * i : stride * (i + 1), stride * i : stride * (i + 1)] = (
                slip_mat_stack
            )

            mat_slip_gradient[
                stride * i : stride * (i + 1), stride * i : stride * (i + 1)
            ] = slip_gradient_mat_stack
        else:
            raise ValueError("Invalid reference frame. Use either 'global' or 'local'")

    return mat_slip, mat_slip_gradient


def get_matrices_slip_slip_gradient_antiplane(els, flag="node"):
    """Assemble design matrix in (x,y) coordinate system for antiplane slip for a
    linear system of equations to calculate quadratic coefficients from applied boundary conditions for an ordered list of fault elements.

    flag = "node" : slip is applied at each node of a fault element
    flag = "mean" : slip is applied as a mean value over the entire fault element, not just at nodes

    Unit vectors for each patch are used to premultiply the input matrices for the global reference frame
    [s   =  [f1   f2   f3 ].[φ1
    ds/dζ]  [f1'  f2'  f3']  φ2
                             φ3]"""

    stride = 3
    n_els = len(els.x1)
    mat_slip = np.zeros((stride * n_els, stride * n_els))
    mat_slip_gradient = np.zeros_like(mat_slip)

    for i in range(n_els):

        x_obs = np.array([-els.half_lengths[i], 0.0, els.half_lengths[i]])

        if flag == "node":
            slip_mat = slip_functions(x_obs, els.half_lengths[i])
        elif flag == "mean":
            slip_mat = slip_functions_mean(x_obs)
        else:
            raise ValueError("Invalid flag. Use either 'node' or 'mean'.")

        slip_gradient_mat = slipgradient_functions(x_obs, els.half_lengths[i])

        mat_slip[stride * i : stride * (i + 1), stride * i : stride * (i + 1)] = (
            slip_mat
        )
        mat_slip_gradient[
            stride * i : stride * (i + 1), stride * i : stride * (i + 1)
        ] = slip_gradient_mat

    return mat_slip, mat_slip_gradient


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


def get_displacement_stress_kernel_constant(x_obs, y_obs, els, mu, nu, flag):
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


def get_displacement_stress_kernel(x_obs, y_obs, els, mu, nu, flag):
    """Function to calculate displacement and stress kernels at a numpy array of locations [x_obs,y_obs]

    flag can either be "shear" or "normal" for kernels resulting shear slip or tensile slip

    kernels returned are stress_xx, stress_yy, stress_xy, u_x, u_y
    """
    n_obs = len(x_obs)
    n_els = len(els.x1)

    kernel_ux = np.zeros((n_obs, 3 * n_els))
    kernel_uy = np.zeros((n_obs, 3 * n_els))
    kernel_sxx = np.zeros((n_obs, 3 * n_els))
    kernel_syy = np.zeros((n_obs, 3 * n_els))
    kernel_sxy = np.zeros((n_obs, 3 * n_els))

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

        # Go through each of the 3 components for a given patch
        # Component 1
        slip_vector = np.array([1.0, 0.0, 0.0])
        strike_slip = slip_vector * flag_strike_slip
        tensile_slip = slip_vector * flag_tensile_slip

        # Calculate displacements and stresses for current element
        (
            displacement_local,
            stress_local,
        ) = displacements_stresses_quadratic_no_rotation(
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
        index = 3 * i
        kernel_sxx[:, index] = stress_eval[0, :]
        kernel_syy[:, index] = stress_eval[1, :]
        kernel_sxy[:, index] = stress_eval[2, :]
        kernel_ux[:, index] = displacement_eval[0, :]
        kernel_uy[:, index] = displacement_eval[1, :]

        # Component 2
        slip_vector = np.array([0.0, 1.0, 0.0])
        strike_slip = slip_vector * flag_strike_slip
        tensile_slip = slip_vector * flag_tensile_slip

        # Calculate displacements and stresses for current element
        (
            displacement_local,
            stress_local,
        ) = displacements_stresses_quadratic_no_rotation(
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
        index = 3 * i + 1
        kernel_sxx[:, index] = stress_eval[0, :]
        kernel_syy[:, index] = stress_eval[1, :]
        kernel_sxy[:, index] = stress_eval[2, :]
        kernel_ux[:, index] = displacement_eval[0, :]
        kernel_uy[:, index] = displacement_eval[1, :]

        # Component 3
        slip_vector = np.array([0.0, 0.0, 1.0])
        strike_slip = slip_vector * flag_strike_slip
        tensile_slip = slip_vector * flag_tensile_slip

        # Calculate displacements and stresses for current element
        (
            displacement_local,
            stress_local,
        ) = displacements_stresses_quadratic_no_rotation(
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
        index = 3 * i + 2
        kernel_sxx[:, index] = stress_eval[0, :]
        kernel_syy[:, index] = stress_eval[1, :]
        kernel_sxy[:, index] = stress_eval[2, :]
        kernel_ux[:, index] = displacement_eval[0, :]
        kernel_uy[:, index] = displacement_eval[1, :]
    return kernel_sxx, kernel_syy, kernel_sxy, kernel_ux, kernel_uy


def rotate_stress_antiplane(stress, inverse_rotation_matrix):
    """Rotate antiplane stress vector from local to global reference frame"""

    stress_rot = np.transpose(
        np.tensordot(stress, inverse_rotation_matrix, axes=(1, 1)), (0, 2, 1)
    )

    return stress_rot


def get_displacement_stress_kernel_slip_antiplane(x_obs, y_obs, els, mu=1):
    """Function to calculate displacement and stress kernels due to a fault source in antiplane geometry
    at a numpy array of locations [x_obs,y_obs]

    kernels returned are stress_xz, stress_yz, u
    """
    n_obs = len(x_obs)
    n_els = len(els.x1)

    kernel_u = np.zeros((n_obs, 3 * n_els))
    kernel_sxz = np.zeros((n_obs, 3 * n_els))
    kernel_syz = np.zeros((n_obs, 3 * n_els))

    for i in range(n_els):  # loop over each element in els()
        # Center observation locations (no translation needed)
        x_trans = x_obs - els.x_centers[i]
        y_trans = y_obs - els.y_centers[i]

        # Rotate observations such that fault element is horizontal
        rotated_coordinates = els.rot_mats_inv[i, :, :] @ np.vstack(
            (x_trans.T, y_trans.T)
        )
        x_rot = rotated_coordinates[0, :].T + els.x_centers[i]
        y_rot = rotated_coordinates[1, :].T + els.y_centers[i]

        # Calculate displacements and stresses for current element
        # returns a 2-d matrix for disp [n_obs x 3 basis functions], 3-d matrix for stress [n_obs x [sx,sy] x 3 basis functions]
        (
            displacement_eval,
            stress_local,
        ) = displacements_stresses_quadratic_slip_no_rotation_antiplane(
            x_rot.reshape(-1, 1),
            y_rot.reshape(-1, 1),
            els.half_lengths[i],
            mu,
            els.x_centers[i],
            els.y_centers[i],
        )
        # rotate stress from local -> global coordinates
        stress_eval = rotate_stress_antiplane(stress_local, els.rot_mats[i, :, :])

        for j in range(3):  # loop over each basis function
            index = 3 * i + j
            kernel_sxz[:, index] = stress_eval[:, 0, j]
            kernel_syz[:, index] = stress_eval[:, 1, j]
            kernel_u[:, index] = displacement_eval[:, j]

    return kernel_sxz, kernel_syz, kernel_u


def get_displacement_stress_kernel_force_antiplane(x_obs, y_obs, els, mu=1):
    """Function to calculate displacement and stress kernels due to a line force source in antiplane geometry
    at a numpy array of locations [x_obs,y_obs]

    kernels returned are 3-d matrices stress_xz, stress_yz, u [Nobs x 2 basis functions x Nsources]
    """
    n_obs = len(x_obs)
    n_els = len(els.x1)

    kernel_u = np.zeros((n_obs, 2, n_els))
    kernel_sxz = np.zeros((n_obs, 2, n_els))
    kernel_syz = np.zeros((n_obs, 2, n_els))

    for i in range(n_els):  # loop over each element in els()
        # Center observation locations (no translation needed)
        x_trans = x_obs - els.x_centers[i]
        y_trans = y_obs - els.y_centers[i]

        # Rotate observations such that fault element is horizontal
        rotated_coordinates = els.rot_mats_inv[i, :, :] @ np.vstack(
            (x_trans.T, y_trans.T)
        )
        x_rot = rotated_coordinates[0, :].T + els.x_centers[i]
        y_rot = rotated_coordinates[1, :].T + els.y_centers[i]

        # Calculate displacements and stresses for current element
        # returns a 2-d matrix for disp [n_obs x 2 basis functions], 3-d matrix for stress [n_obs x [sx,sy] x 2 basis functions]
        (
            displacement_eval,
            stress_local,
        ) = displacements_stresses_linear_force_no_rotation_antiplane(
            x_rot.reshape(-1, 1),
            y_rot.reshape(-1, 1),
            els.half_lengths[i],
            mu,
            els.x_centers[i],
            els.y_centers[i],
        )
        # rotate stress from local -> global coordinates
        stress_eval = rotate_stress_antiplane(stress_local, els.rot_mats[i, :, :])

        kernel_sxz[:, :, i] = stress_eval[:, 0, :]
        kernel_syz[:, :, i] = stress_eval[:, 1, :]
        kernel_u[:, :, i] = displacement_eval[:, :]

    return kernel_sxz, kernel_syz, kernel_u


def get_displacement_stress_kernel_force_planestrain(x_obs, y_obs, els, mu=1, nu=0.25):
    """Function to calculate displacement and stress kernels due to a force source in planestrain
    at a numpy array of locations [x_obs,y_obs]

    kernels returned are stress_xx, stress_xy, stress_yy, u_x, u_y

    4-D kernels [Nobs x (fx,fy) x 2 basis functions x Nsources]
    """

    def rotate_vector(vector, inverse_rotation_matrix):
        """Rotate vector from local to global reference frame"""

        vector_rot = np.transpose(
            np.tensordot(vector, inverse_rotation_matrix, axes=(1, 1)), (0, 1, 3, 2)
        )
        return vector_rot

    def rotate_tensor(tensor, rotation_matrix):
        """Rotate vector from local to global reference frame"""
        inverse_rotation_matrix = rotation_matrix.T
        tensor_rotated = np.zeros_like(tensor)
        t0 = (
            tensor[:, 0, :, :] * inverse_rotation_matrix[0, 0]
            + tensor[:, 1, :, :] * inverse_rotation_matrix[1, 0]
        )
        t1 = (
            tensor[:, 0, :, :] * inverse_rotation_matrix[0, 1]
            + tensor[:, 1, :, :] * inverse_rotation_matrix[1, 1]
        )
        t2 = (
            tensor[:, 1, :, :] * inverse_rotation_matrix[0, 0]
            + tensor[:, 2, :, :] * inverse_rotation_matrix[1, 0]
        )
        t3 = (
            tensor[:, 1, :, :] * inverse_rotation_matrix[0, 1]
            + tensor[:, 2, :, :] * inverse_rotation_matrix[1, 1]
        )

        tensor_rotated[:, 0, :, :] = (
            inverse_rotation_matrix[0, 0] * t0 + inverse_rotation_matrix[1, 0] * t2
        )
        tensor_rotated[:, 1, :, :] = (
            inverse_rotation_matrix[0, 0] * t1 + inverse_rotation_matrix[1, 0] * t3
        )
        tensor_rotated[:, 2, :, :] = (
            inverse_rotation_matrix[0, 1] * t1 + inverse_rotation_matrix[1, 1] * t3
        )

        return tensor_rotated

    n_obs = len(x_obs)
    n_els = len(els.x1)

    kernel_ux = np.zeros((n_obs, 2, 2, n_els))
    kernel_uy = np.zeros((n_obs, 2, 2, n_els))
    kernel_sxx = np.zeros((n_obs, 2, 2, n_els))
    kernel_sxy = np.zeros((n_obs, 2, 2, n_els))
    kernel_syy = np.zeros((n_obs, 2, 2, n_els))

    for i in range(n_els):  # loop over each element in els()
        # Center observation locations (no translation needed)
        x_trans = x_obs - els.x_centers[i]
        y_trans = y_obs - els.y_centers[i]

        # Rotate observations such that fault element is horizontal
        rotated_coordinates = els.rot_mats_inv[i, :, :] @ np.vstack(
            (x_trans.T, y_trans.T)
        )
        x_rot = rotated_coordinates[0, :].T  # + els.x_centers[i]
        y_rot = rotated_coordinates[1, :].T  # + els.y_centers[i]

        # Calculate displacements and stresses for current element
        # returns a 4-d matrix for disp [n_obs x (ux,uy) x (fx,fy) x 2 basis functions],
        # 4-d matrix for stress [n_obs x [sxx,sxy,syy] x (fx,fy) x 2 basis functions]
        Dkernels, Skernels, _ = (
            displacements_stresses_linear_force_no_rotation_planestrain(
                x_rot.reshape(-1, 1),
                y_rot.reshape(-1, 1),
                xf=0,
                yf=0,
                w=els.half_lengths[i],
                nu=nu,
                mu=mu,
            )
        )

        # rotate stress from local -> global coordinates
        Dkernels_rot = rotate_vector(Dkernels, els.rot_mats_inv[i, :, :])
        Skernels_rot = rotate_tensor(Skernels, els.rot_mats_inv[i, :, :])

        kernel_sxx[:, :, :, i] = Skernels_rot[:, 0, :, :]
        kernel_sxy[:, :, :, i] = Skernels_rot[:, 1, :, :]
        kernel_syy[:, :, :, i] = Skernels_rot[:, 2, :, :]
        kernel_ux[:, :, :, i] = Dkernels_rot[:, 0, :, :]
        kernel_uy[:, :, :, i] = Dkernels_rot[:, 1, :, :]

    return kernel_ux, kernel_uy, kernel_sxx, kernel_syy, kernel_sxy


def coeffs_to_disp_stress(kernels_s, kernels_n, coeffs_s, coeffs_n):
    """Function to compute displacements and stresses from 3qn coefficients.

    Provide separate shear (k_s) and normal (k_n) stress kernels and appropriate 3qn coefficients
    """
    ux = kernels_s[3] @ coeffs_s + kernels_n[3] @ coeffs_n
    uy = kernels_s[4] @ coeffs_s + kernels_n[4] @ coeffs_n
    sxx = kernels_s[0] @ coeffs_s + kernels_n[0] @ coeffs_n
    syy = kernels_s[1] @ coeffs_s + kernels_n[1] @ coeffs_n
    sxy = kernels_s[2] @ coeffs_s + kernels_n[2] @ coeffs_n
    return ux, uy, sxx, syy, sxy


def get_traction_kernels(els, kernels, flag="global"):
    """Function to calculate kernels of traction vector from a set of stress kernels and unit vectors.

    Provide elements as a list with ["x_normal"] & ["y_normal"] for the unit normal vector.

    kernels must be provided as kernels[0] = Kxx, kernels[1] = Kyy, kernels[2] = Kxy

    flag can be either "global" for (x,y) coordinates or "local" for (shear,normal) coordinates
    """
    Kxx = kernels[0]
    Kyy = kernels[1]
    Kxy = kernels[2]
    nrows = np.shape(Kxx)[
        0
    ]  # this is typically the same as number of elements because stress kernels are calculated ONLY at the center of a given element

    tx = np.zeros_like(Kxx)
    ty = np.zeros_like(Kxx)
    # unit vector in normal direction
    nvec = np.zeros((nrows, 2))
    # unit vector in shear direction
    svec = np.zeros((nrows, 2))
    svec = np.vstack((els.x_shears, els.y_shears)).T
    for i in range(nrows):
        nvec[i, :] = np.array([els.x_normals[i], els.y_normals[i]])

    nx_matrix = np.zeros_like(Kxx)
    ny_matrix = np.zeros_like(Kxx)
    sx_matrix = np.zeros_like(Kxx)
    sy_matrix = np.zeros_like(Kxx)

    nx_matrix[:, 0::3] = nvec[:, 0].reshape(-1, 1)
    nx_matrix[:, 1::3] = nvec[:, 0].reshape(-1, 1)
    nx_matrix[:, 2::3] = nvec[:, 0].reshape(-1, 1)
    ny_matrix[:, 0::3] = nvec[:, 1].reshape(-1, 1)
    ny_matrix[:, 1::3] = nvec[:, 1].reshape(-1, 1)
    ny_matrix[:, 2::3] = nvec[:, 1].reshape(-1, 1)

    sx_matrix[:, 0::3] = svec[:, 0].reshape(-1, 1)
    sx_matrix[:, 1::3] = svec[:, 0].reshape(-1, 1)
    sx_matrix[:, 2::3] = svec[:, 0].reshape(-1, 1)
    sy_matrix[:, 0::3] = svec[:, 1].reshape(-1, 1)
    sy_matrix[:, 1::3] = svec[:, 1].reshape(-1, 1)
    sy_matrix[:, 2::3] = svec[:, 1].reshape(-1, 1)

    # traction vector t = n.σ
    tx = Kxx * nx_matrix + Kxy * ny_matrix
    ty = Kxy * nx_matrix + Kyy * ny_matrix

    ts = tx * sx_matrix + ty * sy_matrix
    tn = tx * nx_matrix + ty * ny_matrix

    if flag == "global":
        return tx, ty
    elif flag == "local":
        return ts, tn
    else:
        ValueError("flag must be either global or local")


def get_traction_kernels_antiplane(els, kernels, nbasis=3):
    """Function to calculate kernels of traction vector from a set of stress kernels and unit vectors for a given number of basis functions.

    Provide elements as a list with ["x_normal"] & ["y_normal"] for the unit normal vector.

    kernels must be provided as kernels[0] = Kxz, kernels[1] = Kyz

    """
    Kxz = kernels[0]
    Kyz = kernels[1]

    # nbasis = int(np.shape(Kxz)[1]/len(els.x1))

    # unit vector in normal direction
    nvec = np.vstack((els.x_normals, els.y_normals)).T

    nx_matrix = np.zeros_like(Kxz)
    ny_matrix = np.zeros_like(Kxz)
    for i in range(nbasis):
        nx_matrix[:, i::nbasis] = nvec[:, 0].reshape(-1, 1)
        ny_matrix[:, i::nbasis] = nvec[:, 1].reshape(-1, 1)

    # nx_matrix[:, 0::3] = nvec[:, 0].reshape(-1, 1)
    # nx_matrix[:, 1::3] = nvec[:, 0].reshape(-1, 1)
    # nx_matrix[:, 2::3] = nvec[:, 0].reshape(-1, 1)
    # ny_matrix[:, 0::3] = nvec[:, 1].reshape(-1, 1)
    # ny_matrix[:, 1::3] = nvec[:, 1].reshape(-1, 1)
    # ny_matrix[:, 2::3] = nvec[:, 1].reshape(-1, 1)

    # traction vector t = n.σ
    t = Kxz * nx_matrix + Kyz * ny_matrix

    return t


def plot_displacements_stresses_els(
    els, n_obs, ux, uy, sxx, syy, sxy, x_obs, y_obs, n_skip_plot=1
):
    """Plot 2 displacement (ux,uy) and 3 stress fields (sxx,syy,sxy) within a domain x_obs,y_obs

    n_skip_plot is used for plotting displacement vectors - specify the number of points to skip
    """

    def plot_els(els):
        n_els = len(els.x1)
        for i in range(n_els):
            plt.plot(
                [els.x1[i], els.x2[i]],
                [els.y1[i], els.y2[i]],
                ".-k",
                linewidth=1.0,
            )

    # Plot displacements
    plt.figure(figsize=(18, 8))
    plt.subplot(2, 3, 1)
    maxval = np.max(np.abs(ux))
    img = plt.contourf(
        x_obs.reshape(n_obs, n_obs),
        y_obs.reshape(n_obs, n_obs),
        ux.reshape(n_obs, n_obs),
        cmap="coolwarm",
        vmin=-maxval,
        vmax=maxval,
        levels=np.linspace(-maxval, maxval, 11),
    )
    plt.colorbar(img)
    plt.contour(
        x_obs.reshape(n_obs, n_obs),
        y_obs.reshape(n_obs, n_obs),
        ux.reshape(n_obs, n_obs),
        linewidths=0.25,
        colors="k",
        levels=np.linspace(-maxval, maxval, 11),
    )
    plot_els(els)
    plt.xlim([np.min(x_obs), np.max(x_obs)])
    plt.ylim([np.min(y_obs), np.max(y_obs)])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.quiver(
        x_obs[0::n_skip_plot],
        y_obs[0::n_skip_plot],
        ux[0::n_skip_plot],
        uy[0::n_skip_plot],
    )
    plt.title("$u_x$")

    plt.subplot(2, 3, 2)
    maxval = np.max(np.abs(uy))
    img = plt.contourf(
        x_obs.reshape(n_obs, n_obs),
        y_obs.reshape(n_obs, n_obs),
        uy.reshape(n_obs, n_obs),
        cmap="coolwarm",
        levels=np.linspace(-maxval, maxval, 11),
        vmin=-maxval,
        vmax=maxval,
    )
    plt.colorbar(img)
    plt.contour(
        x_obs.reshape(n_obs, n_obs),
        y_obs.reshape(n_obs, n_obs),
        uy.reshape(n_obs, n_obs),
        linewidths=0.25,
        colors="k",
        levels=np.linspace(-maxval, maxval, 11),
    )
    plot_els(els)
    plt.xlim([np.min(x_obs), np.max(x_obs)])
    plt.ylim([np.min(y_obs), np.max(y_obs)])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.quiver(
        x_obs[0::n_skip_plot],
        y_obs[0::n_skip_plot],
        ux[0::n_skip_plot],
        uy[0::n_skip_plot],
    )
    plt.title("$u_y$")

    # Plot stresses
    plt.subplot(2, 3, 4)
    toplot = sxx
    maxval = np.max(np.abs(toplot))
    img = plt.contourf(
        x_obs.reshape(n_obs, n_obs),
        y_obs.reshape(n_obs, n_obs),
        toplot.reshape(n_obs, n_obs),
        cmap="RdYlBu_r",
        vmin=-maxval,
        vmax=maxval,
        levels=np.linspace(-maxval, maxval, 11),
    )
    plt.colorbar(img)
    plt.contour(
        x_obs.reshape(n_obs, n_obs),
        y_obs.reshape(n_obs, n_obs),
        toplot.reshape(n_obs, n_obs),
        linewidths=0.25,
        colors="k",
        levels=np.linspace(-maxval, maxval, 11),
    )
    plt.clim(-maxval, maxval)
    plot_els(els)
    plt.xlim([np.min(x_obs), np.max(x_obs)])
    plt.ylim([np.min(y_obs), np.max(y_obs)])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("$\sigma_{xx}$")

    plt.subplot(2, 3, 5)
    toplot = syy
    maxval = np.max(np.abs(toplot))
    img = plt.contourf(
        x_obs.reshape(n_obs, n_obs),
        y_obs.reshape(n_obs, n_obs),
        toplot.reshape(n_obs, n_obs),
        levels=np.linspace(-maxval, maxval, 11),
        cmap="RdYlBu_r",
        vmin=-maxval,
        vmax=maxval,
    )
    plt.colorbar(img)
    plt.contour(
        x_obs.reshape(n_obs, n_obs),
        y_obs.reshape(n_obs, n_obs),
        toplot.reshape(n_obs, n_obs),
        linewidths=0.25,
        colors="k",
        levels=np.linspace(-maxval, maxval, 11),
    )
    plot_els(els)
    plt.xlim([np.min(x_obs), np.max(x_obs)])
    plt.ylim([np.min(y_obs), np.max(y_obs)])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("$\sigma_{yy}$")

    plt.subplot(2, 3, 6)
    toplot = sxy
    maxval = np.max(np.abs(toplot))
    img = plt.contourf(
        x_obs.reshape(n_obs, n_obs),
        y_obs.reshape(n_obs, n_obs),
        toplot.reshape(n_obs, n_obs),
        levels=np.linspace(-maxval, maxval, 11),
        cmap="RdYlBu_r",
        vmin=-maxval,
        vmax=maxval,
    )
    plt.colorbar(img)
    plt.contour(
        x_obs.reshape(n_obs, n_obs),
        y_obs.reshape(n_obs, n_obs),
        toplot.reshape(n_obs, n_obs),
        levels=np.linspace(-maxval, maxval, 11),
        linewidths=0.25,
        colors="k",
    )
    plot_els(els)
    plt.xlim([np.min(x_obs), np.max(x_obs)])
    plt.ylim([np.min(y_obs), np.max(y_obs)])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("$\sigma_{xy}$")
    plt.show()


def initialize_els():
    els = addict.Dict()
    els.angles = np.array([])
    els.lengths = np.array([])
    els.half_lengths = np.array([])
    els.x_centers = np.array([])
    els.y_centers = np.array([])
    els.rot_mats = np.array([])
    els.rot_mats_inv = np.array([])
    els.x_normals = np.array([])
    els.y_normals = np.array([])
    els.x_shears = np.array([])
    els.y_shears = np.array([])
    els.x_nodes = np.array([])
    els.y_nodes = np.array([])
    return els


def standardize_els_geometry(els, reorder=True):
    for i in range(len(els.x1)):
        # If neccesary change order of end points so that
        # x1 is closest to negative infinity
        if (els.x2[i] < els.x1[i]) & (reorder == True):
            els.x2[i], els.x1[i] = els.x1[i], els.x2[i]
            els.y2[i], els.y1[i] = els.y1[i], els.y2[i]

        dx = els.x2[i] - els.x1[i]
        dy = els.y2[i] - els.y1[i]
        magnitude = np.sqrt(dx**2.0 + dy**2.0)
        els.angles = np.append(els.angles, np.arctan2(dy, dx))
        els.lengths = np.append(els.lengths, magnitude)
        els.half_lengths = np.append(els.half_lengths, 0.5 * els.lengths[i])
        els.x_centers = np.append(els.x_centers, 0.5 * (els.x2[i] + els.x1[i]))
        els.y_centers = np.append(els.y_centers, 0.5 * (els.y2[i] + els.y1[i]))
        els.rot_mats = np.append(
            els.rot_mats,
            np.array(
                [
                    [np.cos(els.angles[i]), -np.sin(els.angles[i])],
                    [np.sin(els.angles[i]), np.cos(els.angles[i])],
                ]
            ),
        ).reshape(-1, 2, 2)
        els.rot_mats_inv = np.append(
            els.rot_mats_inv,
            np.array(
                [
                    [np.cos(-els.angles[i]), -np.sin(-els.angles[i])],
                    [np.sin(-els.angles[i]), np.cos(-els.angles[i])],
                ]
            ),
        ).reshape(-1, 2, 2)
        els.x_normals = np.append(els.x_normals, -dy / magnitude)
        els.y_normals = np.append(els.y_normals, dx / magnitude)
        els.x_shears = np.append(els.x_shears, dx / magnitude)
        els.y_shears = np.append(els.y_shears, dy / magnitude)
        els.x_nodes = np.append(
            els.x_nodes,
            np.array(
                [
                    els.x_centers[i] - (2 / 3 * dx / 2),
                    els.x_centers[i],
                    els.x_centers[i] + (2 / 3 * dx / 2),
                ]
            ),
        ).reshape(-1, 3)
        els.y_nodes = np.append(
            els.y_nodes,
            np.array(
                [
                    els.y_centers[i] - (2 / 3 * dy / 2),
                    els.y_centers[i],
                    els.y_centers[i] + (2 / 3 * dy / 2),
                ]
            ),
        ).reshape(-1, 3)


def get_strainenergy_from_stress(sxx, syy, sxy, mu, nu, conversion="plane_strain"):
    youngs_modulus = 2 * mu * (1 + nu)

    if conversion == "plane_strain":
        # print(f"{conversion=}")
        # Plane strain linear operator. I think this is Crouch and Starfield???
        stress_from_strain_plane_strain = (
            youngs_modulus
            / ((1 + nu) * (1 - 2 * nu))
            * np.array([[1 - nu, nu, 0], [nu, 1 - nu, 0], [0, 0, (1 - 2 * nu) / 2.0]])
        )
        strain_from_stress_plane_strain = np.linalg.inv(stress_from_strain_plane_strain)
        operator = np.copy(strain_from_stress_plane_strain)

    elif conversion == "plane_stress":
        # print(f"{conversion=}")
        # Plane stress linear operator
        strain_from_stress_plane_stress = (
            1
            / youngs_modulus
            * np.array([[1, -nu, 0], [-nu, 1, 0], [0, 0, 2 * (1 + nu)]])
        )
        operator = np.copy(strain_from_stress_plane_stress)

    # print(f"{operator}")

    # Calculate strains
    exx = np.zeros_like(sxx)
    exy = np.zeros_like(syy)
    eyy = np.zeros_like(sxy)
    for i in range(len(sxx)):
        stresses = np.array([sxx[i], syy[i], sxy[i]])
        exx[i], eyy[i], exy[i] = operator @ stresses

    # Calculate strain energy
    strain_energy = sxx * exx + syy * eyy + sxy * exy
    return strain_energy


def get_slipvector_on_fault(els, coeffs, n_eval):
    """Get slip vector evaluated ON the fault in (x,y) coordinate system.

    Args:
        els: fault geometry data structure
        coeffs: quadratic slip coefficients ordered as [3 x shear_slip, 3 x tensile_slip] per fault element
        n_eval: number of points to evaluate slip vector

    Returns:
        x_obs, y_obs: x,y coordinates of locations where slip vector is computed
        fault_slip_x, fault_slip_y: components of slip vector, each of dimension [n_eval x n_els]
    """

    stride = 6
    n_els = len(els.x1)

    # calculate slip as a continuous function
    fault_slip_s = np.zeros(n_els * n_eval)
    fault_slip_n = np.zeros(n_els * n_eval)
    fault_slip_x = np.zeros(n_els * n_eval)
    fault_slip_y = np.zeros(n_els * n_eval)
    # evaluation locations
    x_obs = np.zeros_like(fault_slip_x)
    y_obs = np.zeros_like(fault_slip_x)

    # Extract (s, n) components and store them in two separate vectors
    coeffs_s = np.zeros((3 * n_els))
    coeffs_n = np.zeros((3 * n_els))
    for i in range(n_els):
        coeffs_s[3 * i : 3 * (i + 1)] = coeffs[stride * i : stride * i + 3]
        coeffs_n[3 * i : 3 * (i + 1)] = coeffs[stride * i + 3 : stride * (i + 1)]

    for i in range(n_els):
        xvec = np.linspace(els.x1[i], els.x2[i], n_eval + 1)
        yvec = np.linspace(els.y1[i], els.y2[i], n_eval + 1)
        # evaluation locations on the fault
        x_obs[i * n_eval : (i + 1) * n_eval] = 0.5 * (xvec[1:] + xvec[0:-1])
        y_obs[i * n_eval : (i + 1) * n_eval] = 0.5 * (yvec[1:] + yvec[0:-1])

        # calculate slip in (s,n) coordinates
        xmesh = np.linspace(-0.5, 0.5, n_eval + 1)
        xeval = 0.5 * (xmesh[1:] + xmesh[0:-1])
        s_s = slip_functions(xeval, 0.5) @ coeffs_s[3 * i : 3 * (i + 1)]
        s_n = slip_functions(xeval, 0.5) @ coeffs_n[3 * i : 3 * (i + 1)]
        fault_slip_s[i * n_eval : (i + 1) * n_eval] = s_s
        fault_slip_n[i * n_eval : (i + 1) * n_eval] = s_n

        # rotate from (s,n) to (x,y)
        slip_vector = np.vstack((s_s, s_n)).T
        slip_vector_rotated = slip_vector @ els.rot_mats_inv[i, :, :]
        s_x = slip_vector_rotated[:, 0]
        s_y = slip_vector_rotated[:, 1]
        fault_slip_x[i * n_eval : (i + 1) * n_eval] = s_x
        fault_slip_y[i * n_eval : (i + 1) * n_eval] = s_y

    return x_obs, y_obs, fault_slip_x, fault_slip_y


def get_slipvector_on_fault_antiplane(els, coeffs, n_eval):
    """Get slip scalar evaluated ON the fault in (x,y) coordinate system.

    Args:
        els: fault geometry data structure
        coeffs: quadratic slip coefficients ordered as [3 x shear_slip] per fault element
        n_eval: number of points to evaluate slip scalar

    Returns:
        x_obs, y_obs: x,y coordinates of locations where slip vector is computed
        fault_slip: slip scalar, each of dimension [n_eval x n_els]
    """

    n_els = len(els.x1)

    # calculate slip as a continuous function
    fault_slip = np.zeros(n_els * n_eval)

    # evaluation locations
    x_obs = np.zeros_like(fault_slip)
    y_obs = np.zeros_like(fault_slip)

    for i in range(n_els):
        xvec = np.linspace(els.x1[i], els.x2[i], n_eval + 1)
        yvec = np.linspace(els.y1[i], els.y2[i], n_eval + 1)
        # evaluation locations on the fault
        x_obs[i * n_eval : (i + 1) * n_eval] = 0.5 * (xvec[1:] + xvec[0:-1])
        y_obs[i * n_eval : (i + 1) * n_eval] = 0.5 * (yvec[1:] + yvec[0:-1])

        # calculate slip in (s,n) coordinates
        xmesh = np.linspace(-0.5, 0.5, n_eval + 1)
        xeval = 0.5 * (xmesh[1:] + xmesh[0:-1])
        s = slip_functions(xeval, 0.5) @ coeffs[3 * i : 3 * (i + 1)]
        fault_slip[i * n_eval : (i + 1) * n_eval] = s

    return x_obs, y_obs, fault_slip


# Function to label open nodes, overlapping interior nodes and triple junctions automatically
def label_nodes(els):
    """provide a dictionary of line segments using bemcs.standardize_els_geometry(els),
    to return labelled indices for open nodes, overlapping nodes and triple junctions.

    the indices are provided as a number 3(i-1)<= id <=(3i-1),
    where 'i' corresponds to the mesh element
    """
    n_els = len(els.x1)
    # first find all unique points
    points = np.zeros((2 * n_els, 2))
    x1y1 = np.vstack((els.x1, els.y1)).T
    x2y2 = np.vstack((els.x2, els.y2)).T
    points[0::2, :] = x1y1
    points[1::2, :] = x2y2
    unique_points, id_unique = np.unique(points, axis=0, return_index=True)

    # Find number of open, 2-overlap & triple junction nodes
    index_matrix1 = []  # open
    index_matrix2 = []  # 2-overlap
    index_matrix3 = []  # triple junction

    error_message = "Cannot handle overlapping nodes if their unit normals don't have the same circulation direction "

    for i in range(len(unique_points)):
        pts = unique_points[i, :].reshape(1, -1)

        # Which element(s) contains this point
        id1 = np.where(np.all(pts == x1y1, axis=1))
        id2 = np.where(np.all(pts == x2y2, axis=1))

        # The negative signs are for the triple junction equations
        # s_1 + s_2 + s_3 = 0 with the negative sign going to any 2 elements that are both id1 or id2
        if (np.size(id1) == 2) & (np.size(id2) == 1):  # triple junction
            id_combo = np.hstack((-id1[0] * 3, id2[0] * 3 + 2))
        elif (np.size(id2) == 2) & (np.size(id1) == 1):  # triple junction
            id_combo = np.hstack((id1[0] * 3, -(id2[0] * 3 + 2)))
        elif (np.size(id2) == 1) & (np.size(id1) == 1):  # 2-overlap
            id_combo = np.hstack((id1[0] * 3, -(id2[0] * 3 + 2)))
        elif (np.size(id2) == 2) & (np.size(id1) == 0):  # 2-overlap (problematic)
            id_combo = np.hstack(((id2[0][0] * 3 + 2), -(id2[0][1] * 3 + 2)))
            raise Exception(error_message)
        elif (np.size(id1) == 2) & (np.size(id2) == 0):  # 2-overlap (problematic)
            id_combo = np.hstack(((id1[0][0] * 3), -(id1[0][1] * 3)))
            raise Exception(error_message)
        else:  # open node
            id_combo = np.hstack((id1[0] * 3, (id2[0] * 3 + 2)))

        if np.size(id_combo) == 1:
            index_matrix1.append(id_combo)
        elif np.size(id_combo) == 2:
            index_matrix2.append(id_combo)
        elif np.size(id_combo) == 3:
            index_matrix3.append(id_combo)
        else:
            print(id_combo)
            raise ValueError("Cannot deal with more than 3 lines at a node")

    print("Number of open nodes =", len(index_matrix1))
    print(":", index_matrix1)
    print("Number of 2-overlap nodes =", len(index_matrix2))
    print(":", index_matrix2)
    print("Number of triple junctions =", len(index_matrix3))
    print(":", index_matrix3)

    return index_matrix1, index_matrix2, index_matrix3


def construct_smoothoperator(els, index_open, index_overlap, index_triple):
    """function to construct linear operator that enforces
    continuity and smoothness conditions at non-central nodes

    returns 3 matrices: matrix_system_o, matrix_system_i, matrix_system_t for open, overlapping and triple junctions
    """

    n_els = len(els.x1)
    Nunknowns = 6 * n_els
    # Design matrices (in x,y coordinates) for slip and slip gradients at each 3qn
    matrix_slip, matrix_slip_gradient = get_matrices_slip_slip_gradient(els)

    N_o = 2 * len(index_open)  # open node equations
    N_i = 4 * len(index_overlap)  # overlapping node equations
    N_t = 6 * len(index_triple)  # triple junction equations

    matrix_system_o = np.zeros((N_o, Nunknowns))
    matrix_system_i = np.zeros((N_i, Nunknowns))
    matrix_system_t = np.zeros((N_t, Nunknowns))

    # Linear operator for open nodes
    for i in range(int(N_o / 2)):
        id1 = np.abs(index_open[i])  # node number
        matrix_system_o[2 * i, :] = matrix_slip[2 * id1, :]  # x component
        matrix_system_o[2 * i + 1, :] = matrix_slip[2 * id1 + 1, :]  # y component

    # Linear operator for overlapping nodes
    for i in range(int(N_i / 4)):
        idvals = index_overlap[i]  # node number
        # continuity condition
        if (idvals[0] != 0) & (idvals[1] != 0):
            sign1 = np.sign(idvals[0])
            sign2 = np.sign(idvals[1])
        elif (idvals[0] == 0) & (idvals[1] != 0):
            sign1 = 1
            sign2 = -1
        else:
            sign1 = -1
            sign2 = 1

        matrix_system_i[4 * i, :] = (
            sign1 * matrix_slip[2 * np.abs(idvals[0]), :]
            + sign2 * matrix_slip[2 * np.abs(idvals[1]), :]
        )  # x
        matrix_system_i[4 * i + 1, :] = (
            sign1 * matrix_slip[2 * np.abs(idvals[0]) + 1, :]
            + sign2 * matrix_slip[2 * np.abs(idvals[1]) + 1, :]
        )  # y
        # smoothing constraints
        matrix_system_i[4 * i + 2, :] = (
            sign1 * matrix_slip_gradient[2 * np.abs(idvals[0]), :]
            + sign2 * matrix_slip_gradient[2 * np.abs(idvals[1]), :]
        )  # x
        matrix_system_i[4 * i + 3, :] = (
            sign1 * matrix_slip_gradient[2 * np.abs(idvals[0]) + 1, :]
            + sign2 * matrix_slip_gradient[2 * np.abs(idvals[1]) + 1, :]
        )  # y

    # Linear operator for triple junction nodes
    for k in range(int(N_t / 6)):
        id1 = index_triple[k]
        idvalst = np.abs(id1)

        # node number that need to be subtracted in TJ kinematics
        id_neg = idvalst[id1 < 0]
        # node numbers that need to be added
        id_pos = idvalst[id1 >= 0]
        # triple junction kinematics equations
        if len(id_neg) == 2:
            matrix_system_t[6 * k, :] = (
                matrix_slip[2 * id_pos, :]
                - matrix_slip[2 * id_neg[0], :]
                - matrix_slip[2 * id_neg[1], :]
            )  # x component
            matrix_system_t[6 * k + 1, :] = (
                matrix_slip[2 * id_pos + 1, :]
                - matrix_slip[2 * id_neg[0] + 1, :]
                - matrix_slip[2 * id_neg[1] + 1, :]
            )  # y component
        else:
            matrix_system_t[6 * k, :] = (
                matrix_slip[2 * id_pos[0], :]
                + matrix_slip[2 * id_pos[1], :]
                - matrix_slip[2 * id_neg, :]
            )  # x component
            matrix_system_t[6 * k + 1, :] = (
                matrix_slip[2 * id_pos[0] + 1, :]
                + matrix_slip[2 * id_pos[1] + 1, :]
                - matrix_slip[2 * id_neg + 1, :]
            )  # y component

        # smoothing constraints (2 nodes at a time)
        matrix_system_t[6 * k + 2, :] = (
            matrix_slip_gradient[2 * idvalst[0], :]
            - matrix_slip_gradient[2 * idvalst[1], :]
        )  # x
        matrix_system_t[6 * k + 3, :] = (
            matrix_slip_gradient[2 * idvalst[0] + 1, :]
            - matrix_slip_gradient[2 * idvalst[1] + 1, :]
        )  # y
        matrix_system_t[6 * k + 4, :] = (
            matrix_slip_gradient[2 * idvalst[0], :]
            - matrix_slip_gradient[2 * idvalst[2], :]
        )  # x
        matrix_system_t[6 * k + 5, :] = (
            matrix_slip_gradient[2 * idvalst[0] + 1, :]
            - matrix_slip_gradient[2 * idvalst[2] + 1, :]
        )  # y

    return matrix_system_o, matrix_system_i, matrix_system_t


def construct_smoothoperator_antiplane(els, index_open, index_overlap, index_triple):
    """function to construct linear operator that enforces
    continuity and smoothness conditions at non-central nodes for antiplane geometry

    returns 3 matrices: matrix_system_o, matrix_system_i, matrix_system_t for open, overlapping and triple junctions
    """

    n_els = len(els.x1)
    Nunknowns = 3 * n_els
    # Design matrices (in x,y coordinates) for slip and slip gradients at each 3qn
    matrix_slip, matrix_slip_gradient = get_matrices_slip_slip_gradient_antiplane(els)

    N_o = len(index_open)  # open node equations
    N_i = 2 * len(index_overlap)  # overlapping node equations
    N_t = 3 * len(index_triple)  # triple junction equations

    matrix_system_o = np.zeros((N_o, Nunknowns))
    matrix_system_i = np.zeros((N_i, Nunknowns))
    matrix_system_t = np.zeros((N_t, Nunknowns))

    # Linear operator for open nodes
    for i in range(N_o):
        id1 = np.abs(index_open[i])  # node number
        matrix_system_o[i, :] = matrix_slip[id1, :]  # x component

    # Linear operator for overlapping nodes
    for i in range(int(N_i / 2)):
        idvals = index_overlap[i]  # node number

        if (idvals[0] != 0) & (idvals[1] != 0):
            sign1 = np.sign(idvals[0])
            sign2 = np.sign(idvals[1])
        elif (idvals[0] == 0) & (idvals[1] != 0):
            sign1 = 1
            sign2 = -1
        else:
            sign1 = -1
            sign2 = 1

        # continuity condition
        matrix_system_i[2 * i, :] = (
            sign1 * matrix_slip[np.abs(idvals[0]), :]
            + sign2 * matrix_slip[np.abs(idvals[1]), :]
        )

        # smoothing constraints
        matrix_system_i[2 * i + 1, :] = (
            sign1 * matrix_slip_gradient[np.abs(idvals[0]), :]
            + sign2 * matrix_slip_gradient[np.abs(idvals[1]), :]
        )

    # Linear operator for triple junction nodes
    for k in range(int(N_t / 3)):
        id1 = index_triple[k]
        idvalst = np.abs(id1)

        # node number that need to be subtracted in TJ kinematics
        id_neg = idvalst[id1 < 0]
        # node numbers that need to be added
        id_pos = idvalst[id1 >= 0]
        # triple junction kinematics equations
        if len(id_neg) == 2:
            matrix_system_t[3 * k, :] = (
                matrix_slip[id_pos, :]
                - matrix_slip[id_neg[0], :]
                - matrix_slip[id_neg[1], :]
            )
            # smoothing constraints
            matrix_system_t[3 * k + 1, :] = (
                matrix_slip_gradient[id_pos, :]
                * els.x_shears[int(np.floor(id_pos / 3))]
                - matrix_slip_gradient[id_neg[0], :]
                * els.x_shears[int(np.floor(id_neg[0] / 3))]
                - matrix_slip_gradient[id_neg[1], :]
                * els.x_shears[int(np.floor(id_neg[1] / 3))]
            )

            matrix_system_t[3 * k + 2, :] = (
                matrix_slip_gradient[id_pos, :]
                * els.y_shears[int(np.floor(id_pos / 3))]
                - matrix_slip_gradient[id_neg[0], :]
                * els.y_shears[int(np.floor(id_neg[0] / 3))]
                - matrix_slip_gradient[id_neg[1], :]
                * els.y_shears[int(np.floor(id_neg[1] / 3))]
            )

        else:
            matrix_system_t[3 * k, :] = (
                matrix_slip[id_pos[0], :]
                + matrix_slip[id_pos[1], :]
                - matrix_slip[id_neg, :]
            )
            # smoothing constraints
            matrix_system_t[3 * k + 1, :] = (
                matrix_slip_gradient[id_pos[0], :]
                * els.x_shears[int(np.floor(id_pos[0] / 3))]
                + matrix_slip_gradient[id_pos[1], :]
                * els.x_shears[int(np.floor(id_pos[1] / 3))]
                - matrix_slip_gradient[id_neg, :]
                * els.x_shears[int(np.floor(id_neg / 3))]
            )

            matrix_system_t[3 * k + 2, :] = (
                matrix_slip_gradient[id_pos[0], :]
                * els.y_shears[int(np.floor(id_pos[0] / 3))]
                + matrix_slip_gradient[id_pos[1], :]
                * els.y_shears[int(np.floor(id_pos[1] / 3))]
                - matrix_slip_gradient[id_neg, :]
                * els.y_shears[int(np.floor(id_neg / 3))]
            )

        """# smoothing constraints (2 nodes at a time)
        matrix_system_t[3 * k + 1, :] = (
            matrix_slip_gradient[idvalst[0], :] - matrix_slip_gradient[idvalst[1], :]
        )

        matrix_system_t[3 * k + 2, :] = (
            matrix_slip_gradient[idvalst[0], :] - matrix_slip_gradient[idvalst[2], :]
        )"""

    return matrix_system_o, matrix_system_i, matrix_system_t


def inpolygon(xq, yq, xv, yv):
    """From: https://stackoverflow.com/questions/31542843/inpolygon-examples-of-matplotlib-path-path-contains-points-method

    Args:
        xq : x coordinates of points to test
        yq : y coordinates of points to test
        xv : x coordinates of polygon vertices
        yv : y coordinates of polygon vertices

    Returns:
        _type_: Boolean like for in or out of polygon
    """
    shape = xq.shape
    xq = xq.reshape(-1)
    yq = yq.reshape(-1)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)
    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
    p = matplotlib.path.Path([(xv[i], yv[i]) for i in range(xv.shape[0])])
    return p.contains_points(q).reshape(shape)


def kelvin_point_source_stress(x, y, xoffset, yoffset, fx, fy, mu, nu):
    """
    Calculate the stress components at a point due to a Kelvin point source in an elastic medium.

    This function computes the stress components (sxx, syy, sxy) at a given point (x, y)
    due to a point force applied at an offset location (xoffset, yoffset). The stresses
    are calculated using the Kelvin solution for an infinite elastic medium.

    Parameters:
    -----------
    x : float or numpy.ndarray
        The x-coordinate(s) of the observation point(s).
    y : float or numpy.ndarray
        The y-coordinate(s) of the observation point(s).
    xoffset : float
        The x-coordinate of the point force location.
    yoffset : float
        The y-coordinate of the point force location.
    fx : float
        The x-component of the applied force.
    fy : float
        The y-component of the applied force.
    mu : float
        The shear modulus of the material.
    nu : float
        The Poisson's ratio of the material.

    Returns:
    --------
    sxx : float or numpy.ndarray
        The xx-component of the stress at the observation point(s).
    syy : float or numpy.ndarray
        The yy-component of the stress at the observation point(s).
    sxy : float or numpy.ndarray
        The xy-component of the stress at the observation point(s).

    Notes:
    ------
    The function uses the Kelvin solution for an infinite elastic medium to compute the stresses.
    """

    x = x - xoffset
    y = y - yoffset
    C = 1 / (4 * np.pi * (1 - nu))
    gx = -C * x / (x**2 + y**2)
    gy = -C * y / (x**2 + y**2)
    gxy = C * 2 * x * y / (x**2 + y**2) ** 2
    gxx = C * (x**2 - y**2) / (x**2 + y**2) ** 2
    gyy = -gxx
    sxx = fx * (2 * (1 - nu) * gx - x * gxx) + fy * (2 * nu * gy - y * gxx)
    syy = fx * (2 * nu * gx - x * gyy) + fy * (2 * (1 - nu) * gy - y * gyy)
    sxy = fx * ((1 - 2 * nu) * gy - x * gxy) + fy * ((1 - 2 * nu) * gx - y * gxy)
    return sxx, syy, sxy


def kelvin_point_source_disp(x, y, xoffset, yoffset, fx, fy, mu, nu):
    """
    Calculate the displacements due to a Kelvin point source in an elastic medium.

    This function computes the displacements (ux, uy) at a given point (x, y)
    due to a point force applied at an offset location (xoffset, yoffset). The stresses
    are calculated using the Kelvin solution for an infinite elastic medium.

    Parameters:
    -----------
    x : float or numpy.ndarray
        The x-coordinate(s) of the observation point(s).
    y : float or numpy.ndarray
        The y-coordinate(s) of the observation point(s).
    xoffset : float
        The x-coordinate of the point force location.
    yoffset : float
        The y-coordinate of the point force location.
    fx : float
        The x-component of the applied force.
    fy : float
        The y-component of the applied force.
    mu : float
        The shear modulus of the material.
    nu : float
        The Poisson's ratio of the material.

    Returns:
    --------
    ux : float or numpy.ndarray
        The x-component of the displacements at the observation point(s).
    sy : float or numpy.ndarray
        The y-component of the displacements at the observation point(s).

    Notes:
    ------
    The function uses the Kelvin solution for an infinite elastic medium to compute the displacements.
    """

    x = x - xoffset
    y = y - yoffset
    C = 1 / (4 * np.pi * (1 - nu))
    r = np.sqrt(x**2 + y**2)
    g = -C * np.log(r)
    gx = -C * x / (x**2 + y**2)
    gy = -C * y / (x**2 + y**2)
    ux = fx / (2 * mu) * ((3 - 4 * nu) * g - x * gx) + fy / (2 * mu) * (-y * gx)
    uy = fx / (2 * mu) * (-x * gy) + fy / (2 * mu) * ((3 - 4 * nu) * g - y * gy)
    return ux, uy


def get_triangle_area(lx, ly, dly):
    """
    Calculate the area of a triangle using the lengths of its sides.

    This function calculates the area of a triangle given the coordinates of its vertices using
    Heron's formula. The vertices are assumed to be at (0, 0), (lx, dly), and (0, ly).

    Parameters:
    -----------
    lx : float
        The x-coordinate of the second vertex of the triangle.
    ly : float
        The y-coordinate of the third vertex of the triangle.
    dly : float
        The y-coordinate of the second vertex of the triangle.

    Returns:
    --------
    triangle_area : float
        The area of the triangle.

    Notes:
    ------
    The function performs the following steps:
    1. Calculates the lengths of the sides of the triangle using the distance formula.
    2. Computes the semi-perimeter of the triangle.
    3. Applies Heron's formula to find the area of the triangle.
    """
    a = np.sqrt((lx - 0) ** 2 + (dly - 0) ** 2)
    b = np.sqrt((0 - lx) ** 2 + (ly - dly) ** 2)
    c = np.sqrt((0 - 0) ** 2 + (0 - ly) ** 2)

    # Calculate the semi-perimeter of the triangle
    s = (a + b + c) / 2

    # Calculate the area of the triangle using Heron's formula
    triangle_area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    return triangle_area


def get_transformed_coordinates(vertices, obs_x, obs_y):
    """
    Transform coordinates by translating and rotating the triangle and observation points.

    This function translates the triangle such that the first vertex is at the origin, then rotates
    the triangle and the observation points so that the second vertex aligns with the positive y-axis.

    Parameters:
    -----------
    vertices : numpy.ndarray
        An array of shape (3, 2) representing the coordinates of the triangle's vertices.
    obs_x : numpy.ndarray
        An array of x-coordinates of the observation points.
    obs_y : numpy.ndarray
        An array of y-coordinates of the observation points.

    Returns:
    --------
    rotated_vertices : numpy.ndarray
        The transformed coordinates of the triangle's vertices.
    rotated_obs_x : numpy.ndarray
        The transformed x-coordinates of the observation points.
    rotated_obs_y : numpy.ndarray
        The transformed y-coordinates of the observation points.

    Notes:
    ------
    The function performs the following steps:
    1. Translates the triangle so that the first vertex is at the origin.
    2. Calculates the angle required to rotate the second vertex to align with the positive y-axis.
    3. Constructs a rotation matrix using the calculated angle.
    4. Applies the translation and rotation transformations to both the triangle's vertices and the observation points.
    """
    # Translate the triangle so that the first vertex is at the origin
    translated_vertices = vertices - np.array(vertices[0])
    translated_obs_x = obs_x - vertices[0, 0]
    translated_obs_y = obs_y - vertices[0, 1]

    # Determine the angle to rotate the second vertex to align with the y-axis
    second_vertex = translated_vertices[1]
    angle = np.arctan2(second_vertex[0], second_vertex[1])

    # Rotation matrix to align the second vertex along the positive y-axis
    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    rotated_vertices = np.dot(translated_vertices, rotation_matrix.T)
    rotated_obs = np.dot(
        np.array([translated_obs_x, translated_obs_y]).T, rotation_matrix.T
    )

    return rotated_vertices, rotated_obs[:, 0], rotated_obs[:, 1]


def rotate_vector(vertices, fx, fy, rotdir=1):
    """
    Rotate a vector to a new coordinate system aligned with the triangle's edge.

    This function rotates a force vector (fx, fy) to a new coordinate system such that the
    edge between the first and second vertices of the triangle aligns with the positive y-axis.
    The rotation is performed using the specified rotation direction.

    Parameters:
    -----------
    vertices : numpy.ndarray
        An array of shape (3, 2) representing the coordinates of the triangle's vertices.
    fx : numpy.ndarray
        The x-component of the force vector to be rotated.
    fy : numpy.ndarray
        The y-component of the force vector to be rotated.
    rotdir : int, optional
        The direction of rotation. Default is 1 (counterclockwise). Use -1 for clockwise rotation.

    Returns:
    --------
    fx_rotated : numpy.ndarray
        The rotated x-component of the force vector.
    fy_rotated : numpy.ndarray
        The rotated y-component of the force vector.

    Notes:
    ------
    The function performs the following steps:
    1. Determines the rotation angle required to align the second vertex of the triangle with the positive y-axis.
    2. Constructs the rotation matrix using the calculated angle.
    3. Applies the rotation matrix to the force vector.
    4. Returns the rotated force vector components.
    """
    angle = (
        np.arctan2(vertices[1, 0] - vertices[0, 0], vertices[1, 1] - vertices[0, 1])
    ) * rotdir

    # Rotation matrix to align the second vertex along the positive y-axis
    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    rotated_vector = np.dot(np.hstack([fx, fy]), rotation_matrix.T)
    fx_rotated = rotated_vector[:, 0]
    fy_rotated = rotated_vector[:, 1]

    return fx_rotated, fy_rotated


def rotate_stresses(vertices, sxx, syy, sxy, rotdir=1):
    """
    Rotate stress components to a new coordinate system aligned with the triangle's edge.

    This function rotates the stress components (sxx, syy, sxy) to a new coordinate system such that the
    edge between the first and second vertices of the triangle aligns with the positive y-axis. The rotation
    is performed using the specified rotation direction.

    Parameters:
    -----------
    vertices : numpy.ndarray
        An array of shape (3, 2) representing the coordinates of the triangle's vertices.
    sxx : numpy.ndarray
        The xx-component of the stress at the observation points.
    syy : numpy.ndarray
        The yy-component of the stress at the observation points.
    sxy : numpy.ndarray
        The xy-component of the stress at the observation points.
    rotdir : int, optional
        The direction of rotation. Default is 1 (counterclockwise). Use -1 for clockwise rotation.

    Returns:
    --------
    sxx_rot : numpy.ndarray
        The rotated xx-component of the stress at the observation points.
    syy_rot : numpy.ndarray
        The rotated yy-component of the stress at the observation points.
    sxy_rot : numpy.ndarray
        The rotated xy-component of the stress at the observation points.

    Notes:
    ------
    The function performs the following steps:
    1. Determines the rotation angle required to align the second vertex of the triangle with the positive y-axis.
    2. Constructs the rotation matrix using the calculated angle.
    3. Applies the rotation matrix to the stress tensor at each observation point.
    4. Returns the rotated stress components.
    """
    # Determine the angle to rotate the second vertex to align with the y-axis
    angle = (
        np.arctan2(vertices[1, 0] - vertices[0, 0], vertices[1, 1] - vertices[0, 1])
        * rotdir
    )

    # Rotation matrix to align the second vertex along the positive y-axis
    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )

    sxx_rot = np.zeros_like(sxx)
    syy_rot = np.zeros_like(sxx)
    sxy_rot = np.zeros_like(sxx)
    for i in range(len(sxx)):
        stress_tensor = np.zeros((2, 2))
        stress_tensor[0, 0] = sxx[i]
        stress_tensor[1, 1] = syy[i]
        stress_tensor[0, 1] = sxy[i]
        stress_tensor[1, 0] = sxy[i]

        rotated_stress = rotation_matrix @ stress_tensor @ rotation_matrix.T
        sxx_rot[i] = rotated_stress[0, 0]
        syy_rot[i] = rotated_stress[1, 1]
        sxy_rot[i] = rotated_stress[0, 1]

    return sxx_rot, syy_rot, sxy_rot


def displacements_stresses_triangle_force_planestrain_nearfield(
    triangle, x_obs, y_obs, fx, fy, mu, nu
):
    """
    Calculate the near-field displacements and stresses at observation points due to forces on a triangular element in plane strain.

    This function computes the displacements and stresses (sxx, syy, sxy) at specified observation points
    resulting from forces applied to a triangular element under the assumption of plane strain conditions.
    The calculations use the Kelvin point source solution and integrate over the triangular element using
    double integration.

    Parameters:
    -----------
    triangle : numpy.ndarray
        An array of shape (3, 2) representing the coordinates of the triangle's vertices.
    x_obs : numpy.ndarray
        A 2D array representing the x-coordinates of the observation points.
    y_obs : numpy.ndarray
        A 2D array representing the y-coordinates of the observation points.
    fx : float
        The x-component of the applied force.
    fy : float
        The y-component of the applied force.
    mu : float
        The shear modulus of the material.
    nu : float
        The Poisson's ratio of the material.

    Returns:
    --------
    ux : numpy.ndarray
        The x-component of the displacements at the observation points.
    uy : numpy.ndarray
        The y-component of the displacements at the observation points.
    sxx : numpy.ndarray
        The xx-component of the stress at the observation points.
    syy : numpy.ndarray
        The yy-component of the stress at the observation points.
    sxy : numpy.ndarray
        The xy-component of the stress at the observation points.

    Notes:
    ------
    The function performs the following steps:
    1. Flattens the observation coordinates.
    2. Transforms the triangle and observation coordinates to a local coordinate system.
    3. Rotates the force vector to the local coordinate system.
    4. Defines the integration limits over the transformed triangle.
    5. Performs double integration using the Kelvin point source solution to compute displacements and stresses.
    6. Rotates the results back to the original coordinate system.

    Integration is performed using `scipy.integrate.dblquad` with a specified absolute error tolerance.
    """

    DBLQUAD_TOLERANCE = 1e-3

    # Flatten passed observations coordinates
    x_obs = x_obs.flatten()
    y_obs = y_obs.flatten()

    # Shape forces
    fx = np.array([fx])[:, None]
    fy = np.array([fy])[:, None]

    # Translate and rotate the triangle
    triangle_transformed, obs_x_transformed, obs_y_transformed = (
        get_transformed_coordinates(triangle, x_obs, y_obs)
    )

    # Rotated force vector
    fx_rot, fy_rot = rotate_vector(triangle, fx, fy)

    # Define a triangle region in dblquad style
    lx = triangle_transformed[2, 0]
    dly = triangle_transformed[2, 1]
    ly = triangle_transformed[1, 1]

    triangle_area = get_triangle_area(lx, ly, dly)

    # Definition of integration limits over a triangle and integrate using rotated forces
    ymin = lambda x: dly * x / lx
    ymax = lambda x: ly - (ly - dly) * x / lx

    ux_dblquad = np.zeros_like(obs_x_transformed)
    uy_dblquad = np.zeros_like(obs_x_transformed)
    sxx_dblquad = np.zeros_like(obs_x_transformed)
    syy_dblquad = np.zeros_like(obs_x_transformed)
    sxy_dblquad = np.zeros_like(obs_x_transformed)

    for i in range(0, obs_x_transformed.size):
        # ux velocity integration
        f = lambda y, x: kelvin_point_source_disp(
            obs_x_transformed[i],
            obs_y_transformed[i],
            x,
            y,
            fx_rot,
            fy_rot,
            mu,
            nu,
        )[0]
        sol, err = scipy.integrate.dblquad(
            f, 0, lx, ymin, ymax, epsabs=DBLQUAD_TOLERANCE
        )
        if lx < 0:
            ux_dblquad[i] = -sol / triangle_area
        else:
            ux_dblquad[i] = sol / triangle_area

        # uy velocity integration
        f = lambda y, x: kelvin_point_source_disp(
            obs_x_transformed[i],
            obs_y_transformed[i],
            x,
            y,
            fx_rot,
            fy_rot,
            mu,
            nu,
        )[1]
        sol, err = scipy.integrate.dblquad(
            f, 0, lx, ymin, ymax, epsabs=DBLQUAD_TOLERANCE
        )
        if lx < 0:
            uy_dblquad[i] = -sol / triangle_area
        else:
            uy_dblquad[i] = sol / triangle_area

        # xx stress integration
        f = lambda y, x: kelvin_point_source_stress(
            obs_x_transformed[i],
            obs_y_transformed[i],
            x,
            y,
            fx_rot,
            fy_rot,
            mu,
            nu,
        )[0]
        sol, err = scipy.integrate.dblquad(
            f, 0, lx, ymin, ymax, epsabs=DBLQUAD_TOLERANCE
        )
        sxx_dblquad[i] = sol / triangle_area
        if lx < 0:
            sxx_dblquad[i] = -sol / triangle_area
        else:
            sxx_dblquad[i] = sol / triangle_area

        # yy stress integration
        f = lambda y, x: kelvin_point_source_stress(
            obs_x_transformed[i],
            obs_y_transformed[i],
            x,
            y,
            fx_rot,
            fy_rot,
            mu,
            nu,
        )[1]
        sol, err = scipy.integrate.dblquad(
            f, 0, lx, ymin, ymax, epsabs=DBLQUAD_TOLERANCE
        )
        syy_dblquad[i] = sol / triangle_area
        if lx < 0:
            syy_dblquad[i] = -sol / triangle_area
        else:
            syy_dblquad[i] = sol / triangle_area

        # xy stress integration
        f = lambda y, x: kelvin_point_source_stress(
            obs_x_transformed[i],
            obs_y_transformed[i],
            x,
            y,
            fx_rot,
            fy_rot,
            mu,
            nu,
        )[2]
        sol, err = scipy.integrate.dblquad(
            f, 0, lx, ymin, ymax, epsabs=DBLQUAD_TOLERANCE
        )
        sxy_dblquad[i] = sol / triangle_area
        if lx < 0:
            sxy_dblquad[i] = -sol / triangle_area
        else:
            sxy_dblquad[i] = sol / triangle_area

    # Rotate back to original coordinates
    ux, uy = rotate_vector(
        triangle,
        ux_dblquad.reshape(-1, 1),
        uy_dblquad.reshape(-1, 1),
        -1,
    )

    sxx, syy, sxy = rotate_stresses(
        triangle,
        sxx_dblquad,
        syy_dblquad,
        sxy_dblquad,
        -1,
    )
    return ux, uy, sxx, syy, sxy


def displacements_stresses_triangle_force_planestrain_farfield(
    triangle, x_obs, y_obs, fx, fy, mu, nu
):
    """
    Calculate the far-field displacements and stresses at observation points due to forces on a triangular element in plane strain.

    This function computes the displacements and stresses (sxx, syy, sxy) at specified observation points
    resulting from forces applied to a triangular element under the assumption of plane strain conditions.
    The calculations use the Kelvin point source solution and integrate over the triangular element using
    a quadrature scheme.

    Parameters:
    -----------
    triangle : numpy.ndarray
        An array of shape (3, 2) representing the coordinates of the triangle's vertices.
    x_obs : numpy.ndarray
        A 2D array representing the x-coordinates of the observation points.
    y_obs : numpy.ndarray
        A 2D array representing the y-coordinates of the observation points.
    fx : float
        The x-component of the applied force.
    fy : float
        The y-component of the applied force.
    mu : float
        The shear modulus of the material.
    nu : float
        The Poisson's ratio of the material.

    Returns:
    --------
    ux : numpy.ndarray
        The x-component of the displacements at the observation points.
    uy : numpy.ndarray
        The y-component of the displacements at the observation points.
    sxx : numpy.ndarray
        The xx-component of the stress at the observation points.
    syy : numpy.ndarray
        The yy-component of the stress at the observation points.
    sxy : numpy.ndarray
        The xy-component of the stress at the observation points.

    Notes:
    ------
    The function uses the `quadpy` library to perform numerical integration over the triangular element
    using a quadrature scheme with N_INTEGRATION_POINTS integration points. The Kelvin point source solution is used
    to compute the displacements and stresses due to the applied forces.
    """
    x_obs = (x_obs.flatten(),)
    y_obs = y_obs.flatten()
    ux = np.zeros_like(x_obs)
    uy = np.zeros_like(x_obs)
    sxx = np.zeros_like(x_obs)
    syy = np.zeros_like(x_obs)
    sxy = np.zeros_like(x_obs)

    # quadpy integration scheme
    N_INTEGRATION_POINTS = 20
    scheme = quadpy.t2.get_good_scheme(N_INTEGRATION_POINTS)
    points_new = np.dot(triangle.T, scheme.points)
    n_integration_pts = len(scheme.weights)

    for i in range(n_integration_pts):
        ux_i, uy_i = kelvin_point_source_disp(
            x_obs,
            y_obs,
            points_new[0, i],
            points_new[1, i],
            fx,
            fy,
            mu,
            nu,
        )
        ux += scheme.weights[i] * ux_i
        uy += scheme.weights[i] * uy_i

        sxx_i, syy_i, sxy_i = kelvin_point_source_stress(
            x_obs,
            y_obs,
            points_new[0, i],
            points_new[1, i],
            fx,
            fy,
            mu,
            nu,
        )
        sxx += scheme.weights[i] * sxx_i
        syy += scheme.weights[i] * syy_i
        sxy += scheme.weights[i] * sxy_i
    return ux, uy, sxx, syy, sxy


def displacements_stresses_triangle_force_planestrain(
    triangle, x_obs, y_obs, fx, fy, mu, nu
):
    """
    Calculate displacements and stresses for a given triangular element at specified observation points.

    This function determines the displacements and stresses at observation points by combining near-field
    and far-field solutions based on the distance of the observation points from the centroid of the triangle.

    Parameters:
    -----------
    triangle : numpy.ndarray
        An array of shape (3, 2) representing the coordinates of the triangle's vertices.
    x_obs : numpy.ndarray
        A 2D array representing the x-coordinates of the observation points.
    y_obs : numpy.ndarray
        A 2D array representing the y-coordinates of the observation points.
    fx : float
        The force applied in the x-direction.
    fy : float
        The force applied in the y-direction.
    mu : float
        The shear modulus of the material.
    nu : float
        The Poisson's ratio of the material.

    Returns:
    --------
    ux : numpy.ndarray
        A 2D array of the x-displacements at the observation points.
    uy : numpy.ndarray
        A 2D array of the y-displacements at the observation points.
    sxx : numpy.ndarray
        A 2D array of the normal stress component in the x-direction at the observation points.
    syy : numpy.ndarray
        A 2D array of the normal stress component in the y-direction at the observation points.
    sxy : numpy.ndarray
        A 2D array of the shear stress component at the observation points.

    Notes:
    ------
    The function distinguishes between near-field and far-field observation points using a predefined
    distance cutoff. Near-field solutions are computed using the `get_displacements_stresses_nearfield`
    function, while far-field solutions are computed using the `get_displacements_stresses_farfield` function.

    """
    NEAR_FAR_DISTANCE_CUTOFF = 3.0
    obs_distances_from_centroid = scipy.spatial.distance.cdist(
        np.array([np.mean(triangle[:, 0]), np.mean(triangle[:, 1])])[:, None].T,
        np.array([x_obs.flatten(), y_obs.flatten()]).T,
    ).flatten()
    near_idx = np.where(obs_distances_from_centroid <= NEAR_FAR_DISTANCE_CUTOFF)[0]
    far_idx = np.where(obs_distances_from_centroid > NEAR_FAR_DISTANCE_CUTOFF)[0]

    ux = np.zeros_like(x_obs)
    uy = np.zeros_like(x_obs)
    sxx = np.zeros_like(x_obs)
    syy = np.zeros_like(x_obs)
    sxy = np.zeros_like(x_obs)
    ux_far, uy_far, sxx_far, syy_far, sxy_far = (
        displacements_stresses_triangle_force_planestrain_farfield(
            triangle, x_obs[far_idx], y_obs[far_idx], fx, fy, mu, nu
        )
    )
    ux_near, uy_near, sxx_near, syy_near, sxy_near = (
        displacements_stresses_triangle_force_planestrain_nearfield(
            triangle, x_obs[near_idx], y_obs[near_idx], fx, fy, mu, nu
        )
    )
    ux[far_idx] = ux_far
    uy[far_idx] = uy_far
    sxx[far_idx] = sxx_far
    syy[far_idx] = syy_far
    sxy[far_idx] = sxy_far
    ux[near_idx] = ux_near
    uy[near_idx] = uy_near
    sxx[near_idx] = sxx_near
    syy[near_idx] = syy_near
    sxy[near_idx] = sxy_near

    return ux, uy, sxx, syy, sxy
