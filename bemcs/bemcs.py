import numpy as np
import matplotlib.pyplot as plt

def plot_fields_no_elements(x, y, displacement, stress, sup_title):
    """Contour 2 displacement fields, 3 stress fields, and quiver displacements"""
    x_lim = np.array([x.min(), x.max()])
    y_lim = np.array([y.min(), y.max()])

    def style_plots():
        """Common plot elements"""
        plt.gca().set_aspect("equal")
        plt.xticks([x_lim[0], x_lim[1]])
        plt.yticks([y_lim[0], y_lim[1]])

    def plot_subplot(x, y, idx, field, title):
        """Common elements for each subplot - other than quiver"""
        plt.subplot(2, 3, idx)
        field_max = np.max(np.abs(field))
        scale = 5e-1
        plt.contourf(
            x,
            y,
            field.reshape(x.shape),
            n_contours,
            vmin=-scale * field_max,
            vmax=scale * field_max,
            cmap=plt.get_cmap("RdYlBu"),
        )
        plt.clim(-scale * field_max, scale * field_max)
        plt.colorbar(fraction=0.046, pad=0.04, extend="both")

        plt.contour(
            x,
            y,
            field.reshape(x.shape),
            n_contours,
            vmin=-scale * field_max,
            vmax=scale * field_max,
            linewidths=0.25,
            colors="k",
        )
        plt.title(title)
        style_plots()

    plt.figure(figsize=(12, 8))
    n_contours = 10
    plot_subplot(x, y, 2, displacement[0, :], "x displacement")
    plot_subplot(x, y, 3, displacement[1, :], "y displacement")
    plot_subplot(x, y, 4, stress[0, :], "xx stress")
    plot_subplot(x, y, 5, stress[1, :], "yy stress")
    plot_subplot(x, y, 6, stress[2, :], "xy stress")

    plt.subplot(2, 3, 1)
    plt.quiver(x, y, displacement[0], displacement[1], units="width", color="b")
    plt.title("vector displacement")
    plt.gca().set_aspect("equal")
    plt.xticks([x_lim[0], x_lim[1]])
    plt.yticks([y_lim[0], y_lim[1]])
    plt.suptitle(sup_title)
    plt.tight_layout()
    plt.show(block=False)

def plot_fields(elements, x, y, displacement, stress, sup_title):
    """Contour 2 displacement fields, 3 stress fields, and quiver displacements"""
    x_lim = np.array([x.min(), x.max()])
    y_lim = np.array([y.min(), y.max()])

    def style_plots():
        """Common plot elements"""
        plt.gca().set_aspect("equal")
        plt.xticks([x_lim[0], x_lim[1]])
        plt.yticks([y_lim[0], y_lim[1]])

    def plot_subplot(elements, x, y, idx, field, title):
        """Common elements for each subplot - other than quiver"""
        plt.subplot(2, 3, idx)
        field_max = np.max(np.abs(field))
        scale = 5e-1
        plt.contourf(
            x,
            y,
            field.reshape(x.shape),
            n_contours,
            vmin=-scale * field_max,
            vmax=scale * field_max,
            cmap=plt.get_cmap("RdYlBu"),
        )
        plt.clim(-scale * field_max, scale * field_max)
        plt.colorbar(fraction=0.046, pad=0.04, extend="both")

        plt.contour(
            x,
            y,
            field.reshape(x.shape),
            n_contours,
            vmin=-scale * field_max,
            vmax=scale * field_max,
            linewidths=0.25,
            colors="k",
        )

        for element in elements:
            plt.plot(
                [element["x1"], element["x2"]],
                [element["y1"], element["y2"]],
                "-k",
                linewidth=1.0,
            )
        plt.title(title)
        style_plots()

    plt.figure(figsize=(12, 8))
    n_contours = 10
    plot_subplot(elements, x, y, 2, displacement[0, :], "x displacement")
    plot_subplot(elements, x, y, 3, displacement[1, :], "y displacement")
    plot_subplot(elements, x, y, 4, stress[0, :], "xx stress")
    plot_subplot(elements, x, y, 5, stress[1, :], "yy stress")
    plot_subplot(elements, x, y, 6, stress[2, :], "xy stress")

    plt.subplot(2, 3, 1)
    for element in elements:
        plt.plot(
            [element["x1"], element["x2"]],
            [element["y1"], element["y2"]],
            "-k",
            linewidth=1.0,
        )

    plt.quiver(x, y, displacement[0], displacement[1], units="width", color="b")

    plt.title("vector displacement")
    plt.gca().set_aspect("equal")
    plt.xticks([x_lim[0], x_lim[1]])
    plt.yticks([y_lim[0], y_lim[1]])
    plt.suptitle(sup_title)
    plt.tight_layout()
    plt.show(block=False)


def plot_nine_fields(elements, x, y, displacement, stress, n_pts, sup_title):
    """Contour 3 displacement fields, 6 stress fields"""
    x_lim = np.array([x.min(), x.max()])
    y_lim = np.array([y.min(), y.max()])

    def style_plots():
        """Common plot elements"""
        plt.gca().set_aspect("equal")
        plt.xticks([x_lim[0], x_lim[1]])
        plt.yticks([y_lim[0], y_lim[1]])

    def plot_subplot(elements, x, y, idx, field, n_pts, title):
        """Common elements for each subplot - other than quiver"""
        plt.subplot(3, 3, idx)
        field_max = np.max(np.abs(field))
        scale = 5e-1
        plt.contourf(
            x.reshape(n_pts, n_pts),
            y.reshape(n_pts, n_pts),
            field.reshape(n_pts, n_pts),
            n_contours,
            vmin=-scale * field_max,
            vmax=scale * field_max,
            cmap=plt.get_cmap("RdYlBu"),
        )
        plt.clim(-scale * field_max, scale * field_max)
        plt.colorbar(fraction=0.046, pad=0.04, extend="both")

        plt.contour(
            x.reshape(n_pts, n_pts),
            y.reshape(n_pts, n_pts),
            field.reshape(n_pts, n_pts),
            n_contours,
            vmin=-scale * field_max,
            vmax=scale * field_max,
            linewidths=0.25,
            colors="k",
        )

        for element in elements:
            plt.plot(
                [element["x1"], element["x2"]],
                [element["y1"], element["y2"]],
                "-k",
                linewidth=1.0,
            )
        plt.title(title)
        style_plots()

    # Displacment magntiude
    displacment_magnitude = np.sqrt(displacement[0, :]**2.0 + displacement[1, :]**2.0)

    # Stress invariants
    stress_first_invariant = stress[0, :] + stress[1, :]
    stress_second_invariant = stress[0, :] * stress[1, :] - stress[2, :]**2.0
    stress_third_invariant = np.zeros_like(stress[0, :]) # in 2D

    plt.figure(figsize=(12, 12))
    n_contours = 10
    plot_subplot(elements, x, y, 1, displacment_magnitude, n_pts, "displacement magnitude")
    plot_subplot(elements, x, y, 2, displacement[0, :], n_pts, "x displacement")
    plot_subplot(elements, x, y, 3, displacement[1, :], n_pts, "y displacement")
    plot_subplot(elements, x, y, 4, stress[0, :], n_pts, "xx stress")
    plot_subplot(elements, x, y, 5, stress[1, :], n_pts, "yy stress")
    plot_subplot(elements, x, y, 6, stress[2, :], n_pts, "xy stress")
    plot_subplot(elements, x, y, 7, stress_first_invariant, n_pts, "stress first invariant")
    plot_subplot(elements, x, y, 8, stress_second_invariant, n_pts, "stress second invariant")
    # plot_subplot(elements, x, y, 9, stress_third_invariant, "stress third invariant")
    plt.tight_layout()
    plt.suptitle(sup_title)
    plt.show(block=False)


def plot_element_geometry(elements):
    """Plot element geometry"""
    for element in elements:
        plt.plot(
            [element["x1"], element["x2"]],
            [element["y1"], element["y2"]],
            "-",
            color="r",
            linewidth=0.5,
        )
        plt.plot(
            [element["x1"], element["x2"]],
            [element["y1"], element["y2"]],
            "r.",
            markersize=1,
            linewidth=0.5,
        )

    # Extract and plot unit normal & shear vectors
    x_center = np.array([_["x_center"] for _ in elements])
    y_center = np.array([_["y_center"] for _ in elements])
    x_normal = np.array([_["x_normal"] for _ in elements])
    y_normal = np.array([_["y_normal"] for _ in elements])
    x_shear = np.array([_["x_shear"] for _ in elements])
    y_shear = np.array([_["y_shear"] for _ in elements])
    plt.quiver(x_center, y_center, x_normal, y_normal, units="width", color="gray", width=0.002) 
    plt.quiver(x_center, y_center, x_shear, y_shear, units="width", color="green", width=0.002)       

    for i, element in enumerate(elements):
        plt.text(
            element["x_center"],
            element["y_center"],
            str(i),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=8,
        )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("element geometry and normals")
    plt.gca().set_aspect("equal")
    #plt.show(block=False)

def standardize_elements(elements):
    for element in elements:
        element["angle"] = np.arctan2(
            element["y2"] - element["y1"], element["x2"] - element["x1"]
        )
        element["length"] = np.sqrt(
            (element["x2"] - element["x1"]) ** 2 + (element["y2"] - element["y1"]) ** 2
        )
        element["half_length"] = 0.5 * element["length"]
        element["x_center"] = 0.5 * (element["x2"] + element["x1"])
        element["y_center"] = 0.5 * (element["y2"] + element["y1"])
        element["rotation_matrix"] = np.array(
            [
                [np.cos(element["angle"]), -np.sin(element["angle"])],
                [np.sin(element["angle"]), np.cos(element["angle"])],
            ]
        )
        element["inverse_rotation_matrix"] = np.array(
            [
                [np.cos(-element["angle"]), -np.sin(-element["angle"])],
                [np.sin(-element["angle"]), np.cos(-element["angle"])],
            ]
        )
        dx = element["x2"] - element["x1"]
        dy = element["y2"] - element["y1"]
        mag = np.sqrt(dx**2 + dy**2)
        element["x_normal"] = -dy / mag
        element["y_normal"] = dx / mag
        element["x_shear"] = dx / mag
        element["y_shear"] = dy / mag

        # Evaluations points for quadratic kernels
        element["x_integration_points"] = np.array(
            [
                element["x_center"] - (2 / 3 * dx / 2),
                element["x_center"],
                element["x_center"] + (2 / 3 * dx / 2),
            ]
        )
        element["y_integration_points"] = np.array(
            [
                element["y_center"] - (2 / 3 * dy / 2),
                element["y_center"],
                element["y_center"] + (2 / 3 * dy / 2),
            ]
        )

        # If a local boundary condition is giving convert to global
        # TODO: This is just for convenience there should be flags for real BCs
        if "ux_local" in element:
            u_local = np.array([element["ux_local"], element["uy_local"]])
            u_global = element["rotation_matrix"] @ u_local
            element["ux_global_constant"] = u_global[0]
            element["uy_global_constant"] = u_global[1]
            element["ux_global_quadratic"] = np.repeat(u_global[0], 3)
            element["uy_global_quadratic"] = np.repeat(u_global[1], 3)

    return elements


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
        * (
            np.log(np.sqrt((x - a) ** 2 + y**2))
            - np.log(np.sqrt((x + a) ** 2 + y**2))
        )
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
        * (
            (x - a) / ((x - a) ** 2 + y**2) ** 2
            - (x + a) / ((x + a) ** 2 + y**2) ** 2
        )
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
            * (
                y**5
                + 2 * (a**2 + x**2) * y**3
                + (a**4 - 2 * a**2 * x**2 + x**4) * y
            )
            * np.log(abs(a**2 + 2 * a * x + x**2 + y**2))
            + 3
            * (
                y**5
                + 2 * (a**2 + x**2) * y**3
                + (a**4 - 2 * a**2 * x**2 + x**4) * y
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
            * (
                y**5
                + 2 * (a**2 + x**2) * y**3
                + (a**4 - 2 * a**2 * x**2 + x**4) * y
            )
            * np.log(abs(a**2 + 2 * a * x + x**2 + y**2))
            - 9
            * (
                y**5
                + 2 * (a**2 + x**2) * y**3
                + (a**4 - 2 * a**2 * x**2 + x**4) * y
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
            * (
                y**5
                + 2 * (a**2 + x**2) * y**3
                + (a**4 - 2 * a**2 * x**2 + x**4) * y
            )
            * np.log(abs(a**2 + 2 * a * x + x**2 + y**2))
            - 3
            * (
                y**5
                + 2 * (a**2 + x**2) * y**3
                + (a**4 - 2 * a**2 * x**2 + x**4) * y
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
            * (
                (a**2 + x**2) * arctan_x_plus_a
                + (a**2 + x**2) * arctan_x_minus_a
            )
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
            * (
                (a**2 + x**2) * arctan_x_plus_a
                + (a**2 + x**2) * arctan_x_minus_a
            )
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
            * (
                (a**2 + x**2) * arctan_x_plus_a
                + (a**2 + x**2) * arctan_x_minus_a
            )
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
            + 4
            * (4 * a**6 + 3 * a**5 * x - 12 * a**4 * x**2 + 9 * a * x**5)
            * y**2
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
            + 4
            * (4 * a**6 - 3 * a**5 * x - 12 * a**4 * x**2 - 9 * a * x**5)
            * y**2
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
            * (
                (a**2 + x**2) * arctan_x_plus_a
                + (a**2 + x**2) * arctan_x_minus_a
            )
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
                + (a**6 - a**4 * x**2 - a**2 * x**4 + x**6)
                * arctan_x_minus_a
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
            * (
                a**8
                - 4 * a**6 * x**2
                + 6 * a**4 * x**4
                - 4 * a**2 * x**6
                + x**8
            )
            * arctan_x_plus_a
            + 3
            * (
                a**8
                - 4 * a**6 * x**2
                + 6 * a**4 * x**4
                - 4 * a**2 * x**6
                + x**8
            )
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
            * (
                (a**2 + x**2) * arctan_x_plus_a
                + (a**2 + x**2) * arctan_x_minus_a
            )
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
                + (a**6 - a**4 * x**2 - a**2 * x**4 + x**6)
                * arctan_x_minus_a
            )
            * y**2
            - 2
            * (14 * a**7 + a**5 * x**2 - 24 * a**3 * x**4 + 9 * a * x**6)
            * y
            + 9
            * (
                a**8
                - 4 * a**6 * x**2
                + 6 * a**4 * x**4
                - 4 * a**2 * x**6
                + x**8
            )
            * arctan_x_plus_a
            + 9
            * (
                a**8
                - 4 * a**6 * x**2
                + 6 * a**4 * x**4
                - 4 * a**2 * x**6
                + x**8
            )
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
            * (
                (a**2 + x**2) * arctan_x_plus_a
                + (a**2 + x**2) * arctan_x_minus_a
            )
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
                + (a**6 - a**4 * x**2 - a**2 * x**4 + x**6)
                * arctan_x_minus_a
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
            * (
                a**8
                - 4 * a**6 * x**2
                + 6 * a**4 * x**4
                - 4 * a**2 * x**6
                + x**8
            )
            * arctan_x_plus_a
            + 3
            * (
                a**8
                - 4 * a**6 * x**2
                + 6 * a**4 * x**4
                - 4 * a**2 * x**6
                + x**8
            )
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
    print(f"IN BEMCS - {physical_displacements=}")
    quadratic_coefficients = phicoef(node_coordinates, physical_displacements, a)
    print(f"IN BEMCS - {quadratic_coefficients=}")
    return quadratic_coefficients

# Slip functions
def slip_functions(x, a):
    """ Get pre-multiplier (L) to quadratic coefficients (x) to compute slip (Lx = slip) at any point on the fault patch"""
    design_matrix = np.zeros((len(x), 3))
    f1 = (x / a) * (9 * (x / a) / 8 - 3 / 4)
    f2 = (1 - 3 * (x / a) / 2) * (1 + 3 * (x / a) / 2)
    f3 = (x / a) * (9 * (x / a) / 8 + 3 / 4)
    design_matrix[:,0] = f1
    design_matrix[:,1] = f2
    design_matrix[:,2] = f3
    return design_matrix

def slip_functions_mean(x):
    """ Get pre-multiplier (L) to quadratic coefficients (x) to compute average slip (Lx = mean_slip) over the fault patch"""
    design_matrix = np.zeros((len(x), 3))
    f1 = 3/8
    f2 = 1/4
    f3 = 3/8
    design_matrix[:,0] = f1
    design_matrix[:,1] = f2
    design_matrix[:,2] = f3
    return design_matrix

# Slip gradient functions
def slipgradient_functions(x, a):
    """ Get pre-multiplier (L) to quadratic coefficients (x) to compute slip-gradient (Lx = dslip/dx) at any point on the fault patch. 
    
    Note that the slip gradient is only along the fault."""
    design_matrix = np.zeros((len(x), 3))
    df_1_dx = (9 * x) / (4 * a**2) - 3 / (4 * a)
    df_2_dx = -(9 * x) / (2 * a**2)
    df_3_dx = (9 * x) / (4 * a**2) + 3 / (4 * a)
    design_matrix[:,0] = df_1_dx
    design_matrix[:,1] = df_2_dx
    design_matrix[:,2] = df_3_dx
    return design_matrix

# Compute 3qn coefficients for given slip
def phicoef(x, slip, a):
    """ Get quadratic node coefficients for slip specified at the 3 nodes as an ordered set (x,slip) """
    mat = slip_functions(x,a)
    return np.linalg.inv(mat) @ slip

# compute slip and slip gradients from 3qn coefficients
def get_slip_slipgradient(x, a, phi):
    """ Get slip and slip gradient for a given fault patch at any point (x) within the fault
    from quadratic coefficients (phi)"""
    slip_mat = slip_functions(x,a)
    slipgradient_mat = slipgradient_functions(x,a)
    slip = slip_mat @ phi
    slipgradient = slipgradient_mat @ phi
    return slip, slipgradient


def get_individualdesignmatrix_3qn(elements):
    """Compute design matrix for a linear system of equations to calculate quadratic coefficients from applied boundary conditions for an ordered list of fault elements.
    The resulting matrix is only for 1 component of slip or slip-gradient. Use get_designmatrix_3qn() for the full matrix.

    This function provides 2 design matrices - (1) for slip at every node, (2) for slip gradients at every node"""

    designmatrix_slip = np.zeros((3 * len(elements), 3 * len(elements)))
    designmatrix_slipgradient = np.zeros((3 * len(elements), 3 * len(elements)))

    for i in range(len(elements)):
        slip_matrix = np.zeros((3, 3))
        slipgradient_matrix = np.zeros((3, 3))

        # set x_obs to be oriented along the fault
        x_obs = np.array((-elements[i]["half_length"], 0.0, elements[i]["half_length"]))

        slip_matrix = slip_functions(x_obs, elements[i]["half_length"])
        designmatrix_slip[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = slip_matrix

        slipgradient_matrix = slipgradient_functions(
            x_obs, elements[i]["half_length"]
        )
        designmatrix_slipgradient[
            3 * i : 3 * i + 3, 3 * i : 3 * i + 3
        ] = slipgradient_matrix

    return designmatrix_slip, designmatrix_slipgradient


def get_designmatrix_xy_3qn(elements,flag="node"):
    """Assemble design matrix in (x,y) coordinate system for 2 slip components (s,n) for a
    linear system of equations to calculate quadratic coefficients from applied boundary conditions for an ordered list of fault elements.

    flag = "node" : slip is applied at each node of a fault element
    
    flag = "mean" : slip is applied as a mean value over the entire fault element, not just at nodes

    Unit vectors for each patch are used to premultiply the input matrices
    [dx nx] [f1 f2 f3 0  0  0]
    [dy ny] [0  0  0  f1 f2 f3]"""

    designmatrix_slip = np.zeros((6 * len(elements), 6 * len(elements)))
    designmatrix_slipgradient = np.zeros((6 * len(elements), 6 * len(elements)))

    for i in range(len(elements)):
        slip_matrixstack = np.zeros((6, 6))
        slipgradient_matrixstack = np.zeros((6, 6))

        unitvec_matrix = np.array(
            [
                [elements[i]["x_shear"], elements[i]["x_normal"]],
                [elements[i]["y_shear"], elements[i]["y_normal"]],
            ]
        )
        unitvec_matrixstack = np.kron(np.eye(3), unitvec_matrix)

        # set x_obs to be oriented along the fault
        x_obs = np.array((-elements[i]["half_length"], 0.0, elements[i]["half_length"]))

        if flag == "node":
            slip_matrix = slip_functions(x_obs, elements[i]["half_length"])
        elif flag == "mean":
            slip_matrix = slip_functions_mean(x_obs)
        else:
            raise ValueError("Invalid flag. Use either 'node' or 'mean'.")
        
        slipgradient_matrix = slipgradient_functions(
            x_obs, elements[i]["half_length"]
        )

        slip_matrixstack[0::2, 0:3] = slip_matrix
        slip_matrixstack[1::2, 3:] = slip_matrix
        slipgradient_matrixstack[0::2, 0:3] = slipgradient_matrix
        slipgradient_matrixstack[1::2, 3:] = slipgradient_matrix

        designmatrix_slip[6 * i : 6 * (i + 1), 6 * i : 6 * (i + 1)] = (
            unitvec_matrixstack @ slip_matrixstack
        )
        designmatrix_slipgradient[6 * i : 6 * (i + 1), 6 * i : 6 * (i + 1)] = (
            unitvec_matrixstack @ slipgradient_matrixstack
        )

    return designmatrix_slip, designmatrix_slipgradient

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


def get_quadratic_displacement_stress_kernel(x_obs, y_obs, elements, mu, nu, flag):
    """INPUTS

    x_obs,y_obs - locations to compute kernels
    elements - provide list of elements (geometry & rotation matrices)
    mu, nu - Elastic parameters (Shear Modulus, Poisson ratio)
    flag - 1 for shear, 0 for tensile kernels

    OUTPUTS

    Each stress kernel is a matrix of dimensions

    Kxx = Nobs x 3xNpatches
    Kyy = Nobs x 3xNpatches
    Kxy = Nobs x 3xNpatches

    Each displacement kernel is a matrix of dimensions

    Gx = Nobs x 3xNpatches
    Gy = Nobs x 3xNpatches"""
    Kxx = np.zeros((len(x_obs), 3 * len(elements)))
    Kyy = np.zeros((len(x_obs), 3 * len(elements)))
    Kxy = np.zeros((len(x_obs), 3 * len(elements)))
    Gx = np.zeros((len(x_obs), 3 * len(elements)))
    Gy = np.zeros((len(x_obs), 3 * len(elements)))

    # check for which slip component kernels the user wants
    if flag == 1:
        flag_strikeslip = 1.0
        flag_tensileslip = 0.0
    elif flag == 0:
        flag_strikeslip = 0.0
        flag_tensileslip = 1.0
    else:
        raise ValueError("shear/tensile flag must be 1/0, no other values allowed")

    for i in range(len(elements)):
        # center observation locations (no translation needed)
        x_trans = x_obs - elements[i]["x_center"]
        y_trans = y_obs - elements[i]["y_center"]
        # rotate observations such that fault element is horizontal
        rotated_coordinates = elements[i]["inverse_rotation_matrix"] @ np.vstack(
            (x_trans.T, y_trans.T)
        )
        x_rot = rotated_coordinates[0, :].T + elements[i]["x_center"]
        y_rot = rotated_coordinates[1, :].T + elements[i]["y_center"]

        # go through each of the 3 components for a given patch
        # component 1
        slip_vector = np.array([1.0, 0.0, 0.0])
        strike_slip = slip_vector * flag_strikeslip
        tensile_slip = slip_vector * flag_tensileslip
        # Calculate displacements and stresses for current element
        (
            displacement_local,
            stress_local,
        ) = displacements_stresses_quadratic_no_rotation(
            x_rot,
            y_rot,
            elements[i]["half_length"],
            mu,
            nu,
            strike_slip,
            tensile_slip,
            elements[i]["x_center"],
            elements[i]["y_center"],
        )
        displacement_eval, stress_eval = rotate_displacement_stress(
            displacement_local, stress_local, elements[i]["inverse_rotation_matrix"]
        )
        # displacement_eval,stress_eval = displacement_local,stress_local
        index = 3 * i
        Kxx[:, index] = stress_eval[0, :]
        Kyy[:, index] = stress_eval[1, :]
        Kxy[:, index] = stress_eval[2, :]
        Gx[:, index] = displacement_eval[0, :]
        Gy[:, index] = displacement_eval[1, :]

        # component 2
        slip_vector = np.array([0.0, 1.0, 0.0])
        strike_slip = slip_vector * flag_strikeslip
        tensile_slip = slip_vector * flag_tensileslip
        # Calculate displacements and stresses for current element
        (
            displacement_local,
            stress_local,
        ) = displacements_stresses_quadratic_no_rotation(
            x_rot,
            y_rot,
            elements[i]["half_length"],
            mu,
            nu,
            strike_slip,
            tensile_slip,
            elements[i]["x_center"],
            elements[i]["y_center"],
        )
        displacement_eval, stress_eval = rotate_displacement_stress(
            displacement_local, stress_local, elements[i]["inverse_rotation_matrix"]
        )
        index = 3 * i + 1
        Kxx[:, index] = stress_eval[0, :]
        Kyy[:, index] = stress_eval[1, :]
        Kxy[:, index] = stress_eval[2, :]
        Gx[:, index] = displacement_eval[0, :]
        Gy[:, index] = displacement_eval[1, :]

        # component 3
        slip_vector = np.array([0.0, 0.0, 1.0])
        strike_slip = slip_vector * flag_strikeslip
        tensile_slip = slip_vector * flag_tensileslip
        # Calculate displacements and stresses for current element
        (
            displacement_local,
            stress_local,
        ) = displacements_stresses_quadratic_no_rotation(
            x_rot,
            y_rot,
            elements[i]["half_length"],
            mu,
            nu,
            strike_slip,
            tensile_slip,
            elements[i]["x_center"],
            elements[i]["y_center"],
        )
        displacement_eval, stress_eval = rotate_displacement_stress(
            displacement_local, stress_local, elements[i]["inverse_rotation_matrix"]
        )
        index = 3 * i + 2
        Kxx[:, index] = stress_eval[0, :]
        Kyy[:, index] = stress_eval[1, :]
        Kxy[:, index] = stress_eval[2, :]
        Gx[:, index] = displacement_eval[0, :]
        Gy[:, index] = displacement_eval[1, :]

    return Kxx, Kyy, Kxy, Gx, Gy


def compute_tractionkernels(elements, kernels):
    """Function to calculate kernels of traction vector from a set of stress kernels and unit vectors.

    Provide elements as a list with ["x_normal"] & ["y_normal"] for the unit normal vector.

    kernels must be provided as kernels[0] = Kxx, kernels[1] = Kyy, kernels[2] = Kxy
    """
    Kxx = kernels[0]
    Kyy = kernels[1]
    Kxy = kernels[2]
    nrows = np.shape(Kxx)[0] # this is typically the same as number of elements because stress kernels are calculated ONLY at the center of a given element
    # nrows = len(elements)
    # ncols = np.shape(Kxx)[1]

    tx = np.zeros_like(Kxx)
    ty = np.zeros_like(Kxx)
    # unit vector in normal direction
    nvec = np.zeros((nrows, 2))

    for i in range(nrows):
        nvec[i, :] = np.array((elements[i]["x_normal"], elements[i]["y_normal"]))
    nx_matrix = np.zeros_like(Kxx)
    ny_matrix = np.zeros_like(Kxx)

    # this is definitely incorrect - need to fix
    nx_matrix[:, 0::3] = nvec[:, 0].reshape(-1,1)
    nx_matrix[:, 1::3] = nvec[:, 0].reshape(-1,1)
    nx_matrix[:, 2::3] = nvec[:, 0].reshape(-1,1)
    ny_matrix[:, 0::3] = nvec[:, 1].reshape(-1,1)
    ny_matrix[:, 1::3] = nvec[:, 1].reshape(-1,1)
    ny_matrix[:, 2::3] = nvec[:, 1].reshape(-1,1)

    # traction vector t = n.σ
    tx = Kxx * nx_matrix + Kxy * ny_matrix
    ty = Kxy * nx_matrix + Kyy * ny_matrix

    return tx, ty

