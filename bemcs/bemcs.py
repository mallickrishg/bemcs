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


def plot_element_geometry(elements):
    """Plot element geometry"""
    for element in elements:
        plt.plot(
            [element["x1"], element["x2"]],
            [element["y1"], element["y2"]],
            "-k",
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

    # Extract and plot unit normal vectors
    x_center = np.array([_["x_center"] for _ in elements])
    y_center = np.array([_["y_center"] for _ in elements])
    x_normal = np.array([_["x_normal"] for _ in elements])
    y_normal = np.array([_["y_normal"] for _ in elements])
    plt.quiver(
        x_center, y_center, x_normal, y_normal, units="width", color="gray", width=0.002
    )

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
    plt.show(block=False)

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
        element["x_normal"] = dy / mag
        element["y_normal"] = -dx / mag

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