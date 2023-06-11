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

