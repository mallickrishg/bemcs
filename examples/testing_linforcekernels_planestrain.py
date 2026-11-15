# %%
import numpy as np
import matplotlib.pyplot as plt
import bemcs

# %%
x_offset = 0.0
y_offset = 0.0

mu = 1
nu = 0.25
w = 0.5
nx = 100
ny = 100
x_vec = np.linspace(-2.0, 2.0, nx)
y_vec = np.linspace(-2.0, 2.0, ny)
x_grid, y_grid = np.meshgrid(x_vec, y_vec)
# Displacement kernels [Nobs x (ux or uy) x (fx or fy) x 2 basis functions]

Dkernels, Skernels, Ekernels = (
    bemcs.displacements_stresses_linear_force_no_rotation_planestrain(
        x_grid.reshape(-1, 1),
        y_grid.reshape(-1, 1),
        xf=x_offset,
        yf=y_offset,
        w=w,
        nu=nu,
        mu=mu,
    )
)

# plot displacements
for count in range(0, 2):
    basisfunc = count

    plt.figure(count + 1, figsize=(15, 10))
    for i in range(0, 2):
        for j in range(0, 2):
            plotval = i * 2 + j + 1
            toplot = Dkernels[:, i, j, basisfunc]
            plt.subplot(2, 2, plotval)
            plt.pcolor(
                x_grid,
                y_grid,
                toplot.reshape(ny, nx),
                cmap="seismic_r",
            )
            plt.colorbar()
            if i == 0:
                str1 = "$u_x$"
            else:
                str1 = "$u_y$"
            if j == 0:
                str2 = "$f_x$"
            else:
                str2 = "$f_y$"
            plt.title(str1 + "," + str2 + " (basis = " + str(basisfunc + 1) + ")")
            plt.gca().set_aspect("equal", adjustable="box")
    plt.show()

# kernels: [Nobs x (ux or uy) x (fx or fy) x 2 basis functions]
# fx displacements
plt.figure(figsize=(10, 6))
for basisfunc in range(2):
    plt.subplot(1, 2, basisfunc + 1)
    ux = Dkernels[:, 0, 0, basisfunc]
    uy = Dkernels[:, 1, 0, basisfunc]
    plt.quiver(x_grid, y_grid, ux, uy, scale=0.2, units="xy")
    plt.gca().set_aspect("equal", adjustable="box")
plt.show()

# fy displacements
plt.figure(figsize=(10, 6))
for basisfunc in range(2):
    plt.subplot(1, 2, basisfunc + 1)
    ux = Dkernels[:, 0, 1, basisfunc]
    uy = Dkernels[:, 1, 1, basisfunc]
    plt.quiver(x_grid, y_grid, ux, uy, scale=0.2, units="xy")
    plt.gca().set_aspect("equal", adjustable="box")
plt.show()

# %%
# construct mesh
els = bemcs.initialize_els()

# els.x1 = np.array([-w])
# els.y1 = np.array([0.0])
# els.x2 = np.array([w])
# els.y2 = np.array([0.0])
els.x1 = np.array([0.0])
els.y1 = np.array([w])
els.x2 = np.array([0.0])
els.y2 = np.array([-w])

bemcs.standardize_els_geometry(els, reorder=False)
# bemcs.plot_els_geometry(els)

# kernel shape: [Nobs x (fx,fy) x 2 basis functions x Nels]
K_ux, K_uy, K_sxx, K_syy, K_sxy = (
    bemcs.bemAssembly.get_displacement_stress_kernel_force_planestrain(
        x_grid.flatten(),
        y_grid.flatten(),
        els.x_centers,
        els.y_centers,
        els.half_lengths,
        els.rot_mats,
        els.rot_mats_inv,
        mu=mu,
        nu=nu,
    )
)

nskip = 13
# plot displacements and stresses for [fx,fy] by summing over basis components
for force_comp in range(2):
    if force_comp == 0:
        force_str = "$f_s$"
    else:
        force_str = "$f_n$"
    plt.figure(figsize=(15, 10))
    for i in range(0, 2):
        for j in range(0, 2):
            plotval = i * 2 + j + 1
            if i == 0 and j == 0:
                ux = K_ux[:, force_comp, 0, :] + 1 * K_ux[:, force_comp, 1, :]
                uy = K_uy[:, force_comp, 0, :] + 1 * K_uy[:, force_comp, 1, :]
                toplot = np.sqrt(ux**2 + uy**2)
                str1 = "$u$"
            elif i == 0 and j == 1:
                toplot = K_sxy[:, force_comp, 0, :] + 1 * K_sxy[:, force_comp, 1, :]
                str1 = "$\sigma_{xy}$"
            elif i == 1 and j == 0:
                toplot = K_sxx[:, force_comp, 0, :] + 1 * K_sxx[:, force_comp, 1, :]
                str1 = "$\sigma_{xx}$"
            else:
                toplot = K_syy[:, force_comp, 0, :] + 1 * K_syy[:, force_comp, 1, :]
                str1 = "$\sigma_{yy}$"
            plt.subplot(2, 2, plotval)
            plt.pcolor(
                x_grid,
                y_grid,
                toplot.reshape(ny, nx),
                cmap="coolwarm",
                vmin=-0.5,
                vmax=0.5,
            )
            plt.colorbar()
            if plotval == 1:
                plt.quiver(
                    x_grid.flatten()[0::nskip],
                    y_grid.flatten()[0::nskip],
                    ux[0::nskip],
                    uy[0::nskip],
                    scale=0.2,
                    units="xy",
                )

            plt.title(str1 + " due to " + force_str)
            plt.gca().set_aspect("equal", adjustable="box")
    plt.show()

# Plot pressure field by summing p = (sxx+syy)/2
plt.figure(figsize=(14, 6))
for force_comp in range(2):
    plt.subplot(1, 2, force_comp + 1)
    p = 0.5 * (
        K_sxx[:, force_comp, 0, :]
        + 1 * K_sxx[:, force_comp, 1, :]
        + K_syy[:, force_comp, 0, :]
        + 1 * K_syy[:, force_comp, 1, :]
    )
    plt.pcolor(
        x_grid,
        y_grid,
        p.reshape(ny, nx),
        cmap="coolwarm",
        vmin=-0.5,
        vmax=0.5,
    )
    plt.colorbar()
    if force_comp == 0:
        force_str = "$f_s$"
    else:
        force_str = "$f_n$"
    plt.title("Pressure field due to " + force_str)
    plt.gca().set_aspect("equal", adjustable="box")
plt.show()
