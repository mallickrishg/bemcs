# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import bemcs
import utilfunctions as UF
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# %% Define mesh and boundary conditions
# construct mesh
Lscale = 30
N_top = 50  # always choose odd npts so that 0 can be included symmetrically
hmax = 1
# xvals = np.linspace(-Lscale, Lscale, npts)
# build symmetric log-spaced x array from –Lscale → 0 → +Lscale
epsmin = 5e-3 * Lscale
h_fixed = 2.0  # uniform spacing for left/right/bottom
L_y = -5.0  # <-- USER-DEFINED bottom value

# ------------------------------------------------------------
# 1. Top boundary (logistic spacing)
# ------------------------------------------------------------
half = np.logspace(np.log10(epsmin), np.log10(Lscale), N_top)
x_top = np.concatenate([-half[::-1], [0.0], half])
y_top = UF.logistic(x_top, L=hmax, k=2, x0=0)
N_top_pts = len(x_top)

# ------------------------------------------------------------
# 2. Left boundary (fixed spacing)
# bottom → top
# ------------------------------------------------------------
y_left_vals = np.arange(L_y, y_top[0] - h_fixed, h_fixed)
x_left_vals = np.full_like(y_left_vals, x_top[0])

# ------------------------------------------------------------
# 3. Right boundary (fixed spacing)
# top → bottom
# ------------------------------------------------------------
y_right_vals = np.arange(y_top[-1], L_y, -h_fixed)
x_right_vals = np.full_like(y_right_vals, x_top[-1])

# ------------------------------------------------------------
# 4. Bottom boundary (fixed spacing)
# right → left
# ------------------------------------------------------------
x_bottom_vals = np.arange(Lscale, -Lscale - h_fixed, -h_fixed)
y_bottom_vals = np.full_like(x_bottom_vals, L_y)

# ------------------------------------------------------------
# 5. Concatenate boundaries in reversed circulation:
# left → top → right → bottom
# ------------------------------------------------------------
xvals = np.concatenate([x_left_vals, x_top, x_right_vals[1:], x_bottom_vals[0:]])
yvals = np.concatenate([y_left_vals, y_top, y_right_vals[1:], y_bottom_vals[0:]])

x1 = xvals[0:-1]
x2 = xvals[1:]
y1 = yvals[0:-1]
y2 = yvals[1:]

# construct mesh
els = bemcs.initialize_els()

els.x1 = x1
els.y1 = y1
els.x2 = x2
els.y2 = y2

bemcs.standardize_els_geometry(els, reorder=False)
bemcs.plot_els_geometry(els)

# ------------------------------------------------------------
# 8. Assign BC types
# top = "t_global", others = "u_global"
# ------------------------------------------------------------
BCtype = np.empty_like(x1, dtype=object)
N_left = len(x_left_vals)
N_top_elems = N_top_pts - 1
# left boundary
BCtype[:N_left] = "u_global"
# top boundary
BCtype[N_left : N_left + N_top_elems] = "t_global"
# right + bottom boundaries
BCtype[N_left + N_top_elems :] = "u_global"

# apply a localized Gaussian load at the top boundary
BCval = np.exp(-(els.x_centers**2) / (2 * np.pi * 0.5**2))
# BCval = 1.0 * (els.y_centers - 0.0)
BCval[BCtype == "u_global"] = 0.0

# provide connections to compute trapeoidal basis functions
# connectivity matrix: each row is a group of 3 consecutive elements
connect_matrix = np.vstack(
    [np.array([i, i + 1, i + 2]) for i in range(0, len(els.x1) - 2, 2)]
)

# Define material properties
# Elastic parameters and lithostatic effects
mu = 1  # normalized units (mostly does not matter for topographic problem)
nu = 0.25  # Poisson's ratio
rho = 1  # rho*g/mu
ydatum = 1  # provide a datum for verticals

# %% Compute displacement & stress kernels (slip & force basis)
# Observation points
n_obs = 100
xlimits = [-1.5 * np.abs(L_y), 1.5 * np.abs(L_y)]
ylimits = [0.99 * L_y, 1.2 * hmax]
x_obs = np.linspace(xlimits[0], xlimits[1], n_obs)
# y_obs = np.linspace(-2 * hmax, 1.2 * hmax, n_obs)
y_obs = np.linspace(ylimits[0], ylimits[1], n_obs)
x_obs, y_obs = np.meshgrid(x_obs, y_obs)
x_obs = x_obs.flatten()
y_obs = y_obs.flatten()

# Compute shear and tensile stress kernels (slip basis)
kernels_s = bemcs.get_displacement_stress_kernel(x_obs, y_obs, els, mu, nu, "shear")
kernels_n = bemcs.get_displacement_stress_kernel(x_obs, y_obs, els, mu, nu, "normal")

# create a mask to remove all plotted values above topography (outside the domain)
index = bemcs.inpolygon(
    x_obs,
    y_obs,
    np.hstack((els.x1)),
    np.hstack((els.y1)),
)
# index = np.ones_like(x_obs, dtype=bool)

# compute kernels [Nobs x 2*N_trapz]
K_ux, K_uy, K_sxx, K_syy, K_sxy = (
    bemcs.bemAssembly.get_kernels_trapezoidalforce_planestrain(
        x_obs, y_obs, els, connect_matrix
    )
)

# Solve topographic load problem
bc_x = BCval * els.x_normals
bc_y = BCval * els.y_normals

# solve for slip coefficients
quadratic_coefs_s, quadratic_coefs_n = UF.solve_bem_system_slip(
    els, bc_x, bc_y, BCtype, BCtype, mu, nu
)
# Compute stresses
ux, uy, sxx, syy, sxy = bemcs.coeffs_to_disp_stress(
    kernels_s, kernels_n, quadratic_coefs_s, quadratic_coefs_n
)
# # plot displacements & stresses
# n_skip_plot = 3
# bemcs.plot_displacements_stresses_els(
#     els, n_obs, ux, uy, sxx, syy, sxy, x_obs, y_obs, n_skip_plot
# )


# solve for force coefficients
fcoefs_s, fcoefs_n = UF.solve_bem_system_force(
    els,
    connect_matrix,
    bc_x[connect_matrix[:, 1]],
    bc_y[connect_matrix[:, 1]],
    BCtype[connect_matrix[:, 1]],
    BCtype[connect_matrix[:, 1]],
    mu,
    nu,
)
# compute displacements & stresses from force coefficients
ux_f = K_ux[:, 0::2] @ fcoefs_s + K_ux[:, 1::2] @ fcoefs_n
uy_f = K_uy[:, 0::2] @ fcoefs_s + K_uy[:, 1::2] @ fcoefs_n
sxx_f = K_sxx[:, 0::2] @ fcoefs_s + K_sxx[:, 1::2] @ fcoefs_n
syy_f = K_syy[:, 0::2] @ fcoefs_s + K_syy[:, 1::2] @ fcoefs_n
sxy_f = K_sxy[:, 0::2] @ fcoefs_s + K_sxy[:, 1::2] @ fcoefs_n
# # plot displacements & stresses
# bemcs.plot_displacements_stresses_els(
#     els, n_obs, ux_f, uy_f, sxx_f, syy_f, sxy_f, x_obs, y_obs, n_skip_plot
# )
# compute overpressure in the medium
pressure = (sxx + syy) / 2  # - rho * (y_obs.reshape(-1, 1) - ydatum)
pressure_f = (sxx_f + syy_f) / 2  # - rho * (y_obs.reshape(-1, 1) - ydatum)
# compute deviatoric stress
deviatoric_stress = np.sqrt(1 / 4 * ((sxx - syy) ** 2) + sxy**2)
deviatoric_stress_f = np.sqrt(1 / 4 * ((sxx_f - syy_f) ** 2) + sxy_f**2)

# %% Plot results in terms of deviatoric stress & pressure
plt.figure(figsize=(15, 7))
# plot results using slip coefficients
plt.subplot(2, 2, 1)
# plot deviatoric stress due to topography
toplot = deviatoric_stress
toplot[~index, 0] = np.nan
maxval = 0.5
bemcs.bemAssembly.plot_BEM_field(
    toplot,
    els,
    x_obs,
    y_obs,
    xlimits,
    ylimits,
    maxval=maxval,
    n_levels=11,
    cmap="coolwarm",
)
plt.title("$\\sigma_{II}$ (slip)", fontsize=15)
plt.ylabel("z", fontsize=12)

plt.subplot(2, 2, 2)
toplot = np.abs(pressure)
toplot[~index, 0] = np.nan
bemcs.bemAssembly.plot_BEM_field(
    toplot,
    els,
    x_obs,
    y_obs,
    xlimits,
    ylimits,
    maxval=maxval,
    n_levels=11,
    cmap="coolwarm",
)
plt.title("$p=|\\frac{\\sigma_{kk}}{2}|$ (slip)", fontsize=15)
plt.ylabel("z", fontsize=12)

# Using force coefficients
plt.subplot(2, 2, 3)
# plot deviatoric stress due to topography
toplot = deviatoric_stress_f
toplot[~index, 0] = np.nan
bemcs.bemAssembly.plot_BEM_field(
    toplot,
    els,
    x_obs,
    y_obs,
    xlimits,
    ylimits,
    maxval=maxval,
    n_levels=11,
    cmap="coolwarm",
)
plt.title("$\\sigma_{II}$ (force)", fontsize=15)
plt.xlabel("x", fontsize=12)
plt.ylabel("z", fontsize=12)

plt.subplot(2, 2, 4)
toplot = np.abs(pressure_f)
toplot[~index, 0] = np.nan
bemcs.bemAssembly.plot_BEM_field(
    toplot,
    els,
    x_obs,
    y_obs,
    xlimits,
    ylimits,
    maxval=maxval,
    n_levels=11,
    cmap="coolwarm",
)
plt.title("$p=|\\frac{\\sigma_{kk}}{2}|$ (force)", fontsize=15)
plt.xlabel("x", fontsize=12)
plt.ylabel("z", fontsize=12)
plt.show()

# now plot stress components using slip coefficients
_, _, sxx_f, syy_f, sxy_f = bemcs.coeffs_to_disp_stress(
    kernels_s, kernels_n, quadratic_coefs_s, quadratic_coefs_n
)
# plotting parameters
maxval = 2.0
plt.figure(figsize=(15, 5))
for count in range(3):
    plt.subplot(2, 3, count + 1)
    if count == 0:
        toplot = sxx_f
        labelname = "$\\sigma_{xx}$"
    elif count == 1:
        toplot = syy_f
        labelname = "$\\sigma_{yy}$"
    else:
        toplot = sxy_f
        labelname = "$\\sigma_{xy}$"
    bemcs.bemAssembly.plot_BEM_field(
        toplot,
        els,
        x_obs,
        y_obs,
        xlimits,
        ylimits,
        maxval=maxval,
        n_levels=21,
        cmap="Spectral",
    )
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title(labelname + " (slip)")

# now plot stress components using force coefficients
sxx_f = K_sxx[:, 0::2] @ fcoefs_s + K_sxx[:, 1::2] @ fcoefs_n
syy_f = K_syy[:, 0::2] @ fcoefs_s + K_syy[:, 1::2] @ fcoefs_n
sxy_f = K_sxy[:, 0::2] @ fcoefs_s + K_sxy[:, 1::2] @ fcoefs_n

for count in range(3):
    plt.subplot(2, 3, 3 + count + 1)
    if count == 0:
        toplot = sxx_f
        labelname = "$\\sigma_{xx}$"
    elif count == 1:
        toplot = syy_f
        labelname = "$\\sigma_{yy}$"
    else:
        toplot = sxy_f
        labelname = "$\\sigma_{xy}$"
    bemcs.bemAssembly.plot_BEM_field(
        toplot,
        els,
        x_obs,
        y_obs,
        xlimits,
        ylimits,
        maxval=maxval,
        n_levels=21,
        cmap="Spectral",
    )
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title(labelname + " (force)")
plt.show()

# %% evaluate stress fields at the surface
npts_per_elt = 10
dr = -1e-9  # small offset to avoid discontinuity
x_obs = np.zeros((len(els.x1[BCtype == "t_global"]) * npts_per_elt,))
y_obs = np.zeros((len(els.x1[BCtype == "t_global"]) * npts_per_elt,))
for i in range(len(els.x1[BCtype == "t_global"])):
    xpts = np.linspace(
        els.x1[BCtype == "t_global"][i],
        els.x2[BCtype == "t_global"][i],
        npts_per_elt + 2,
    )[1:-1]
    ypts = np.linspace(
        els.y1[BCtype == "t_global"][i],
        els.y2[BCtype == "t_global"][i],
        npts_per_elt + 2,
    )[1:-1]
    x_obs[i * npts_per_elt : (i + 1) * npts_per_elt] = (
        xpts + els.x_normals[BCtype == "t_global"][i] * dr
    )
    y_obs[i * npts_per_elt : (i + 1) * npts_per_elt] = (
        ypts + els.y_normals[BCtype == "t_global"][i] * dr
    )

# Compute shear and tensile stress kernels
kernels_s = bemcs.get_displacement_stress_kernel(x_obs, y_obs, els, mu, nu, "shear")
kernels_n = bemcs.get_displacement_stress_kernel(x_obs, y_obs, els, mu, nu, "normal")
# Compute stresses
_, _, sxx, syy, sxy = bemcs.coeffs_to_disp_stress(
    kernels_s, kernels_n, quadratic_coefs_s, quadratic_coefs_n
)
# compute kernels [Nobs x 2*N_trapz]
_, _, K_sxx, K_syy, K_sxy = bemcs.bemAssembly.get_kernels_trapezoidalforce_planestrain(
    x_obs, y_obs, els, connect_matrix
)

# compute stresses from force coefficients
sxx_f = K_sxx[:, 0::2] @ fcoefs_s + K_sxx[:, 1::2] @ fcoefs_n
syy_f = K_syy[:, 0::2] @ fcoefs_s + K_syy[:, 1::2] @ fcoefs_n
sxy_f = K_sxy[:, 0::2] @ fcoefs_s + K_sxy[:, 1::2] @ fcoefs_n

plt.figure(figsize=(15, 1.5))
plt.subplot(1, 3, 1)
plt.plot(x_obs, sxx, ".", label="slip")
plt.plot(x_obs, sxx_f, "-", label="force")
plt.xlabel("x", fontsize=10)
plt.ylabel("$\\sigma_{xx}$", fontsize=10)
plt.legend()
plt.xlim(xlimits)
plt.ylim([-1, 1])
plt.subplot(1, 3, 2)
plt.plot(x_obs, syy, ".")
plt.plot(x_obs, syy_f, "-")
plt.xlabel("x", fontsize=10)
plt.ylabel("$\\sigma_{yy}$", fontsize=10)
plt.xlim(xlimits)
plt.ylim([-1, 1])
plt.subplot(1, 3, 3)
plt.plot(x_obs, sxy, ".")
plt.plot(x_obs, sxy_f, "-")
plt.xlabel("x", fontsize=10)
plt.ylabel("$\\sigma_{xy}$", fontsize=10)
plt.xlim(xlimits)
plt.ylim([-1, 1])
plt.show()

# evaluate stresses & tractions at element centers
x_obs = (els.x_centers + els.x_normals * dr)[BCtype == "t_global"]
y_obs = (els.y_centers + els.y_normals * dr)[BCtype == "t_global"]
# Compute shear and tensile stress kernels
kernels_s = bemcs.get_displacement_stress_kernel(x_obs, y_obs, els, mu, nu, "shear")
kernels_n = bemcs.get_displacement_stress_kernel(x_obs, y_obs, els, mu, nu, "normal")
# Compute stresses
ux, uy, sxx, syy, sxy = bemcs.coeffs_to_disp_stress(
    kernels_s, kernels_n, quadratic_coefs_s, quadratic_coefs_n
)

# compute kernels [Nobs x 2*N_trapz]
_, _, K_sxx, K_syy, K_sxy = bemcs.bemAssembly.get_kernels_trapezoidalforce_planestrain(
    x_obs, y_obs, els, connect_matrix
)
# compute stresses from force coefficients
sxx_f = K_sxx[:, 0::2] @ fcoefs_s + K_sxx[:, 1::2] @ fcoefs_n
syy_f = K_syy[:, 0::2] @ fcoefs_s + K_syy[:, 1::2] @ fcoefs_n
sxy_f = K_sxy[:, 0::2] @ fcoefs_s + K_sxy[:, 1::2] @ fcoefs_n
# plot traction components
plt.figure(figsize=(10, 2))
plt.subplot(1, 2, 1)
plt.plot(
    x_obs,
    sxx.flatten() * els.x_normals[BCtype == "t_global"]
    + sxy.flatten() * els.y_normals[BCtype == "t_global"],
    "o-",
    label="slip",
)
plt.plot(
    x_obs,
    sxx_f.flatten() * els.x_normals[BCtype == "t_global"]
    + sxy_f.flatten() * els.y_normals[BCtype == "t_global"],
    ".-",
    label="force",
)
plt.xlabel("x", fontsize=10)
plt.ylabel("$\\tau_x$", fontsize=10)
plt.legend()
plt.xlim(xlimits)
plt.subplot(1, 2, 2)
plt.plot(
    x_obs,
    sxy.flatten() * els.x_normals[BCtype == "t_global"]
    + syy.flatten() * els.y_normals[BCtype == "t_global"],
    "o-",
)
plt.plot(
    x_obs,
    sxy_f.flatten() * els.x_normals[BCtype == "t_global"]
    + syy_f.flatten() * els.y_normals[BCtype == "t_global"],
    ".-",
)
plt.xlabel("x", fontsize=10)
plt.ylabel("$\\tau_y$", fontsize=10)
plt.xlim(xlimits)
plt.show()

# %%
