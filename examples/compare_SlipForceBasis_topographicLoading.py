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
Lscale = 200
npts = 50  # always choose an odd npts so that 0 can be included symmetrically
hmax = 1.5
# xvals = np.linspace(-Lscale, Lscale, npts)
# build symmetric log-spaced x array from –Lscale → 0 → +Lscale
epsmin = 1e-3 * Lscale
half = np.logspace(np.log10(epsmin), np.log10(Lscale), (npts + 1) // 2)
xvals = np.concatenate([-half[::-1], [0.0], half])
yvals = UF.logistic(xvals, L=hmax, k=2, x0=0)

x1 = xvals[0:-1]
x2 = xvals[1:]
y1 = yvals[0:-1]
y2 = yvals[1:]
# Default BC is traction everywhere
BCtype = np.array(["t_global"] * len(x1))
# Set first and last 2 elements to displacement BC
# BCtype[0:2] = "u_global"
# BCtype[-2:] = "u_global"

# construct mesh
els = bemcs.initialize_els()

els.x1 = x1
els.y1 = y1
els.x2 = x2
els.y2 = y2

bemcs.standardize_els_geometry(els, reorder=False)
bemcs.plot_els_geometry(els)
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
xlimits = [-5, 5]
x_obs = np.linspace(xlimits[0], xlimits[1], n_obs)
y_obs = np.linspace(-2 * hmax, 1.2 * hmax, n_obs)
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
    np.hstack((els.x_centers, 100, 100, -100)),
    np.hstack((els.y_centers, 10, -10, -10)),
)

# compute kernels [Nobs x 2*N_trapz]
K_ux, K_uy, K_sxx, K_syy, K_sxy = (
    bemcs.bemAssembly.get_kernels_trapezoidalforce_planestrain(
        x_obs, y_obs, els, connect_matrix
    )
)

# Solve topographic load problem
ydatum = 0
rhog = 1  # normalized

# BCval = rhog * (els.y_centers - ydatum)
BCval = np.exp(-(els.x_centers**2) / (2 * np.pi * 1**2))
# BCval[0:2] = 0.0  # first element
# BCval[-2:] = 0.0  # last element

bc_x = BCval * els.x_normals
bc_y = BCval * els.y_normals

# solve for slip coefficients
quadratic_coefs_s, quadratic_coefs_n = UF.solve_bem_system_slip(
    els, bc_x, bc_y, BCtype, BCtype, mu=mu, nu=nu
)
# Compute stresses
_, _, sxx, syy, sxy = bemcs.coeffs_to_disp_stress(
    kernels_s, kernels_n, quadratic_coefs_s, quadratic_coefs_n
)


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
# compute stresses from force coefficients
sxx_f = K_sxx[:, 0::2] @ fcoefs_s + K_sxx[:, 1::2] @ fcoefs_n
syy_f = K_syy[:, 0::2] @ fcoefs_s + K_syy[:, 1::2] @ fcoefs_n
sxy_f = K_sxy[:, 0::2] @ fcoefs_s + K_sxy[:, 1::2] @ fcoefs_n

# compute overpressure in the medium
pressure = (sxx + syy) / 2  # - rho * (y_obs.reshape(-1, 1) - ydatum)
pressure_f = (sxx_f + syy_f) / 2  # - rho * (y_obs.reshape(-1, 1) - ydatum)
# compute deviatoric stress
deviatoric_stress = np.sqrt(1 / 4 * ((sxx - syy) ** 2) + sxy**2)
deviatoric_stress_f = np.sqrt(1 / 4 * ((sxx_f - syy_f) ** 2) + sxy_f**2)

# %% Plot results
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
    ylimits=None,
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
    ylimits=None,
    maxval=maxval,
    n_levels=11,
    cmap="Blues",
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
    ylimits=None,
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
    ylimits=None,
    maxval=maxval,
    n_levels=11,
    cmap="Blues",
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
        ylimits=None,
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
        ylimits=None,
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
x_obs = np.zeros((len(els.x1) * npts_per_elt,))
y_obs = np.zeros((len(els.x1) * npts_per_elt,))
for i in range(len(els.x1)):
    xpts = np.linspace(els.x1[i], els.x2[i], npts_per_elt + 2)[1:-1]
    ypts = np.linspace(els.y1[i], els.y2[i], npts_per_elt + 2)[1:-1]
    x_obs[i * npts_per_elt : (i + 1) * npts_per_elt] = xpts + els.x_normals[i] * dr
    y_obs[i * npts_per_elt : (i + 1) * npts_per_elt] = ypts + els.y_normals[i] * dr

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

plt.figure(figsize=(15, 2))
plt.subplot(1, 3, 1)
plt.plot(x_obs, sxx, ".-", label="slip")
plt.plot(x_obs, sxx_f, ".-", label="force")
plt.xlabel("x", fontsize=10)
plt.ylabel("$\\sigma_{xx}$", fontsize=10)
plt.legend()
plt.xlim(xlimits)
# plt.ylim([-1, 1])
plt.subplot(1, 3, 2)
plt.plot(x_obs, syy, ".-")
plt.plot(x_obs, syy_f, ".-")
plt.xlabel("x", fontsize=10)
plt.ylabel("$\\sigma_{yy}$", fontsize=10)
plt.xlim(xlimits)
# plt.ylim([-1, 1])
plt.subplot(1, 3, 3)
plt.plot(x_obs, sxy, ".-")
plt.plot(x_obs, sxy_f, ".-")
plt.xlabel("x", fontsize=10)
plt.ylabel("$\\sigma_{xy}$", fontsize=10)
plt.xlim(xlimits)
# plt.ylim([-1, 1])
plt.show()

# evaluate stresses & tractions at element centers
x_obs = els.x_centers + els.x_normals * dr
y_obs = els.y_centers + els.y_normals * dr
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
# plot traction components
plt.figure(figsize=(10, 2))
plt.subplot(1, 2, 1)
plt.plot(
    x_obs,
    sxx.flatten() * els.x_normals + sxy.flatten() * els.y_normals,
    "o-",
    label="slip",
)
plt.plot(
    x_obs,
    sxx_f.flatten() * els.x_normals + sxy_f.flatten() * els.y_normals,
    ".-",
    label="force",
)
plt.xlabel("x", fontsize=10)
plt.ylabel("$\\tau_x$", fontsize=10)
plt.legend()
plt.xlim(xlimits)
plt.subplot(1, 2, 2)
plt.plot(x_obs, sxy.flatten() * els.x_normals + syy.flatten() * els.y_normals, "o-")
plt.plot(x_obs, sxy_f.flatten() * els.x_normals + syy_f.flatten() * els.y_normals, ".-")
plt.xlabel("x", fontsize=10)
plt.ylabel("$\\tau_y$", fontsize=10)
plt.xlim(xlimits)
plt.show()

# %%
