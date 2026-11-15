# %% import libraries
import numpy as np
import matplotlib.pyplot as plt
import bemcs
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# %% Build a simple test mesh
els = bemcs.initialize_els()
# 3 fault elements (slip)
x1_s = np.array([-1.5, -0.5, 0.5])
x2_s = np.array([-0.5, 0.5, 1.5])
y1_s = np.array([0.0, 0.0, 0.0])
y2_s = np.array([0.0, 0.0, 0.0])

# 3 heterogeneous interface elements (force)
x1_f = np.array([-1.5, -0.5, 0.5])
x2_f = np.array([-0.5, 0.5, 1.5])
y1_f = np.array([-1.0, -1.0, -1.0])
y2_f = np.array([-1.0, -1.0, -1.0])

els.x1 = np.hstack((x1_s, x1_f))
els.x2 = np.hstack((x2_s, x2_f))
els.y1 = np.hstack((y1_s, y1_f))
els.y2 = np.hstack((y2_s, y2_f))

bemcs.standardize_els_geometry(els, reorder=False)
bemcs.plot_els_geometry(els)
plt.show()

# Connectivities for quadratic hat (slip) and trapezoidal (force)
connect_slip = np.array([[0, 1, 2]])
connect_force = np.array([[3, 4, 5]])

BCtype_slip = ['s']
BCval_slip = np.array([[1.0, 0.0]]) # 1.0 shear slip, 0.0 normal slip
alpha_force = np.array([0.5]) # dmu/dn / mu

# %% Solve BEM system
mu = 1.0
nu = 0.25

print("Solving Plane Strain Heterogeneous BEM system...")
coefs_slip, coefs_force = bemcs.bemAssembly.solve_planestrain_heterogeneous(
    els,
    connect_slip, BCtype_slip, BCval_slip,
    connect_force, alpha_force,
    mu=mu, nu=nu, verbose=True,
)

print("Slip coefficients:", coefs_slip)
print("Force coefficients:", coefs_force)

# %% Compute and plot displacements and stresses
nx_obs = 100
ny_obs = 100
x_vec = np.linspace(-3.0, 3.0, nx_obs)
y_vec = np.linspace(-3.0, 1.0, ny_obs)
x_obs, y_obs = np.meshgrid(x_vec, y_vec)
xo = x_obs.flatten()
yo = y_obs.flatten()

# Mesh: hat slip on elements 0-2, trapezoidal force on elements 3-5
mesh = bemcs.Mesh(
    els,
    ["s", "s", "s", "f", "f", "f"],
    basis={"slip": "hat", "force": "trapezoid"},
    connectivity={"slip": connect_slip, "force": connect_force},
)
U, S, info = bemcs.get_kernels(xo, yo, mesh, geometry="planestrain", mu=mu, nu=nu)
print(info)

# Total fields: the solution vector is [slip coefficients; force coefficients]
x_sol = np.concatenate((coefs_slip, coefs_force))
n = len(xo)
ux, uy = (U[:, :, k, :].reshape(n, -1) @ x_sol for k in range(2))
sxx, syy, sxy = (S[:, :, k, :].reshape(n, -1) @ x_sol for k in range(3))

# Plot
xlimits = [-3, 3]
ylimits = [-3, 1]

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
bemcs.plot_BEM_field(
    ux.reshape(ny_obs, nx_obs), els, xo, yo, xlimits, ylimits, maxval=0.5, n_levels=11, cmap="Spectral_r"
)
plt.title("ux")

plt.subplot(2, 2, 2)
bemcs.plot_BEM_field(
    uy.reshape(ny_obs, nx_obs), els, xo, yo, xlimits, ylimits, maxval=0.5, n_levels=11, cmap="Spectral_r"
)
plt.title("uy")

plt.subplot(2, 2, 3)
bemcs.plot_BEM_field(
    sxx.reshape(ny_obs, nx_obs), els, xo, yo, xlimits, ylimits, maxval=1.0, n_levels=11, cmap="Spectral_r"
)
plt.title("sxx")

plt.subplot(2, 2, 4)
bemcs.plot_BEM_field(
    sxy.reshape(ny_obs, nx_obs), els, xo, yo, xlimits, ylimits, maxval=1.0, n_levels=11, cmap="Spectral_r"
)
plt.title("sxy")

plt.tight_layout()
plt.show()
