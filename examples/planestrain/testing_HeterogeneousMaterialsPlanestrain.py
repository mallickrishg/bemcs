# %% import libraries
import numpy as np
import matplotlib.pyplot as plt
import bemcs
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# %% [markdown]
# # Plane-strain fault above a material interface
# A short fault (3 slip elements, unit shear slip in the middle element) lies at y = 0, in a layer
# with shear modulus `mu_top`; a straight interface (3 force elements) at y = -1 separates it from
# a half-space with `mu_bottom`. The interface carries equivalent body forces on a trapezoidal
# patch (`BC_type = 'h'`). Every element lists the modulus on its -n and +n side: elements run
# left to right, so their unit normal points up (+n = above).

# %% Build the mesh
els = bemcs.initialize_els()
x1 = np.array([-1.5, -0.5, 0.5, -1.5, -0.5, 0.5])
x2 = np.array([-0.5, 0.5, 1.5, -0.5, 0.5, 1.5])
y1 = np.array([0.0, 0.0, 0.0, -1.0, -1.0, -1.0])
els.x1, els.x2, els.y1, els.y2 = x1, x2, y1, y1.copy()
bemcs.standardize_els_geometry(els, reorder=False)

mu = 1.0  # reference shear modulus of the kernels
nu = 0.25
mu_top, mu_bottom = np.exp(0.5), 1.0  # modulus above and below the interface
mesh = bemcs.Mesh(
    els,
    bc_type=["s", "s", "s", "h", "h", "h"],
    bc_value=np.array([[0, 0], [1.0, 0.0], [0, 0], [0, 0], [0, 0], [0, 0]]),  # shear slip on the middle element
    mu_minus=[mu_top] * 3 + [mu_bottom] * 3,
    mu_plus=[mu_top] * 3 + [mu_top] * 3,
    basis={"slip": "hat", "force": "trapezoid"},
)
plt.figure(figsize=(6, 3))
bemcs.plot_mesh(mesh)
plt.title(r"elements, normals and $\mu$ on the $-n$ / $+n$ side")
plt.show()

# %% Solve (one-ring hat for slip, trapezoid for force)
print("Solving Plane Strain Heterogeneous BEM system...")
x_sol, (A, b, rows) = bemcs.solve(mesh, geometry="planestrain", mu=mu, nu=nu, return_system=True)
print("condition number:", np.linalg.cond(A))
coefs_slip = x_sol[mesh.columns("slip", geometry="planestrain")]
coefs_force = x_sol[mesh.columns("force", geometry="planestrain")]
print("Slip coefficients:", coefs_slip)
print("Force coefficients:", coefs_force)

# %% Displacements and (true) stresses
def mu_field(x, y):
    return np.where(y > -1.0, mu_top, mu_bottom)


fig, axes, fields = bemcs.plot_fields(
    mesh, x_sol, geometry="planestrain", mu=mu, nu=nu, xlim=(-3, 3), ylim=(-3, 1), n=(100, 100),
    components=["ux", "uy", "sxx", "sxy"], mu_field=mu_field,
)
plt.show()
ux, uy, sxx, sxy = (fields[k] for k in ("ux", "uy", "sxx", "sxy"))
