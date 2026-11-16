# %% import libraries
import numpy as np
import matplotlib.pyplot as plt
import bemcs

# %% [markdown]
# # Antiplane fault in a layered (heterogeneous) medium
# The material interfaces are represented by force elements (`BC_type = 'h'`) carrying equivalent
# body forces. Every element lists the shear modulus on its -n side (`mu_minus`) and +n side
# (`mu_plus`); for an interface this is the jump in modulus. `bemcs.solve` imposes the exact
# sharp-interface condition (continuity of the true traction) on trapezoidal force patches.
#
# Run `ConstructMesh.py` in this folder first: it writes `HeterogeneousDomainMesh.csv`.

# %% Read the mesh
mesh = bemcs.Mesh.from_csv(
    "HeterogeneousDomainMesh.csv", basis={"slip": "quadratic", "force": "trapezoid"}
)
els = mesh.els
geometry = "antiplane"
mu = 1.0  # reference shear modulus of the kernels
print(mesh)

# layered structure (same as in ConstructMesh.py): used to plot true stresses
zlayer, mulayer = np.array([-2.0, -1.0]), np.array([5.0, 3.0, 1.0])


def mu_field(x, y):
    return np.where(y < zlayer[0], mulayer[0], np.where(y < zlayer[1], mulayer[1], mulayer[2]))


plt.figure(figsize=(10, 4))
bemcs.plot_mesh(mesh, show_normals=False)
plt.xlim(-4, 4)
plt.ylim(-2.5, 0.5)
plt.title(r"mesh; dots: $\mu$ on the $-n$ and $+n$ side of each interface element")
plt.show()

# %% Solve: quadratic slip (general form) on fault and free surface, trapezoids on interfaces
x = bemcs.solve(mesh, geometry=geometry, mu=mu)
quadcoefs = x[mesh.columns("slip", geometry=geometry)]
forcecoefs = x[mesh.columns("force", geometry=geometry)]

# %% Displacement and (true) stress fields inside the domain
fig, axes, fields = bemcs.plot_fields(
    mesh, x, geometry=geometry, mu=mu, xlim=(-4, 4), ylim=(-6, -1e-3), n=(200, 150),
    mu_field=mu_field,
)
plt.show()
u, sx, sy = fields["uz"], fields["sxz"], fields["syz"]

# %% Fields at the free surface
xs = np.linspace(-4, 4, 400)
surface = bemcs.evaluate_fields(mesh, x, xs, np.full_like(xs, -1e-3), geometry=geometry, mu=mu,
                                mu_field=mu_field)
plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.plot(xs, surface["uz"], "-")
plt.ylabel("u")
plt.grid()
plt.title("Surface displacement")
plt.subplot(2, 1, 2)
plt.plot(xs, surface["sxz"], "-", label=r"$\sigma_{xz}$")
plt.plot(xs, surface["syz"], "-", label=r"$\sigma_{yz}$")
plt.xlabel("x")
plt.grid()
plt.legend()
plt.show()

# %% Slip on the fault
xf, yf, slipnodes = mesh.values_on_elements(x, "slip", geometry=geometry, n_eval=10)
on_fault = np.repeat(mesh.bc_type[mesh.elements("slip"), 0] == "s", 10)
plt.figure(figsize=(4, 4))
plt.plot(slipnodes[on_fault], yf[on_fault], ".-")
plt.ylabel("y")
plt.xlabel("slip")
plt.grid()
plt.show()
# %% end of file
