# %%
import numpy as np
import matplotlib.pyplot as plt
import bemcs

# %% [markdown]
# # BEM Problem Description
# Provide an input file with mesh endpoints, boundary condition labels and the shear modulus on both
# sides of every element. Below we consider a standard problem: the displacement and stress of a
# strike-slip fault segment below a free surface.
#
# The boundary conditions are:
# - $t$: traction-free ($\hat{n}\cdot \sigma = 0$) on the top surface (free surface)
# - $s$: imposed slip on the fault segments
#
# The boundary conditions are imposed at the centre of each element. In addition, the slip and its
# gradient are continuous between neighbouring elements, and at the fault tip both are zero (the
# tip element is a taper and carries no boundary condition). With quadratic slip (3 coefficients per
# element) this gives as many equations as unknowns.
#
# ## Input file and mesh geometry
# Columns: `x1, z1, x2, z2, BC_type, value, mu_minus, mu_plus` (the shear modulus on the -n and +n
# side of each element; equal for a homogeneous medium).

# %%
mu = 30  # reference shear modulus (GPa); also the modulus on both sides of every element
mesh = bemcs.Mesh.from_csv("example_mesh.csv", length_scale=1e3)  # coordinates in km
els = mesh.els
geometry = "antiplane"
print(mesh)

plt.figure(figsize=(8, 4))
bemcs.plot_mesh(mesh)
plt.title("mesh: slip elements and normals")
plt.show()

# %% [markdown]
# ## Solve
# `bemcs.solve` assembles the boundary-condition rows and the continuity rows and solves for the
# quadratic slip coefficients (general form: 3 per element).

# %%
quadcoefs = bemcs.solve(mesh, geometry=geometry, mu=mu)

# %% [markdown]
# # Visualization of results
# ## Fault slip distribution

# %%
index = mesh.bc_type[:, 0] == "s"  # fault elements
xs, ys, slip = mesh.values_on_elements(quadcoefs, "slip", geometry=geometry, n_eval=10)
on_fault = np.repeat(index, 10)
plt.figure(figsize=(5, 5))
plt.plot(slip[on_fault], ys[on_fault], ".-")
plt.xlabel("slip (m)")
plt.ylabel("depth (km)")
plt.show()

# %% [markdown]
# ## Internal displacement and stress field
# `bemcs.plot_fields` evaluates the kernels on a grid and plots every component (antiplane: $u_z$,
# $\sigma_{xz}$, $\sigma_{yz}$).

# %%
fig, axes, fields = bemcs.plot_fields(
    mesh, quadcoefs, geometry=geometry, mu=mu, xlim=(-20, 20), ylim=(-20, 4.41), n=(200, 120),
)
plt.show()
u, sx, sy = fields["uz"], fields["sxz"], fields["syz"]
