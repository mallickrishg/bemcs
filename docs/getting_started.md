# Getting started

## Installation

`bemcs` is not on PyPI yet. Clone the repository and install it in development mode. The conda environment file
sets up all dependencies:

```bash
git clone https://github.com/mallickrishg/bemcs.git
cd bemcs
conda config --prepend channels conda-forge
conda env create
conda activate bemcs
pip install -e .
```

[numba](https://numba.pydata.org) is optional. With it installed, pass `use_numba=True` to the kernel functions
and `bemcs.solve` for faster kernel evaluation on large meshes.

## A first solve

A straight crack of length 2 in an infinite plane-strain medium, loaded by a uniform shear traction:

```python
import numpy as np
import bemcs

# A straight crack from x = -1 to 1, split into 20 elements
n = 20
xs = np.linspace(-1.0, 1.0, n + 1)
els = bemcs.initialize_els()
els.x1, els.y1 = xs[:-1], np.zeros(n)
els.x2, els.y2 = xs[1:], np.zeros(n)
bemcs.standardize_els_geometry(els, reorder=False)

# Uniform shear traction (local shear, normal) on every element; one modulus everywhere
mesh = bemcs.Mesh(
    els,
    basis={"slip": "hat"},
    bc_type=["t_local"] * n,
    bc_value=np.tile([1.0, 0.0], (n, 1)),
    mu_minus=np.ones(n),
    mu_plus=np.ones(n),
)
x = bemcs.solve(mesh, geometry="planestrain", nu=0.25)

# Slip along the crack (slip[:, 0] is x, slip[:, 1] is y), and the stress field around it
xs_eval, ys_eval, slip = mesh.values_on_elements(x, "slip", geometry="planestrain")
fig, axes, fields = bemcs.plot_fields(
    mesh, x, geometry="planestrain", nu=0.25,
    xlim=(-2, 2), ylim=(-2, 2), components=["sxy"],
)
```

What happened:

1. **Geometry.** `initialize_els` and `standardize_els_geometry` build the element container: centres, lengths,
   unit tangents and normals. Element orientation matters; see {doc}`user_guide/concepts`.
2. **Mesh.** `bemcs.Mesh` attaches a boundary condition to every element, the shear modulus on each side, and the
   basis. `"hat"` is the reduced form (one unknown per element and direction); `"quadratic"` (the default) is the
   general form with explicit continuity conditions. Both give the same answer.
3. **Solve.** `bemcs.solve` assembles and solves the collocation system and returns the coefficient vector `x`.
4. **Results.** `values_on_elements` gives slip along the elements; `evaluate_fields` and `plot_fields` give
   displacements and stresses anywhere.

The two crack-tip elements carry no boundary condition: fault tips are clamped (slip and slip gradient go to zero),
so the tip elements only taper the slip. See {doc}`user_guide/solving`.

## Next steps

- {doc}`user_guide/index`: conventions, the mesh CSV format, boundary condition types, heterogeneous materials.
