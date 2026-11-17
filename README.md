# bemcs - **B**oundary **E**lement **M**ethod with **C**ontinuous **S**ources

[![DOI](https://zenodo.org/badge/651550360.svg)](https://zenodo.org/doi/10.5281/zenodo.10676299)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-≥3.12-blue.svg)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue?logo=github)](https://mallickrishg.github.io/bemcs/)

A two-dimensional boundary element method (antiplane and plane strain) built on **continuous sources**: quadratic
slip elements whose slip and slip gradient are continuous across nodes, and linear line forces. Continuous slip
removes the stress singularities at element ends that constant-slip elements have, and line forces on material
interfaces model **piecewise-uniform heterogeneous media** with the kernels of a homogeneous reference medium.

- Faults and cracks with prescribed slip, displacement or traction, in global or local components, including kinks
  and triple junctions.
- Material interfaces between regions of different shear modulus, free surfaces, and fault crossings.
- Remote (background) stress fields; true stresses in every material.
- Meshes in Python or from CSV files, with optional connectivity files; numba-accelerated kernels.

>  *Developed for geophysical and tectonics modeling, with a focus on earthquake cycle deformation.*

📘 **Documentation: [https://mallickrishg.github.io/bemcs/](https://mallickrishg.github.io/bemcs/)**

### Where singularities cancel, and where they don't

Each element's stress kernel is singular at the element's end points. When slip and slip gradient are continuous
across a node, the singular parts of neighbouring elements cancel. The effect depends on the geometry at the node:

* **Co-planar elements (straight fault segments): full cancellation.** Stresses stay bounded up to and at the shared
  node.
* **Non-planar elements (kinks, triple junctions): a singularity remains.** The mesh is made of straight elements, so the
  fault geometry is only C⁰: its direction jumps at every kink. Smooth slip removes the singularity due to the slip
  field, but not the one due to the geometry. A weak (logarithmic) stress singularity remains at the node, and its
  strength grows with the angle between neighbouring elements (the discrete curvature). Refining the mesh reduces
  the angles and therefore the singularity, but does not remove it.

The kernels are evaluated so that this cancellation also holds in floating point. Relative to the stress magnitude,
the rounding error at a distance d from a node is about 1e-16·L/d (for element length L), so ~1e-8 even at d = 1e-8 L.
On a kinked fault, a stress that grows as you approach a node is the geometric singularity described above, not
numerical noise.

---
# Getting started

`bemcs` is not on PyPI. Clone the repository and create the conda environment, which also installs `bemcs` in
editable mode:
```
git clone https://github.com/mallickrishg/bemcs.git
cd bemcs
conda config --prepend channels conda-forge
conda env create
conda activate bemcs
```

A crack under uniform shear traction, solved and plotted:
```python
import numpy as np
import bemcs

n = 20
xs = np.linspace(-1.0, 1.0, n + 1)
els = bemcs.initialize_els()
els.x1, els.y1, els.x2, els.y2 = xs[:-1], np.zeros(n), xs[1:], np.zeros(n)
bemcs.standardize_els_geometry(els, reorder=False)

mesh = bemcs.Mesh(els, basis={"slip": "hat"}, bc_type=["t_local"] * n,
                  bc_value=np.tile([1.0, 0.0], (n, 1)), mu_minus=np.ones(n), mu_plus=np.ones(n))
x = bemcs.solve(mesh, geometry="planestrain", nu=0.25)
fig, axes, fields = bemcs.plot_fields(mesh, x, geometry="planestrain", nu=0.25,
                                      xlim=(-2, 2), ylim=(-2, 2), components=["sxy"])
```

Then read 👉 [**Getting started**](https://mallickrishg.github.io/bemcs/getting_started.html) and the [user guide](https://mallickrishg.github.io/bemcs/user_guide/index.html).
Worked [examples](https://mallickrishg.github.io/bemcs/examples_index.html) are in `examples/antiplane` and `examples/planestrain`.

Run the tests with `pytest` (`pytest --no-numba` for the plain-numpy kernels).

---

## 📘 Documentation

The documentation is at **[https://mallickrishg.github.io/bemcs/](https://mallickrishg.github.io/bemcs/)**:

* **[Getting started](https://mallickrishg.github.io/bemcs/getting_started.html)**: installation and a first solve
* **[User guide](https://mallickrishg.github.io/bemcs/user_guide/index.html)**: mesh input and boundary conditions, connectivity files, solving,
  heterogeneous materials
* **[Examples](https://mallickrishg.github.io/bemcs/examples_index.html)**: worked antiplane and plane-strain problems, run on every build
* **[Theory](https://mallickrishg.github.io/bemcs/theory/index.html)**: continuous slip and force sources, the heterogeneous formulation
* **[API reference](https://mallickrishg.github.io/bemcs/api.html)**

The sources are in `docs/`. The site is rebuilt and published on every push to `main`. To build it locally:

```
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
open docs/_build/html/index.html
```

## 🧩 Package Structure

| Module                    | Description                                             |
| ------------------------- | ------------------------------------------------------- |
| `bemcs.bemAssembly`       | `Mesh`, `solve`, field evaluation and plotting, bases, kernels and assembly |
| `bemcs.bemcs`             | Element geometry, closed-form element kernels, point sources                |

The public functions are re-exported at the top level (`bemcs.Mesh`, `bemcs.solve`, ...); see the
[API reference](https://mallickrishg.github.io/bemcs/api.html).

---

## 📄 Citation

If you use **bemcs** in your research, please cite:

> **Mallick, R. & Meade, B. J.** (2025). *Smooth slip is all you need: A singularity-free boundary element method for fault slip problems.*
> Computers & Geosciences, 196, 105820.
> [https://doi.org/10.1016/j.cageo.2024.105820](https://doi.org/10.1016/j.cageo.2024.105820)

---
