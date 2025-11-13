# bemcs - **B**oundary **E**lement **M**ethod with **C**ontinuous **S**ources

[![DOI](https://zenodo.org/badge/651550360.svg)](https://zenodo.org/doi/10.5281/zenodo.10676299)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-≥3.9-blue.svg)
[![Docs](https://img.shields.io/badge/docs-Wiki-blue?logo=github)](https://github.com/mallickrishg/bemcs/wiki)

Solutions to various boundary value problems in linear elasticity using quadratic polynomial slip boundary elements that are continuous in slip gradient at element boundaries. We also offer linear polynomial force boundary elements or distributed forces as constant sources over triangular elements. 

>  *Developed for geophysical and tectonics modeling, with a focus on earthquake cycle deformation.*

---
# Getting started

To set up a development conda environment, run the following commands in the `bemcs` folder.
```
conda config --prepend channels conda-forge
conda env create
conda activate bemcs
pip install --no-use-pep517 -e .
```

For a full step-by-step walkthrough, see the

👉 [**Tutorial: Setting up and solving an elastostatic plane strain problem**](https://github.com/mallickrishg/bemcs/wiki/Tutorial-on-setting-up-and-solving-an-elastostatic-problem-using-the-Boundary-Element-Method)

👉 [**Tutorial: Solving an antiplane problem in a heterogeneous elastic medium**](https://github.com/mallickrishg/bemcs/wiki/Tutorial-on-antiplane-slip-In-a-heterogeneous-elastic-medium)

---

## 📘 Documentation

* **[Wiki Home](https://github.com/mallickrishg/bemcs/wiki)** — overview, setup, and examples
* **[Elastostatic BEM tutorial](https://github.com/mallickrishg/bemcs/wiki/Tutorial-on-setting-up-and-solving-an-elastostatic-problem-using-the-Boundary-Element-Method)**
* (More tutorials coming soon…)

## 🧩 Package Structure

| Module                    | Description                                             |
| ------------------------- | ------------------------------------------------------- |
| `bemcs.bemcs`             | Core classes and functions for BEM modeling             |
| `bemcs.bemAssembly`       | Helper routines for assembling system matrices          |

---

## 📄 Citation

If you use **bemcs** in your research, please cite:

> **Mallick, R. & Meade, B. J.** (2025). *Smooth slip is all you need: A singularity-free boundary element method for fault slip problems.*
> Computers & Geosciences, 196, 105820.
> [https://doi.org/10.1016/j.cageo.2024.105820](https://doi.org/10.1016/j.cageo.2024.105820)

---
