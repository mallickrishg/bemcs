# bemcs
Continuous slip boundary element models

Solutions to various boundary value problems in linear elasticity using quadratic polynomial slip boundary elements that are continuous in slip gradient at element boundaries.

# Getting started

To set up a development conda environment, run the following commands in the `bemcs` folder.
```
conda config --prepend channels conda-forge
conda env create
conda activate bemcs
pip install --no-use-pep517 -e .
```

# Initial release
[![DOI](https://zenodo.org/badge/651550360.svg)](https://zenodo.org/doi/10.5281/zenodo.10676299)
