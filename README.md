# bemcs
Continuous slip boundary element models

Solutions to various boundary value problems in linear elasticity using quadratic polynomial slip boundary elements

# Getting started

To set up a development conda environment, run the following commands in the `bemcs` folder.
```
conda config --prepend channels conda-forge
conda env create
conda activate bemcs
pip install --no-use-pep517 -e .
```