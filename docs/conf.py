"""Sphinx configuration for the bemcs documentation.

Build locally from the repository root:

    pip install -r docs/requirements.txt
    sphinx-build -b html docs docs/_build/html

The site is built and published to GitHub Pages by .github/workflows/docs.yml on every push to main.

Examples are not in the docs yet (which examples to share is undecided). To add them later:
switch "myst_parser" to "myst_nb", copy the chosen example folders into docs/examples/ at build
time, and read the percent-format scripts as notebooks with
    nb_custom_formats = {".py": "percent_reader.reads"}
(docs/_ext/percent_reader.py keeps commented-out IPython magics as comments and adds kernel
metadata). Each example then needs a Markdown title cell, and examples that read generated CSV
meshes need those generated first.
"""

import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DOCS / "_ext"))

# quadpy is only used inside the triangle-source functions; the docs build (CI) does not install it
try:
    import quadpy  # noqa: F401
except ImportError:
    import types

    sys.modules["quadpy"] = types.ModuleType("quadpy")

import bemcs  # noqa: E402

# -- Project -------------------------------------------------------------------
project = "bemcs"
author = "Rishav Mallick and Brendan Meade"
copyright = "2023-2026, Rishav Mallick and Brendan Meade"
version = release = bemcs.__version__

# -- Extensions ----------------------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "_ext", "**.ipynb_checkpoints"]

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_heading_anchors = 3

# -- API reference -------------------------------------------------------------
autosummary_generate = True
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": False}
autodoc_typehints = "none"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- HTML ----------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "bemcs"
html_theme_options = {
    "github_url": "https://github.com/mallickrishg/bemcs",
    "navigation_with_keys": False,
    "show_toc_level": 2,
}
html_static_path = []
