"""Sphinx configuration for the bemcs documentation.

Build locally from the repository root:

    pip install -r docs/requirements.txt
    sphinx-build -b html docs docs/_build/html

The site is built and published to GitHub Pages by .github/workflows/docs.yml on every push to main.

Examples: the scripts listed in EXAMPLES are copied into docs/examples/ at build time (with the
files they read) and run as notebooks by MyST-NB (percent-format "# %%" cells, read by
docs/_ext/percent_reader.py). Each runs in its own folder, so relative paths work as in the repo.
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
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

exclude_patterns = ["_build", "_ext", "figures", "conf.py", "**.ipynb_checkpoints"]

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_heading_anchors = 3

# -- Examples ------------------------------------------------------------------
# (folder, script, page title or None to keep the script's own, extra files it reads)
EXAMPLES = [
    ("antiplane", "antiplane101.py", "Antiplane basics: a strike-slip fault below a free surface", ["example_mesh.csv"]),
    ("planestrain", "example_freesurface_fault.py", None, []),
    ("antiplane", "example_Rybicki_bimaterial.py", None, []),
    ("planestrain", "example_subduction_bimaterial.py", None, ["ConstructMesh_subduction.py"]),
    ("", "example_greens_functions.py", None, []),  # examples/ itself; built under docs/examples/general/
]
GITHUB = "https://github.com/mallickrishg/bemcs/blob/main/examples"


def _example_page(text, folder, script, title):
    """The script as a docs page: a title cell (given, or the script's first H1), a link to the
    source, every later H1 demoted to H2 so the page has one title, and a "# %%" before code that
    follows a markdown cell directly."""
    out, in_md, seen_title = [], False, title is not None
    for line in text.splitlines():
        if line.startswith("# %%"):
            in_md = "[markdown]" in line
        elif in_md and line.strip() and not line.startswith("#"):
            # code right after a markdown cell without its own "# %%": start a code cell, or the
            # notebook would take the code as part of the text and never run it
            out.append("# %%")
            in_md = False
        elif in_md and line.startswith("# # "):
            if seen_title:
                line = "# ## " + line[4:]
            seen_title = True
        out.append(line)
    path = f"{folder}/{script}" if folder else script
    source = f"*Source: [`examples/{path}`]({GITHUB}/{path})*"
    head = ["# %% [markdown]", f"# # {title}", "#", f"# {source}", ""] if title else []
    body = "\n".join(out)
    if not title:  # put the source link right after the first markdown cell's title line
        i = body.index("\n", body.index("# # ")) + 1
        body = body[:i] + f"#\n# {source}\n#\n" + body[i:]
    return "\n".join(head) + body + "\n"


def _copy_examples():
    import shutil

    dest = DOCS / "examples"
    for folder, script, title, extra in EXAMPLES:
        src, out = ROOT / "examples" / folder, dest / (folder or "general")
        out.mkdir(parents=True, exist_ok=True)
        page = _example_page((src / script).read_text(), folder, script, title)
        target = out / script
        if not target.exists() or target.read_text() != page:  # keep mtimes, so the cache holds
            target.write_text(page)
        for f in extra:
            shutil.copy2(src / f, out / f)


_copy_examples()
exclude_patterns += [f"examples/{d or 'general'}/{f}" for d, _, _, extra in EXAMPLES for f in extra]  # support files, not pages

nb_custom_formats = {".py": "percent_reader.reads"}
nb_execution_mode = "cache"
nb_execution_timeout = 900
nb_execution_raise_on_error = True
nb_execution_show_tb = True
nb_merge_streams = True

# -- API reference -------------------------------------------------------------
autosummary_generate = True
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": False}
autodoc_typehints = "none"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_ivar = True  # "Attributes" sections as field lists, not duplicates of dataclass fields

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
