"""Static checks on the public API of bemcs: names resolve, nothing is
redefined, and the two core modules do not duplicate each other."""

import ast
import builtins
import pathlib
import symtable

import pytest

import bemcs
import bemcs.bemAssembly as GF
import bemcs.bemcs as core

REPO = pathlib.Path(__file__).resolve().parents[1]
MODULES = {"bemcs.bemcs": core, "bemcs.bemAssembly": GF}

# Names defined in BOTH modules (none since stage 3). The test below fails if a
# duplicate appears.
KNOWN_DUPLICATES = set()


def _top_level_functions(module):
    tree = ast.parse(pathlib.Path(module.__file__).read_text())
    return [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]


def _undefined_global_names(path):
    """Names read as globals inside functions/classes but never bound at module level."""
    src = pathlib.Path(path).read_text()
    top = symtable.symtable(src, str(path), "exec")
    module_names = {
        s.get_name()
        for s in top.get_symbols()
        if s.is_assigned() or s.is_imported() or s.is_namespace()
    }
    declared_global_and_assigned = set()
    missing = set()

    def walk(table):
        for sym in table.get_symbols():
            if sym.is_declared_global() and sym.is_assigned():
                declared_global_and_assigned.add(sym.get_name())
        for sym in table.get_symbols():
            name = sym.get_name()
            if (
                table is not top
                and sym.is_global()
                and sym.is_referenced()
                and name not in module_names
                and not hasattr(builtins, name)
            ):
                missing.add(name)
        for child in table.get_children():
            walk(child)

    walk(top)
    return sorted(missing - declared_global_and_assigned)


def _init_exports():
    tree = ast.parse((REPO / "bemcs" / "__init__.py").read_text())
    names = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "bemcs":
            names += [a.name for a in node.names]
    return names


@pytest.mark.parametrize("name", _init_exports())
def test_init_exports_resolve_to_callables(name):
    assert hasattr(bemcs, name), f"bemcs.__init__ exports {name!r} but it is missing"
    assert callable(getattr(bemcs, name))


@pytest.mark.parametrize("modname", MODULES)
def test_no_function_defined_twice_in_module(modname):
    names = _top_level_functions(MODULES[modname])
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert not dupes, f"{modname} redefines {dupes}; the later one silently wins"


@pytest.mark.parametrize("modname", MODULES)
def test_no_undefined_global_names(modname):
    missing = _undefined_global_names(MODULES[modname].__file__)
    assert not missing, f"{modname} references undefined names: {missing}"


def test_duplicates_between_modules_match_known_list():
    actual = set(_top_level_functions(core)) & set(_top_level_functions(GF))
    assert actual == KNOWN_DUPLICATES, (
        f"new duplicates: {sorted(actual - KNOWN_DUPLICATES)}; "
        f"stale entries in KNOWN_DUPLICATES: {sorted(KNOWN_DUPLICATES - actual)}"
    )


def test_no_redundancy_between_modules():
    assert not set(_top_level_functions(core)) & set(_top_level_functions(GF))


# Public functions of bemcs.bemcs that bemcs/__init__.py does not re-export
# (none since stage 3): export new public functions or give them a leading "_".
KNOWN_UNEXPORTED = set()


def test_unexported_public_functions_are_listed():
    """Public functions of bemcs.bemcs that are not re-exported in __init__
    must be a deliberate choice; list them so additions are noticed."""
    exported = set(_init_exports())
    public = {n for n in _top_level_functions(core) if not n.startswith("_")}
    unexported = public - exported
    assert unexported == KNOWN_UNEXPORTED, (
        f"newly unexported: {sorted(unexported - KNOWN_UNEXPORTED)}; "
        f"now exported: {sorted(KNOWN_UNEXPORTED - unexported)}"
    )


def test_core_module_does_not_import_bemAssembly():
    """Dependency is one-way: bemAssembly builds on bemcs.bemcs, never the reverse."""
    tree = ast.parse(pathlib.Path(core.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported |= {a.name for a in node.names}
    assert not any("bemAssembly" in name for name in imported)


def _run_python(code):
    import subprocess
    import sys

    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, timeout=300
    )


def test_import_is_quiet():
    r = _run_python("import bemcs")
    assert r.returncode == 0, r.stderr
    assert r.stdout == ""


def test_package_works_without_numba():
    """numba is optional: the numpy path runs, use_numba=True raises ImportError."""
    code = """
import sys
sys.modules["numba"] = None  # make 'import numba' fail
import numpy as np
import bemcs
assert not bemcs.bemAssembly.NUMBA_AVAILABLE
els = bemcs.initialize_els()
els.x1, els.y1, els.x2, els.y2 = (np.array(v) for v in ([0.0, 1.0], [0.0, 0.0], [1.0, 2.0], [0.0, 0.5]))
bemcs.standardize_els_geometry(els, reorder=False)
U, S, _ = bemcs.get_slip_kernels([0.3], [0.7], els, geometry="planestrain", mu=1.0, nu=0.25)
assert np.isfinite(S).all()
U, S, _ = bemcs.get_force_kernels([0.3], [0.7], els, geometry="antiplane", mu=1.0)
assert np.isfinite(S).all()
try:
    bemcs.get_slip_kernels([0.3], [0.7], els, geometry="antiplane", mu=1.0, use_numba=True)
except ImportError:
    print("ok")
"""
    r = _run_python(code)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"
