import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

import bemcs


@pytest.fixture(scope="session")
def els_curved():
    """Four connected, non-collinear elements (a gently curved fault)."""
    els = bemcs.initialize_els()
    els.x1 = np.array([-2.0, 0.0, 2.0, 4.0])
    els.x2 = np.array([0.0, 2.0, 4.0, 6.0])
    els.y1 = np.array([0.0, 0.3, 0.2, -0.4])
    els.y2 = np.array([0.3, 0.2, -0.4, -1.0])
    bemcs.standardize_els_geometry(els, reorder=False)
    return els


@pytest.fixture(scope="session")
def obs_points():
    """Observation points away from the element end points."""
    xo = np.linspace(-3.0, 7.0, 11) + 0.13
    yo = np.linspace(-2.0, 1.0, 11)
    return xo, yo


# --- numba: the default backend when installed (cold compile ~7 s, then cached on disk) ---
# `pytest --no-numba` runs everything on the plain-numpy path (BEMCS_NUMBA=0) and skips the
# tests that compare the two backends.


def pytest_addoption(parser):
    parser.addoption(
        "--no-numba",
        action="store_true",
        default=False,
        help="use the plain-numpy kernels (BEMCS_NUMBA=0) and skip numba-vs-numpy tests",
    )


def pytest_configure(config):
    if config.getoption("--no-numba"):
        os.environ["BEMCS_NUMBA"] = "0"


def pytest_collection_modifyitems(config, items):
    from bemcs.bemAssembly import NUMBA_AVAILABLE

    if config.getoption("--no-numba") or not NUMBA_AVAILABLE:
        reason = "--no-numba" if NUMBA_AVAILABLE else "numba is not installed"
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if "numba" in item.keywords:
                item.add_marker(skip)
