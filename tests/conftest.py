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
