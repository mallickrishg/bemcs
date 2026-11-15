"""Floating-point precision of the plane-strain element formulas near element
endpoints (tasks.md issue 16).

The formula source (`_quadratic_slip_f`, `_linear_force_g`) is evaluated twice:
in float64, and with 50-digit mpmath numbers by executing the same source with
an mpmath stand-in for numpy. The difference is the rounding error.
"""

import inspect
import textwrap

import mpmath as mp
import numpy as np
import pytest

import bemcs.bemAssembly as GF

mp.mp.dps = 50


class _MpNumpy:
    pi = mp.pi
    log = staticmethod(mp.log)
    arctan = staticmethod(mp.atan)
    sign = staticmethod(mp.sign)


def _mp_version(jitted):
    src = textwrap.dedent(inspect.getsource(jitted.py_func))
    src = "\n".join(l for l in src.splitlines() if not l.startswith("@"))
    ns = {"np": _MpNumpy}
    exec(src, ns)
    return ns[jitted.py_func.__name__]


def _rel_error(fn, shape, *args):
    lo = np.zeros(shape)
    with np.errstate(divide="ignore"):  # numpy scalars: x/0 -> inf, as in numba
        fn.py_func(*[np.float64(a) for a in args], lo)
    hi = np.empty(shape, dtype=object)
    x, *rest = [mp.mpf(a) for a in args]
    if x == rest[1]:  # directly above the endpoint: avoid y / (a - x) with a - x == 0
        x += mp.mpf("1e-45")
    _mp_version(fn)(x, *rest, hi)
    hi = np.array(hi.tolist(), dtype=float)
    return np.abs(lo - hi).max() / np.abs(hi).max()


A, NU, MU = 0.7, 0.27, 1.3
# (distance from the endpoint x = a in units of a, max relative error allowed).
# The thresholds are set from measured behaviour; the pre-revamp formulas were
# O(1) wrong from about 1e-4 a inward. Closer than ~1e-6 a the remaining
# polynomial x log cancellation dominates (tasks.md section 8.4).
DISTANCES = [(1e-1, 1e-12), (1e-2, 1e-10), (1e-3, 1e-8), (1e-4, 1e-6), (1e-5, 5e-5)]


@pytest.mark.parametrize("d, tol", DISTANCES)
@pytest.mark.parametrize("direction", [(0.6, 0.8), (0.0, 1.0), (1.0, -1e-3)])
def test_quadratic_slip_formula_precision_near_endpoint(d, tol, direction):
    x = A + direction[0] * d * A
    y = direction[1] * d * A
    assert _rel_error(GF._quadratic_slip_f, (7, 3), x, y, A, NU) < tol


@pytest.mark.parametrize("d, tol", DISTANCES)
@pytest.mark.parametrize("direction", [(0.6, 0.8), (0.0, 1.0), (1.0, -1e-3)])
def test_linear_force_formula_precision_near_endpoint(d, tol, direction):
    x = A + direction[0] * d * A
    y = direction[1] * d * A
    assert _rel_error(GF._linear_force_g, (2, 2, 5), x, y, A, MU, NU) < tol
