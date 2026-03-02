"""
Comparison tests between blastwave and the original jetsimpy.

Each test runs both implementations with identical parameters and asserts
that the relative error is below a threshold (5e-3).  The ~0.2% differences
arise from improved adaptive quadrature (forced refinement of all-zero
intervals) and zero-value interpolation that make blastwave slightly more
accurate than the original C++ implementation in edge cases.
"""

import numpy as np
import pytest

import blastwave
import jetsimpy

# Shared parameters (from jetsimpy examples/quick_start.py)
P = dict(
    Eiso=1e52,
    lf=300,
    theta_c=0.1,
    n0=1,
    A=0,
    eps_e=0.1,
    eps_b=0.01,
    p=2.17,
    theta_v=0.0,
    d=474.33,
    z=0.1,
    s=6,
)

# Sparse time array for speed (20 points, 0.1 to 100 days)
T_SECONDS = np.logspace(-1, 2, 20) * 86400
NU = 1e18  # Hz (X-ray)


def _relative_error(a, b):
    """Element-wise relative error, ignoring near-zero values."""
    mask = np.abs(b) > 1e-30
    return np.max(np.abs(a[mask] - b[mask]) / np.abs(b[mask]))


def test_tophat_flux_density():
    fd_rs = blastwave.FluxDensity_tophat(T_SECONDS, NU, P)
    fd_orig = jetsimpy.FluxDensity_tophat(T_SECONDS, NU, P)

    assert fd_rs.shape == fd_orig.shape
    err = _relative_error(fd_rs, fd_orig)
    assert err < 5e-3, f"Top-hat relative error too large: {err:.6e}"


def test_gaussian_flux_density():
    fd_rs = blastwave.FluxDensity_gaussian(T_SECONDS, NU, P)
    fd_orig = jetsimpy.FluxDensity_gaussian(T_SECONDS, NU, P)

    assert fd_rs.shape == fd_orig.shape
    err = _relative_error(fd_rs, fd_orig)
    assert err < 5e-3, f"Gaussian relative error too large: {err:.6e}"


def test_powerlaw_flux_density():
    fd_rs = blastwave.FluxDensity_powerlaw(T_SECONDS, NU, P)
    fd_orig = jetsimpy.FluxDensity_powerlaw(T_SECONDS, NU, P)

    assert fd_rs.shape == fd_orig.shape
    err = _relative_error(fd_rs, fd_orig)
    assert err < 5e-3, f"Power-law relative error too large: {err:.6e}"
