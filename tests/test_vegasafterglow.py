"""
Test: blastwave forward-shock flux vs VegasAfterglow reference implementation.

Compares sync_ssa_smooth (blastwave) against VegasAfterglow's default synchrotron
+ SSA model across multiple parameter configurations. Both codes are run in
no-spread mode for a clean comparison of the radiation physics.

Requires VegasAfterglow to be installed / importable.
Run with:  pytest tests/test_vegasafterglow.py -v
"""
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Try to import VegasAfterglow; skip all tests if unavailable
# ---------------------------------------------------------------------------
va = pytest.importorskip("VegasAfterglow")

import blastwave

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MPC_CM = 3.0856775814913673e24   # 1 Mpc in cm
MJY_CGS = 1e-26                  # 1 mJy in erg/cm²/s/Hz

# Common observing setup
TDAY = np.logspace(-2, 3, 30)
TSEC = TDAY * 86400.0
NU_RADIO_100M = 1e8    # 100 MHz
NU_RADIO_1G   = 1e9    # 1 GHz
NU_RADIO_5G   = 5e9    # 5 GHz
NU_OPTICAL    = 1e14   # optical
NU_XRAY      = 2.42e17 # 1 keV

# Fixed geometry / distance
THETA_V = 0.0
D_MPC   = 474.33
Z       = 0.1


def _run_comparison(theta_c, E_iso, Gamma0, n0, eps_e, eps_B, p_val):
    """Run both codes and return dict of {band: (bw_flux, va_flux)} in mJy."""

    # -- VegasAfterglow --
    jet_v   = va.TophatJet(theta_c=theta_c, E_iso=E_iso, Gamma0=Gamma0)
    medium  = va.ISM(n_ism=n0)
    obs     = va.Observer(lumi_dist=D_MPC * MPC_CM, z=Z, theta_obs=THETA_V)
    rad_fwd = va.Radiation(eps_e=eps_e, eps_B=eps_B, p=p_val)
    model_v = va.Model(jet=jet_v, medium=medium, observer=obs, fwd_rad=rad_fwd)

    freqs = np.array([NU_RADIO_100M, NU_RADIO_1G, NU_RADIO_5G, NU_OPTICAL, NU_XRAY])
    result_v = model_v.flux_density_grid(TSEC, freqs)
    va_fluxes = [np.array(result_v.total[i]) / MJY_CGS for i in range(len(freqs))]

    # -- blastwave --
    P = dict(
        Eiso=E_iso, lf=Gamma0, theta_c=theta_c,
        n0=n0, A=0.0,
        eps_e=eps_e, eps_b=eps_B, p=p_val,
        theta_v=THETA_V, d=D_MPC, z=Z, s=6,
    )

    jet_j = blastwave.Jet(
        blastwave.TopHat(theta_c, E_iso, lf0=Gamma0),
        0.0, n0,
        tmin=10.0, tmax=1e10,
        grid=blastwave.ForwardJetRes(theta_c, 129),
        spread=False,
        eps_e=eps_e,
        eps_b=eps_B,
        p_fwd=p_val,
    )

    bw_fluxes = []
    for nu_val in freqs:
        flux = jet_j.FluxDensity(TSEC, nu_val, P, model="sync_ssa_smooth")
        bw_fluxes.append(np.array(flux))

    band_names = ["100MHz", "1GHz", "5GHz", "Optical", "Xray"]
    return {
        name: (bw, va_f)
        for name, bw, va_f in zip(band_names, bw_fluxes, va_fluxes)
    }


def _median_ratio(bw_flux, va_flux):
    """Compute median BW/VA ratio where both are positive and finite."""
    mask = (va_flux > 0) & (bw_flux > 0) & np.isfinite(va_flux) & np.isfinite(bw_flux)
    if mask.sum() == 0:
        return float('nan')
    return float(np.median(bw_flux[mask] / va_flux[mask]))


# ---------------------------------------------------------------------------
# Parameter configurations to test
# ---------------------------------------------------------------------------
CONFIGS = [
    pytest.param(
        dict(theta_c=0.1, E_iso=1e52, Gamma0=300.0, n0=1.0,
             eps_e=0.1, eps_B=0.01, p_val=2.17),
        id="default",
    ),
    pytest.param(
        dict(theta_c=0.1, E_iso=1e52, Gamma0=300.0, n0=0.01,
             eps_e=0.1, eps_B=0.01, p_val=2.17),
        id="n0=0.01",
    ),
    pytest.param(
        dict(theta_c=0.1, E_iso=1e52, Gamma0=300.0, n0=100.0,
             eps_e=0.1, eps_B=0.01, p_val=2.17),
        id="n0=100",
    ),
    pytest.param(
        dict(theta_c=0.1, E_iso=1e52, Gamma0=300.0, n0=1.0,
             eps_e=0.1, eps_B=0.1, p_val=2.17),
        id="eps_B=0.1",
    ),
    pytest.param(
        dict(theta_c=0.1, E_iso=1e52, Gamma0=300.0, n0=1.0,
             eps_e=0.3, eps_B=0.01, p_val=2.17),
        id="eps_e=0.3",
    ),
    pytest.param(
        dict(theta_c=0.1, E_iso=1e52, Gamma0=300.0, n0=1.0,
             eps_e=0.1, eps_B=0.01, p_val=2.5),
        id="p=2.5",
    ),
    pytest.param(
        dict(theta_c=0.1, E_iso=1e52, Gamma0=300.0, n0=1.0,
             eps_e=0.1, eps_B=0.01, p_val=2.05),
        id="p=2.05",
    ),
]

# Tolerance: median ratio must be within this factor of 1.0
# Radio is harder (SSA transition), so we allow more slack.
RADIO_TOL = 0.50   # within factor 1.5
OPTX_TOL  = 0.30   # within factor 1.3


@pytest.mark.parametrize("params", CONFIGS)
def test_forward_shock_flux(params):
    """BW sync_ssa_smooth matches VegasAfterglow within tolerance."""
    result = _run_comparison(**params)

    failures = []
    for band, (bw, va_f) in result.items():
        ratio = _median_ratio(bw, va_f)
        if np.isnan(ratio):
            continue

        tol = RADIO_TOL if band in ("100MHz", "1GHz", "5GHz") else OPTX_TOL
        lo, hi = 1.0 - tol, 1.0 + tol

        if not (lo <= ratio <= hi):
            failures.append(f"{band}: median BW/VA = {ratio:.3f} (expected {lo:.2f}-{hi:.2f})")
        else:
            print(f"  {band}: BW/VA = {ratio:.3f}  OK")

    assert not failures, "Flux mismatch:\n" + "\n".join(failures)
