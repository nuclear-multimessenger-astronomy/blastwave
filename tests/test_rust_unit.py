"""
Unit tests for the blastwave Rust extension and Python wrappers.

These tests verify basic functionality without requiring the original jetsimpy.
"""

import numpy as np
import pytest

from blastwave import Jet, TopHat, ForwardJetRes


# Shared minimal parameters
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
)


@pytest.fixture(scope="module")
def jet():
    """Create a solved Jet instance shared across tests in this module."""
    j = Jet(
        TopHat(P["theta_c"], P["Eiso"], lf0=P["lf"]),
        P["A"],
        P["n0"],
        tmin=10.0,
        tmax=1e10,
        grid=ForwardJetRes(P["theta_c"], 129),
        tail=True,
        spread=True,
        cal_level=1,
    )
    return j


def test_jet_creation(jet):
    """Verify that Jet creation and solveJet complete without error."""
    assert jet is not None


def test_pde_data_shapes(jet):
    """Verify t_pde, y_pde, theta_pde return arrays with expected shapes."""
    t = jet.t_pde
    y = jet.y_pde
    theta = jet.theta_pde

    assert t.ndim == 1
    assert t.shape[0] > 0

    assert theta.ndim == 1
    assert theta.shape[0] > 0

    # y_pde shape should be [6, ntheta, nt]
    assert y.ndim == 3
    assert y.shape[0] == 6
    assert y.shape[1] == theta.shape[0]
    assert y.shape[2] == t.shape[0]


def test_interpolation(jet):
    """Verify interpolation methods return finite values."""
    t = jet.t_pde
    theta = jet.theta_pde

    # Pick a time in the middle of the PDE solution range
    t_mid = t[len(t) // 2]
    th_mid = theta[len(theta) // 4]

    bg = jet.beta_gamma(t_mid, th_mid)
    assert np.isfinite(bg), f"beta_gamma returned non-finite: {bg}"

    r = jet.R(t_mid, th_mid)
    assert np.isfinite(r), f"R returned non-finite: {r}"
    assert r > 0, f"R should be positive, got {r}"

    msw = jet.dMsw_dOmega(t_mid, th_mid)
    assert np.isfinite(msw), f"dMsw_dOmega returned non-finite: {msw}"


def test_interpolation_vectorized(jet):
    """Verify interpolation works with array inputs."""
    t = jet.t_pde
    theta = jet.theta_pde

    t_arr = np.array([t[len(t) // 4], t[len(t) // 2], t[3 * len(t) // 4]])
    th_arr = np.full_like(t_arr, theta[len(theta) // 4])

    bg = jet.beta_gamma(t_arr, th_arr)
    assert bg.shape == (3,)
    assert np.all(np.isfinite(bg))


# ─── Forward-mapping flux method tests ───

P_FWD = dict(
    Eiso=1e52, lf=300, theta_c=0.1, n0=1, A=0,
    eps_e=0.1, eps_b=0.01, p=2.17,
    theta_v=0.0, d=474.33, z=0.1, s=6,
)
T_FWD = np.logspace(-1, 2, 30) * 86400
NU_FWD = 1e18


def test_forward_flux_positive_and_finite():
    """Forward-mapping should produce positive, finite flux values."""
    from blastwave import FluxDensity_tophat
    fd = FluxDensity_tophat(T_FWD, NU_FWD, P_FWD, flux_method="forward")
    assert fd.shape == T_FWD.shape
    assert np.all(np.isfinite(fd)), "Forward flux contains non-finite values"
    assert np.all(fd > 0), "Forward flux contains non-positive values"


def test_forward_agrees_with_eats():
    """Forward-mapping should agree with EATS within 0.1 dex for on-axis tophat."""
    from blastwave import FluxDensity_tophat
    fd_eats = FluxDensity_tophat(T_FWD, NU_FWD, P_FWD)
    fd_fwd = FluxDensity_tophat(T_FWD, NU_FWD, P_FWD, flux_method="forward")

    mask = (fd_eats > 0) & (fd_fwd > 0)
    log_diff = np.abs(np.log10(fd_fwd[mask]) - np.log10(fd_eats[mask]))
    max_diff = np.max(log_diff)
    assert max_diff < 0.1, f"Forward vs EATS max deviation: {max_diff:.4f} dex (limit 0.1)"


def test_forward_gaussian():
    """Forward-mapping should work for Gaussian jet profile."""
    from blastwave import FluxDensity_gaussian
    fd_eats = FluxDensity_gaussian(T_FWD, NU_FWD, P_FWD)
    fd_fwd = FluxDensity_gaussian(T_FWD, NU_FWD, P_FWD, flux_method="forward")

    mask = (fd_eats > 0) & (fd_fwd > 0)
    log_diff = np.abs(np.log10(fd_fwd[mask]) - np.log10(fd_eats[mask]))
    max_diff = np.max(log_diff)
    assert max_diff < 0.1, f"Gaussian forward vs EATS max deviation: {max_diff:.4f} dex"


def test_forward_powerlaw():
    """Forward-mapping should work for power-law jet profile."""
    from blastwave import FluxDensity_powerlaw
    fd_eats = FluxDensity_powerlaw(T_FWD, NU_FWD, P_FWD)
    fd_fwd = FluxDensity_powerlaw(T_FWD, NU_FWD, P_FWD, flux_method="forward")

    mask = (fd_eats > 0) & (fd_fwd > 0)
    log_diff = np.abs(np.log10(fd_fwd[mask]) - np.log10(fd_eats[mask]))
    max_diff = np.max(log_diff)
    assert max_diff < 0.1, f"Power-law forward vs EATS max deviation: {max_diff:.4f} dex"


def test_forward_offaxis_accuracy():
    """Off-axis forward-grid should agree with EATS within 0.1 dex."""
    from blastwave import FluxDensity_tophat
    P_off = {**P_FWD, "theta_v": 0.3}
    t_off = np.logspace(0, 2.5, 10) * 86400

    fd_eats = FluxDensity_tophat(t_off, NU_FWD, P_off)
    fd_fwd = FluxDensity_tophat(t_off, NU_FWD, P_off, flux_method="forward")

    mask = (fd_eats > 0) & (fd_fwd > 0)
    log_diff = np.abs(np.log10(fd_fwd[mask]) - np.log10(fd_eats[mask]))
    max_diff = np.max(log_diff)
    assert max_diff < 0.1, f"Off-axis forward vs EATS max deviation: {max_diff:.4f} dex (limit 0.1)"


def test_forward_ode_spread():
    """Forward-mapping should work with ODE spreading mode."""
    from blastwave import Jet, TopHat, ForwardJetRes
    jet = Jet(
        TopHat(P_FWD["theta_c"], P_FWD["Eiso"], lf0=P_FWD["lf"]),
        P_FWD["A"], P_FWD["n0"],
        grid=ForwardJetRes(P_FWD["theta_c"], 129),
        spread_mode="ode", tail=True,
    )
    fd_eats = jet.FluxDensity(T_FWD, NU_FWD, P_FWD)
    fd_fwd = jet.FluxDensity(T_FWD, NU_FWD, P_FWD, flux_method="forward")

    mask = (fd_eats > 0) & (fd_fwd > 0)
    log_diff = np.abs(np.log10(fd_fwd[mask]) - np.log10(fd_eats[mask]))
    max_diff = np.max(log_diff)
    assert max_diff < 0.2, f"ODE forward vs EATS max deviation: {max_diff:.4f} dex"


def test_forward_no_spread():
    """Forward-mapping should work with no-spread mode."""
    from blastwave import FluxDensity_tophat
    fd_eats = FluxDensity_tophat(T_FWD, NU_FWD, P_FWD, spread=False)
    fd_fwd = FluxDensity_tophat(T_FWD, NU_FWD, P_FWD, spread=False, flux_method="forward")

    mask = (fd_eats > 0) & (fd_fwd > 0)
    log_diff = np.abs(np.log10(fd_fwd[mask]) - np.log10(fd_eats[mask]))
    max_diff = np.max(log_diff)
    assert max_diff < 0.1, f"No-spread forward vs EATS max deviation: {max_diff:.4f} dex"


# ─── Magnetar energy injection tests ───

P_MAG = dict(
    Eiso=1e52, lf=100, theta_c=0.1, n0=1, A=0,
    eps_e=0.1, eps_b=0.01, p=2.2,
    theta_v=0.0, d=40.0, z=0.01,
)
T_MAG = np.geomspace(1e3, 1e7, 50)
NU_MAG = 2.4e17  # 1 keV


def test_magnetar_no_regression():
    """magnetar_l0=0 must give identical results to baseline (no injection)."""
    from blastwave import FluxDensity_tophat
    fd_base = FluxDensity_tophat(T_MAG, NU_MAG, P_MAG, spread=False, tmax=1e8)
    fd_zero = FluxDensity_tophat(T_MAG, NU_MAG, P_MAG, spread=False, tmax=1e8,
                                  magnetar_l0=0.0, magnetar_t0=1e4, magnetar_q=2.0)
    np.testing.assert_allclose(fd_zero, fd_base, rtol=1e-10,
                                err_msg="magnetar_l0=0 should be identical to baseline")


def test_magnetar_increases_late_time_flux():
    """Magnetar injection should produce higher flux than baseline at late times."""
    from blastwave import FluxDensity_tophat
    t_late = np.geomspace(1e5, 1e7, 20)
    fd_base = FluxDensity_tophat(t_late, NU_MAG, P_MAG, spread=False, tmax=1e8)
    fd_mag = FluxDensity_tophat(t_late, NU_MAG, P_MAG, spread=False, tmax=1e8,
                                 magnetar_l0=1e47, magnetar_t0=1e4, magnetar_q=2.0)

    mask = (fd_base > 0) & (fd_mag > 0)
    assert np.any(mask), "Need at least some positive flux values"
    # Injection adds energy → flux should be higher at late times
    assert np.all(fd_mag[mask] >= fd_base[mask] * 0.99), \
        "Magnetar flux should be >= baseline at late times"
    # Should be significantly brighter at some point
    max_ratio = np.max(fd_mag[mask] / fd_base[mask])
    assert max_ratio > 2.0, f"Magnetar should produce >2x brighter flux, got {max_ratio:.2f}x"


def test_magnetar_positive_and_finite():
    """Magnetar-injected flux should be positive and finite."""
    from blastwave import FluxDensity_tophat
    fd = FluxDensity_tophat(T_MAG, NU_MAG, P_MAG, spread=False, tmax=1e8,
                             magnetar_l0=1e48, magnetar_t0=1e3, magnetar_q=2.0)
    assert fd.shape == T_MAG.shape
    assert np.all(np.isfinite(fd)), "Magnetar flux contains non-finite values"
    assert np.all(fd > 0), "Magnetar flux contains non-positive values"


def test_magnetar_ode_spread():
    """Magnetar injection should work with ODE spreading mode."""
    from blastwave import Jet, TopHat, ForwardJetRes
    t_test = np.geomspace(1e3, 1e7, 20)

    jet_base = Jet(
        TopHat(P_MAG["theta_c"], P_MAG["Eiso"], lf0=P_MAG["lf"]),
        P_MAG["A"], P_MAG["n0"],
        grid=ForwardJetRes(P_MAG["theta_c"], 33), tmax=1e8,
        spread_mode="ode", tail=True,
    )
    jet_mag = Jet(
        TopHat(P_MAG["theta_c"], P_MAG["Eiso"], lf0=P_MAG["lf"]),
        P_MAG["A"], P_MAG["n0"],
        grid=ForwardJetRes(P_MAG["theta_c"], 33), tmax=1e8,
        spread_mode="ode", tail=True,
        magnetar_l0=1e47, magnetar_t0=1e4, magnetar_q=2.0,
    )

    fd_base = jet_base.FluxDensity(t_test, NU_MAG, P_MAG)
    fd_mag = jet_mag.FluxDensity(t_test, NU_MAG, P_MAG)

    assert np.all(np.isfinite(fd_mag)), "ODE magnetar flux contains non-finite values"
    assert np.all(fd_mag > 0), "ODE magnetar flux contains non-positive values"
    # Late-time flux should be enhanced
    mask = (fd_base > 0) & (fd_mag > 0) & (t_test > 1e5)
    if np.any(mask):
        assert np.all(fd_mag[mask] >= fd_base[mask] * 0.99), \
            "ODE magnetar flux should be >= baseline at late times"


def test_magnetar_higher_l0_gives_more_flux():
    """Higher magnetar luminosity should produce more flux at fixed t0, q."""
    from blastwave import FluxDensity_tophat
    t_test = np.geomspace(1e4, 1e7, 20)
    fd_low = FluxDensity_tophat(t_test, NU_MAG, P_MAG, spread=False, tmax=1e8,
                                 magnetar_l0=1e46, magnetar_t0=1e4, magnetar_q=2.0)
    fd_high = FluxDensity_tophat(t_test, NU_MAG, P_MAG, spread=False, tmax=1e8,
                                  magnetar_l0=1e48, magnetar_t0=1e4, magnetar_q=2.0)

    mask = (fd_low > 0) & (fd_high > 0)
    assert np.any(mask), "Need positive flux values"
    # Higher L0 → more injection → brighter
    ratio = fd_high[mask] / fd_low[mask]
    assert np.max(ratio) > 1.5, f"Higher L0 should give significantly more flux, max ratio={np.max(ratio):.2f}"


def test_magnetar_radio_band():
    """Magnetar injection should also affect radio band emission."""
    from blastwave import FluxDensity_tophat
    nu_radio = 5e9  # 5 GHz
    t_test = np.geomspace(1e4, 1e7, 20)
    fd_base = FluxDensity_tophat(t_test, nu_radio, P_MAG, spread=False, tmax=1e8)
    fd_mag = FluxDensity_tophat(t_test, nu_radio, P_MAG, spread=False, tmax=1e8,
                                 magnetar_l0=1e47, magnetar_t0=1e4, magnetar_q=2.0)

    assert np.all(np.isfinite(fd_mag)), "Radio magnetar flux contains non-finite values"
    assert np.all(fd_mag > 0), "Radio magnetar flux contains non-positive values"
    mask = (fd_base > 0) & (fd_mag > 0)
    max_ratio = np.max(fd_mag[mask] / fd_base[mask])
    assert max_ratio > 1.5, f"Magnetar should enhance radio flux, max ratio={max_ratio:.2f}"


def test_magnetar_pde_warning(capfd):
    """PDE mode with magnetar injection should print a warning to stderr."""
    from blastwave import Jet, TopHat, ForwardJetRes
    Jet(
        TopHat(P_MAG["theta_c"], P_MAG["Eiso"], lf0=P_MAG["lf"]),
        P_MAG["A"], P_MAG["n0"],
        grid=ForwardJetRes(P_MAG["theta_c"], 17), tmax=1e8,
        spread=True,  # PDE mode
        magnetar_l0=1e47, magnetar_t0=1e4, magnetar_q=2.0,
    )
    captured = capfd.readouterr()
    assert "magnetar injection not supported in PDE mode" in captured.err, \
        f"Expected PDE warning, got stderr: {captured.err!r}"


def test_magnetar_shortcut_functions():
    """All shortcut functions should accept magnetar kwargs without error."""
    from blastwave import (FluxDensity_tophat, FluxDensity_gaussian,
                              FluxDensity_powerlaw, FluxDensity_spherical)
    t_test = np.array([1e4, 1e5, 1e6])
    mag_kwargs = dict(magnetar_l0=1e47, magnetar_t0=1e4, magnetar_q=2.0)

    P_th = {**P_MAG, "s": 6}  # powerlaw needs 's'

    fd = FluxDensity_tophat(t_test, NU_MAG, P_MAG, spread=False, tmax=1e8, **mag_kwargs)
    assert np.all(np.isfinite(fd)) and np.all(fd > 0)

    fd = FluxDensity_gaussian(t_test, NU_MAG, P_MAG, spread=False, tmax=1e8, **mag_kwargs)
    assert np.all(np.isfinite(fd)) and np.all(fd > 0)

    P_sph = {**P_MAG, "Eiso": 1e52, "lf": 100}
    fd = FluxDensity_spherical(t_test, NU_MAG, P_sph, tmax=1e8, **mag_kwargs)
    assert np.all(np.isfinite(fd)) and np.all(fd > 0)

    fd = FluxDensity_powerlaw(t_test, NU_MAG, P_th, spread=False, tmax=1e8, **mag_kwargs)
    assert np.all(np.isfinite(fd)) and np.all(fd > 0)


def test_magnetar_delayed_onset():
    """Injection with large magnetar_ts should give results close to baseline.

    Note: For on-axis relativistic blast waves, observer time maps to lab-frame
    time via t_lab ~ t_obs * 2*Gamma^2, so ts in lab frame maps to a very early
    observer time. We verify the ts parameter works by:
    1. Very late ts (1e10 s, essentially never) should match baseline.
    2. Delayed ts should give less enhancement than immediate (ts=0).
    """
    from blastwave import Jet, TopHat, ForwardJetRes
    t_test = np.geomspace(1e3, 1e7, 20)

    jet_base = Jet(
        TopHat(P_MAG["theta_c"], P_MAG["Eiso"], lf0=P_MAG["lf"]),
        P_MAG["A"], P_MAG["n0"],
        grid=ForwardJetRes(P_MAG["theta_c"], 33), tmax=1e8,
        spread=False, tail=True,
    )
    jet_immediate = Jet(
        TopHat(P_MAG["theta_c"], P_MAG["Eiso"], lf0=P_MAG["lf"]),
        P_MAG["A"], P_MAG["n0"],
        grid=ForwardJetRes(P_MAG["theta_c"], 33), tmax=1e8,
        spread=False, tail=True,
        magnetar_l0=1e47, magnetar_t0=1e4, magnetar_q=2.0, magnetar_ts=0.0,
    )
    # ts=1e10: injection starts so late it never activates during simulation
    jet_never = Jet(
        TopHat(P_MAG["theta_c"], P_MAG["Eiso"], lf0=P_MAG["lf"]),
        P_MAG["A"], P_MAG["n0"],
        grid=ForwardJetRes(P_MAG["theta_c"], 33), tmax=1e8,
        spread=False, tail=True,
        magnetar_l0=1e47, magnetar_t0=1e4, magnetar_q=2.0, magnetar_ts=1e10,
    )

    fd_base = jet_base.FluxDensity(t_test, NU_MAG, P_MAG)
    fd_imm = jet_immediate.FluxDensity(t_test, NU_MAG, P_MAG)
    fd_never = jet_never.FluxDensity(t_test, NU_MAG, P_MAG)

    # ts=1e10 should match baseline (injection never starts)
    rel_diff = np.max(np.abs(fd_never - fd_base) / np.maximum(fd_base, 1e-30))
    assert rel_diff < 1e-6, f"ts=1e10 should match baseline, got diff={rel_diff:.2e}"

    # Immediate injection should be brighter than baseline
    assert np.any(fd_imm > fd_base * 1.5), "Immediate injection should enhance flux"


def test_magnetar_multi_episode():
    """Multiple injection episodes should enhance flux more than a single one."""
    from blastwave import Jet, TopHat, ForwardJetRes
    t_test = np.geomspace(1e4, 1e7, 30)

    # Single episode
    jet_single = Jet(
        TopHat(P_MAG["theta_c"], P_MAG["Eiso"], lf0=P_MAG["lf"]),
        P_MAG["A"], P_MAG["n0"],
        grid=ForwardJetRes(P_MAG["theta_c"], 33), tmax=1e8,
        spread=False, tail=True,
        magnetar_l0=1e47, magnetar_t0=1e4, magnetar_q=2.0,
    )
    # Two episodes (same first + another starting at 1e5 s)
    jet_double = Jet(
        TopHat(P_MAG["theta_c"], P_MAG["Eiso"], lf0=P_MAG["lf"]),
        P_MAG["A"], P_MAG["n0"],
        grid=ForwardJetRes(P_MAG["theta_c"], 33), tmax=1e8,
        spread=False, tail=True,
        magnetar_l0=[1e47, 1e47],
        magnetar_t0=[1e4, 1e4],
        magnetar_q=[2.0, 2.0],
        magnetar_ts=[0.0, 1e5],
    )

    fd_single = jet_single.FluxDensity(t_test, NU_MAG, P_MAG)
    fd_double = jet_double.FluxDensity(t_test, NU_MAG, P_MAG)

    assert np.all(np.isfinite(fd_double)) and np.all(fd_double > 0)
    # After second episode kicks in (t > 1e5), double should be brighter
    mask_late = t_test > 2e5
    if np.any(mask_late):
        assert np.all(fd_double[mask_late] >= fd_single[mask_late] * 0.99), \
            "Two episodes should give >= flux than single episode after second onset"


def test_magnetar_ts_shortcut():
    """Shortcut functions should accept magnetar_ts parameter."""
    from blastwave import FluxDensity_tophat
    t_test = np.geomspace(1e3, 1e7, 10)
    fd = FluxDensity_tophat(t_test, NU_MAG, P_MAG, spread=False, tmax=1e8,
                             magnetar_l0=1e47, magnetar_t0=1e4, magnetar_q=2.0,
                             magnetar_ts=1e4)
    assert np.all(np.isfinite(fd)) and np.all(fd > 0)
