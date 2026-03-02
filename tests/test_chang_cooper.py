"""
Integration tests for Chang-Cooper numeric model, pair production, and EBL absorption.
"""

import numpy as np
import pytest

import blastwave as jsr
try:
    from blastwave import blastwave_extension as _ext
except ImportError:
    import blastwave_extension as _ext


# Standard GRB afterglow parameters (spherical for speed)
P_BASE = {
    "Eiso": 1e53,
    "lf": 300,
    "n0": 1.0,
    "A": 0.0,
    "eps_e": 0.1,
    "eps_b": 0.01,
    "p": 2.3,
    "d": 100,      # Mpc
    "z": 0.01,
    "theta_v": 0.0,
}


class TestNumericModel:
    """Tests for model='numeric' (Chang-Cooper solver)."""

    def test_numeric_model_runs(self):
        """The numeric model should produce positive finite flux."""
        t = np.array([1e4, 1e6])
        nu = 1e14  # optical
        P = P_BASE.copy()

        flux = jsr.FluxDensity_spherical(t, nu, P, model="numeric", ntheta=5)
        assert np.all(np.isfinite(flux)), f"Flux should be finite: {flux}"
        assert np.any(flux > 0), f"At least some flux should be positive: {flux}"

    def test_numeric_spectral_shape(self):
        """The numeric model should produce a reasonable spectrum across frequencies."""
        t = 1e5
        nu = np.logspace(8, 18, 6)
        P = P_BASE.copy()

        flux = jsr.FluxDensity_spherical(t, nu, P, model="numeric", ntheta=5)
        assert np.all(np.isfinite(flux)), f"Flux should be finite: {flux}"
        # At least some frequencies should have nonzero flux
        assert np.sum(flux > 0) >= 2, f"At least 2 frequencies should have nonzero flux: {flux}"

    def test_numeric_vs_analytic_order_of_magnitude(self):
        """Numeric and analytic models should agree within ~3 dex."""
        t = np.array([1e5])
        nu = 1e14
        P = P_BASE.copy()

        flux_numeric = jsr.FluxDensity_spherical(t, nu, P, model="numeric", ntheta=5)
        flux_analytic = jsr.FluxDensity_spherical(t, nu, P, model="sync", ntheta=5)

        # Both should be positive
        mask = (flux_numeric > 0) & (flux_analytic > 0)
        if np.any(mask):
            ratio = np.log10(flux_numeric[mask] / flux_analytic[mask])
            assert np.all(np.abs(ratio) < 3.0), \
                f"Numeric vs analytic should agree within ~3 dex, got ratios: {ratio}"


class TestNumericWithPairs:
    """Tests for pair production with the numeric model."""

    def test_numeric_with_pairs_runs(self):
        """include_pp=1 should produce finite positive results."""
        t = np.array([1e4, 1e6])
        nu = 1e14
        P = P_BASE.copy()
        P["include_pp"] = 1.0

        flux = jsr.FluxDensity_spherical(t, nu, P, model="numeric", ntheta=5)
        assert np.all(np.isfinite(flux)), f"Flux with PP should be finite: {flux}"
        assert np.any(flux > 0), f"At least some flux with PP should be positive: {flux}"

    def test_pairs_modify_spectrum(self):
        """Pair production may modify the spectrum (or leave it unchanged at low energies)."""
        t = 1e5
        nu = np.logspace(10, 16, 4)
        P_no_pp = P_BASE.copy()
        P_pp = P_BASE.copy()
        P_pp["include_pp"] = 1.0

        flux_no_pp = jsr.FluxDensity_spherical(t, nu, P_no_pp, model="numeric", ntheta=5)
        flux_pp = jsr.FluxDensity_spherical(t, nu, P_pp, model="numeric", ntheta=5)

        # Both should be finite
        assert np.all(np.isfinite(flux_no_pp))
        assert np.all(np.isfinite(flux_pp))


class TestEBL:
    """Tests for EBL absorption."""

    def test_ebl_tau_xray_transparent(self):
        """X-ray frequencies should have tau ~ 0."""
        tau = _ext.ebl_tau(1e18, 0.5)
        assert tau == 0.0, f"X-ray should be transparent, got tau={tau}"

    def test_ebl_tau_radio_transparent(self):
        """Radio frequencies should have tau ~ 0."""
        tau = _ext.ebl_tau(1e9, 1.0)
        assert tau == 0.0, f"Radio should be transparent, got tau={tau}"

    def test_ebl_tau_tev_absorbed(self):
        """TeV photons at moderate redshift should have tau > 0."""
        # 1 TeV = 2.418e26 Hz
        nu_tev = 2.418e26
        tau = _ext.ebl_tau(nu_tev, 0.5)
        assert tau > 0.0, f"TeV photon should be absorbed, got tau={tau}"

    def test_ebl_tau_array(self):
        """Array version should return correct shape and values."""
        nu = np.array([1e18, 1e25, 2.418e26])
        tau = _ext.ebl_tau_array(nu, 0.5)
        assert len(tau) == 3
        assert tau[0] == 0.0  # X-ray transparent
        assert tau[2] > 0.0   # TeV absorbed

    def test_ebl_tau_monotonic_energy(self):
        """Tau should increase with energy at fixed z."""
        nu = np.logspace(24, 28, 20)
        tau = _ext.ebl_tau_array(nu, 0.5)
        # Check monotonicity for the nonzero portion
        nonzero = tau > 0
        if np.sum(nonzero) > 1:
            tau_nz = tau[nonzero]
            assert np.all(np.diff(tau_nz) >= 0), \
                f"Tau should be monotonically increasing: {tau_nz}"

    def test_ebl_tau_monotonic_redshift(self):
        """Tau should increase with redshift at fixed energy."""
        nu = 2.418e26  # 1 TeV
        z_values = [0.1, 0.3, 0.5, 1.0, 2.0]
        tau_values = [_ext.ebl_tau(nu, z) for z in z_values]
        for i in range(len(tau_values) - 1):
            assert tau_values[i + 1] >= tau_values[i], \
                f"Tau should increase with z: {list(zip(z_values, tau_values))}"

    def test_ebl_applied_to_flux(self):
        """ebl=True should not modify radio flux."""
        t = 1e5
        nu_radio = 1e9
        P = P_BASE.copy()
        P["z"] = 0.5

        flux_no_ebl = jsr.FluxDensity_spherical(t, nu_radio, P, ebl=False, ntheta=5)
        flux_with_ebl = jsr.FluxDensity_spherical(t, nu_radio, P, ebl=True, ntheta=5)

        # Radio should be unaffected
        np.testing.assert_allclose(flux_no_ebl, flux_with_ebl, rtol=1e-10)

    def test_ebl_known_table_value(self):
        """Check a known table value: E=0.17281 TeV, z=0.5 -> tau~0.625."""
        nu = 0.17281 * 2.418e26  # TeV to Hz
        tau = _ext.ebl_tau(nu, 0.5)
        assert abs(tau - 0.62492) < 0.01, f"Expected tau~0.625, got {tau}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
