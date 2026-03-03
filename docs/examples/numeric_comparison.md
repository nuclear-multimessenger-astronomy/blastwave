# Numeric Model Comparison: Chang-Cooper vs Analytic

This example compares the analytic synchrotron model (`model="sync"`) with the full numeric Chang-Cooper electron distribution solver (`model="numeric"`) to validate consistency and highlight where the numeric model captures additional physics.

## Background

The default `"sync"` model uses the standard analytic broken-power-law synchrotron spectrum (Sari, Piran & Narayan 1998). The `"numeric"` model instead solves the electron energy distribution using a Chang-Cooper finite-difference scheme, naturally capturing:

- Smooth spectral transitions (no sharp breaks)
- Synchrotron cooling pile-up near the cooling break
- Pair production at very high energies
- Accurate spectral shapes in the deep cooling regime

For most afterglow applications the two models agree to within a few percent, but the numeric model becomes important for precision spectral fitting and at extreme frequencies.

## Physical parameters

We use a standard on-axis top-hat jet:

| Parameter | Value | Notes |
|-----------|-------|-------|
| \(E_\mathrm{iso}\) | \(10^{53}\) erg | Isotropic-equivalent energy |
| \(\Gamma_0\) | 300 | Initial Lorentz factor |
| \(\theta_c\) | 0.1 rad | Jet half-opening angle |
| \(n_0\) | 1.0 cm\(^{-3}\) | ISM density |
| \(\varepsilon_e\) | 0.1 | Electron energy fraction |
| \(\varepsilon_B\) | 0.01 | Magnetic energy fraction |
| \(p\) | 2.3 | Electron spectral index |
| \(d\) | 1000 Mpc | Luminosity distance |
| \(z\) | 0.2 | Redshift |

## Computing the models

```python
import numpy as np
from blastwave import FluxDensity_tophat

DAY = 86400.0

P = {
    "Eiso":    1e53,
    "lf":      300.0,
    "theta_c": 0.1,
    "A":       0.0,
    "n0":      1.0,
    "eps_e":   0.1,
    "eps_b":   0.01,
    "p":       2.3,
    "theta_v": 0.0,
    "d":       1000.0,
    "z":       0.2,
}

t_radio = np.geomspace(0.1 * DAY, 1000.0 * DAY, 150)
t_xray = np.geomspace(100.0, 300.0 * DAY, 150)

# Analytic synchrotron
F_radio_sync = FluxDensity_tophat(t_radio, 3e9 * np.ones_like(t_radio), P,
                                   tmin=1.0, tmax=1500 * DAY, model="sync")
F_xray_sync = FluxDensity_tophat(t_xray, 2.418e17 * np.ones_like(t_xray), P,
                                  tmin=1.0, tmax=1500 * DAY, model="sync")

# Chang-Cooper numeric
F_radio_numeric = FluxDensity_tophat(t_radio, 3e9 * np.ones_like(t_radio), P,
                                      tmin=1.0, tmax=1500 * DAY, model="numeric")
F_xray_numeric = FluxDensity_tophat(t_xray, 2.418e17 * np.ones_like(t_xray), P,
                                     tmin=1.0, tmax=1500 * DAY, model="numeric")
```

## Plotting

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

# Panel 1: Overlaid light curves
ax1.plot(t_radio / DAY, F_radio_sync, '-', color='C0', lw=2, label='Radio sync')
ax1.plot(t_radio / DAY, F_radio_numeric, '--', color='C0', lw=2, label='Radio numeric')
ax1.plot(t_xray / DAY, F_xray_sync, '-', color='C3', lw=2, label='X-ray sync')
ax1.plot(t_xray / DAY, F_xray_numeric, '--', color='C3', lw=2, label='X-ray numeric')
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlabel('Time (days)'); ax1.set_ylabel('Flux density (mJy)')
ax1.set_title('Light Curves'); ax1.legend()

# Panel 2: Fractional residuals
resid_radio = (F_radio_numeric - F_radio_sync) / F_radio_sync * 100
resid_xray = (F_xray_numeric - F_xray_sync) / F_xray_sync * 100
ax2.plot(t_radio / DAY, resid_radio, '-', color='C0', lw=2, label='Radio 3 GHz')
ax2.plot(t_xray / DAY, resid_xray, '-', color='C3', lw=2, label='X-ray 1 keV')
ax2.axhline(0, color='gray', ls='--', lw=1)
ax2.set_xscale('log')
ax2.set_xlabel('Time (days)'); ax2.set_ylabel('Residual (%)')
ax2.set_title('(Numeric − Sync) / Sync'); ax2.legend()

# Panel 3: High-energy comparison (10 keV)
nu_high = 2.418e18
t_high = np.geomspace(100.0, 100.0 * DAY, 100)
F_high_sync = FluxDensity_tophat(t_high, nu_high * np.ones_like(t_high), P,
                                  tmin=1.0, tmax=200 * DAY, model="sync")
F_high_numeric = FluxDensity_tophat(t_high, nu_high * np.ones_like(t_high), P,
                                     tmin=1.0, tmax=200 * DAY, model="numeric")
ax3.plot(t_high / DAY, F_high_sync, '-', color='C4', lw=2, label='Sync (10 keV)')
ax3.plot(t_high / DAY, F_high_numeric, '--', color='C4', lw=2, label='Numeric (10 keV)')
ax3.set_xscale('log'); ax3.set_yscale('log')
ax3.set_xlabel('Time (days)'); ax3.set_ylabel('Flux density (mJy)')
ax3.set_title('High Energy (10 keV)'); ax3.legend()

plt.suptitle('Numeric (Chang-Cooper) vs Analytic Synchrotron', fontweight='bold')
plt.tight_layout()
plt.savefig('numeric_comparison.png', dpi=150)
```

![Numeric vs analytic comparison](img/numeric_comparison.png)

## Discussion

**Left panel**: At radio and 1 keV X-ray frequencies, the analytic and numeric models produce nearly identical light curves. The solid (sync) and dashed (numeric) curves overlap, confirming that the analytic approximation is excellent for standard afterglow modeling.

**Center panel**: The fractional residuals quantify the differences. At radio frequencies the agreement is typically within a few percent. At X-ray energies the residuals can be somewhat larger, particularly near spectral transitions where the numeric model produces smoother breaks.

**Right panel**: At 10 keV, the numeric model can capture pair production effects that suppress the high-energy flux. The analytic model does not include pair production, so it overpredicts the flux at early times when the source is compact and pair-opaque.

!!! tip "When to use `model=\"numeric\"`"
    Use the numeric model when:

    - Fitting broadband SEDs across spectral breaks where smooth transitions matter
    - Working at very high energies ($\gtrsim 10$ keV) where pair production is relevant
    - Needing sub-percent-level accuracy in the spectral shape

    For standard light curve fitting at a single frequency, `model="sync"` is faster and equally accurate.

## Full script

The complete analysis script is at [`examples/numeric_comparison.py`](https://github.com/nuclear-multimessenger-astronomy/blastwave/blob/main/examples/numeric_comparison.py). To regenerate the plot:

```bash
python examples/numeric_comparison.py
```
