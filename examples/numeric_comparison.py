"""
Numeric model comparison: Chang-Cooper (numeric) vs analytic synchrotron.

Compares model="sync" and model="numeric" at the same physical parameters
to show agreement and highlight differences (e.g., pair production effects
at high frequencies).
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, "..")
from blastwave import FluxDensity_tophat

# ---------- Plot style ----------
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 15,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 11,
    'figure.titlesize': 17,
})

# ---------- Constants ----------
DAY = 86400.0

# ---------- Model parameters ----------
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
    "d":       1000.0,    # 1 Gpc
    "z":       0.2,
}

# ---------- Time grids ----------
t_radio = np.geomspace(0.1 * DAY, 1000.0 * DAY, 150)
t_xray = np.geomspace(100.0, 300.0 * DAY, 150)

nu_radio = 3e9    # 3 GHz
nu_xray = 2.418e17  # 1 keV

# ---------- Compute models ----------
print("Computing analytic synchrotron (sync)...")
print("  Radio 3 GHz...")
F_radio_sync = FluxDensity_tophat(t_radio, nu_radio * np.ones_like(t_radio), P,
                                   tmin=1.0, tmax=1500 * DAY, model="sync")
print("  X-ray 1 keV...")
F_xray_sync = FluxDensity_tophat(t_xray, nu_xray * np.ones_like(t_xray), P,
                                  tmin=1.0, tmax=1500 * DAY, model="sync")

print("Computing Chang-Cooper numeric model...")
print("  Radio 3 GHz...")
F_radio_numeric = FluxDensity_tophat(t_radio, nu_radio * np.ones_like(t_radio), P,
                                      tmin=1.0, tmax=1500 * DAY, model="numeric")
print("  X-ray 1 keV...")
F_xray_numeric = FluxDensity_tophat(t_xray, nu_xray * np.ones_like(t_xray), P,
                                     tmin=1.0, tmax=1500 * DAY, model="numeric")

# ---------- Compute X-ray with pair production ----------
print("Computing numeric model with pair production at X-ray...")
# Use higher energy to show pair effects more clearly
nu_high = 2.418e18  # 10 keV
t_high = np.geomspace(100.0, 100.0 * DAY, 100)
F_high_sync = FluxDensity_tophat(t_high, nu_high * np.ones_like(t_high), P,
                                  tmin=1.0, tmax=200 * DAY, model="sync")
F_high_numeric = FluxDensity_tophat(t_high, nu_high * np.ones_like(t_high), P,
                                     tmin=1.0, tmax=200 * DAY, model="numeric")

# ---------- Plot ----------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

# Panel 1: Overlaid light curves
t_radio_days = t_radio / DAY
t_xray_days = t_xray / DAY

ax1.plot(t_radio_days, F_radio_sync, '-', color='C0', lw=2, label='Radio sync')
ax1.plot(t_radio_days, F_radio_numeric, '--', color='C0', lw=2, label='Radio numeric')
ax1.plot(t_xray_days, F_xray_sync, '-', color='C3', lw=2, label='X-ray sync')
ax1.plot(t_xray_days, F_xray_numeric, '--', color='C3', lw=2, label='X-ray numeric')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Time (days)')
ax1.set_ylabel('Flux density (mJy)')
ax1.set_title('Light Curves')
ax1.legend(fontsize=9, ncol=1)
ax1.tick_params(which='both', direction='in', top=True, right=True)

# Panel 2: Fractional residuals
# Radio
mask_radio = (F_radio_sync > 0) & (F_radio_numeric > 0)
resid_radio = np.full_like(F_radio_sync, np.nan)
resid_radio[mask_radio] = (F_radio_numeric[mask_radio] - F_radio_sync[mask_radio]) / F_radio_sync[mask_radio]

# X-ray
mask_xray = (F_xray_sync > 0) & (F_xray_numeric > 0)
resid_xray = np.full_like(F_xray_sync, np.nan)
resid_xray[mask_xray] = (F_xray_numeric[mask_xray] - F_xray_sync[mask_xray]) / F_xray_sync[mask_xray]

ax2.plot(t_radio_days, resid_radio * 100, '-', color='C0', lw=2, label='Radio 3 GHz')
ax2.plot(t_xray_days, resid_xray * 100, '-', color='C3', lw=2, label='X-ray 1 keV')
ax2.axhline(0, color='gray', ls='--', lw=1)
ax2.axhspan(-10, 10, color='gray', alpha=0.1)
ax2.set_xscale('log')
ax2.set_xlabel('Time (days)')
ax2.set_ylabel('Residual (%)')
ax2.set_title('(Numeric $-$ Sync) / Sync')
ax2.legend()
ax2.set_ylim(-50, 50)
ax2.tick_params(which='both', direction='in', top=True, right=True)

# Panel 3: High-energy comparison (pair production effects)
t_high_days = t_high / DAY
ax3.plot(t_high_days, F_high_sync, '-', color='C4', lw=2, label='Sync (10 keV)')
ax3.plot(t_high_days, F_high_numeric, '--', color='C4', lw=2, label='Numeric (10 keV)')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel('Time (days)')
ax3.set_ylabel('Flux density (mJy)')
ax3.set_title('High Energy (10 keV)')
ax3.legend()
ax3.tick_params(which='both', direction='in', top=True, right=True)

fig.suptitle('Numeric (Chang-Cooper) vs Analytic Synchrotron', fontweight='bold')
plt.tight_layout()

outdir = os.path.join(os.path.dirname(__file__), '..', 'docs', 'examples', 'img')
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, 'numeric_comparison.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved {outpath}")
plt.show()
