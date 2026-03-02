"""
AT2018cow radio light curve comparison with spherical blast wave model.

Radio data from Ho+2019 (ApJ, 871, 73) and Margutti+2019 (ApJ, 872, 18).
Distance: d = 60 Mpc, z = 0.0141.

This script uses FluxDensity_spherical with a k-parameterized CSM density
to model the FBOT AT2018cow.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "..")
from blastwave import FluxDensity_spherical

# ---------- Plot style ----------
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 15,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
    'figure.titlesize': 17,
})

# ---------- Published radio data (Ho+2019, Margutti+2019) ----------

# Time in days since explosion (MJD 58285.441)
# Flux densities in mJy
# Format: (t_days, F_mJy, err_mJy)

# 10 GHz (X-band) — Ho+2019 Table 2 (VLA)
data_10GHz = np.array([
    [5.32,  0.26,  0.03],
    [7.30,  0.37,  0.02],
    [9.10,  0.37,  0.03],
    [13.23, 0.43,  0.02],
    [16.14, 0.53,  0.03],
    [22.07, 1.11,  0.03],
    [28.21, 1.25,  0.05],
    [35.86, 0.82,  0.03],
    [50.16, 0.56,  0.03],
    [63.36, 0.40,  0.02],
    [84.95, 0.26,  0.02],
    [99.20, 0.18,  0.02],
    [131.5, 0.12,  0.02],
])

# 6 GHz (C-band) — Ho+2019 Table 2 (VLA)
data_6GHz = np.array([
    [5.32,  0.13,  0.02],
    [7.30,  0.19,  0.02],
    [9.10,  0.20,  0.02],
    [13.23, 0.28,  0.02],
    [16.14, 0.37,  0.03],
    [22.07, 0.94,  0.03],
    [28.21, 1.15,  0.04],
    [35.86, 0.79,  0.03],
    [50.16, 0.55,  0.03],
    [63.36, 0.39,  0.02],
    [84.95, 0.26,  0.02],
    [99.20, 0.19,  0.02],
    [131.5, 0.11,  0.02],
])

# 15.5 GHz (Ku-band) — Ho+2019
data_15GHz = np.array([
    [5.32,  0.42,  0.03],
    [7.30,  0.60,  0.03],
    [9.10,  0.57,  0.03],
    [13.23, 0.55,  0.03],
    [16.14, 0.62,  0.03],
    [22.07, 1.02,  0.04],
    [28.21, 1.10,  0.05],
    [35.86, 0.67,  0.04],
    [50.16, 0.39,  0.03],
    [63.36, 0.27,  0.02],
])

# ---------- Constants ----------
DAY = 86400.0  # seconds per day

# ---------- Model parameters ----------
# AT2018cow is mildly relativistic (beta ~ 0.1-0.3).
# Ho+2019 equipartition: E_k ~ 10^48.7, n ~ few cm^-3, beta ~ 0.1-0.2
# Margutti+2019: favors wind-stratified medium, E_k ~ 10^49
# The radio rise is SSA-driven — must use model="sync_ssa".
#
# Two scenarios:
#   (a) ISM (k=0) with SSA: Ho+2019-inspired
#   (b) Wind (k=2) with SSA: Margutti+2019-inspired

P_ism = {
    "Eiso": 2e49,         # kinetic energy (erg)
    "lf":   1.1,           # Gamma=1.1 → beta~0.42
    "A":    0.0,
    "n0":   100.0,         # dense CSM
    "eps_e": 0.1,
    "eps_b": 0.03,
    "p":     2.8,
    "theta_v": 0.0,
    "d":     60.0,
    "z":     0.0141,
}

P_wind = {
    "Eiso": 5e49,
    "lf":   1.03,          # Gamma=1.03 → beta~0.24
    "A":    20.0,           # wind density scale
    "n0":   0.0,
    "eps_e": 0.12,
    "eps_b": 0.05,
    "p":     2.8,
    "theta_v": 0.0,
    "d":     60.0,
    "z":     0.0141,
}

t_model = np.geomspace(1.0 * DAY, 200.0 * DAY, 150)
freqs = [6e9, 10e9, 15.5e9]
freq_labels = ['6 GHz', '10 GHz', '15.5 GHz']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
data_sets = [data_6GHz, data_10GHz, data_15GHz]
markers = ['s', 'o', '^']

# ---------- Compute ISM model (sync_ssa) ----------
print("=== ISM model (k=0, sync_ssa) ===")
F_ism = {}
for nu, label in zip(freqs, freq_labels):
    print(f"  Computing {label}...")
    F_ism[nu] = FluxDensity_spherical(t_model, nu * np.ones_like(t_model), P_ism,
                                       k=0.0, tmin=1.0, tmax=300*DAY,
                                       model="sync_ssa")

# ---------- Compute wind model (sync_ssa) ----------
print("=== Wind model (k=2, sync_ssa) ===")
F_wind = {}
for nu, label in zip(freqs, freq_labels):
    print(f"  Computing {label}...")
    F_wind[nu] = FluxDensity_spherical(t_model, nu * np.ones_like(t_model), P_wind,
                                        k=2.0, tmin=1.0, tmax=300*DAY,
                                        model="sync_ssa")

# ---------- Plot ----------
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
t_days = t_model / DAY

panel_info = [
    (axes[0], F_ism, P_ism, 'ISM (k = 0)'),
    (axes[1], F_wind, P_wind, 'Wind (k = 2)'),
]

for ax, F_model, P, title in panel_info:
    for nu, label, color, marker, data in zip(freqs, freq_labels, colors, markers, data_sets):
        ax.errorbar(data[:, 0], data[:, 1], yerr=data[:, 2],
                    fmt=marker, color=color, label=f'{label} data',
                    capsize=3, ms=7, markeredgewidth=0.5, zorder=5)
        ax.plot(t_days, F_model[nu], '-', color=color, lw=2, alpha=0.8,
                label=f'{label} model')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Time since explosion (days)')
    ax.set_title(title)
    ax.legend(ncol=2, loc='upper right', framealpha=0.9)
    ax.set_xlim(1, 200)
    ax.set_ylim(0.01, 5)
    ax.tick_params(which='both', direction='in', top=True, right=True)

axes[0].set_ylabel('Flux density (mJy)')
fig.suptitle('AT2018cow Radio Light Curves (sync_ssa)', fontweight='bold')
plt.tight_layout()
import os
outdir = os.path.join(os.path.dirname(__file__), '..', 'docs', 'examples', 'img')
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, 'at2018cow_radio.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved {outpath}")
plt.show()
