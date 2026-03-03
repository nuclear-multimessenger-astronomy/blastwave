"""
GW170817: Off-axis Gaussian jet afterglow.

Models the radio (3 GHz VLA) and X-ray (1 keV Chandra) afterglow of
the binary neutron star merger GW170817 using a Gaussian structured jet
viewed off-axis.

Radio data from Mooley+2018 (Nature, 554, 207) and Hallinan+2017.
X-ray data from Hajela+2019 (ApJL, 886, L17) and Troja+2020.
Distance: d ~ 40 Mpc, z = 0.0098.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, "..")
from blastwave import FluxDensity_gaussian

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

# ---------- Approximate published data ----------
# Radio 3 GHz (VLA) — compiled from Mooley+2018, Hallinan+2017, Margutti+2018
# Format: (t_days, F_microJy, err_microJy)
data_radio_uJy = np.array([
    [16.4,   15.1,   3.4],
    [22.2,   18.0,   4.5],
    [35.2,   22.0,   4.0],
    [46.3,   28.5,   4.0],
    [54.3,   33.0,   4.5],
    [65.5,   40.0,   4.5],
    [75.4,   44.0,   4.0],
    [92.6,   52.0,   5.0],
    [107.5,  60.0,   4.5],
    [115.2,  62.0,   5.0],
    [135.8,  67.0,   5.0],
    [152.2,  70.0,   5.0],
    [163.1,  68.0,   6.0],
    [185.5,  56.0,   5.5],
    [207.4,  52.0,   5.0],
    [230.0,  48.0,   6.0],
    [260.0,  40.0,   5.0],
    [298.0,  32.0,   5.0],
    [362.0,  24.0,   5.0],
    [581.0,  12.0,   3.5],
    [900.0,   5.5,   2.5],
])
# Convert microJy -> mJy
data_radio = data_radio_uJy.copy()
data_radio[:, 1] *= 1e-3
data_radio[:, 2] *= 1e-3

# X-ray 1 keV (Chandra) — compiled from Hajela+2019, Troja+2020
# Format: (t_days, F_microJy, err_microJy)  — converted from flux via standard spectral assumptions
data_xray_uJy = np.array([
    [9.2,    0.85,   0.45],
    [15.4,   1.9,    0.5],
    [109.2,  5.5,    0.8],
    [135.0,  5.6,    0.7],
    [153.5,  6.4,    0.8],
    [163.0,  6.5,    0.6],
    [186.0,  5.0,    0.7],
    [209.0,  4.2,    0.6],
    [260.0,  3.3,    0.5],
    [358.0,  1.7,    0.4],
    [581.0,  0.80,   0.30],
    [743.0,  0.50,   0.25],
])
data_xray = data_xray_uJy.copy()
data_xray[:, 1] *= 1e-3
data_xray[:, 2] *= 1e-3

# ---------- Model parameters ----------
# Off-axis Gaussian jet: Mooley+2018, Ghirlanda+2019, Hotokezaka+2019
P = {
    "Eiso":    2e52,        # isotropic-equivalent energy (erg)
    "lf":      300.0,       # initial Lorentz factor
    "theta_c": 0.07,        # jet core half-opening angle (rad)
    "A":       0.0,
    "n0":      5e-3,        # ISM density (cm^-3)
    "eps_e":   0.1,
    "eps_b":   1e-3,
    "p":       2.15,
    "theta_v": 0.35,        # viewing angle (rad) ~ 20 deg
    "d":       40.0,        # luminosity distance (Mpc)
    "z":       0.0098,
}

# ---------- Compute model ----------
t_model = np.geomspace(1.0 * DAY, 1200.0 * DAY, 200)

print("Computing 3 GHz radio light curve...")
nu_radio = 3e9 * np.ones_like(t_model)
F_radio = FluxDensity_gaussian(t_model, nu_radio, P,
                                tmin=1.0, tmax=1500 * DAY)

print("Computing 1 keV X-ray light curve...")
nu_xray = 2.418e17 * np.ones_like(t_model)  # 1 keV in Hz
F_xray = FluxDensity_gaussian(t_model, nu_xray, P,
                               tmin=1.0, tmax=1500 * DAY)

# ---------- Plot ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
t_days = t_model / DAY

# Left panel: Radio 3 GHz
ax1.errorbar(data_radio[:, 0], data_radio[:, 1] * 1e3, yerr=data_radio[:, 2] * 1e3,
             fmt='o', color='C0', label='3 GHz (VLA)', capsize=3, ms=5, zorder=5)
ax1.plot(t_days, F_radio * 1e3, '-', color='C0', lw=2, alpha=0.8, label='Model')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Time since merger (days)')
ax1.set_ylabel(r'Flux density ($\mu$Jy)')
ax1.set_title('Radio 3 GHz')
ax1.legend()
ax1.set_xlim(5, 1200)
ax1.set_ylim(1, 200)
ax1.tick_params(which='both', direction='in', top=True, right=True)

# Right panel: X-ray 1 keV
ax2.errorbar(data_xray[:, 0], data_xray[:, 1] * 1e3, yerr=data_xray[:, 2] * 1e3,
             fmt='s', color='C3', label='1 keV (Chandra)', capsize=3, ms=5, zorder=5)
ax2.plot(t_days, F_xray * 1e3, '-', color='C3', lw=2, alpha=0.8, label='Model')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Time since merger (days)')
ax2.set_ylabel(r'Flux density ($\mu$Jy)')
ax2.set_title('X-ray 1 keV')
ax2.legend()
ax2.set_xlim(5, 1200)
ax2.set_ylim(0.1, 20)
ax2.tick_params(which='both', direction='in', top=True, right=True)

fig.suptitle('GW170817 — Off-Axis Gaussian Jet Afterglow', fontweight='bold')
plt.tight_layout()

outdir = os.path.join(os.path.dirname(__file__), '..', 'docs', 'examples', 'img')
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, 'gw170817.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved {outpath}")
plt.show()
