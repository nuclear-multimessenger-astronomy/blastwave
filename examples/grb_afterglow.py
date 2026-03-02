"""
Physics Showcase: GRB Afterglow Effects

Demonstrates how different physical ingredients affect GRB afterglow light curves:
  1. Multi-band light curves (radio through X-ray)
  2. Viewing angle effects (tophat vs Gaussian jets)
  3. Lateral spreading (with vs without)
  4. Jet structure comparison (tophat vs Gaussian vs power-law)

Uses standard GRB afterglow parameters in an ISM environment.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "..")
from blastwave import FluxDensity_tophat, FluxDensity_gaussian, FluxDensity_powerlaw

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

# ---------- Physical parameters ----------
# Typical long GRB afterglow in an ISM environment
P = dict(
    Eiso=1e53,       # isotropic equivalent energy (erg)
    lf=300,          # initial Lorentz factor
    theta_c=0.1,     # half-opening angle (rad) ~ 5.7 deg
    n0=1.0,          # ISM density (cm^-3)
    A=0.0,           # no wind component
    eps_e=0.1,       # electron energy fraction
    eps_b=0.01,      # magnetic energy fraction
    p=2.2,           # electron power-law index
    theta_v=0.0,     # viewing angle (on-axis)
    d=1000.0,        # luminosity distance (Mpc)
    z=0.2,           # redshift
)

# Time grid
t_seconds = np.geomspace(10, 1e8, 300)
t_days = t_seconds / 86400
outdir = os.path.join(os.path.dirname(__file__), '..', 'docs', 'examples', 'img')
os.makedirs(outdir, exist_ok=True)

# ═════════════════════════════════════════════════════
# Figure 1: Multi-band on-axis light curves
# ═════════════════════════════════════════════════════
print("Computing multi-band light curves (on-axis tophat)...")

bands = [
    (3e9,    '3 GHz (radio)',   '#1f77b4', '-'),
    (6e9,    '6 GHz (radio)',   '#2ca02c', '-'),
    (1e14,   'R-band (optical)','#ff7f0e', '-'),
    (2.4e17, '1 keV (X-ray)',   '#d62728', '-'),
]

fig, ax = plt.subplots(figsize=(10, 7))

for nu, label, color, ls in bands:
    flux = FluxDensity_tophat(t_seconds, nu, P, spread=True)
    ax.plot(t_days, flux, ls, color=color, lw=2, label=label)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Flux density (mJy)')
ax.set_title('GRB Afterglow: On-axis Tophat Jet', fontweight='bold')
ax.legend(loc='upper right', framealpha=0.9)
ax.set_xlim(1e-4, 1e3)
ax.set_ylim(1e-7, 1e2)
ax.tick_params(which='both', direction='in', top=True, right=True)

plt.tight_layout()
outpath = os.path.join(outdir, 'grb_afterglow_multiband.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved {outpath}")
plt.close()

# ═════════════════════════════════════════════════════
# Figure 2: Viewing angle effects (X-ray)
# ═════════════════════════════════════════════════════
print("Computing viewing angle comparison (X-ray)...")

nu_xray = 2.4e17  # 1 keV

angles_deg = [0, 3, 6, 10, 20]
colors_angle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: Tophat jet
ax = axes[0]
for theta_v_deg, color in zip(angles_deg, colors_angle):
    P_v = {**P, 'theta_v': np.radians(theta_v_deg)}
    flux = FluxDensity_tophat(t_seconds, nu_xray, P_v, spread=True)
    ax.plot(t_days, flux, '-', color=color, lw=2,
            label=rf'$\theta_v = {theta_v_deg}^\circ$')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Flux density at 1 keV (mJy)')
ax.set_title(r'Tophat jet ($\theta_c = 5.7^\circ$)')
ax.legend(loc='upper right', framealpha=0.9)
ax.set_xlim(1e-4, 1e3)
ax.set_ylim(1e-10, 1e1)
ax.tick_params(which='both', direction='in', top=True, right=True)

# Right: Gaussian jet
ax = axes[1]
for theta_v_deg, color in zip(angles_deg, colors_angle):
    P_v = {**P, 'theta_v': np.radians(theta_v_deg)}
    flux = FluxDensity_gaussian(t_seconds, nu_xray, P_v, spread=True)
    ax.plot(t_days, flux, '-', color=color, lw=2,
            label=rf'$\theta_v = {theta_v_deg}^\circ$')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Time (days)')
ax.set_title(r'Gaussian jet ($\theta_c = 5.7^\circ$)')
ax.legend(loc='upper right', framealpha=0.9)
ax.set_xlim(1e-4, 1e3)
ax.set_ylim(1e-10, 1e1)
ax.tick_params(which='both', direction='in', top=True, right=True)

fig.suptitle('Viewing Angle Effects on X-ray Afterglow', fontweight='bold')
plt.tight_layout()
outpath = os.path.join(outdir, 'grb_afterglow_viewing.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved {outpath}")
plt.close()

# ═════════════════════════════════════════════════════
# Figure 3: Effect of lateral spreading
# ═════════════════════════════════════════════════════
print("Computing spreading comparison (X-ray)...")

fig, ax = plt.subplots(figsize=(10, 7))

P_on = {**P, 'theta_v': 0.0}
P_off = {**P, 'theta_v': np.radians(15)}

for spread, ls, alpha in [(True, '-', 1.0), (False, '--', 0.7)]:
    label_suf = 'with spreading' if spread else 'no spreading'

    flux_on = FluxDensity_tophat(t_seconds, nu_xray, P_on, spread=spread)
    ax.plot(t_days, flux_on, ls, color='#1f77b4', lw=2, alpha=alpha,
            label=rf'On-axis, {label_suf}')

    flux_off = FluxDensity_tophat(t_seconds, nu_xray, P_off, spread=spread)
    ax.plot(t_days, flux_off, ls, color='#d62728', lw=2, alpha=alpha,
            label=rf'$\theta_v = 15^\circ$, {label_suf}')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Flux density at 1 keV (mJy)')
ax.set_title('Effect of Lateral Spreading', fontweight='bold')
ax.legend(loc='upper right', framealpha=0.9)
ax.set_xlim(1e-4, 1e3)
ax.set_ylim(1e-10, 1e1)
ax.tick_params(which='both', direction='in', top=True, right=True)

plt.tight_layout()
outpath = os.path.join(outdir, 'grb_afterglow_spreading.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved {outpath}")
plt.close()

# ═════════════════════════════════════════════════════
# Figure 4: Jet structure comparison
# ═════════════════════════════════════════════════════
print("Computing jet structure comparison...")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# For power-law jet, need 's' parameter (energy profile steepness)
P_pl = {**P, 's': 4.0}  # E(theta) ~ (1 + (theta/theta_c)^2)^{-s/2}

# Left: On-axis (all structures look similar)
ax = axes[0]
P_v = {**P, 'theta_v': 0.0}
P_pl_v = {**P_pl, 'theta_v': 0.0}

flux_th = FluxDensity_tophat(t_seconds, nu_xray, P_v, spread=True)
flux_ga = FluxDensity_gaussian(t_seconds, nu_xray, P_v, spread=True)
flux_pl = FluxDensity_powerlaw(t_seconds, nu_xray, P_pl_v, spread=True)

ax.plot(t_days, flux_th, '-',  color='#1f77b4', lw=2, label='Tophat')
ax.plot(t_days, flux_ga, '--', color='#ff7f0e', lw=2, label='Gaussian')
ax.plot(t_days, flux_pl, '-.', color='#2ca02c', lw=2, label='Power-law')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Flux density at 1 keV (mJy)')
ax.set_title(r'On-axis ($\theta_v = 0^\circ$)')
ax.legend(loc='upper right', framealpha=0.9)
ax.set_xlim(1e-4, 1e3)
ax.set_ylim(1e-7, 1e1)
ax.tick_params(which='both', direction='in', top=True, right=True)

# Right: Off-axis (structures diverge dramatically)
ax = axes[1]
theta_off = np.radians(15)
P_v = {**P, 'theta_v': theta_off}
P_pl_v = {**P_pl, 'theta_v': theta_off, 's': 4.0}

flux_th = FluxDensity_tophat(t_seconds, nu_xray, P_v, spread=True)
flux_ga = FluxDensity_gaussian(t_seconds, nu_xray, P_v, spread=True)
flux_pl = FluxDensity_powerlaw(t_seconds, nu_xray, P_pl_v, spread=True)

ax.plot(t_days, flux_th, '-',  color='#1f77b4', lw=2, label='Tophat')
ax.plot(t_days, flux_ga, '--', color='#ff7f0e', lw=2, label='Gaussian')
ax.plot(t_days, flux_pl, '-.', color='#2ca02c', lw=2, label='Power-law')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Time (days)')
ax.set_title(r'Off-axis ($\theta_v = 15^\circ$)')
ax.legend(loc='upper right', framealpha=0.9)
ax.set_xlim(1e-4, 1e3)
ax.set_ylim(1e-10, 1e1)
ax.tick_params(which='both', direction='in', top=True, right=True)

fig.suptitle('Jet Structure Comparison (1 keV X-ray)', fontweight='bold')
plt.tight_layout()
outpath = os.path.join(outdir, 'grb_afterglow_structure.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved {outpath}")
plt.close()

print("\nDone! Generated 4 figures.")
