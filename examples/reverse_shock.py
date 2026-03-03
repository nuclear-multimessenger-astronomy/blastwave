"""
Reverse shock: Forward vs reverse shock decomposition.

Uses the Jet class directly to access FluxDensity_forward() and
FluxDensity_reverse(), showing the individual contributions at
radio and optical frequencies.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, "..")
from blastwave import Jet, TopHat, ForwardJetRes

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

# ---------- Jet configuration ----------
# Top-hat jet with reverse shock enabled
# duration=1000 s gives a thick shell with a strong RS
theta_c = 0.1

jet = Jet(
    TopHat(theta_c, 1e52, lf0=300.0),
    0.0,                    # nwind
    1.0,                    # nism
    tmin=10.0,
    tmax=500 * DAY,
    grid=ForwardJetRes(theta_c, 129),
    tail=True,
    spread=True,
    cal_level=1,
    include_reverse_shock=True,
    sigma=0.0,              # unmagnetized ejecta (strong RS)
    eps_e_rs=0.1,
    eps_b_rs=0.01,
    p_rs=2.2,
    duration=1000.0,        # engine duration [s] → thick shell RS
)

# ---------- Physical parameters ----------
P = {
    "Eiso":    1e52,
    "lf":      300.0,
    "theta_c": theta_c,
    "A":       0.0,
    "n0":      1.0,
    "eps_e":   0.1,
    "eps_b":   0.01,
    "p":       2.3,
    "theta_v": 0.0,
    "d":       474.33,   # ~z=0.1
    "z":       0.1,
}

# ---------- Time grid ----------
t = np.geomspace(10.0, 300.0 * DAY, 200)

# ---------- Frequencies ----------
nu_radio = 3e9             # 3 GHz
nu_optical = 4.56e14       # R-band (~658 nm)

# ---------- Compute flux components ----------
print("Computing radio 3 GHz...")
F_radio_total   = jet.FluxDensity(t, nu_radio, P)
F_radio_forward = jet.FluxDensity_forward(t, nu_radio, P)
F_radio_reverse = jet.FluxDensity_reverse(t, nu_radio, P)

print("Computing optical R-band...")
F_opt_total   = jet.FluxDensity(t, nu_optical, P)
F_opt_forward = jet.FluxDensity_forward(t, nu_optical, P)
F_opt_reverse = jet.FluxDensity_reverse(t, nu_optical, P)

# ---------- Plot ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
t_days = t / DAY

# Left panel: Radio
ax1.plot(t_days, F_radio_total, '-', color='black', lw=2.5, label='Total')
ax1.plot(t_days, F_radio_forward, '--', color='C0', lw=2, label='Forward shock')
ax1.plot(t_days, F_radio_reverse, '--', color='C3', lw=2, label='Reverse shock')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Time (days)')
ax1.set_ylabel('Flux density (mJy)')
ax1.set_title('Radio 3 GHz')
ax1.legend()
ax1.set_xlim(1e-4, 300)
ax1.tick_params(which='both', direction='in', top=True, right=True)

# Right panel: Optical
ax2.plot(t_days, F_opt_total, '-', color='black', lw=2.5, label='Total')
ax2.plot(t_days, F_opt_forward, '--', color='C0', lw=2, label='Forward shock')
ax2.plot(t_days, F_opt_reverse, '--', color='C3', lw=2, label='Reverse shock')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Time (days)')
ax2.set_ylabel('Flux density (mJy)')
ax2.set_title('Optical R-band')
ax2.legend()
ax2.set_xlim(1e-4, 300)
ax2.tick_params(which='both', direction='in', top=True, right=True)

fig.suptitle('Forward vs Reverse Shock Decomposition', fontweight='bold')
plt.tight_layout()

outdir = os.path.join(os.path.dirname(__file__), '..', 'docs', 'examples', 'img')
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, 'reverse_shock.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved {outpath}")
plt.show()
