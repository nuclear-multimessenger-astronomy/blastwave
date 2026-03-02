"""
GRB 200826A: The Shortest Collapsar GRB

Demonstrates FluxDensity_tophat for modeling a real GRB afterglow with an
ISM density profile.  Multi-wavelength data from Ahumada et al. 2021,
Nature Astronomy, 5, 917 (arXiv:2105.05067).

GRB 200826A was the shortest-duration gamma-ray burst conclusively associated
with a collapsar (massive star core collapse), with T90 ~ 1 s at z = 0.748.
Its multi-band afterglow (X-ray, optical, radio) is well-modeled by a
relativistic tophat jet expanding into a constant-density ISM.

Produces one figure:
  Multi-band afterglow light curves (X-ray, optical, radio) with data
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "..")
from blastwave import FluxDensity_tophat
from scipy.integrate import quad

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

DAY = 86400.0

# ---------- Cosmology ----------
z = 0.748

def luminosity_distance(z, H0=70.0, Om=0.3):
    """Luminosity distance in Mpc (flat LCDM)."""
    OL = 1.0 - Om
    c_km_s = 299792.458
    result, _ = quad(lambda zp: 1.0 / np.sqrt(Om * (1 + zp)**3 + OL), 0, z)
    return (c_km_s / H0) * (1 + z) * result

d_L = luminosity_distance(z)

# ---------- Physical parameters ----------
# Ahumada+2021 best-fit afterglowpy parameters (Table S6)
# Tophat jet, ISM-like density (k=0)
P = dict(
    Eiso=6.0e52,        # isotropic kinetic energy (erg)
    lf=300,             # initial Lorentz factor
    theta_c=0.24,       # jet half-opening angle (rad) ~ 13.7 deg
    n0=0.055,           # ISM density (cm^-3)
    A=0.0,              # no wind component
    eps_e=0.42,         # electron energy fraction
    eps_b=6.4e-5,       # magnetic energy fraction
    p=2.4,              # electron power-law index
    theta_v=0.0,        # viewing angle (on-axis approximation)
    d=d_L,              # luminosity distance (Mpc)
    z=z,                # redshift
)

# ---------- Observational data ----------
# All data from Ahumada+2021

# Optical data: extinction-corrected AB magnitudes -> mJy
# F_nu(mJy) = 3631e3 * 10^(-m_AB/2.5) where m_corrected = m_AB - A_nu
opt_data = {
    'g': {  # g-band: nu ~ 6.29e14 Hz
        't_days': np.array([0.21, 0.28, 1.15]),
        'mag':    np.array([20.86, 20.95, 22.75]),
        'mag_err': np.array([0.04, 0.16, 0.26]),
        'A_nu':   0.21,
    },
    'r': {  # r-band: nu ~ 4.83e14 Hz
        't_days': np.array([0.23, 3.23]),
        'mag':    np.array([20.69, 24.46]),
        'mag_err': np.array([0.05, 0.12]),
        'A_nu':   0.15,
    },
    'i': {  # i-band: nu ~ 3.93e14 Hz
        't_days': np.array([28.28]),
        'mag':    np.array([25.45]),
        'mag_err': np.array([0.15]),
        'A_nu':   0.11,
    },
}

def ab_to_mJy(mag, A_nu=0.0):
    """Convert extinction-corrected AB magnitude to mJy."""
    m_corr = mag - A_nu
    return 3631e3 * 10**(-m_corr / 2.5)  # mJy

# X-ray: Swift XRT count rates -> flux density at 1 keV (mJy)
# Photon index Gamma=1.5, unabsorbed ECF = 4.27e-11 erg/cm^2/ct
# F(0.3-10 keV) = count_rate * ECF
# For Gamma=1.5: F_nu(1keV) = F(0.3-10 keV) / (5.228) * 4.136e-18 / 1e-26
xrt_ecf = 4.27e-11         # erg/cm^2 per count (unabsorbed)
xrt_bandpass_integral = 5.228  # integral of E^{-0.5} dE from 0.3 to 10 keV
h_keV_s = 4.136e-18        # Planck constant in keV*s

xrt_t_days = np.array([0.70, 0.75, 0.83, 1.75, 2.60, 6.40, 16.66])
xrt_rate   = np.array([2.16e-2, 1.91e-2, 1.66e-2, 5.64e-3, 5.36e-3, 3.33e-3, 1.22e-3])
xrt_rate_err = np.array([0.53e-2, 0.49e-2, 0.41e-2, 1.21e-3, 1.22e-3, 0.77e-3, 0.82e-3])

# Convert count rate -> F_nu(1 keV) in mJy
xrt_flux_mJy = xrt_rate * xrt_ecf / xrt_bandpass_integral * h_keV_s / 1e-26
xrt_flux_err = xrt_rate_err * xrt_ecf / xrt_bandpass_integral * h_keV_s / 1e-26

# Radio: VLA 6 GHz (Fong+2020 GCN)
radio_t_days = np.array([2.28])
radio_flux_mJy = np.array([0.040])     # 40 uJy
radio_flux_err = np.array([0.010])     # estimated

# GMRT 1.256 GHz upper limits
gmrt_t_days = np.array([14.46, 19.79])
gmrt_ulim_mJy = np.array([0.0486, 0.0574])  # 3-sigma, uJy -> mJy

# ---------- Model computation ----------
print("Computing model light curves...")

t_seconds = np.geomspace(100, 200 * DAY, 300)  # 100 s to 200 days
t_days = t_seconds / DAY

# Frequencies
nu_xray = 2.4e17     # 1 keV
nu_g = 6.29e14       # g-band
nu_r = 4.83e14       # r-band
nu_radio6 = 6e9      # 6 GHz

flux_xray = FluxDensity_tophat(t_seconds, nu_xray, P, spread=True)
flux_g = FluxDensity_tophat(t_seconds, nu_g, P, spread=True)
flux_r = FluxDensity_tophat(t_seconds, nu_r, P, spread=True)
flux_radio6 = FluxDensity_tophat(t_seconds, nu_radio6, P, spread=True)

# ---------- Plot ----------
print("Plotting...")

fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

# --- Panel 1: X-ray (1 keV) ---
ax = axes[0]
ax.errorbar(xrt_t_days, xrt_flux_mJy, yerr=xrt_flux_err,
            fmt='o', color='#d62728', ms=6, capsize=3,
            label='Swift XRT (1 keV)')
ax.plot(t_days, flux_xray, '-', color='#d62728', lw=2, alpha=0.7)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Time since GRB (days)')
ax.set_ylabel('Flux density (mJy)')
ax.set_title('X-ray (1 keV)')
ax.legend(loc='upper right')
ax.set_xlim(0.3, 100)
ax.tick_params(which='both', direction='in', top=True, right=True)

# --- Panel 2: Optical ---
ax = axes[1]

# g-band data
g = opt_data['g']
g_flux = ab_to_mJy(g['mag'], g['A_nu'])
g_flux_err = g_flux * g['mag_err'] * np.log(10) / 2.5  # error propagation
ax.errorbar(g['t_days'], g_flux, yerr=g_flux_err,
            fmt='s', color='#2ca02c', ms=6, capsize=3, label='ZTF g-band')

# r-band data
r = opt_data['r']
r_flux = ab_to_mJy(r['mag'], r['A_nu'])
r_flux_err = r_flux * r['mag_err'] * np.log(10) / 2.5
ax.errorbar(r['t_days'], r_flux, yerr=r_flux_err,
            fmt='o', color='#1f77b4', ms=6, capsize=3, label='LDT/ZTF r-band')

# i-band data
i_dat = opt_data['i']
i_flux = ab_to_mJy(i_dat['mag'], i_dat['A_nu'])
i_flux_err = i_flux * i_dat['mag_err'] * np.log(10) / 2.5
ax.errorbar(i_dat['t_days'], i_flux, yerr=i_flux_err,
            fmt='^', color='#ff7f0e', ms=6, capsize=3, label='Gemini i-band')

# Model
ax.plot(t_days, flux_g, '-', color='#2ca02c', lw=2, alpha=0.7, label='g model')
ax.plot(t_days, flux_r, '--', color='#1f77b4', lw=2, alpha=0.7, label='r model')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Time since GRB (days)')
ax.set_title('Optical')
ax.legend(loc='upper right', fontsize=9.5)
ax.set_xlim(0.1, 100)
ax.tick_params(which='both', direction='in', top=True, right=True)

# --- Panel 3: Radio (6 GHz) ---
ax = axes[2]
ax.errorbar(radio_t_days, radio_flux_mJy, yerr=radio_flux_err,
            fmt='D', color='#1f77b4', ms=7, capsize=3,
            label='VLA 6 GHz')

# GMRT upper limits
ax.plot(gmrt_t_days, gmrt_ulim_mJy, 'v', color='#9467bd', ms=7,
        label='GMRT 1.3 GHz (3$\\sigma$)')

ax.plot(t_days, flux_radio6, '-', color='#1f77b4', lw=2, alpha=0.7)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Time since GRB (days)')
ax.set_title('Radio (6 GHz)')
ax.legend(loc='upper left')
ax.set_xlim(0.3, 100)
ax.tick_params(which='both', direction='in', top=True, right=True)

fig.suptitle('GRB 200826A — Shortest Collapsar GRB ($z = 0.748$)',
             fontweight='bold', fontsize=16)
plt.tight_layout()

outdir = os.path.join(os.path.dirname(__file__), '..', 'docs', 'examples', 'img')
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, 'grb200826a.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved {outpath}")
plt.close()

print("\nDone!")
