"""
Compare blastwave forward shock light curves against VegasAfterglow
for identical physical parameters.

VegasAfterglow outputs flux density in erg/cm²/s/Hz (CGS).
blastwave outputs flux density in mJy.
1 mJy = 1e-26 erg/cm²/s/Hz
"""
import numpy as np
import time

# ── Shared physical parameters ──
theta_c = 0.1       # half-opening angle [rad]
E_iso   = 1e52      # isotropic equivalent energy [erg]
Gamma0  = 300.0     # initial Lorentz factor
n0      = 1.0       # ISM number density [cm^-3]
eps_e   = 0.1       # electron energy fraction
eps_B   = 0.01      # magnetic energy fraction
p_val   = 2.17      # electron spectral index
theta_v = 0.0       # viewing angle [rad]
d_Mpc   = 474.33    # luminosity distance [Mpc]
z       = 0.1       # redshift

# Observing grid
tday = np.logspace(-2, 3, 50)
tsecond = tday * 86400.0
nu_radio  = 1e9   # 1 GHz
nu_optical = 1e14  # optical
nu_xray   = 1e18  # X-ray

MPC_cm = 3.0856775814913673e24  # 1 Mpc in cm
mJy_cgs = 1e-26  # 1 mJy in erg/cm²/s/Hz

# ──────────────────────────────────────────────────────────────────────
# VegasAfterglow
# ──────────────────────────────────────────────────────────────────────
from VegasAfterglow import TophatJet, ISM, Observer, Radiation, Model

jet_v   = TophatJet(theta_c=theta_c, E_iso=E_iso, Gamma0=Gamma0)
medium  = ISM(n_ism=n0)
obs     = Observer(lumi_dist=d_Mpc * MPC_cm, z=z, theta_obs=theta_v)
rad_fwd = Radiation(eps_e=eps_e, eps_B=eps_B, p=p_val)

t0 = time.time()
model_v = Model(jet=jet_v, medium=medium, observer=obs, fwd_rad=rad_fwd)
freqs = np.array([nu_radio, nu_optical, nu_xray])
result_v = model_v.flux_density_grid(tsecond, freqs)
vegas_time = time.time() - t0
print(f"VegasAfterglow: {vegas_time:.2f}s")

# result_v.total is in erg/cm²/s/Hz → convert to mJy
flux_v_radio   = np.array(result_v.total[0]) / mJy_cgs
flux_v_optical = np.array(result_v.total[1]) / mJy_cgs
flux_v_xray    = np.array(result_v.total[2]) / mJy_cgs

# ──────────────────────────────────────────────────────────────────────
# blastwave (run each band separately — each creates a new Jet)
# ──────────────────────────────────────────────────────────────────────
import blastwave

P = dict(
    Eiso=E_iso, lf=Gamma0, theta_c=theta_c,
    n0=n0, A=0.0,
    eps_e=eps_e, eps_b=eps_B, p=p_val,
    theta_v=theta_v, d=d_Mpc, z=z, s=6,
)

t0 = time.time()
# Use a single Jet instance for all bands
jet_j = blastwave.Jet(
    blastwave.TopHat(theta_c, E_iso, lf0=Gamma0),
    0.0,   # nwind
    n0,    # nism
    tmin=10.0, tmax=1e10,
    grid=blastwave.ForwardJetRes(theta_c, 129),
    spread=True,
)
flux_j_radio   = jet_j.FluxDensity(tsecond, nu_radio,   P)
flux_j_optical = jet_j.FluxDensity(tsecond, nu_optical, P)
flux_j_xray    = jet_j.FluxDensity(tsecond, nu_xray,    P)
jetrs_time = time.time() - t0
print(f"blastwave:    {jetrs_time:.2f}s")

# ──────────────────────────────────────────────────────────────────────
# Comparison
# ──────────────────────────────────────────────────────────────────────
def compare(name, t, flux_v, flux_j):
    """Print comparison statistics for a single band."""
    # Only compare where both are positive and finite
    mask = (flux_v > 0) & (flux_j > 0) & np.isfinite(flux_v) & np.isfinite(flux_j)
    if mask.sum() == 0:
        print(f"  {name}: no valid comparison points!")
        return

    ratio = flux_j[mask] / flux_v[mask]
    log_ratio = np.log10(ratio)

    print(f"  {name}:")
    print(f"    valid points: {mask.sum()}/{len(t)}")
    print(f"    ratio range:  [{ratio.min():.4f}, {ratio.max():.4f}]")
    print(f"    median ratio: {np.median(ratio):.4f}")
    print(f"    max |log10 ratio|: {np.max(np.abs(log_ratio)):.3f} dex")
    print(f"    mean |log10 ratio|: {np.mean(np.abs(log_ratio)):.3f} dex")

print("\n=== Forward Shock Comparison: blastwave / VegasAfterglow ===")
compare("Radio (1 GHz)",   tday, flux_v_radio,   flux_j_radio)
compare("Optical (1e14 Hz)", tday, flux_v_optical, flux_j_optical)
compare("X-ray (1 keV)",   tday, flux_v_xray,    flux_j_xray)

# Print sample values side by side
print("\n=== Sample values (X-ray, 1 keV) ===")
print(f"{'t [day]':>10s}  {'VegasAG [mJy]':>15s}  {'blastwave [mJy]':>18s}  {'ratio':>8s}")
for idx in [0, len(tday)//4, len(tday)//2, 3*len(tday)//4, len(tday)-1]:
    v, j = flux_v_xray[idx], flux_j_xray[idx]
    r = j / v if v > 0 else float('nan')
    print(f"{tday[idx]:10.3e}  {v:15.6e}  {j:18.6e}  {r:8.4f}")

print("\n=== Sample values (Radio, 1 GHz) ===")
print(f"{'t [day]':>10s}  {'VegasAG [mJy]':>15s}  {'blastwave [mJy]':>18s}  {'ratio':>8s}")
for idx in [0, len(tday)//4, len(tday)//2, 3*len(tday)//4, len(tday)-1]:
    v, j = flux_v_radio[idx], flux_j_radio[idx]
    r = j / v if v > 0 else float('nan')
    print(f"{tday[idx]:10.3e}  {v:15.6e}  {j:18.6e}  {r:8.4f}")

# Try to save a plot
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    bands = [
        ("Radio (1 GHz)", flux_v_radio, flux_j_radio),
        ("Optical (1e14 Hz)", flux_v_optical, flux_j_optical),
        ("X-ray (1 keV)", flux_v_xray, flux_j_xray),
    ]

    for i, (name, fv, fj) in enumerate(bands):
        ax = axes[0, i]
        mask_v = fv > 0
        mask_j = fj > 0
        if mask_v.any():
            ax.loglog(tday[mask_v], fv[mask_v], 'b-', label='VegasAfterglow', linewidth=2)
        if mask_j.any():
            ax.loglog(tday[mask_j], fj[mask_j], 'r--', label='blastwave', linewidth=2)
        ax.set_xlabel('Time [days]')
        ax.set_ylabel('Flux density [mJy]')
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax2 = axes[1, i]
        mask = (fv > 0) & (fj > 0) & np.isfinite(fv) & np.isfinite(fj)
        if mask.any():
            ratio = fj[mask] / fv[mask]
            ax2.semilogx(tday[mask], ratio, 'k-', linewidth=1.5)
            ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
            ax2.set_ylim(0.5, 2.0)
        ax2.set_xlabel('Time [days]')
        ax2.set_ylabel('Ratio (blastwave / VegasAG)')
        ax2.set_title(f'{name} ratio')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/mcoughli/blastwave/tests/compare_vegas.png', dpi=150)
    print("\nPlot saved to tests/compare_vegas.png")
except ImportError:
    print("\nmatplotlib not available, skipping plot")
