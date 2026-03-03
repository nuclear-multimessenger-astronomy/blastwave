"""
Compare blastwave reverse shock light curves against VegasAfterglow
for identical physical parameters (tophat jet, on-axis, ISM, thick shell RS).

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
p_val   = 2.3       # electron spectral index
theta_v = 0.0       # viewing angle [rad]
d_Mpc   = 474.33    # luminosity distance [Mpc]  (~z=0.1)
z       = 0.1       # redshift
duration = 1000.0   # engine duration [s] → thick shell RS

# RS microphysics (same as FS for clean comparison)
eps_e_rs = 0.1
eps_B_rs = 0.01
p_rs     = 2.2

# Observing grid
DAY = 86400.0
tday = np.logspace(-4, np.log10(300), 100)
tsecond = tday * DAY
nu_radio = 3e9    # 3 GHz

MPC_cm = 3.0856775814913673e24  # 1 Mpc in cm
mJy_cgs = 1e-26  # 1 mJy in erg/cm²/s/Hz

# ──────────────────────────────────────────────────────────────────────
# VegasAfterglow
# ──────────────────────────────────────────────────────────────────────
from VegasAfterglow import TophatJet, ISM, Observer, Radiation, Model

jet_v   = TophatJet(theta_c=theta_c, E_iso=E_iso, Gamma0=Gamma0,
                    duration=duration)
medium  = ISM(n_ism=n0)
obs     = Observer(lumi_dist=d_Mpc * MPC_cm, z=z, theta_obs=theta_v)
rad_fwd = Radiation(eps_e=eps_e, eps_B=eps_B, p=p_val)
rad_rvs = Radiation(eps_e=eps_e_rs, eps_B=eps_B_rs, p=p_rs)

t0 = time.time()
model_v = Model(jet=jet_v, medium=medium, observer=obs,
                fwd_rad=rad_fwd, rvs_rad=rad_rvs)
freqs = np.array([nu_radio])
result_v = model_v.flux_density_grid(tsecond, freqs)
vegas_time = time.time() - t0
print(f"VegasAfterglow: {vegas_time:.2f}s")

# Extract component fluxes (erg/cm²/s/Hz → mJy)
flux_v_total = np.array(result_v.total[0]) / mJy_cgs
flux_v_fwd   = np.array(result_v.fwd.sync[0]) / mJy_cgs
flux_v_rvs   = np.array(result_v.rvs.sync[0]) / mJy_cgs

print(f"  VegasAG total range:   [{flux_v_total.min():.3e}, {flux_v_total.max():.3e}] mJy")
print(f"  VegasAG forward range: [{flux_v_fwd.min():.3e}, {flux_v_fwd.max():.3e}] mJy")
print(f"  VegasAG reverse range: [{flux_v_rvs.min():.3e}, {flux_v_rvs.max():.3e}] mJy")

# ──────────────────────────────────────────────────────────────────────
# blastwave
# ──────────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, "/home/mcoughli/blastwave")
from blastwave import Jet, TopHat, ForwardJetRes

MODEL = "sync_ssa_smooth"

t0 = time.time()
jet_b = Jet(
    TopHat(theta_c, E_iso, lf0=Gamma0),
    0.0,   # nwind
    n0,    # nism
    tmin=10.0,
    tmax=500 * DAY,
    grid=ForwardJetRes(theta_c, 129),
    tail=True,
    spread=False,
    cal_level=1,
    include_reverse_shock=True,
    sigma=0.0,
    eps_e=eps_e,
    eps_b=eps_B,
    p_fwd=p_val,
    eps_e_rs=eps_e_rs,
    eps_b_rs=eps_B_rs,
    p_rs=p_rs,
    duration=duration,
)

P = dict(
    Eiso=E_iso, lf=Gamma0, theta_c=theta_c,
    n0=n0, A=0.0,
    eps_e=eps_e, eps_b=eps_B, p=p_val,
    theta_v=theta_v, d=d_Mpc, z=z,
)

flux_b_total   = jet_b.FluxDensity(tsecond, nu_radio, P, model=MODEL)
flux_b_forward = jet_b.FluxDensity_forward(tsecond, nu_radio, P, model=MODEL)
flux_b_reverse = jet_b.FluxDensity_reverse(tsecond, nu_radio, P, model=MODEL)
bw_time = time.time() - t0
print(f"\nblastwave: {bw_time:.2f}s")
print(f"  blastwave total range:   [{flux_b_total.min():.3e}, {flux_b_total.max():.3e}] mJy")
print(f"  blastwave forward range: [{flux_b_forward.min():.3e}, {flux_b_forward.max():.3e}] mJy")
print(f"  blastwave reverse range: [{flux_b_reverse.min():.3e}, {flux_b_reverse.max():.3e}] mJy")

# ──────────────────────────────────────────────────────────────────────
# Comparison
# ──────────────────────────────────────────────────────────────────────
def compare(name, t, flux_ref, flux_test):
    """Print comparison statistics for a single band."""
    mask = (flux_ref > 0) & (flux_test > 0) & np.isfinite(flux_ref) & np.isfinite(flux_test)
    if mask.sum() == 0:
        print(f"  {name}: no valid comparison points!")
        return

    ratio = flux_test[mask] / flux_ref[mask]
    log_ratio = np.log10(ratio)

    print(f"  {name}:")
    print(f"    valid points: {mask.sum()}/{len(t)}")
    print(f"    ratio range:  [{ratio.min():.4f}, {ratio.max():.4f}]")
    print(f"    median ratio: {np.median(ratio):.4f}")
    print(f"    max |log10 ratio|: {np.max(np.abs(log_ratio)):.3f} dex")
    print(f"    mean |log10 ratio|: {np.mean(np.abs(log_ratio)):.3f} dex")

print(f"\n=== Reverse Shock Comparison (model={MODEL}, spread=False) ===")
print(f"=== blastwave / VegasAfterglow (3 GHz) ===")
compare("Total flux", tday, flux_v_total, flux_b_total)
compare("Forward shock", tday, flux_v_fwd, flux_b_forward)
compare("Reverse shock", tday, flux_v_rvs, flux_b_reverse)

# Sample values
print("\n=== Sample values (Total, 3 GHz) ===")
print(f"{'t [day]':>10s}  {'VegasAG [mJy]':>15s}  {'blastwave [mJy]':>18s}  {'ratio':>8s}")
indices = np.linspace(0, len(tday)-1, 8, dtype=int)
for idx in indices:
    v, j = flux_v_total[idx], flux_b_total[idx]
    r = j / v if v > 0 else float('nan')
    print(f"{tday[idx]:10.3e}  {v:15.6e}  {j:18.6e}  {r:8.4f}")

print("\n=== Sample values (Reverse shock, 3 GHz) ===")
print(f"{'t [day]':>10s}  {'VegasAG [mJy]':>15s}  {'blastwave [mJy]':>18s}  {'ratio':>8s}")
for idx in indices:
    v, j = flux_v_rvs[idx], flux_b_reverse[idx]
    r = j / v if v > 0 and j > 0 else float('nan')
    print(f"{tday[idx]:10.3e}  {v:15.6e}  {j:18.6e}  {r:8.4f}")

# ──────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    components = [
        ("Total", flux_v_total, flux_b_total),
        ("Forward shock", flux_v_fwd, flux_b_forward),
        ("Reverse shock", flux_v_rvs, flux_b_reverse),
    ]

    for i, (name, fv, fb) in enumerate(components):
        ax = axes[0, i]
        mask_v = fv > 0
        mask_b = fb > 0
        if mask_v.any():
            ax.loglog(tday[mask_v], fv[mask_v], 'b-',
                      label='VegasAfterglow', linewidth=2)
        if mask_b.any():
            ax.loglog(tday[mask_b], fb[mask_b], 'r--',
                      label='blastwave', linewidth=2)
        ax.set_xlabel('Time [days]')
        ax.set_ylabel('Flux density [mJy]')
        ax.set_title(f'{name} (3 GHz)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax2 = axes[1, i]
        mask = (fv > 0) & (fb > 0) & np.isfinite(fv) & np.isfinite(fb)
        if mask.any():
            ratio = fb[mask] / fv[mask]
            ax2.semilogx(tday[mask], ratio, 'k-', linewidth=1.5)
            ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
            ax2.set_ylim(0.01, 100)
            ax2.set_yscale('log')
        ax2.set_xlabel('Time [days]')
        ax2.set_ylabel('Ratio (blastwave / VegasAG)')
        ax2.set_title(f'{name} ratio')
        ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f'Reverse Shock Comparison (3 GHz, model={MODEL}, spread=False)\n'
        f'VegasAG: {vegas_time:.2f}s | blastwave: {bw_time:.2f}s',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    outpath = '/home/mcoughli/blastwave/tests/compare_vegas_rs.png'
    plt.savefig(outpath, dpi=150)
    print(f"\nPlot saved to {outpath}")
except ImportError:
    print("\nmatplotlib not available, skipping plot")
