"""
Compare blastwave ODE spreading mode against PDE mode and VegasAfterglow
for identical physical parameters (tophat jet, on-axis, ISM).

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
nu_radio   = 1e9   # 1 GHz
nu_optical = 1e14  # optical
nu_xray    = 1e18  # X-ray

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

freqs = np.array([nu_radio, nu_optical, nu_xray])

# Warmup run
model_v = Model(jet=jet_v, medium=medium, observer=obs, fwd_rad=rad_fwd)
_ = model_v.flux_density_grid(tsecond, freqs)

# Timed run
t0 = time.time()
model_v = Model(jet=jet_v, medium=medium, observer=obs, fwd_rad=rad_fwd)
result_v = model_v.flux_density_grid(tsecond, freqs)
vegas_time = time.time() - t0
print(f"VegasAfterglow:     {vegas_time:.4f}s")

flux_v_radio   = np.array(result_v.total[0]) / mJy_cgs
flux_v_optical = np.array(result_v.total[1]) / mJy_cgs
flux_v_xray    = np.array(result_v.total[2]) / mJy_cgs

# ──────────────────────────────────────────────────────────────────────
# blastwave  — PDE mode (spread=True, default)
# ──────────────────────────────────────────────────────────────────────
import blastwave

P = dict(
    Eiso=E_iso, lf=Gamma0, theta_c=theta_c,
    n0=n0, A=0.0,
    eps_e=eps_e, eps_b=eps_B, p=p_val,
    theta_v=theta_v, d=d_Mpc, z=z, s=6,
)

ncells = 129

# Warmup
jet_pde_w = blastwave.Jet(
    blastwave.TopHat(theta_c, E_iso, lf0=Gamma0),
    0.0, n0, tmin=10.0, tmax=1e10,
    grid=blastwave.ForwardJetRes(theta_c, ncells),
    spread=True,
)
_ = jet_pde_w.FluxDensity(tsecond, nu_xray, P)

# Timed
t0 = time.time()
jet_pde = blastwave.Jet(
    blastwave.TopHat(theta_c, E_iso, lf0=Gamma0),
    0.0, n0, tmin=10.0, tmax=1e10,
    grid=blastwave.ForwardJetRes(theta_c, ncells),
    spread=True,
)
flux_pde_radio   = jet_pde.FluxDensity(tsecond, nu_radio,   P)
flux_pde_optical = jet_pde.FluxDensity(tsecond, nu_optical, P)
flux_pde_xray    = jet_pde.FluxDensity(tsecond, nu_xray,    P)
pde_time = time.time() - t0
print(f"blastwave PDE:    {pde_time:.4f}s  ({ncells} cells)")

# ──────────────────────────────────────────────────────────────────────
# blastwave  — ODE mode (spread_mode="ode")
# ──────────────────────────────────────────────────────────────────────

# Warmup
jet_ode_w = blastwave.Jet(
    blastwave.TopHat(theta_c, E_iso, lf0=Gamma0),
    0.0, n0, tmin=10.0, tmax=1e10,
    grid=blastwave.ForwardJetRes(theta_c, ncells),
    spread_mode="ode",
)
_ = jet_ode_w.FluxDensity(tsecond, nu_xray, P)

# Timed
t0 = time.time()
jet_ode = blastwave.Jet(
    blastwave.TopHat(theta_c, E_iso, lf0=Gamma0),
    0.0, n0, tmin=10.0, tmax=1e10,
    grid=blastwave.ForwardJetRes(theta_c, ncells),
    spread_mode="ode",
)
flux_ode_radio   = jet_ode.FluxDensity(tsecond, nu_radio,   P)
flux_ode_optical = jet_ode.FluxDensity(tsecond, nu_optical, P)
flux_ode_xray    = jet_ode.FluxDensity(tsecond, nu_xray,    P)
ode_time = time.time() - t0
print(f"blastwave ODE:    {ode_time:.4f}s  ({ncells} cells)")

# ──────────────────────────────────────────────────────────────────────
# Comparison
# ──────────────────────────────────────────────────────────────────────
def compare(name, t, flux_ref, flux_test, ref_label, test_label):
    """Print comparison statistics for a single band."""
    mask = (flux_ref > 0) & (flux_test > 0) & np.isfinite(flux_ref) & np.isfinite(flux_test)
    if mask.sum() == 0:
        print(f"  {name}: no valid comparison points!")
        return

    ratio = flux_test[mask] / flux_ref[mask]
    log_ratio = np.log10(ratio)

    print(f"  {name}:")
    print(f"    valid points: {mask.sum()}/{len(t)}")
    print(f"    max |log10 ratio|: {np.max(np.abs(log_ratio)):.3f} dex")
    print(f"    mean |log10 ratio|: {np.mean(np.abs(log_ratio)):.3f} dex")

print("\n=== Timing Summary ===")
print(f"  VegasAfterglow:  {vegas_time:.4f}s")
print(f"  blastwave PDE: {pde_time:.4f}s  (ratio: {pde_time/vegas_time:.1f}x)")
print(f"  blastwave ODE: {ode_time:.4f}s  (ratio: {ode_time/vegas_time:.1f}x)")

print("\n=== ODE vs VegasAfterglow ===")
compare("Radio (1 GHz)",     tday, flux_v_radio,   flux_ode_radio,   "VegasAG", "ODE")
compare("Optical (1e14 Hz)", tday, flux_v_optical, flux_ode_optical, "VegasAG", "ODE")
compare("X-ray (1 keV)",     tday, flux_v_xray,    flux_ode_xray,   "VegasAG", "ODE")

print("\n=== ODE vs PDE ===")
compare("Radio (1 GHz)",     tday, flux_pde_radio,   flux_ode_radio,   "PDE", "ODE")
compare("Optical (1e14 Hz)", tday, flux_pde_optical, flux_ode_optical, "PDE", "ODE")
compare("X-ray (1 keV)",     tday, flux_pde_xray,    flux_ode_xray,   "PDE", "ODE")

print("\n=== PDE vs VegasAfterglow ===")
compare("Radio (1 GHz)",     tday, flux_v_radio,   flux_pde_radio,   "VegasAG", "PDE")
compare("Optical (1e14 Hz)", tday, flux_v_optical, flux_pde_optical, "VegasAG", "PDE")
compare("X-ray (1 keV)",     tday, flux_v_xray,    flux_pde_xray,   "VegasAG", "PDE")

# ──────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    bands = [
        ("Radio (1 GHz)", flux_v_radio, flux_pde_radio, flux_ode_radio),
        ("Optical (1e14 Hz)", flux_v_optical, flux_pde_optical, flux_ode_optical),
        ("X-ray (1 keV)", flux_v_xray, flux_pde_xray, flux_ode_xray),
    ]

    for i, (name, fv, fp, fo) in enumerate(bands):
        ax = axes[0, i]
        mask_v = fv > 0
        mask_p = fp > 0
        mask_o = fo > 0
        if mask_v.any():
            ax.loglog(tday[mask_v], fv[mask_v], 'b-', label='VegasAfterglow', linewidth=2)
        if mask_p.any():
            ax.loglog(tday[mask_p], fp[mask_p], 'r--', label='blastwave PDE', linewidth=2)
        if mask_o.any():
            ax.loglog(tday[mask_o], fo[mask_o], 'g:', label='blastwave ODE', linewidth=2)
        ax.set_xlabel('Time [days]')
        ax.set_ylabel('Flux density [mJy]')
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax2 = axes[1, i]
        # ODE/VegasAG ratio
        mask_ov = (fv > 0) & (fo > 0) & np.isfinite(fv) & np.isfinite(fo)
        if mask_ov.any():
            ax2.semilogx(tday[mask_ov], fo[mask_ov]/fv[mask_ov], 'g-', label='ODE/VegasAG', linewidth=1.5)
        # PDE/VegasAG ratio
        mask_pv = (fv > 0) & (fp > 0) & np.isfinite(fv) & np.isfinite(fp)
        if mask_pv.any():
            ax2.semilogx(tday[mask_pv], fp[mask_pv]/fv[mask_pv], 'r--', label='PDE/VegasAG', linewidth=1.5)
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_ylim(0.3, 3.0)
        ax2.set_xlabel('Time [days]')
        ax2.set_ylabel('Ratio vs VegasAfterglow')
        ax2.set_title(f'{name} ratio')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    plt.suptitle(f'TopHat jet, on-axis, {ncells} cells\n'
                 f'VegasAG: {vegas_time:.3f}s | PDE: {pde_time:.3f}s | ODE: {ode_time:.3f}s',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/mcoughli/blastwave/tests/compare_vegas_ode.png', dpi=150)
    print("\nPlot saved to tests/compare_vegas_ode.png")
except ImportError:
    print("\nmatplotlib not available, skipping plot")
