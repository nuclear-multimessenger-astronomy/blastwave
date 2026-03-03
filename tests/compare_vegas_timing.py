"""
Timing comparison: blastwave PDE vs ODE vs VegasAfterglow
across different cell counts and jet profiles.
"""
import numpy as np
import time

# ── Shared physical parameters ──
theta_c = 0.1
E_iso   = 1e52
Gamma0  = 300.0
n0      = 1.0
eps_e   = 0.1
eps_B   = 0.01
p_val   = 2.17
theta_v = 0.0
d_Mpc   = 474.33
z       = 0.1

tday = np.logspace(-2, 3, 50)
tsecond = tday * 86400.0
nu_xray = 1e18

MPC_cm = 3.0856775814913673e24
mJy_cgs = 1e-26

P = dict(
    Eiso=E_iso, lf=Gamma0, theta_c=theta_c,
    n0=n0, A=0.0,
    eps_e=eps_e, eps_b=eps_B, p=p_val,
    theta_v=theta_v, d=d_Mpc, z=z, s=6,
)

# ── VegasAfterglow baseline ──
from VegasAfterglow import TophatJet, GaussianJet, ISM, Observer, Radiation, Model
import blastwave

def time_vegas_tophat():
    jet_v = TophatJet(theta_c=theta_c, E_iso=E_iso, Gamma0=Gamma0)
    medium = ISM(n_ism=n0)
    obs = Observer(lumi_dist=d_Mpc * MPC_cm, z=z, theta_obs=theta_v)
    rad_fwd = Radiation(eps_e=eps_e, eps_B=eps_B, p=p_val)
    # warmup
    m = Model(jet=jet_v, medium=medium, observer=obs, fwd_rad=rad_fwd)
    _ = m.flux_density_grid(tsecond, np.array([nu_xray]))
    # timed
    t0 = time.time()
    m = Model(jet=jet_v, medium=medium, observer=obs, fwd_rad=rad_fwd)
    _ = m.flux_density_grid(tsecond, np.array([nu_xray]))
    return time.time() - t0

def time_vegas_gaussian():
    jet_v = GaussianJet(theta_c=theta_c, E_iso=E_iso, Gamma0=Gamma0)
    medium = ISM(n_ism=n0)
    obs = Observer(lumi_dist=d_Mpc * MPC_cm, z=z, theta_obs=theta_v)
    rad_fwd = Radiation(eps_e=eps_e, eps_B=eps_B, p=p_val)
    # warmup
    m = Model(jet=jet_v, medium=medium, observer=obs, fwd_rad=rad_fwd)
    _ = m.flux_density_grid(tsecond, np.array([nu_xray]))
    # timed
    t0 = time.time()
    m = Model(jet=jet_v, medium=medium, observer=obs, fwd_rad=rad_fwd)
    _ = m.flux_density_grid(tsecond, np.array([nu_xray]))
    return time.time() - t0

def time_blastwave(profile, ncells, spread_mode):
    """Time blastwave with given profile, cells, and spread mode."""
    kwargs = {}
    if spread_mode == "pde":
        kwargs["spread"] = True
    elif spread_mode == "ode":
        kwargs["spread_mode"] = "ode"
    else:
        kwargs["spread"] = False

    # warmup
    jet = blastwave.Jet(
        profile, 0.0, n0, tmin=10.0, tmax=1e10,
        grid=blastwave.ForwardJetRes(theta_c, ncells),
        **kwargs,
    )
    _ = jet.FluxDensity(tsecond, nu_xray, P)

    # timed
    t0 = time.time()
    jet = blastwave.Jet(
        profile, 0.0, n0, tmin=10.0, tmax=1e10,
        grid=blastwave.ForwardJetRes(theta_c, ncells),
        **kwargs,
    )
    _ = jet.FluxDensity(tsecond, nu_xray, P)
    return time.time() - t0

# ── Run timing ──
cell_counts = [17, 33, 65, 129, 257]

print("=" * 70)
print("TOPHAT JET (on-axis, X-ray 1 keV)")
print("=" * 70)

vegas_th = time_vegas_tophat()
print(f"  VegasAfterglow:  {vegas_th:.4f}s")
print()

tophat_profile = blastwave.TopHat(theta_c, E_iso, lf0=Gamma0)

print(f"  {'Cells':>6s}  {'PDE':>10s}  {'ODE':>10s}  {'No-spread':>10s}  {'PDE/VA':>8s}  {'ODE/VA':>8s}")
print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}")

for nc in cell_counts:
    t_pde = time_blastwave(tophat_profile, nc, "pde")
    t_ode = time_blastwave(tophat_profile, nc, "ode")
    t_none = time_blastwave(tophat_profile, nc, "none")
    print(f"  {nc:6d}  {t_pde:10.4f}s  {t_ode:10.4f}s  {t_none:10.4f}s  {t_pde/vegas_th:8.1f}x  {t_ode/vegas_th:8.1f}x")

print()
print("=" * 70)
print("GAUSSIAN JET (on-axis, X-ray 1 keV)")
print("=" * 70)

vegas_g = time_vegas_gaussian()
print(f"  VegasAfterglow:  {vegas_g:.4f}s")
print()

gaussian_profile = blastwave.Gaussian(theta_c, E_iso, lf0=Gamma0)

print(f"  {'Cells':>6s}  {'PDE':>10s}  {'ODE':>10s}  {'No-spread':>10s}  {'PDE/VA':>8s}  {'ODE/VA':>8s}")
print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}")

for nc in cell_counts:
    t_pde = time_blastwave(gaussian_profile, nc, "pde")
    t_ode = time_blastwave(gaussian_profile, nc, "ode")
    t_none = time_blastwave(gaussian_profile, nc, "none")
    print(f"  {nc:6d}  {t_pde:10.4f}s  {t_ode:10.4f}s  {t_none:10.4f}s  {t_pde/vegas_g:8.1f}x  {t_ode/vegas_g:8.1f}x")
