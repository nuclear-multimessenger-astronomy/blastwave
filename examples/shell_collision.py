"""
Shell collision (refreshed shock) example.

Demonstrates the trailing_shells parameter for modeling refreshed shocks
from delayed collisions between relativistic shells, following the
framework of Akl et al. (2026, arXiv:2603.08555) for GRB 250129A.

Produces two figures:
  1. shell_collision_demo.png — baseline vs 2-shell collision at radio, optical, X-ray
  2. shell_collision_parameter_study.png — varying shell energy and Lorentz factor
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

from blastwave import Jet, TopHat, ForwardJetRes

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
})

outdir = os.path.join(os.path.dirname(__file__), "..", "docs", "examples", "img")
os.makedirs(outdir, exist_ok=True)

DAY = 86400.0

# ══════════════════════════════════════════════════════════════════════
# Physical parameters (inspired by GRB 250129A, Akl et al. 2026 Table 4)
# ══════════════════════════════════════════════════════════════════════
theta_j  = np.radians(6.0)
n_ism    = 1.0
E1       = 1.7e53      # leading shell
G1       = 100.0
eps_e    = 0.09
eps_b    = 2.5e-3
p_val    = 2.3
z        = 1.0
d_Mpc    = 6700.0      # approximate for z~1

# Trailing shells
E2 = 1e54              # 1st collision (energetic)
G2 = 40.0
t_launch_2 = 83.0      # seconds after leading shell

E3 = 8e53              # 2nd collision
G3 = 25.0
t_launch_3 = 83.0 + 0.5 * DAY  # 0.5 days after first shell

P = dict(
    Eiso=E1, lf=G1, theta_c=theta_j, A=0.0, n0=n_ism,
    p=p_val, eps_e=eps_e, eps_b=eps_b,
    theta_v=0.0, z=z, d=d_Mpc,
)

ncells = 65
t = np.geomspace(1e2, 5e7, 400)
t_days = t / DAY

# ══════════════════════════════════════════════════════════════════════
# Figure 1: Multi-band comparison (baseline vs shell collision)
# ══════════════════════════════════════════════════════════════════════
print("Building jets...")
t0 = time.time()

jet_base = Jet(
    TopHat(theta_j, E1, lf0=G1), 0.0, n_ism,
    tmin=10.0, tmax=5e7, spread=False,
    grid=ForwardJetRes(theta_j, ncells), tail=True,
)
print(f"  Baseline jet: {time.time() - t0:.2f}s")

t0 = time.time()
jet_coll = Jet(
    TopHat(theta_j, E1, lf0=G1), 0.0, n_ism,
    tmin=10.0, tmax=5e7, spread=False,
    grid=ForwardJetRes(theta_j, ncells), tail=True,
    trailing_shells=[
        (E2, G2, t_launch_2),
        (E3, G3, t_launch_3),
    ],
)
print(f"  Collision jet: {time.time() - t0:.2f}s")

bands = [
    (1e9,     "Radio 1 GHz",  "#1f77b4"),
    (4.68e14, "Optical R-band", "#ff7f0e"),
    (4.84e17, "X-ray 2 keV",  "#d62728"),
]

print("Computing light curves...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (nu, label, color) in zip(axes, bands):
    t0 = time.time()
    f_base = np.array(jet_base.FluxDensity(t, nu, P))
    f_coll = np.array(jet_coll.FluxDensity(t, nu, P))
    print(f"  {label}: {time.time() - t0:.2f}s")

    mask_b = f_base > 0
    mask_c = f_coll > 0

    ax.loglog(t_days[mask_b], f_base[mask_b], '--', color=color, lw=1.5,
              alpha=0.6, label='No collision')
    ax.loglog(t_days[mask_c], f_coll[mask_c], '-', color=color, lw=2.5,
              label='Shell collision')

    # Mark collision times
    for tc_s, lbl in [(t_launch_2, r'$t_1$'), (t_launch_3, r'$t_2$')]:
        # Approximate lab-frame collision time: shell catches blast wave
        # (for annotation — actual collision is computed dynamically)
        ax.axvline(tc_s / DAY, color='gray', ls=':', lw=0.8, alpha=0.5)

    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Flux density [mJy]')
    ax.set_title(label)
    ax.legend(loc='best')
    ax.set_xlim(1e-3, 500)
    ax.grid(True, alpha=0.2)

fig.suptitle('Shell Collision: Refreshed Shock Rebrightening', fontweight='bold')
plt.tight_layout()
outpath = os.path.join(outdir, "shell_collision_demo.png")
plt.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"Saved {outpath}")
plt.close()

# ══════════════════════════════════════════════════════════════════════
# Figure 2: Parameter study — varying shell energy and Lorentz factor
# ══════════════════════════════════════════════════════════════════════
print("\nParameter study...")
nu_opt = 4.68e14  # R-band

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Left panel: vary shell energy (fixed Gamma2=40)
f_base = np.array(jet_base.FluxDensity(t, nu_opt, P))
ax1.loglog(t_days, f_base, 'k--', lw=1.5, alpha=0.5, label='Baseline')

for E_shell, ls, alpha in [(3e53, '-', 0.5), (1e54, '-', 0.7), (3e54, '-', 1.0)]:
    t0 = time.time()
    jet_e = Jet(
        TopHat(theta_j, E1, lf0=G1), 0.0, n_ism,
        tmin=10.0, tmax=5e7, spread=False,
        grid=ForwardJetRes(theta_j, ncells), tail=True,
        trailing_shells=[(E_shell, 40.0, 83.0)],
    )
    f = np.array(jet_e.FluxDensity(t, nu_opt, P))
    mask = f > 0
    ax1.loglog(t_days[mask], f[mask], ls, lw=2, alpha=alpha,
               label=rf'$E_2 = {E_shell:.0e}$ erg')
    print(f"  E_shell={E_shell:.0e}: {time.time() - t0:.2f}s")

ax1.set_xlabel('Time [days]')
ax1.set_ylabel('Flux density [mJy]')
ax1.set_title(r'Varying shell energy ($\Gamma_2 = 40$)')
ax1.legend(fontsize=9)
ax1.set_xlim(1e-3, 500)
ax1.grid(True, alpha=0.2)

# Right panel: vary shell Lorentz factor (fixed E2=1e54)
ax2.loglog(t_days, f_base, 'k--', lw=1.5, alpha=0.5, label='Baseline')

for G_shell, ls, alpha in [(20.0, '-', 0.5), (40.0, '-', 0.7), (80.0, '-', 1.0)]:
    t0 = time.time()
    jet_g = Jet(
        TopHat(theta_j, E1, lf0=G1), 0.0, n_ism,
        tmin=10.0, tmax=5e7, spread=False,
        grid=ForwardJetRes(theta_j, ncells), tail=True,
        trailing_shells=[(1e54, G_shell, 83.0)],
    )
    f = np.array(jet_g.FluxDensity(t, nu_opt, P))
    mask = f > 0
    ax2.loglog(t_days[mask], f[mask], ls, lw=2, alpha=alpha,
               label=rf'$\Gamma_2 = {G_shell:.0f}$')
    print(f"  G_shell={G_shell:.0f}: {time.time() - t0:.2f}s")

ax2.set_xlabel('Time [days]')
ax2.set_ylabel('Flux density [mJy]')
ax2.set_title(r'Varying shell Lorentz factor ($E_2 = 10^{54}$ erg)')
ax2.legend(fontsize=9)
ax2.set_xlim(1e-3, 500)
ax2.grid(True, alpha=0.2)

fig.suptitle('Shell Collision Parameter Study (Optical R-band)', fontweight='bold')
plt.tight_layout()
outpath = os.path.join(outdir, "shell_collision_parameter_study.png")
plt.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"Saved {outpath}")
plt.close()

print("\nDone!")
