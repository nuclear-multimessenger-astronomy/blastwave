"""
Benchmarks: Performance comparison across spread modes, angular resolution,
jet profiles, and radiation models.

Times each configuration with warmup + averaging, producing a 2x2 horizontal
bar chart.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
sys.path.insert(0, "..")
from blastwave import Jet, TopHat, Gaussian, PowerLaw, ForwardJetRes, FluxDensity_tophat, FluxDensity_gaussian, FluxDensity_powerlaw

# ---------- Plot style ----------
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 17,
})

# ---------- Constants ----------
DAY = 86400.0

# ---------- Base parameters ----------
P_base = {
    "Eiso":    1e53,
    "lf":      300.0,
    "theta_c": 0.1,
    "A":       0.0,
    "n0":      1.0,
    "eps_e":   0.1,
    "eps_b":   0.01,
    "p":       2.3,
    "theta_v": 0.0,
    "d":       1000.0,
    "z":       0.2,
}

P_powerlaw = {**P_base, "s": 4.0}

t_bench = np.geomspace(0.1 * DAY, 100.0 * DAY, 50)
nu_bench = 3e9 * np.ones_like(t_bench)

N_WARMUP = 1
N_REPEAT = 3

def bench(func, *args, **kwargs):
    """Run func with warmup + averaging, return mean time in seconds."""
    # Warmup
    for _ in range(N_WARMUP):
        func(*args, **kwargs)
    # Timed runs
    times = []
    for _ in range(N_REPEAT):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.mean(times)

# ---------- 1. Spread modes ----------
print("=== Spread modes ===")
spread_labels = ['PDE', 'ODE', 'No spread']
spread_times = []

# PDE spread
print("  PDE...")
t_pde = bench(FluxDensity_tophat, t_bench, nu_bench, P_base,
              tmin=1.0, tmax=200*DAY, spread=True)
spread_times.append(t_pde)
print(f"    {t_pde:.3f} s")

# ODE spread — use Jet class directly with spread_mode="ode"
print("  ODE...")
def run_ode():
    jet = Jet(
        TopHat(0.1, 1e53, lf0=300.0), 0.0, 1.0,
        tmin=1.0, tmax=200*DAY,
        grid=ForwardJetRes(0.1, 129), tail=True,
        spread=True, spread_mode="ode", cal_level=1,
    )
    return jet.FluxDensity(t_bench, nu_bench, P_base)
t_ode = bench(run_ode)
spread_times.append(t_ode)
print(f"    {t_ode:.3f} s")

# No spread
print("  No spread...")
t_nospread = bench(FluxDensity_tophat, t_bench, nu_bench, P_base,
                   tmin=1.0, tmax=200*DAY, spread=False)
spread_times.append(t_nospread)
print(f"    {t_nospread:.3f} s")

# ---------- 2. Angular resolution ----------
print("=== Angular resolution ===")
ntheta_values = [17, 33, 65, 129, 257]
res_times = []

for n in ntheta_values:
    print(f"  {n} cells...")
    def run_res(n=n):
        jet = Jet(
            TopHat(0.1, 1e53, lf0=300.0), 0.0, 1.0,
            tmin=1.0, tmax=200*DAY,
            grid=ForwardJetRes(0.1, n), tail=True,
            spread=True, cal_level=1,
        )
        return jet.FluxDensity(t_bench, nu_bench, P_base)
    t_res = bench(run_res)
    res_times.append(t_res)
    print(f"    {t_res:.3f} s")

# ---------- 3. Jet profiles ----------
print("=== Jet profiles ===")
profile_labels = ['TopHat', 'Gaussian', 'Power-law']
profile_times = []

print("  TopHat...")
t_th = bench(FluxDensity_tophat, t_bench, nu_bench, P_base,
             tmin=1.0, tmax=200*DAY)
profile_times.append(t_th)
print(f"    {t_th:.3f} s")

print("  Gaussian...")
t_gauss = bench(FluxDensity_gaussian, t_bench, nu_bench, P_base,
                tmin=1.0, tmax=200*DAY)
profile_times.append(t_gauss)
print(f"    {t_gauss:.3f} s")

print("  Power-law...")
t_pl = bench(FluxDensity_powerlaw, t_bench, nu_bench, P_powerlaw,
             tmin=1.0, tmax=200*DAY)
profile_times.append(t_pl)
print(f"    {t_pl:.3f} s")

# ---------- 4. Radiation models ----------
print("=== Radiation models ===")
model_labels = ['sync', 'sync_ssa', 'numeric']
model_times = []

for model in model_labels:
    print(f"  {model}...")
    t_model = bench(FluxDensity_tophat, t_bench, nu_bench, P_base,
                    tmin=1.0, tmax=200*DAY, model=model)
    model_times.append(t_model)
    print(f"    {t_model:.3f} s")

# ---------- Plot ----------
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
bar_color = '#4C72B0'
bar_height = 0.5

# Panel 1: Spread modes
ax = axes[0, 0]
y_pos = np.arange(len(spread_labels))
ax.barh(y_pos, spread_times, height=bar_height, color=bar_color)
ax.set_yticks(y_pos)
ax.set_yticklabels(spread_labels)
ax.set_xlabel('Time (s)')
ax.set_title('Spread Mode')
for i, v in enumerate(spread_times):
    ax.text(v + max(spread_times) * 0.02, i, f'{v:.2f}s', va='center', fontsize=11)

# Panel 2: Angular resolution
ax = axes[0, 1]
y_pos = np.arange(len(ntheta_values))
labels_res = [str(n) for n in ntheta_values]
ax.barh(y_pos, res_times, height=bar_height, color=bar_color)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels_res)
ax.set_xlabel('Time (s)')
ax.set_title('Angular Resolution (cells)')
for i, v in enumerate(res_times):
    ax.text(v + max(res_times) * 0.02, i, f'{v:.2f}s', va='center', fontsize=11)

# Panel 3: Jet profiles
ax = axes[1, 0]
y_pos = np.arange(len(profile_labels))
ax.barh(y_pos, profile_times, height=bar_height, color=bar_color)
ax.set_yticks(y_pos)
ax.set_yticklabels(profile_labels)
ax.set_xlabel('Time (s)')
ax.set_title('Jet Profile')
for i, v in enumerate(profile_times):
    ax.text(v + max(profile_times) * 0.02, i, f'{v:.2f}s', va='center', fontsize=11)

# Panel 4: Radiation models
ax = axes[1, 1]
y_pos = np.arange(len(model_labels))
ax.barh(y_pos, model_times, height=bar_height, color=bar_color)
ax.set_yticks(y_pos)
ax.set_yticklabels(model_labels)
ax.set_xlabel('Time (s)')
ax.set_title('Radiation Model')
for i, v in enumerate(model_times):
    ax.text(v + max(model_times) * 0.02, i, f'{v:.2f}s', va='center', fontsize=11)

fig.suptitle('blastwave Performance Benchmarks', fontweight='bold', fontsize=17)
plt.tight_layout()

outdir = os.path.join(os.path.dirname(__file__), '..', 'docs', 'examples', 'img')
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, 'benchmarks.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"\nSaved {outpath}")
plt.show()
