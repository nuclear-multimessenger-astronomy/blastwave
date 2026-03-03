# Reverse Shock: Forward vs Reverse Decomposition

This example demonstrates the reverse shock capability by decomposing the afterglow into forward and reverse shock contributions using the `Jet` class directly.

## Background

When a GRB jet decelerates into the circumburst medium, two shocks form:

- **Forward shock (FS)** --- propagates into the ambient medium, producing the standard long-lived afterglow
- **Reverse shock (RS)** --- propagates back into the ejecta, producing a bright, short-lived flash

The reverse shock emission peaks early (minutes to hours) and fades rapidly, while the forward shock dominates at late times. The relative strength depends on the ejecta magnetization ($\sigma$): unmagnetized ejecta ($\sigma = 0$) produce a strong RS; highly magnetized ejecta suppress the RS.

Key references:

- Sari & Piran 1999, ApJ, 520, 641 --- reverse shock theory
- Kobayashi 2000, ApJ, 545, 807 --- reverse shock emission model
- Zhang, Kobayashi & Meszaros 2003, ApJ, 595, 950 --- magnetized reverse shocks

## Using the Jet class directly

To access the individual forward and reverse shock components, we use the `Jet` class rather than the shortcut functions:

```python
import numpy as np
from blastwave import Jet, TopHat, ForwardJetRes

DAY = 86400.0
theta_c = 0.1

jet = Jet(
    TopHat(theta_c, 1e53, lf0=300.0),
    0.0,                    # nwind (no wind)
    1.0,                    # nism
    tmin=1.0,
    tmax=500 * DAY,
    grid=ForwardJetRes(theta_c, 129),
    tail=True,
    spread=True,
    cal_level=1,
    include_reverse_shock=True,
    sigma=0.0,              # unmagnetized ejecta → strong RS
    eps_e_rs=0.1,
    eps_b_rs=0.01,
    p_rs=2.2,
)
```

Key parameters:

- **`include_reverse_shock=True`** --- enables RS dynamics and emission
- **`sigma=0.0`** --- unmagnetized ejecta, producing the strongest possible reverse shock
- **`eps_e_rs`, `eps_b_rs`, `p_rs`** --- microphysical parameters for the RS (can differ from the FS)

## Computing forward and reverse components

The `Jet` class provides three flux methods:

```python
P = {
    "Eiso":    1e53,
    "lf":      300.0,
    "theta_c": theta_c,
    "A":       0.0,
    "n0":      1.0,
    "eps_e":   0.1,
    "eps_b":   0.01,
    "p":       2.3,
    "theta_v": 0.0,
    "d":       1000.0,
    "z":       0.2,
}

t = np.geomspace(10.0, 300.0 * DAY, 200)
nu_radio = 3e9 * np.ones_like(t)

# Total (FS + RS)
F_total   = jet.FluxDensity(t, nu_radio, P)

# Forward shock only
F_forward = jet.FluxDensity_forward(t, nu_radio, P)

# Reverse shock only
F_reverse = jet.FluxDensity_reverse(t, nu_radio, P)
```

## Plotting

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
t_days = t / DAY

# Radio 3 GHz
ax1.plot(t_days, F_total, '-', color='black', lw=2.5, label='Total')
ax1.plot(t_days, F_forward, '--', color='C0', lw=2, label='Forward shock')
ax1.plot(t_days, F_reverse, '--', color='C3', lw=2, label='Reverse shock')
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlabel('Time (days)'); ax1.set_ylabel('Flux density (mJy)')
ax1.set_title('Radio 3 GHz'); ax1.legend()

# Optical R-band
nu_opt = 4.56e14 * np.ones_like(t)
F_opt_total   = jet.FluxDensity(t, nu_opt, P)
F_opt_forward = jet.FluxDensity_forward(t, nu_opt, P)
F_opt_reverse = jet.FluxDensity_reverse(t, nu_opt, P)

ax2.plot(t_days, F_opt_total, '-', color='black', lw=2.5, label='Total')
ax2.plot(t_days, F_opt_forward, '--', color='C0', lw=2, label='Forward shock')
ax2.plot(t_days, F_opt_reverse, '--', color='C3', lw=2, label='Reverse shock')
ax2.set_xscale('log'); ax2.set_yscale('log')
ax2.set_xlabel('Time (days)'); ax2.set_ylabel('Flux density (mJy)')
ax2.set_title('Optical R-band'); ax2.legend()

fig.suptitle('Forward vs Reverse Shock Decomposition', fontweight='bold')
plt.tight_layout()
plt.savefig('reverse_shock.png', dpi=150)
```

![Reverse shock decomposition](img/reverse_shock.png)

## Discussion

**Radio (3 GHz)**: The reverse shock produces a bright early flash that peaks within the first day, then fades as the RS-heated electrons cool. The forward shock rises more gradually and dominates after $\sim$1 day. At late times ($\gtrsim 10$ days) the light curve is entirely FS-dominated.

**Optical (R-band)**: The RS optical flash is even more pronounced relative to the FS at early times, since the RS electrons are initially very energetic. The optical RS fades faster than the radio RS due to stronger synchrotron cooling at higher frequencies.

!!! note "Magnetization effects"
    Setting `sigma > 0` introduces ejecta magnetization, which suppresses the reverse shock. For $\sigma \gtrsim 1$, the RS becomes negligible and the afterglow is entirely FS-dominated. Try varying `sigma` from 0 to 10 to see this transition.

!!! tip "RS microphysical parameters"
    The RS microphysical parameters (`eps_e_rs`, `eps_b_rs`, `p_rs`) are independent of the FS values. In practice, the RS magnetic field may be stronger than the FS field (higher `eps_b_rs`) if the ejecta carry ordered magnetic fields, while `eps_e_rs` is often assumed similar to `eps_e`.

## Full script

The complete analysis script is at [`examples/reverse_shock.py`](https://github.com/nuclear-multimessenger-astronomy/blastwave/blob/main/examples/reverse_shock.py). To regenerate the plot:

```bash
python examples/reverse_shock.py
```
