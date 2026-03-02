# Quick Start

## Core workflow

1. Define a **jet profile** (energy and Lorentz factor vs. angle)
2. Create a `Jet` object with medium density, time range, and solver options
3. Call `FluxDensity()` with observer parameters to compute light curves

## Shortcut functions

For common jet profiles, shortcut functions handle steps 1--2 automatically:

```python
import numpy as np
from blastwave import FluxDensity_tophat, FluxDensity_gaussian, FluxDensity_spherical

# Observer times and frequencies
t = np.geomspace(1e3, 1e8, 200)  # seconds
nu = 1e9 * np.ones_like(t)        # 1 GHz

# Physical parameters
P = {
    "Eiso": 1e52,       # isotropic equivalent energy (erg)
    "lf": 300,           # initial Lorentz factor
    "theta_c": 0.1,      # jet core half-opening angle (rad)
    "A": 0.0,            # wind density scale (cm^-3)
    "n0": 1.0,           # ISM number density (cm^-3)
    "eps_e": 0.1,        # electron energy fraction
    "eps_b": 0.01,       # magnetic energy fraction
    "p": 2.2,            # electron spectral index
    "theta_v": 0.3,      # viewing angle (rad)
    "d": 1000.0,         # luminosity distance (Mpc)
    "z": 0.1,            # redshift
}

flux = FluxDensity_tophat(t, nu, P)
```

### Available shortcuts

| Function | Profile | Required keys in `P` |
|----------|---------|---------------------|
| `FluxDensity_tophat` | Top-hat jet | `Eiso`, `lf`, `theta_c` |
| `FluxDensity_gaussian` | Gaussian jet | `Eiso`, `lf`, `theta_c` |
| `FluxDensity_powerlaw` | Power-law jet | `Eiso`, `lf`, `theta_c`, `s` |
| `FluxDensity_spherical` | Spherical explosion | `Eiso`, `lf` |

All shortcuts accept common keyword arguments:

| Keyword | Default | Description |
|---------|---------|-------------|
| `tmin` | `10.0` | Simulation start time (s) |
| `tmax` | `1e10` | Simulation end time (s) |
| `model` | `"sync"` | Radiation model |
| `rtol` | `1e-3` | Integration tolerance |
| `flux_method` | `None` | `"forward"` for forward-mapping flux |
| `cal_level` | `1` | Calibration level (0, 1, or 2) |

`FluxDensity_spherical` additionally accepts `k` (CSM density power-law index, default 2.0) and `ntheta` (number of angular cells, default 17).

## Using the Jet class directly

For more control, use the `Jet` class:

```python
from blastwave import Jet, TopHat, Gaussian, Spherical
from blastwave import Uniform, ForwardJetRes

# Create a Gaussian jet with ODE spreading
jet = Jet(
    Gaussian(theta_c=0.1, Eiso=1e52, lf0=300),
    nwind=0.0,
    nism=1.0,
    tmin=10.0,
    tmax=1e8,
    grid=ForwardJetRes(0.1, 129),
    spread_mode="ode",
    cal_level=1,
)

# Compute flux density
flux = jet.FluxDensity(t, nu, P, model="sync", rtol=1e-3)
```

### Jet profiles

```python
TopHat(theta_c, Eiso, lf0=1e100)        # uniform energy within theta_c
Gaussian(theta_c, Eiso, lf0=1e100)      # E ~ exp(-theta^2 / 2 theta_c^2)
PowerLaw(theta_c, Eiso, lf0=1e100, s=4) # E ~ (1 + (theta/theta_c)^2)^{-s/2}
Spherical(Eiso, lf0=1e100)              # uniform energy at all angles
```

### Angular grids

```python
Uniform(npoints)                          # equal spacing [0, pi]
ForwardJetRes(theta_c, npoints)          # arcsinh spacing, refined near theta=0
CounterJetRes(theta_c, npoints)          # refined near theta=pi
ForwardCounterJetRes(theta_c, npoints)   # refined near both poles
```

### Spreading modes

| Mode | `spread_mode=` | Description |
|------|----------------|-------------|
| No spread | `"none"` | Cells evolve independently (fastest) |
| ODE spread | `"ode"` | Per-cell ODE spreading (VegasAfterglow-style) |
| PDE spread | `"pde"` | Finite-volume PDE (most accurate, CFL-limited) |

For backward compatibility, `spread=True` uses PDE and `spread=False` uses no-spread. The `spread_mode` keyword overrides `spread` when set.

### Radiation models

| Model string | Description |
|--------------|-------------|
| `"sync"` | Synchrotron (optically thin) |
| `"sync_ssa"` | Synchrotron + self-absorption |
| `"sync_dnp"` | Deep Newtonian phase synchrotron |

### CSM density profile

The circumburst medium density is:

$$n(r) = n_\mathrm{wind} \left(\frac{r}{10^{17}\,\mathrm{cm}}\right)^{-k} + n_\mathrm{ISM}$$

The power-law index $k$ is set via the `k` parameter on `Jet()` or `FluxDensity_spherical()`:

- $k = 0$: constant density (ISM)
- $k = 2$: stellar wind ($\dot{M}/v_w$ profile)
- Intermediate values for more complex environments

```python
# ISM-like environment
jet = Jet(Spherical(1e49, lf0=2.0), nwind=0.0, nism=1.0, k=0.0, ...)

# Wind environment
jet = Jet(Spherical(1e49, lf0=2.0), nwind=1.0, nism=0.0, k=2.0, ...)

# Intermediate CSM (k=1.5)
jet = Jet(Spherical(1e49, lf0=2.0), nwind=1.0, nism=0.0, k=1.5, ...)
```

## Plotting a multi-frequency light curve

```python
import numpy as np
import matplotlib.pyplot as plt
from blastwave import FluxDensity_tophat

P = {
    "Eiso": 1e52, "lf": 300, "theta_c": 0.1,
    "A": 0.0, "n0": 1.0,
    "eps_e": 0.1, "eps_b": 0.01, "p": 2.2,
    "theta_v": 0.3, "d": 1000.0, "z": 0.1,
}

t = np.geomspace(1e3, 1e8, 200)

for nu_val, label in [(1e9, "1 GHz"), (1e14, "Optical"), (1e17, "X-ray")]:
    nu = nu_val * np.ones_like(t)
    flux = FluxDensity_tophat(t, nu, P)
    plt.plot(t / 86400, flux, label=label)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Time (days)")
plt.ylabel("Flux density (mJy)")
plt.legend()
plt.show()
```

## Accessing hydrodynamic data

The `Jet` object stores the PDE/ODE solution and supports interpolation:

```python
jet = Jet(TopHat(0.1, 1e52, 300), nwind=0, nism=1, ...)

# Raw PDE solution arrays
t_pde = jet.t_pde              # time grid (s)
y_pde = jet.y_pde              # [5, ntheta, nt]: Msw, Mej, beta_gamma_sq, beta_th, R
theta_pde = jet.theta_pde      # cell centers (rad)

# Interpolated quantities at arbitrary (t, theta)
bg = jet.beta_gamma(1e5, 0.05)  # four-velocity at t=1e5 s, theta=0.05 rad
r = jet.R(1e5, 0.05)            # radius (cm)
```
