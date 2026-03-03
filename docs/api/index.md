# API Reference

## Python API (`blastwave`)

### Jet profiles

```python
TopHat(theta_c, Eiso, lf0=1e100)
```
Top-hat jet: uniform energy `Eiso` within half-opening angle `theta_c`, zero outside.

```python
Gaussian(theta_c, Eiso, lf0=1e100)
```
Gaussian jet: $E(\theta) = E_\mathrm{iso} \exp(-\theta^2 / 2\theta_c^2)$.

```python
PowerLaw(theta_c, Eiso, lf0=1e100, s=4.0)
```
Power-law jet: $E(\theta) = E_\mathrm{iso} (1 + (\theta/\theta_c)^2)^{-s/2}$.

```python
Spherical(Eiso, lf0=1e100)
```
Spherical explosion: uniform energy at all angles. Triggers the tophat fast path (1-cell solve) for efficiency.

### Angular grids

```python
Uniform(npoints) -> ndarray
```
Uniform spacing from $0$ to $\pi$.

```python
ForwardJetRes(theta_c, npoints) -> ndarray
```
Arcsinh spacing refined near $\theta = 0$ (forward jet). Standard choice for on-axis and moderately off-axis jets.

```python
CounterJetRes(theta_c, npoints) -> ndarray
```
Refined near $\theta = \pi$ (counter-jet).

```python
ForwardCounterJetRes(theta_c, npoints) -> ndarray
```
Refined near both poles.

### `Jet` class

```python
class Jet:
    def __init__(
        self,
        profiles,             # (theta, energy, lorentz_factor) arrays
        nwind,                # wind density scale (cm^-3)
        nism,                 # ISM density (cm^-3)
        tmin=10.0,            # simulation start time (s)
        tmax=1e10,            # simulation end time (s)
        grid=Uniform(257),    # angular cell edges
        tail=True,            # add isotropic low-energy tail
        spread=True,          # enable lateral spreading
        spread_mode=None,     # "none", "ode", or "pde"
        k=2.0,                # CSM density power-law index
        cal_level=1,          # calibration level (0, 1, 2)
        rtol=1e-6,            # ODE solver tolerance
        cfl=0.9,              # CFL number (PDE only)
        include_reverse_shock=False,
        sigma=0.0,            # ejecta magnetization
        eps_e_rs=0.1,         # RS electron energy fraction
        eps_b_rs=0.01,        # RS magnetic energy fraction
        p_rs=2.3,             # RS electron spectral index
        t0_injection=0.0,     # energy injection timescale (s)
        l_injection=0.0,      # injection luminosity (erg/s)
        m_dot_injection=0.0,  # mass injection rate (g/s)
    ): ...
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `profiles` | tuple | `(theta, energy, lf)` arrays from a profile function |
| `nwind` | float | Wind density scale: $n_\mathrm{wind}$ in $n(r) = n_\mathrm{wind}(r/10^{17})^{-k} + n_\mathrm{ISM}$ |
| `nism` | float | ISM number density (cm$^{-3}$) |
| `k` | float | CSM density power-law index. $k=0$ (ISM), $k=2$ (wind). Default 2.0 |
| `spread_mode` | str | `"none"`, `"ode"`, or `"pde"`. Overrides `spread` flag |
| `tail` | bool | Add isotropic low-energy tail (safety for off-axis). Set `False` for spherical |
| `grid` | ndarray | Cell edge angles from 0 to $\pi$ |

#### Methods

```python
Jet.FluxDensity(t, nu, P, model="sync", rtol=1e-3,
                max_iter=100, force_return=True, flux_method=None) -> ndarray
```
Compute flux density in mJy at observer times `t` (s) and frequencies `nu` (Hz).

`P` is a dict with keys:

| Key | Description |
|-----|-------------|
| `eps_e` | Electron energy fraction |
| `eps_b` | Magnetic energy fraction |
| `p` | Electron spectral index |
| `theta_v` | Viewing angle (rad) |
| `d` | Luminosity distance (Mpc) |
| `z` | Redshift |

```python
Jet.FluxDensity_forward(t, nu, P, ...) -> ndarray
Jet.FluxDensity_reverse(t, nu, P, ...) -> ndarray
```
Forward-shock-only and reverse-shock-only flux density (when `include_reverse_shock=True`).

#### Properties

```python
Jet.t_pde -> ndarray           # time grid (s), shape (nt,)
Jet.y_pde -> ndarray           # solution [5, ntheta, nt]: Msw, Mej, beta_gamma_sq, beta_th, R
Jet.theta_pde -> ndarray       # cell centers (rad), shape (ntheta,)
```

#### Interpolation

```python
Jet.beta_gamma(t, theta) -> float   # four-velocity
Jet.beta_theta(t, theta) -> float   # lateral velocity
Jet.R(t, theta) -> float            # radius (cm)
Jet.dMsw_dOmega(t, theta) -> float  # swept mass per sr (g/sr)
Jet.dMej_dOmega(t, theta) -> float  # ejecta mass per sr (g/sr)
Jet.dE0_dOmega(t, theta) -> float   # energy per sr (erg/sr)
```

### Shortcut functions

```python
FluxDensity_tophat(t, nu, P, tmin=10.0, tmax=1e10, spread=True,
                   cal_level=1, model="sync", rtol=1e-3, flux_method=None,
                   include_rs=False, rs_params=None, ebl=False,
                   magnetar_l0=None, magnetar_t0=None,
                   magnetar_q=None, magnetar_ts=None) -> ndarray
```

```python
FluxDensity_gaussian(t, nu, P, ...) -> ndarray
FluxDensity_powerlaw(t, nu, P, ...) -> ndarray
```

```python
FluxDensity_spherical(t, nu, P, k=2.0, tmin=10.0, tmax=1e10, ntheta=17,
                      cal_level=1, model="sync", rtol=1e-3,
                      flux_method=None, include_rs=False,
                      rs_params=None) -> ndarray
```
Spherical explosion shortcut. Uses `spread=False`, `tail=False`, `Uniform(ntheta)` grid. The `k` parameter sets the CSM density power-law index.

### Radiation models

| String | Description | See |
|--------|-------------|-----|
| `"sync"` | Optically thin synchrotron (Sari+ 1998) | [Radiation](../physics/radiation.md) |
| `"sync_dnp"` | Deep Newtonian phase correction | [Radiation](../physics/radiation.md) |
| `"sync_ssa"` | Synchrotron + self-absorption | [Radiation](../physics/radiation.md) |
| `"sync_ssc"` | Synchrotron self-Compton (+ Klein-Nishina) | [Radiation](../physics/radiation.md) |
| `"sync_thermal"` | Thermal + non-thermal synchrotron (MQ21) | [Radiation](../physics/radiation.md) |
| `"numeric"` | Chang-Cooper electron kinetic equation | [Numeric Model](numeric.md) |

### Parameter dictionary `P`

All shortcut functions and `Jet.FluxDensity()` take a parameter dictionary `P`:

| Key | Units | Description |
|-----|-------|-------------|
| `Eiso` | erg | Isotropic equivalent energy |
| `lf` | --- | Initial Lorentz factor |
| `theta_c` | rad | Jet core half-opening angle (not needed for `Spherical`) |
| `s` | --- | Power-law structure index (only for `PowerLaw`) |
| `A` | cm$^{-3}$ | Wind density scale ($n_\mathrm{wind}$) |
| `n0` | cm$^{-3}$ | ISM number density ($n_\mathrm{ISM}$) |
| `eps_e` | --- | Electron energy fraction |
| `eps_b` | --- | Magnetic energy fraction |
| `p` | --- | Electron spectral index ($p > 2$) |
| `theta_v` | rad | Viewing angle |
| `d` | Mpc | Luminosity distance |
| `z` | --- | Redshift |

#### Model-specific parameters

| Key | Units | Models | Description |
|-----|-------|--------|-------------|
| `eps_T` | --- | `sync_thermal` | Electron thermalization efficiency (default 1.0) |
| `delta` | --- | `sync_thermal` | Power-law energy fraction (default `eps_e/eps_T`) |
| `full_volume` | --- | `sync_thermal` | Set to 1.0 for FM25 full-volume post-shock |
| `k` | --- | `sync_thermal` | CSM power-law index for full-volume mode |
| `ssc_kn` | --- | `sync_ssc` | Set to 1 for Klein-Nishina corrections |
| `n_gamma` | --- | `numeric` | Number of gamma bins (default 300) |
| `gamma_max` | --- | `numeric` | Maximum electron Lorentz factor (default 1e8) |
| `include_pp` | --- | `numeric` | Set to 1 for pair production |

#### Energy injection (magnetar spin-down)

The shortcut functions accept optional magnetar parameters:

| Keyword | Units | Description |
|---------|-------|-------------|
| `magnetar_l0` | erg/s | Initial spin-down luminosity |
| `magnetar_t0` | s | Spin-down timescale |
| `magnetar_q` | --- | Temporal decay index (default 2.0) |
| `magnetar_ts` | s | Start time for injection |

Injection luminosity: $L(t) = L_0 (1 + (t - t_s) / t_0)^{-q}$.
