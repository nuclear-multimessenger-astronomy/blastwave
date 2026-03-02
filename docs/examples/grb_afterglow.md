# Physics Showcase: GRB Afterglow Effects

This example demonstrates how different physical ingredients affect GRB afterglow light curves. Using standard afterglow parameters, we show the impact of observing frequency, viewing angle, lateral spreading, and jet structure.

## Physical parameters

We use a typical long GRB afterglow in an ISM environment:

| Parameter | Value | Description |
|-----------|-------|-------------|
| \(E_\mathrm{iso}\) | \(10^{53}\) erg | Isotropic equivalent energy |
| \(\Gamma_0\) | 300 | Initial Lorentz factor |
| \(\theta_c\) | 0.1 rad (5.7 deg) | Half-opening angle |
| \(n_0\) | 1 cm\(^{-3}\) | ISM number density |
| \(\varepsilon_e\) | 0.1 | Electron energy fraction |
| \(\varepsilon_B\) | 0.01 | Magnetic energy fraction |
| \(p\) | 2.2 | Electron power-law index |
| \(d_L\) | 1000 Mpc | Luminosity distance |
| \(z\) | 0.2 | Redshift |

## Multi-band light curves

The on-axis tophat jet model produces light curves spanning from radio to X-ray:

- **X-ray** (1 keV): Decays as a power law from the start, since \(\nu_X > \nu_c > \nu_m\)
- **Optical** (R-band): Peaks at \(\sim 10^{-3}\) days as \(\nu_m\) passes through the band, then decays
- **Radio** (3, 6 GHz): Rises over days to weeks as the self-absorption frequency \(\nu_a\) drops below the observing band
- **Jet break**: Visible as a steepening at \(\sim 10\) days when the jet edge becomes visible

```python
import numpy as np
from blastwave import FluxDensity_tophat

P = dict(
    Eiso=1e53, lf=300, theta_c=0.1,
    n0=1.0, A=0.0, eps_e=0.1, eps_b=0.01, p=2.2,
    theta_v=0.0, d=1000.0, z=0.2,
)

t = np.geomspace(10, 1e8, 300)  # seconds
flux_radio = FluxDensity_tophat(t, 3e9, P, spread=True)
flux_xray  = FluxDensity_tophat(t, 2.4e17, P, spread=True)
```

![Multi-band GRB afterglow](img/grb_afterglow_multiband.png)

## Viewing angle effects

The observed light curve depends strongly on the observer's viewing angle. For **off-axis** observers (\(\theta_v > \theta_c\)):

1. The early emission is suppressed by relativistic beaming
2. The light curve rises as the jet decelerates and the beaming cone widens
3. Eventually the off-axis light curve converges with the on-axis prediction

The jet structure also matters: a **Gaussian** jet produces smoother transitions between viewing angles compared to a sharp-edged **tophat** jet, because the gradual energy profile provides partial illumination at all angles.

```python
from blastwave import FluxDensity_tophat, FluxDensity_gaussian

P_offaxis = {**P, 'theta_v': np.radians(10)}  # 10 degrees
flux_tophat  = FluxDensity_tophat(t, 2.4e17, P_offaxis, spread=True)
flux_gaussian = FluxDensity_gaussian(t, 2.4e17, P_offaxis, spread=True)
```

![Viewing angle effects](img/grb_afterglow_viewing.png)

## Lateral spreading

Lateral spreading causes the jet to expand sideways as it decelerates, significantly affecting the light curve shape:

- **On-axis**: Spreading steepens the post-jet-break decay by redistributing energy over a wider solid angle
- **Off-axis**: Spreading causes the jet to widen toward the observer's line of sight earlier, producing a brighter and earlier peak

The `spread` parameter controls this behavior:

```python
flux_spread   = FluxDensity_tophat(t, 2.4e17, P, spread=True)
flux_nospread = FluxDensity_tophat(t, 2.4e17, P, spread=False)
```

![Spreading effects](img/grb_afterglow_spreading.png)

The solid lines include spreading; the dashed lines do not. On-axis (blue), the jet break steepening at \(\sim 10\) days is enhanced by spreading. Off-axis (red, \(\theta_v = 15^\circ\)), spreading brightens and advances the peak.

## Jet structure comparison

Three built-in jet structures are available:

- **Tophat** (`FluxDensity_tophat`): Uniform energy within \(\theta_c\), zero outside
- **Gaussian** (`FluxDensity_gaussian`): \(E(\theta) \propto \exp(-\theta^2 / 2\theta_c^2)\)
- **Power-law** (`FluxDensity_powerlaw`): \(E(\theta) \propto (1 + (\theta/\theta_c)^2)^{-s/2}\)

```python
from blastwave import FluxDensity_tophat, FluxDensity_gaussian, FluxDensity_powerlaw

P_pl = {**P, 's': 4.0}  # power-law steepness
flux_pl = FluxDensity_powerlaw(t, 2.4e17, P_pl, spread=True)
```

![Jet structure comparison](img/grb_afterglow_structure.png)

**On-axis** (left): All three structures produce nearly identical light curves, since the observer sees the jet core in all cases.

**Off-axis** (right, \(\theta_v = 15^\circ\)): The structures diverge dramatically. The tophat jet shows a sharp rise and the latest peak due to its hard edge. The Gaussian and power-law jets rise earlier because their energy profiles extend beyond \(\theta_c\), providing emission toward off-axis observers at earlier times. This distinction is critical for modeling events like GW170817.

## Full script

The complete script is at [`examples/grb_afterglow.py`](https://github.com/nuclear-multimessenger-astronomy/blastwave/blob/main/examples/grb_afterglow.py). To regenerate the plots:

```bash
python examples/grb_afterglow.py
```
