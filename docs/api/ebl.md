# EBL Absorption

## Overview

Extragalactic Background Light (EBL) absorption attenuates high-energy photons (E > ~10 GeV) via pair production with the diffuse EBL photon field. This is significant for TeV observations of distant sources.

blastwave implements the Franceschini & Rodighiero (2018) optical depth table with bilinear interpolation.

## Usage

### Apply to flux calculation

Add `ebl=True` to any flux density call:

```python
import blastwave as jsr

P = {
    "Eiso": 1e53, "lf": 300, "theta_c": 0.1,
    "n0": 1.0, "A": 0.0,
    "eps_e": 0.1, "eps_b": 0.01, "p": 2.3,
    "d": 100, "z": 0.5, "theta_v": 0.0,
}

# Without EBL
flux = jsr.FluxDensity_tophat(t, nu, P)

# With EBL absorption
flux_ebl = jsr.FluxDensity_tophat(t, nu, P, ebl=True)
```

The `ebl` parameter is available on all flux density methods:

- `jet.FluxDensity(..., ebl=True)`
- `jet.FluxDensity_forward(..., ebl=True)`
- `jet.FluxDensity_reverse(..., ebl=True)`
- `jsr.FluxDensity_tophat(..., ebl=True)`
- `jsr.FluxDensity_gaussian(..., ebl=True)`
- `jsr.FluxDensity_spherical(..., ebl=True)`
- `jsr.FluxDensity_powerlaw(..., ebl=True)`

### Standalone function

Compute optical depth directly:

```python
from blastwave import blastwave_extension as ext
import numpy as np

# Single frequency
tau = ext.ebl_tau(1e26, z=0.5)  # tau at 1e26 Hz, z=0.5

# Array of frequencies
nu = np.logspace(24, 28, 100)
tau_arr = ext.ebl_tau_array(nu, z=0.5)
```

## Table Details

- 56 energy bins: 5.2 GeV to 232 TeV
- 10 redshift bins: z = 0.01 to z = 3.0
- Interpolation: bilinear in (ln E, z)
- Below the table minimum energy or z < 0.01: tau = 0 (transparent)
- NaN entries (extreme absorption) are stored as infinity: exp(-inf) = 0

## Physical Notes

- X-ray and radio frequencies are completely transparent to EBL (tau = 0)
- EBL absorption becomes significant above ~50 GeV for z > 0.1
- At TeV energies, the optical depth can be >> 1, meaning complete absorption
- The observed flux is attenuated by exp(-tau(E, z))

## References

- Franceschini, A. & Rodighiero, G. (2018). "The extragalactic background light revisited and the cosmic photon-photon opacity." A&A, 614, C1 (erratum to A&A 603, A34).
- Nedora, V. et al. PyBlastAfterglowMag.
