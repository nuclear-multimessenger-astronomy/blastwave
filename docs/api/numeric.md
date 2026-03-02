# Numeric Synchrotron Model (Chang-Cooper)

## Overview

The `"numeric"` radiation model uses a Chang-Cooper implicit finite-difference scheme to evolve the electron energy distribution N(gamma) and compute synchrotron emissivity + self-absorption from the full distribution, rather than using analytic piecewise power-law approximations.

This approach is more accurate for non-standard electron distributions and enables self-consistent pair production feedback.

## Algorithm

1. Build a log-spaced gamma grid (default 300 bins, up to gamma_max = 1e8)
2. Inject a power-law electron source Q(gamma) ~ gamma^{-p} between gamma_m and gamma_max
3. Compute synchrotron + adiabatic cooling rates at half-grid points
4. Compute Chang-Cooper upwind parameters for numerical stability
5. Assemble tridiagonal system (implicit backward Euler)
6. Solve via Thomas algorithm in O(n_bins) time
7. Compute j_nu = integral of P(nu, gamma) N(gamma) dgamma using the Dermer (2009) synchrotron kernel
8. Compute alpha_nu for SSA from the derivative of N(gamma)/gamma^2
9. Return intensity with SSA: j/alpha * (1 - exp(-alpha*dr))

## Usage

```python
import blastwave as jsr

P = {
    "Eiso": 1e53, "lf": 300, "theta_c": 0.1,
    "n0": 1.0, "A": 0.0,
    "eps_e": 0.1, "eps_b": 0.01, "p": 2.3,
    "d": 100, "z": 0.01, "theta_v": 0.0,
}

flux = jsr.FluxDensity_tophat(t, nu, P, model="numeric")
```

## Parameters

Set these in the P dictionary:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `eps_e` | required | Electron energy fraction |
| `eps_b` | required | Magnetic energy fraction |
| `p` | required | Electron spectral index |
| `n_gamma` | 300 | Number of gamma bins |
| `gamma_max` | 1e8 | Maximum electron Lorentz factor |
| `include_pp` | 0 | Set to 1 to enable pair production (gamma+gamma -> e+e-) |

## Pair Production

When `include_pp=1`, the solver performs an additional iteration:

1. Solve N(gamma) without pairs
2. Compute synchrotron photon density n_nu = j_nu * dr / (h nu c)
3. Compute pair injection source using the Miceli & Nava (2022) cross-section kernel
4. Re-solve N(gamma) with the updated source

## Synchrotron Kernel

Uses the Dermer (2009) fitting formula for the synchrotron spectral function F(x):

```
F(x) = 1.808 * x^{1/3} / sqrt(1 + 3.4*x^{2/3})
      * (1 + 2.21*x^{2/3} + 0.347*x^{4/3}) / (1 + 1.353*x^{2/3} + 0.217*x^{4/3})
      * exp(-x)
```

where x = nu / nu_c(gamma).

## References

- Chang, J. & Cooper, G. (1970). "A practical difference scheme for Fokker-Planck equations." Journal of Computational Physics, 6(1), 1-16.
- Dermer, C.D. (2009). "High Energy Radiation from Black Holes." Princeton University Press.
- Miceli, D. & Nava, L. (2022). "Pair production in GRB afterglows." MNRAS, 510(2), 2391-2404.
- Nedora, V. et al. PyBlastAfterglowMag.
