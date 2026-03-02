# blastwave

A high-performance relativistic blast wave simulator in Rust with Python bindings. Computes hydrodynamic evolution and multi-band synchrotron radiation for gamma-ray burst (GRB) afterglows, fast blue optical transients (FBOTs), neutron star merger afterglows, and other explosive transients.

**blastwave** originated as a Rust reimplementation of [jetsimpy](https://github.com/haowang-astro/jetsimpy) (Wang et al. 2024) but has since grown into a standalone code incorporating physics and numerical methods from several afterglow modeling frameworks:

- **[jetsimpy](https://github.com/haowang-astro/jetsimpy)** (Wang et al. 2024) --- finite-volume PDE lateral spreading, analytic synchrotron models, EATS integration
- **[VegasAfterglow](https://github.com/YihanWangAstro/VegasAfterglow)** (Wang, Zhang & Huang 2025) --- per-cell ODE lateral spreading, forward-mapping flux computation
- **[PyBlastAfterglowMag](https://github.com/vsevolodnedora/PyBlastAfterglowMag)** (Nedora et al.) --- Chang-Cooper electron kinetic equation solver, pair production, EBL absorption
- **Margalit & Quataert (2021)** / **Ferguson & Margalit (2025)** --- thermal + non-thermal synchrotron model with full-volume post-shock extension

## Features

### Hydrodynamics
- Three lateral spreading modes: finite-volume **PDE** (CFL-limited RK2), per-cell **ODE** spreading (adaptive RK45), and **no-spread**
- Top-hat, Gaussian, power-law, and spherical explosion jet profiles
- General CSM density: n(r) = n_wind (r / 10^17 cm)^{-k} + n_ISM with arbitrary power-law index k
- Forward + reverse shock dynamics with magnetization
- Energy injection via magnetar spin-down (single or multiple episodes)

### Radiation
- **Analytic synchrotron** (`sync`) --- Sari, Piran & Narayan (1998) piecewise power-law spectra
- **Synchrotron self-absorption** (`sync_ssa`) --- min(I_thin, I_blackbody) Rayleigh-Jeans approach
- **Synchrotron self-Compton** (`sync_ssc`) --- inverse Compton with Klein-Nishina corrections
- **Thermal synchrotron** (`sync_thermal`) --- Margalit & Quataert (2021) thermal + non-thermal electron model with self-consistent self-absorption; optional Ferguson & Margalit (2025) full-volume post-shock mode for trans-relativistic shocks
- **Numeric electron distribution** (`numeric`) --- Chang-Cooper implicit finite-difference solver for the electron kinetic equation N(gamma), with optional pair production (Miceli & Nava 2022)
- **EBL absorption** (`ebl=True`) --- Franceschini & Rodighiero (2018) optical depth table

### Flux Computation
- **EATS** (default) --- adaptive quadrature on the equal arrival time surface
- **Forward-mapping** (`flux_method="forward"`) --- pre-computed radiation grid with binary search + interpolation; eliminates EATS overhead for on-axis viewing angles

### Performance
- Rust core with Rayon multithreading for batch luminosity computations
- Tophat fast path: detects uniform initial conditions, solves 1 cell, replicates
- On-axis EATS: 1D adaptive quadrature (2.4x faster than 2D)
- Forward-mapping reduces flux computation from milliseconds to ~0.1ms

## Project Structure

```
blastwave/
├── src/              # Pure Rust core library
│   ├── hydro/        # Hydrodynamic solvers (PDE, ODE, no-spread)
│   ├── afterglow/    # Radiation models, EATS, forward-mapping, Chang-Cooper, SSA, IC, EBL
│   ├── math/         # Numerical utilities (integration, root-finding)
│   └── constants.rs  # Physical constants
├── rust/             # PyO3 Python extension (blastwave_extension)
├── blastwave/        # Python wrappers (Jet, FluxDensity_*, grids)
├── tests/            # Comparison and unit tests
└── docs/             # Documentation (MkDocs)
```

## Installation

### Requirements

- Rust toolchain (1.70+)
- Python 3.8+
- [maturin](https://github.com/PyO3/maturin) (`pip install maturin`)

### Build and install

```bash
cd rust && maturin develop --release && cd ..
pip install .
```

## Quick Start

```python
import numpy as np
import blastwave

# Define physical parameters
P = dict(
    Eiso=1e52,       # isotropic equivalent energy (erg)
    lf=300,          # core Lorentz factor
    theta_c=0.1,     # half-opening angle (rad)
    n0=1,            # ISM number density (cm^-3)
    A=0,             # wind density parameter
    eps_e=0.1,       # electron energy fraction
    eps_b=0.01,      # magnetic energy fraction
    p=2.17,          # electron power-law index
    theta_v=0.0,     # viewing angle (rad)
    d=474.33,        # luminosity distance (Mpc)
    z=0.1,           # redshift
)

# Observing times and frequency
tday = np.logspace(-3, 3, 100)
tsecond = tday * 86400
nu = 1e18  # Hz (X-ray)

# Compute flux densities for different jet profiles
fd_tophat = blastwave.FluxDensity_tophat(tsecond, nu, P)
fd_gaussian = blastwave.FluxDensity_gaussian(tsecond, nu, P)
```

### Lateral Spreading Modes

```python
# PDE spreading (default)
jet = blastwave.Jet(profiles, nwind, nism, spread=True)

# ODE spreading (VegasAfterglow-style, faster for large cell counts)
jet = blastwave.Jet(profiles, nwind, nism, spread_mode="ode")

# No spreading
jet = blastwave.Jet(profiles, nwind, nism, spread=False)
```

### Forward-Mapping Flux

```python
# On-axis: forward-mapping (falls back to EATS for off-axis)
fd = blastwave.FluxDensity_tophat(t, nu, P, flux_method="forward")
```

### Thermal Synchrotron

```python
P_thermal = {**P, 'eps_T': 1.0, 'delta': 0.1}
flux = jet.FluxDensity(t, nu, P_thermal, model='sync_thermal')

# Full-volume post-shock (Ferguson & Margalit 2025)
P_fv = {**P_thermal, 'full_volume': 1.0, 'k': 0.0}
flux = jet.FluxDensity(t, nu, P_fv, model='sync_thermal')
```

### EBL Absorption

```python
flux = blastwave.FluxDensity_tophat(t, nu, P, ebl=True)
```

## References

- Wang, H., Bhattacharya, M., Gill, R., & Giannios, D. (2024). "jetsimpy: A Highly Efficient Hydrodynamic Code for Gamma-Ray Burst Afterglow." *ApJS*, 273(1), 17. [arXiv:2402.19359](https://arxiv.org/abs/2402.19359) | [GitHub](https://github.com/haowang-astro/jetsimpy)
- Wang, Y., Zhang, B., & Huang, B. (2025). "VegasAfterglow: A Broad-Spectrum Gamma-Ray Burst Afterglow Modeling Package." *ApJ*. [arXiv:2507.10829](https://arxiv.org/abs/2507.10829) | [GitHub](https://github.com/YihanWangAstro/VegasAfterglow)
- Nedora, V. et al. "PyBlastAfterglowMag." [GitHub](https://github.com/vsevolodnedora/PyBlastAfterglowMag)
- Margalit, B. & Quataert, E. (2021). "Thermal Electrons in Mildly-relativistic Synchrotron Blast-waves." *ApJ*. [arXiv:2111.00012](https://arxiv.org/abs/2111.00012)
- Ferguson, R. & Margalit, B. (2025). "Numerical Modeling of Relativistic Effects in Synchrotron-Emitting Shocks." [arXiv:2509.16313](https://arxiv.org/abs/2509.16313)
- Sari, R., Piran, T., & Narayan, R. (1998). "Spectra and Light Curves of Gamma-Ray Burst Afterglows." *ApJL*, 497(1), L17.
- Wijers, R. A. M. J., & Galama, T. J. (1999). "Physical Parameters of GRB 970508 and GRB 971214 from Their Afterglow Synchrotron Emission." *ApJ*, 523(1), 177.
- Franceschini, A. & Rodighiero, G. (2018). "The extragalactic background light revisited and the cosmic photon-photon opacity." *A&A*, 614, C1.
- Miceli, D. & Nava, L. (2022). "Pair production in GRB afterglows." *MNRAS*, 510(2), 2391-2404.
- Dermer, C. D. (2009). "High Energy Radiation from Black Holes." Princeton University Press.
- Chang, J. & Cooper, G. (1970). "A practical difference scheme for Fokker-Planck equations." *JCP*, 6(1), 1-16.

## License

See [LICENSE](LICENSE) for details.
