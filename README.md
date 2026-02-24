# jetsimpy-rs

A Rust reimplementation of [jetsimpy](https://github.com/haowang-astro/jetsimpy), a GRB (Gamma-Ray Burst) afterglow simulator. This project provides a high-performance Rust core library with Python bindings via PyO3, offering the same physics and API as the original C++/pybind11 implementation.

## Features

- Hydrodynamic simulation of relativistic jet evolution
- Synchrotron radiation calculation with EATS (Equal Arrival Time Surface) integration
- Support for top-hat, Gaussian, and power-law jet profiles
- Python API compatible with the original jetsimpy

## Project Structure

```
jetsimpy-rs/
├── src/              # Pure Rust core library
│   ├── hydro/        # Hydrodynamic solver (PDE, interpolation)
│   ├── afterglow/    # Radiation models (synchrotron, EATS)
│   ├── math/         # Numerical utilities (integration, root-finding)
│   └── constants.rs  # Physical constants
├── rust/             # PyO3 Python extension (jetsimpy_extension)
├── jetsimpy_rs/      # Python wrappers (Jet, FluxDensity_*, grids)
└── tests/            # Comparison and unit tests
```

## Installation

### Requirements

- Rust toolchain (1.70+)
- Python 3.8+
- [maturin](https://github.com/PyO3/maturin) (`pip install maturin`)

### Build the Rust library

```bash
cargo build --release
```

### Build and install the Python extension

```bash
# Build the PyO3 extension
cd rust && maturin develop --release && cd ..

# Install the Python package
pip install .
```

## Quick Start

```python
import numpy as np
import jetsimpy_rs

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
    s=6,             # power-law jet slope
)

# Observing times and frequency
tday = np.logspace(-3, 3, 100)
tsecond = tday * 86400
nu = 1e18  # Hz (X-ray)

# Compute flux densities for different jet profiles
fd_tophat = jetsimpy_rs.FluxDensity_tophat(tsecond, nu, P)
fd_gaussian = jetsimpy_rs.FluxDensity_gaussian(tsecond, nu, P)
fd_powerlaw = jetsimpy_rs.FluxDensity_powerlaw(tsecond, nu, P)
```

## Zero-value Interpolation in Luminosity Computation

For extreme physical parameters (e.g. very high isotropic energy E0 ~ 1e57 erg or very low ISM density n0 ~ 1e-5 cm^-3), the EATS solver can fail to find a solution at certain observation times because the required source-frame time exceeds the PDE evolution maximum (`tmax`). When this happens, `luminosity()` returns 0.0 for those time points, producing isolated gaps in an otherwise smooth lightcurve.

The original C++ jetsimpy raises an error and aborts entirely in these cases. jetsimpy-rs instead returns 0.0 for the affected points and continues, but these spurious zeros contaminate downstream applications (e.g. surrogate model training data where `log10(flux)` maps to -300).

### How it works

After the parallel luminosity computation in `calculateLuminosity` (`rust/src/lib.rs`), a post-processing step identifies zero-valued entries that are bounded on both sides by positive values at the same observing frequency. These interior zeros are filled using **log-log interpolation** (linear in ln(luminosity) vs ln(time)), which is the natural interpolation scheme for power-law lightcurves.

The algorithm:

1. **Groups results by frequency** — uses bit-exact floating-point equality, with a fast path for the common single-frequency case.
2. **Sorts by time** within each group (skipped if already sorted).
3. **Scans for zero runs** — for each contiguous block of zeros, checks for positive neighbors on both sides.
4. **Interpolates** — fills the gap with `exp(lerp(ln(L_left), ln(L_right), frac))` where `frac` is computed in log-time space.

Boundary zeros (leading or trailing, where there is no neighbor on one side) are left untouched, since these typically represent genuine physical boundaries (observation times before the jet becomes visible or after the PDE domain ends).

### Impact

On worst-case parameters from the GRB training set:
- **Before**: up to 35/250 time points returned zero luminosity, producing -300 floors in log-space training data.
- **After**: 0 interior zeros remain. Only boundary points (e.g. the very last time exceeding `tmax`) may still be zero.

The interpolated values blend smoothly with computed values (max log10 jump ~ 0.5 dex between consecutive points on a 250-point log-spaced grid).

### Adaptive Integration Refinement

A complementary fix in the adaptive quadrature (`src/math/integral.rs`) forces refinement of all-zero intervals on the first iteration. This prevents the integrator from skipping regions where all 5 sample points happen to miss a narrow emission beam, which could otherwise cause the 2D EATS integral to return zero even when the physical flux is nonzero.

## References

- Wang, H., Bhattacharya, M., Gill, R., & Giannios, D. (2024). "jetsimpy: A Highly Efficient Hydrodynamic Code for Gamma-Ray Burst Afterglow." *The Astrophysical Journal Supplement Series*, 273(1), 17. [arXiv:2404.11365](https://arxiv.org/abs/2404.11365)
- Original implementation: [haowang-astro/jetsimpy](https://github.com/haowang-astro/jetsimpy)

## License

See [LICENSE](LICENSE) for details.
