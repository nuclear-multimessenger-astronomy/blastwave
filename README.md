# jetsimpy-rs

A Rust reimplementation of [jetsimpy](https://github.com/haowang-astro/jetsimpy), a GRB (Gamma-Ray Burst) afterglow simulator. This project provides a high-performance Rust core library with Python bindings via PyO3.

## Core Features (from original jetsimpy)

The following features reproduce the original C++/pybind11 jetsimpy implementation:

- Hydrodynamic simulation of relativistic jet evolution with PDE finite-volume lateral spreading
- Synchrotron radiation (`sync`, `sync_dnp`) following Sari, Piran & Narayan (1998) with EATS integration
- Top-hat, Gaussian, and power-law jet profiles
- No-spread mode (`spread=False`)
- Weighted average observables (centroid offset, image size)
- Python API: `FluxDensity_tophat`, `FluxDensity_gaussian`, `FluxDensity_powerlaw` shortcut functions

To reproduce the original jetsimpy physics, use the default radiation model (`sync`) with PDE spreading (`spread=True`). The comparison tests in `tests/test_comparison.py` verify agreement within 0.2% (5e-3 relative error) across all three jet profiles:

```bash
pytest tests/test_comparison.py
```

## New in jetsimpy-rs

The following features have been added beyond the original jetsimpy:

- **ODE lateral spreading** (`spread_mode="ode"`): VegasAfterglow-style per-cell adaptive RK45 spreading, scaling linearly with cell count instead of O(n_θ²) for PDE
- **Synchrotron self-absorption** (`sync_ssa`): SSA radiation model using min(I_thin, I_blackbody) Rayleigh-Jeans approach
- **Synchrotron self-Compton** (`sync_ssc`): Inverse Compton scattering with Klein-Nishina cross-section corrections
- **Reverse shock**: Forward + reverse shock dynamics and emission, with separate `FluxDensity_forward()` and `FluxDensity_reverse()` diagnostics
- **Zero-value interpolation**: Log-log interpolation of spurious zero-luminosity points at extreme parameters (original jetsimpy aborts in these cases)
- **Adaptive integration refinement**: Forces refinement of all-zero EATS quadrature intervals to prevent missed narrow emission beams
- **Forward-mapping flux computation** (`flux_method="forward"`): VegasAfterglow-style pre-computed radiation grid with binary search + interpolation, eliminating EATS adaptive quadrature overhead for on-axis viewing angles
- **EATS quadrature optimizations**: On-axis peak-finding skip (peak is always at θ=0 by symmetry), off-axis peak subsampling (20 points instead of all cells), and pre-allocated integration buffers
- **Primitive-variable no-spread solver**: No-spread mode now uses per-cell primitive-variable RK45 (shared with ODE mode), eliminating conservative-to-primitive root-finding and heap allocations per time step

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

## Lateral Spreading Modes

jetsimpy-rs supports three lateral spreading modes, selectable via the `spread_mode` parameter:

| Mode | Description | Algorithm | Origin |
|------|-------------|-----------|--------|
| `"pde"` | Finite-volume PDE spreading (default when `spread=True`) | CFL-limited RK2 with lateral flux transport | Original jetsimpy |
| `"ode"` | Per-cell ODE spreading | Adaptive RK45 per cell, independent evolution | New (VegasAfterglow-style) |
| `"none"` | No lateral spreading (default when `spread=False`) | Adaptive RK45, cells fully independent | Original jetsimpy |

### Usage

```python
# Default: PDE spreading (backward compatible with original jetsimpy)
jet = jetsimpy_rs.Jet(profiles, nwind, nism, spread=True)

# ODE spreading (new, faster for large cell counts)
jet = jetsimpy_rs.Jet(profiles, nwind, nism, spread_mode="ode")

# No spreading
jet = jetsimpy_rs.Jet(profiles, nwind, nism, spread=False)
# or equivalently:
jet = jetsimpy_rs.Jet(profiles, nwind, nism, spread_mode="none")
```

### ODE Spreading Physics

The ODE spreading mode implements the lateral spreading model from Wang et al. (2024):

- **Spreading equation**: `dθ/dt = F(u) · β_f · c / (2Γr)` where `F(u) = 1/(1 + 7·u·θ_s)` is a suppression function that enforces causality at high Lorentz factors
- **Solid angle correction**: Mass sweeping rate is modified by `f = (1 - cos θ) / (1 - cos θ₀)` to account for the changing solid angle as the jet spreads
- **Independent cells**: Each theta cell evolves its own ODE system with adaptive Dormand-Prince RK45 (no lateral flux transport between cells)
- **Tophat fast path**: For uniform initial conditions, solves 1 cell and replicates

### Accuracy

ODE and PDE modes agree within ~0.1 dex for tophat jets and ~0.13 dex for structured (Gaussian) jets across all frequencies and times.

## Forward-Mapping Flux Method

For on-axis viewing angles (θ_v ≈ 0), jetsimpy-rs supports a forward-mapping flux computation method that replaces the default EATS adaptive quadrature with a pre-computed radiation grid. This is selected via `flux_method="forward"`.

### How it works

Instead of inverse-mapping (root-finding per observation point to map observer time back to source-frame time), forward-mapping:

1. **Pre-computes** observer-frame times and radiation at each hydro grid point (θ, t_src) after the hydro solve
2. **Queries** via binary search + log-linear interpolation for each desired observation time

For each theta cell j and source time step k:
- `t_obs[j][k] = t_src[k] - r[j][k] * cos(θ_j) / c`
- `dL/dΩ[j][k] = I(ν_src) * r² * D³` (radiation intensity times geometric and Doppler factors)

Total luminosity: `L = 2π Σ_j dL/dΩ_j(t_obs) × [cos(θ_{j-1/2}) - cos(θ_{j+1/2})]`

### Usage

```python
# Forward-mapping (on-axis only, falls back to EATS for off-axis)
fd = jetsimpy_rs.FluxDensity_tophat(t, nu, P, flux_method="forward")

# Or via the Jet class directly
jet = jetsimpy_rs.Jet(profiles, nwind, nism)
fd = jet.FluxDensity(t, nu, P, flux_method="forward")

# Off-axis: flux_method="forward" silently falls back to EATS
P_offaxis = {**P, "theta_v": 0.3}
fd = jetsimpy_rs.FluxDensity_tophat(t, nu, P_offaxis, flux_method="forward")  # uses EATS
```

### Optimizations

- **Cell skipping**: Cells with negligible energy (peak β·γ² < 10⁻⁶ × max across all time steps) are excluded from pre-computation
- **Time-grid subsampling**: When the hydro output exceeds 300 time steps (PDE mode produces ~1000), the time grid is subsampled to limit pre-computation cost

### Accuracy

Forward-mapping agrees with EATS to within:
- **0.002 dex** for PDE and no-spread modes
- **0.04 dex** for no-spread mode
- **0.097 dex** for ODE mode (due to coarser hydro time grid)

Verified across tophat (0.002 dex), Gaussian (0.002 dex), and power-law (0.004 dex) jet profiles.

### Performance

At 17 cells (tophat, on-axis, X-ray, 100 time points):

| Method | Total | Hydro | Flux | vs VegasAfterglow |
|--------|-------|-------|------|-------------------|
| no-spread + forward | 2.4ms | 2.3ms | 0.1ms | 1.6× |
| no-spread + EATS | 6.6ms | 2.3ms | 4.8ms | 4.4× |
| ODE + forward | 4.1ms | 4.2ms | 0.1ms | 2.7× |
| ODE + EATS | 8.9ms | 3.5ms | 3.8ms | 5.9× |

Forward-mapping reduces flux computation from milliseconds to ~0.1ms, making the remaining performance gap vs VegasAfterglow (~1.5ms) entirely hydro solve overhead.

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

## Comparison with VegasAfterglow

jetsimpy-rs and [VegasAfterglow](https://github.com/YihanWangAstro/VegasAfterglow) (Wang et al. 2025) are both GRB afterglow codes implementing the same core physics (relativistic blast wave deceleration + synchrotron radiation), but differ in several modeling choices and computational strategies. This section documents the differences.

### Physics Modeling Differences

The synchrotron radiation models differ in three coupled ways. These differences produce a ~0.3–1.3 dex luminosity offset between the codes; importantly, they form a self-consistent package within each framework and cannot be mixed independently.

#### 1. Minimum electron Lorentz factor (γ_m)

| Code | Formula | Approach |
|------|---------|----------|
| jetsimpy-rs | γ_m = [(p−2)/(p−1)] ε_e (m_p/m_e) (Γ−1) | Uses the bulk Lorentz factor Γ (Sari, Piran & Narayan 1998) |
| VegasAfterglow | γ_m = [(p−2)/(p−1)] ε_e (m_p/m_e) (Γ_th−1) | Uses the downstream thermal Lorentz factor Γ_th = U_th/(mc²) + 1, tracked as an ODE state variable with shock heating and adiabatic cooling |

At early times (Γ ≈ 300), Γ_th/Γ ≈ 0.69 due to the BM self-similar profile averaging. This propagates into the magnetic field (B ∝ √e_density) and characteristic frequencies (ν_m ∝ γ_m² B).

#### 2. Inverse Compton cooling in γ_c

| Code | Formula |
|------|---------|
| jetsimpy-rs (`sync` model) | γ_c = 6π m_e Γ c / (σ_T B² t) |
| VegasAfterglow | γ_c = 6π m_e Γ c / (σ_T B² t (1+Y)) where Y = ½(√(1 + 4η_e ε_e/ε_B) − 1) |

The Thomson Y parameter accounts for inverse Compton losses on the cooling break. For typical parameters (ε_e=0.1, ε_B=0.01), Y ≈ 1.8 at early times, reducing γ_c by a factor of ~3. jetsimpy-rs's `sync_ssc` model does include IC corrections but uses a different (Klein-Nishina) treatment.

#### 3. Peak spectral power normalization

| Code | P_ν,max expression |
|------|---------------------|
| jetsimpy-rs | P_ν,max = √3 e³ B n / (m_e c²) |
| VegasAfterglow | P_ν,max = (π/4) × 0.92 × √3 e³ B n / (m_e c²) |

The factor π/4 ≈ 0.785 comes from averaging over the pitch angle distribution (isotropic electrons in a tangled magnetic field), and 0.92 is the spectral peak correction factor from Wijers & Galama (1999). The combined factor is ~0.72.

### Performance Comparison

Benchmarks for a tophat jet, on-axis (θ_v=0), X-ray (10¹⁸ Hz), 100 time points. Total time includes both the hydrodynamic solve and flux computation. Measured on a single core (Intel Xeon, HPC node).

#### Default (EATS flux method)

| Cells | PDE | ODE | No-spread | VegasAfterglow |
|-------|-----|-----|-----------|----------------|
| 17 | 0.004s | 0.005s | 0.004s | 0.0015s |
| 33 | 0.008s | 0.009s | 0.006s | — |
| 129 | 0.099s | 0.028s | 0.018s | — |
| 257 | 0.398s | 0.054s | 0.034s | — |

#### With forward-mapping (`flux_method="forward"`)

| Cells | PDE | ODE | No-spread | VegasAfterglow |
|-------|-----|-----|-----------|----------------|
| 17 | 0.004s | 0.004s | 0.002s | 0.0015s |
| 129 | 0.096s | 0.020s | 0.010s | — |

Forward-mapping eliminates flux computation as a bottleneck (~0.1ms vs ~1.5–5ms for EATS), bringing 17-cell no-spread to within 1.6× of VegasAfterglow. The remaining gap is hydro solve overhead.

**Scaling**: PDE mode scales as O(n_θ²) due to CFL-limited time stepping; ODE and no-spread modes scale approximately linearly with cell count.

**ODE vs PDE crossover**: ODE spreading becomes faster than PDE at ~33–65 cells, with physics accuracy within ~0.1 dex of PDE for tophat jets and ~0.13 dex for structured (Gaussian) jets.

## References

- Wang, H., Bhattacharya, M., Gill, R., & Giannios, D. (2024). "jetsimpy: A Highly Efficient Hydrodynamic Code for Gamma-Ray Burst Afterglow." *The Astrophysical Journal Supplement Series*, 273(1), 17. [arXiv:2402.19359](https://arxiv.org/abs/2402.19359)
- Wang, Y., Zhang, B., & Huang, B. (2025). "VegasAfterglow: A Broad-Spectrum Gamma-Ray Burst Afterglow Modeling Package." *The Astrophysical Journal*. [arXiv:2507.10829](https://arxiv.org/abs/2507.10829) | [GitHub](https://github.com/YihanWangAstro/VegasAfterglow)
- Sari, R., Piran, T., & Narayan, R. (1998). "Spectra and Light Curves of Gamma-Ray Burst Afterglows." *The Astrophysical Journal Letters*, 497(1), L17.
- Wijers, R. A. M. J., & Galama, T. J. (1999). "Physical Parameters of GRB 970508 and GRB 971214 from Their Afterglow Synchrotron Emission." *The Astrophysical Journal*, 523(1), 177.
- Original implementation: [haowang-astro/jetsimpy](https://github.com/haowang-astro/jetsimpy)

## License

See [LICENSE](LICENSE) for details.
