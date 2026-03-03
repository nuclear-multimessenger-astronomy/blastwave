# Benchmarks

This page presents performance benchmarks for `blastwave` across different configurations: spread modes, angular resolution, jet profiles, and radiation models.

## Setup

All benchmarks use a standard on-axis top-hat jet (unless varying the profile) with 50 time points and 3 GHz radio frequency. Each configuration is run with 1 warmup iteration followed by 3 timed iterations; the mean is reported.

Base parameters:

| Parameter | Value |
|-----------|-------|
| \(E_\mathrm{iso}\) | \(10^{53}\) erg |
| \(\Gamma_0\) | 300 |
| \(\theta_c\) | 0.1 rad |
| \(n_0\) | 1.0 cm\(^{-3}\) |
| \(\varepsilon_e\) | 0.1 |
| \(\varepsilon_B\) | 0.01 |
| \(p\) | 2.3 |
| Resolution | 129 cells |

## Results

![Benchmark results](examples/img/benchmarks.png)

### Spread modes

Three lateral spreading modes are available:

- **PDE** (default) --- full finite-volume lateral spreading solver. Most accurate but slowest.
- **ODE** --- simplified ODE-based spreading (VegasAfterglow-style). Good balance of accuracy and speed.
- **No spread** --- lateral spreading disabled. Fastest, appropriate for early times or on-axis viewing.

### Angular resolution

The number of angular cells controls the trade-off between accuracy and speed. The top-hat profile benefits from a fast path that makes it insensitive to resolution, but structured jets (Gaussian, power-law) scale more significantly with cell count.

Recommended resolutions:

- **17 cells** --- quick estimates, parameter space exploration
- **65 cells** --- standard analysis
- **129 cells** --- publication-quality (default for shortcut functions)
- **257 cells** --- convergence testing

### Jet profiles

- **TopHat** --- uses a 1-cell fast path when on-axis, making it significantly faster
- **Gaussian** --- smooth structure requires full angular integration
- **Power-law** --- similar cost to Gaussian

### Radiation models

- **`sync`** --- analytic synchrotron. Fastest, sufficient for most applications.
- **`sync_ssa`** --- synchrotron + self-absorption. Modest overhead from the absorption calculation.
- **`numeric`** --- full Chang-Cooper electron distribution. Most expensive but captures smooth spectral transitions, pair production, and cooling effects.

## Running the benchmarks

To reproduce these results on your system:

```bash
python examples/benchmarks.py
```

!!! note
    Absolute times depend on hardware. These benchmarks were run on an HPC compute node. Relative comparisons between configurations are more meaningful than absolute values.

## Reproducing the script

The complete benchmarking script is at [`examples/benchmarks.py`](https://github.com/nuclear-multimessenger-astronomy/blastwave/blob/main/examples/benchmarks.py).
