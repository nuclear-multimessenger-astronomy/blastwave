# blastwave

Fast Rust GRB afterglow simulator with Python bindings.

A high-performance relativistic blast wave simulator in Rust with Python bindings. Computes hydrodynamic evolution and multi-band synchrotron radiation from relativistic blast waves interacting with a circumburst medium. Designed for modeling gamma-ray burst (GRB) afterglows, fast blue optical transients (FBOTs), and other explosive transients.

## Key Features

- **Hydrodynamic solvers** --- three spreading modes: finite-volume PDE (CFL-limited RK2), per-cell ODE spreading (VegasAfterglow-style RK45), and no-spread
- **Jet profiles** --- TopHat, Gaussian, PowerLaw, and Spherical explosion profiles
- **General CSM density** --- $n(r) = n_\mathrm{wind} (r / 10^{17}\,\mathrm{cm})^{-k} + n_\mathrm{ISM}$ with arbitrary power-law index $k$ (ISM: $k=0$, wind: $k=2$)
- **Radiation models** --- synchrotron, synchrotron + self-absorption (SSA), deep Newtonian phase, inverse Compton, thermal synchrotron
- **Reverse shock** --- coupled forward-reverse shock dynamics with magnetization
- **Forward-mapping flux** --- pre-computed radiation grid for fast multi-frequency light curves
- **Parallel** --- Rayon multithreading for batch luminosity computations
- **Python bindings** --- `blastwave` package via PyO3/maturin with numpy arrays

## Quick Example

```python
import numpy as np
from blastwave import FluxDensity_tophat

P = {
    "Eiso": 1e52, "lf": 300, "theta_c": 0.1,
    "A": 0.0, "n0": 1.0,
    "eps_e": 0.1, "eps_b": 0.01, "p": 2.2,
    "theta_v": 0.3, "d": 1000.0, "z": 0.1,
}

t = np.geomspace(1e3, 1e8, 200)
nu = 1e9 * np.ones_like(t)
flux = FluxDensity_tophat(t, nu, P)
```

## Getting Started

See the [Installation](getting-started/installation.md) and [Quick Start](getting-started/quickstart.md) guides.

## Citation

If you use blastwave in your research, please cite:

```bibtex
@software{blastwave,
  author = {Coughlin, Michael},
  title = {blastwave: Fast Rust GRB afterglow simulator with Python bindings},
  url = {https://github.com/nuclear-multimessenger-astronomy/blastwave},
}
```

blastwave builds on the following works:

- Peer 2012, ApJL, 752, L8 --- hydrodynamic model
- Sari, Piran & Narayan 1998, ApJL, 497, L17 --- synchrotron radiation
- Margalit & Quataert 2021, ApJL, 923, L14 --- thermal synchrotron
- Ferguson & Margalit 2025 --- full-volume post-shock extension ([GitHub](https://github.com/RossFerguson1/synchrotron_shock_model))
