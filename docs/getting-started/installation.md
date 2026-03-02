# Installation

## Prerequisites

- [Rust toolchain](https://rustup.rs/) (stable)

For Python bindings:

- Python >= 3.9
- [maturin](https://github.com/PyO3/maturin)
- numpy

## Python bindings

```bash
pip install maturin numpy
cd blastwave/rust
maturin develop --release
```

This builds and installs the `blastwave` package into the active virtualenv.

!!! tip "Virtual environment"
    Maturin requires a virtualenv or conda environment. Create one with:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

!!! note "HPC / module systems"
    On HPC clusters you may need to load compiler and Python modules first:
    ```bash
    module load gcccore/13.2.0 python/3.11.5 cmake/3.27.6
    source /path/to/your/venv/bin/activate
    cd blastwave/rust
    maturin develop --release
    ```

## Building a wheel

To build a distributable wheel:

```bash
cd blastwave/rust
maturin build --release
pip install target/wheels/blastwave_cpu-*.whl
```

## Rebuilding after code changes

If Rust source changes aren't taking effect, clean and rebuild:

```bash
cd blastwave/rust
cargo clean
maturin develop --release
```

## Controlling parallelism

blastwave uses [Rayon](https://github.com/rayon-rs/rayon) for multi-core parallelism in batch luminosity computations. By default it uses all available cores. To limit thread count:

```bash
export RAYON_NUM_THREADS=4
```
