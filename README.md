# CDR Phase I Validation (Ising Toy-Model)

Reference implementation scaffold for **Core Distinguishability Relativity (CDR)** Phase I validation in a controlled toy domain.

## Scientific objective

Validate the CDR detection protocol in a **2-component binary Ising-like conditional kernel** where the ground-truth selection parameter `epsilon` is known.

The implementation supports:

- baseline conditional kernel `P0(I' | I)`
- kernel-local integration gain `Δχ(I'; I)`
- reweighted kernel `Pε(I' | I)` (CDR exponential-family tilt)
- deterministic trajectory simulation with fixed seeds
- reproducible artifacts for downstream inference and gates

## Scope (current milestone)

This initial milestone implements the **kernel foundation**:

- `src/ising_kernel.py`
- exact state enumeration for `{0,1}^2`
- exact probabilities (no Monte Carlo approximation at kernel level)
- deterministic trajectory sampling
- per-step `Δχ` logging for later inference modules

## Next modules (planned)

- `statistics.py` (Fisher/Hessian, rank, conditioning)
- `estimators.py` (MLE/MAP for epsilon)
- `validators.py` (G1–G7 gates)
- `controls.py` (time-shuffle and surrogates)
- `phase1_runner.py` (orchestration)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt