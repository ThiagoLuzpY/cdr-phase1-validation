# Core Distinguishability Relativity (CDR): Phase I Validation

**Status:** Phase I ✅ **COMPLETE (7/7 gates PASS)** | Phase II (empirical validation) in planning

---

## What is CDR?

CDR is a **pre-registered framework for detecting information-driven selection bias** in observed dynamics without ensnaring yourself in p-hacking or circular reasoning.

### Simple version:

When you observe a system (EEG, quantum RNG, neural activity), how do you know if what you're seeing is:

- **Real effect** (selection bias is happening)
- **Statistical artifact** (just noise)
- **Model misspecification** (your baseline assumption is wrong)

### CDR answers this by:

1. **Pre-registering** your hypothesis before looking at data
2. **Running 7 mandatory validation gates** (G1–G7) instead of just p-values
3. **Using negative controls** (time-shuffle, adversarial baseline, out-of-sample) to catch artifacts
4. **Being transparent** about what you can and cannot conclude

---

## Phase I: Toy-Model Validation ✅

We validate CDR on a **2-component Ising conditional kernel** where we control everything:

- Ground-truth selection parameter `epsilon` is known (we set it)
- State space is small and enumerable (2 binary components = 4 states)
- No confounders, no real-world mess

**Result:** All 7 gates pass. Method works in controlled conditions.

### What this means:

✅ Protocol is operationally valid  
✅ Detection works when effect is real (H₁: epsilon > 0)  
✅ Detection rejects when no effect (H₀: epsilon = 0)  
✅ Controls collapse appropriately  

❌ Does NOT prove consciousness causes RNG deviation (Phase II tests that)  
❌ Does NOT prove CDR is better than alternatives (comparative study needed)  

---

## Phase II: Empirical Validation (Planned)

**Hypothesis:** Consciousness (measured via EEG) drives information-driven selection in quantum RNG output.

- **Data:** 20–30 meditators + 20–30 controls, EEG + quantum RNG, pre-registered protocol
- **Timeline:** 12 months

---

## Project Structure
```
cdr-phase1-validation/
├── config/
│   ├── __init__.py
│   └── phase1_config.py                # Pre-registered hyperparameters
│
├── results/
│   ├── _test_phase1_plus/              # Test runs (temporary)
│   ├── golden_run_phase1_plus_v1/      # Reproducible artifact bundle
│   └── phase1_plus_full/               # Full Phase I+ validation results
│
├── scripts/
│   ├── __init__.py
│   ├── make_audit_bundle.py            # Generate Zenodo-ready audit package
│   └── run_phase1_plus_full.py         # Run complete Phase I+ validation
│
├── src/
│   ├── __init__.py
│   ├── adversarial_kernel.py           # Adversarial baseline flexibility tests
│   ├── artifacts.py                    # Save JSON, plots, manifests
│   ├── controls.py                     # Time-shuffle, surrogates, control procedures
│   ├── estimators.py                   # MLE for epsilon
│   ├── ising_kernel.py                 # Ising 2-component kernel
│   ├── model_selection.py              # Model comparison utilities
│   ├── phase1_plus_runner.py           # Phase I+ full orchestrator
│   ├── phase1_runner.py                # Phase I basic orchestrator
│   ├── statistics.py                   # Fisher, Hessian, rank, conditioning
│   └── validators.py                   # Gates G1–G7
│
├── tests/
│   ├── __init__.py
│   ├── test_controls.py                # Unit tests for controls
│   ├── test_estimators.py              # Unit tests for estimators
│   ├── test_ising.py                   # Unit tests for Ising kernel
│   ├── test_phase1_plus_runner.py      # Integration tests
│   ├── test_statistics.py              # Unit tests for statistics
│   └── test_validators.py              # Unit tests for gates
│
├── .gitignore                          # Git exclusions
├── main.py                             # Entry point
├── README.md                           # This file
└── requirements.txt                    # Python dependencies
```

---

## Installation

### Requirements

- Python 3.8+
- numpy, scipy, pandas, matplotlib

### Setup

```bash
# Clone repository
git clone https://github.com/ThiagoLuzpY/cdr-phase1-validation.git
cd cdr-phase1-validation

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Run Phase I validation (reproduces 7/7 gates PASS)

```bash
cd cdr-phase1-validation
python -m src.phase1_runner
```

**Expected output:**

```
CDR Phase I+ — Gates G1–G7
────────────────────────────────────────────
G1_H0_recovery: PASS
G2_H1_recovery: PASS
G3_controls_collapse: PASS
G4_identifiability: PASS
G5_stability: PASS
G6_adversarial: PASS
G7_out_of_sample: PASS
FINAL: PASS ✅
```

Results saved to `results/phase1_plus_full/`.

### Run tests

```bash
pytest tests/ -v
```

---

## Key Results

### All 7 Gates Pass

| Gate | Status | Meaning |
|------|--------|---------|
| **G1** | ✅ PASS | H₀ (no selection) → ε̂ ≈ 0 |
| **G2** | ✅ PASS | H₁ (selection present) → ε̂ ≈ 0.3 (true value) |
| **G3** | ✅ PASS | Time-shuffle control → ε̂ collapses to 0 |
| **G4** | ✅ PASS | Parameters identifiable (Schur complement ≥ 0.005) |
| **G5** | ✅ PASS | Stable across minor config perturbations |
| **G6** | ✅ PASS | Effect not absorbed by flexible baseline |
| **G7** | ✅ PASS | Generalizes to held-out test data |

### Reproducibility

- **Fixed seeds:** H₀ uses seeds 42–61, H₁ uses 100–119
- **Audit bundle:** SHA-256 hashes in `artifacts_manifest.json`
- **Pre-registration:** Hyperparameters frozen in `config/phase1_config.py`
- **Zenodo DOI:** [Pending upload]

---

## Methodology

### The CDR Protocol (Simplified)

#### Step 1: Pre-register
- Specify baseline kernel P₀(I'|I) (Ising transition probs)
- Specify hypothesis: "Selection present?" (ε > 0)
- Specify 7 gates that **MUST** pass

#### Step 2: Run experiment
- Generate/collect trajectories
- Estimate coupling parameter ε
- Compute Fisher information matrix

#### Step 3: Apply gates
- **G1–G2:** Can we recover ε under H₀ and H₁?
- **G3:** Do negative controls work?
- **G4–G5:** Are parameters identifiable and stable?
- **G6–G7:** Is effect real or artifact?

#### Step 4: Binary decision
- All gates pass? → Effect is detectable
- Any gate fails? → Method didn't work

---

## Limitations (Phase I)

### Toy-model scope:

✓ 2-component Ising (4 states total)  
✓ No confounders, no noise beyond trajectory sampling  
✗ Not consciousness (that's Phase II)  
✗ Not validated against alternatives (no benchmark)  

### What Phase I proves:

- Method is operationally sound
- Detection works in controlled conditions

### What Phase I does NOT prove:

- Consciousness causes selection (Phase II tests this)
- CDR is better than other methods (requires comparative study)
- Real-world application will work (Phase II will test)

---

## Phase II: Next Steps

**Hypothesis:** Consciousness (high CFQ from EEG) drives RNG deviation (TSQP connection).

### Data needed:

- 20–30 experienced meditators (high consciousness)
- 20–30 control group (baseline consciousness)
- 30 min EEG + quantum RNG per person
- Pre-registered protocol (OSF registry)

### Expected outcome:

CDR detects selection in meditators but not controls.

### Timeline:

12–18 months

---

## How to Contribute / Extend

### For Phase II empirical validation:

1. Pre-register your protocol on OSF (osf.io)
2. Collect EEG + RNG data
3. Apply CDR gates
4. Report results (pass/fail, no p-hacking)

### For methodological improvements:

1. Fork repository
2. Add new gate or control procedure
3. Test on toy-model (Phase I)
4. Submit PR with clear documentation

### For theory extensions:

1. Extend CDR to larger state spaces (n > 2 components)
2. Generalize to continuous-state systems
3. Connect to information geometry

---

## References

### Core CDR Papers

- Luz, T. (2026). *Core Distinguishability Relativity: Parts I–IV*. Zenodo. [DOI pending]
- Petina, A. (2025). *The Minimal Architecture of Distinguishability (ANAM)*. Zenodo.

### Methodological Inspiration

- Popper, K.R. (1959). *The Logic of Scientific Discovery*. Hutchinson.
- Lakatos, I. (1978). *The Methodology of Scientific Research Programmes*. Cambridge University Press.
- Rosen, R. (1991). *Life Itself: A Comprehensive Inquiry*. Columbia University Press.

### Related Work (Consciousness & RNG)

- Radin, D. (2006). *Entangled Minds*. Paraview Press. (Global Consciousness Project)
- May, E.C., & Utts, J.M. (2022). *Parapsychology: A Handbook for the 21st Century*. McFarland.

---

## Citation

If you use CDR in your work, please cite:

```bibtex
@software{luz2026cdr,
  title={Core Distinguishability Relativity: Phase I Validation},
  author={Luz, Thiago},
  year={2026},
  url={https://github.com/ThiagoLuzpY/cdr-phase1-validation},
  doi={[Zenodo DOI pending]}
}
```

---

## License

**CC0 1.0 Universal (Public Domain)**

You are free to use, modify, and distribute this code without restriction.

---

## Contact

- **Author:** Thiago Luz
- **Email:** thiago.luz.silva1@gmail.com
- **GitHub:** https://github.com/ThiagoLuzpY/
- **Affiliation:** Independent researcher

---

## Acknowledgments

- Peer feedback: [Colleagues who reviewed early versions]
- Computational support: [NumPy, SciPy, Pandas, Matplotlib]
- Inspiration: Alena V. Petina (ANAM generative architecture)

---

## FAQ

**Q: Is this ready for real data?**

A: The *method* is validated in Phase I. Phase II will test on real consciousness + RNG data.

**Q: Why not just use p-values?**

A: P-values don't catch artifact, flexible baseline absorption, or out-of-sample failure. CDR's 7 gates do.

**Q: Can I apply CDR to my domain?**

A: Yes, but you'll need to: (1) define your baseline kernel P₀, (2) implement gates for your state space, (3) pre-register. Phase II protocol template coming soon.

**Q: What's the connection to TSQP?**

A: TSQP is the *hypothesis* (consciousness modulates quantum RNG). CDR is the *method* to test it. Phase II combines both.

---

**Last updated:** February 2026  
**Status:** Phase I complete. Phase II planning in progress.
