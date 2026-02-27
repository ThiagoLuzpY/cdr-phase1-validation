Core Distinguishability Relativity (CDR): Phase I Validation
Status: Phase I ✅ COMPLETE (7/7 gates PASS) | Phase II (empirical validation) in planning

What is CDR?
CDR is a pre-registered framework for detecting information-driven selection bias in observed dynamics without ensnaring yourself in p-hacking or circular reasoning.
Simple version:
When you observe a system (EEG, quantum RNG, neural activity), how do you know if what you're seeing is:

Real effect (selection bias is happening)
Statistical artifact (just noise)
Model misspecification (your baseline assumption is wrong)

CDR answers this by:

Pre-registering your hypothesis before looking at data
Running 7 mandatory validation gates (G1–G7) instead of just p-values
Using negative controls (time-shuffle, adversarial baseline, out-of-sample) to catch artifacts
Being transparent about what you can and cannot conclude


Phase I: Toy-Model Validation ✅
We validate CDR on a 2-component Ising conditional kernel where we control everything:

Ground-truth selection parameter epsilon is known (we set it)
State space is small and enumerable (2 binary components = 4 states)
No confounders, no real-world mess

Result: All 7 gates pass. Method works in controlled conditions.
What this means:

✅ Protocol is operationally valid
✅ Detection works when effect is real (H₁: epsilon > 0)
✅ Detection rejects when no effect (H₀: epsilon = 0)
✅ Controls collapse appropriately
❌ Does NOT prove consciousness causes RNG deviation (Phase II tests that)
❌ Does NOT prove CDR is better than alternatives (comparative study needed)


Phase II: Empirical Validation (Planned)
Hypothesis: Consciousness (measured via EEG) drives information-driven selection in quantum RNG output.
Data: 20–30 meditators + 20–30 controls, EEG + quantum RNG, pre-registered protocol.
Timeline: 12 months.

Project Structure
cdr-phase1-validation/
├── README.md                           ← You are here
├── requirements.txt                    ← Python dependencies
├── .gitignore                          ← Excludes __pycache__, results/, etc
│
├── src/                                ← Main code
│   ├── __init__.py
│   ├── ising_kernel.py                 ← Ising 2-component kernel
│   ├── likelihood.py                   ← Log-likelihood computation
│   ├── estimators.py                   ← MLE for epsilon
│   ├── statistics.py                   ← Fisher, Hessian, rank, conditioning
│   ├── controls.py                     ← Time-shuffle, surrogates
│   ├── validators.py                   ← Gates G1–G7
│   ├── artifacts.py                    ← Save JSON, plots, manifests
│   └── phase1_runner.py                ← Orchestrator (main entry point)
│
├── tests/                              ← Unit tests
│   ├── __init__.py
│   ├── test_ising_kernel.py
│   ├── test_estimators.py
│   ├── test_statistics.py
│   ├── test_controls.py
│   └── test_validators.py
│
├── results/                            ← Outputs (Phase I golden run)
│   └── golden_run_phase1_plus_v1/      ← Reproducible artifact bundle
│       ├── phase1plus_summary.json
│       ├── gate_report.txt
│       ├── seed_manifest.json
│       ├── artifacts_manifest.json
│       └── plots/
│
├── audit_bundle/                       ← Zenodo-ready audit package
│   ├── phase1_plus_golden_v1.zip
│   ├── artifacts_manifest.json         ← SHA-256 hashes
│   └── BUNDLE_METADATA.txt
│
├── config/
│   └── phase1_config.py                ← Pre-registered hyperparameters
│
└── notebooks/
    └── phase1_analysis.ipynb           ← Exploratory visualization

Installation
Requirements

Python 3.8+
numpy, scipy, pandas, matplotlib

Setup
bash# Clone repository
git clone https://github.com/ThiagoLuzpY/cdr-phase1-validation.git
cd cdr-phase1-validation

# Create virtual environment
python -m venv .venv

# Activate
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

Quick Start
Run Phase I validation (reproduces 7/7 gates PASS)
bashcd cdr-phase1-validation
python -m src.phase1_runner
Output:
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
Results saved to results/phase1_plus_full/.

Run tests
bashpytest tests/ -v

Key Results
All 7 Gates Pass
GateStatusMeaningG1✅ PASSH₀ (no selection) → ε̂ ≈ 0G2✅ PASSH₁ (selection present) → ε̂ ≈ 0.3 (true value)G3✅ PASSTime-shuffle control → ε̂ collapses to 0G4✅ PASSParameters identifiable (Schur complement ≥ 0.005)G5✅ PASSStable across minor config perturbationsG6✅ PASSEﬀect not absorbed by flexible baselineG7✅ PASSGeneralizes to held-out test data
Reproducibility

Fixed seeds: H₀ uses seeds 42–61, H₁ uses 100–119
Audit bundle: SHA-256 hashes of all artifacts in artifacts_manifest.json
Pre-registration: Hyperparameters frozen in config/phase1_config.py
Zenodo DOI: [DOI will be assigned upon upload]


Methodology
The CDR Protocol (Simplified)
Step 1: Pre-register

Specify baseline kernel P₀(I'|I) (Ising transition probs)
Specify hypothesis: "Selection present?" (ε > 0)
Specify 7 gates that MUST pass

Step 2: Run experiment

Generate/collect trajectories
Estimate coupling parameter ε
Compute Fisher information matrix

Step 3: Apply gates

G1–G2: Can we recover ε under H₀ and H₁?
G3: Do negative controls work?
G4–G5: Are parameters identifiable and stable?
G6–G7: Is effect real or artifact?

Step 4: Binary decision

All gates pass? → Effect is detectable
Any gate fails? → Method didn't work


Limitations (Phase I)
Toy-model scope:

✓ 2-component Ising (4 states total)
✓ No confounders, no noise beyond trajectory sampling
✗ Not consciousness (that's Phase II)
✗ Not validated against alternatives (no benchmark)

What Phase I proves:

Method is operationally sound
Detection works in controlled conditions

What Phase I does NOT prove:

Consciousness causes selection (Phase II tests this)
CDR is better than other methods (requires comparative study)
Real-world application will work (Phase II will test)


Phase II: Next Steps
Hypothesis: Consciousness (high CFQ from EEG) drives RNG deviation (TSQP connection).
Data needed:

20–30 experienced meditators (high consciousness)
20–30 control group (baseline consciousness)
30 min EEG + quantum RNG per person
Pre-registered protocol (OSF registry)

Expected outcome: CDR detects selection in meditators but not controls.
Timeline: 12–18 months.

How to Contribute / Extend
For Phase II empirical validation:

Pre-register your protocol on OSF (osf.io)
Collect EEG + RNG data
Apply CDR gates
Report results (pass/fail, no p-hacking)

For methodological improvements:

Fork repository
Add new gate or control procedure
Test on toy-model (Phase I)
Submit PR with clear documentation

For theory extensions:

Extend CDR to larger state spaces (n > 2 components)
Generalize to continuous-state systems
Connect to information geometry


References
Core CDR Papers

Petina, A. (2025). Core Distinguishability Relativity: Parts I–IV. Zenodo.
Petina, A. (2025). The Minimal Architecture of Distinguishability (ANAM). Zenodo.

Methodological Inspiration

Popper, K.R. (1959). The Logic of Scientific Discovery.
Lakatos, I. (1978). The Methodology of Scientific Research Programmes.
Rosen, R. (1991). Life Itself: A Comprehensive Inquiry.

Related Work (Consciousness & RNG)

Radin, D. (2006). Entangled Minds. (Global Consciousness Project)
May, E.C., Utts, J.M. (2022). Parapsychology: A Handbook for the 21st Century.


Citation
If you use CDR in your work, please cite:
bibtex@software{luz2026cdr,
  title={Core Distinguishability Relativity: Phase I Validation},
  author={Luz, Thiago},
  year={2026},
  url={https://github.com/ThiagoLuzpY/cdr-phase1-validation},
  doi={[Zenodo DOI]}
}

License
CC0 1.0 Universal (Public Domain)
You are free to use, modify, and distribute this code without restriction.

Contact
Author: Thiago Luz
Email: thiago.luz.silva1@gmail.com
GitHub: https://github.com/ThiagoLuzpY/
Affiliation: Independent researcher

Acknowledgments

Peer feedback: [Colleagues who reviewed early versions]
Computational support: [Tools used: NumPy, SciPy, etc]
Inspiration: Alena V. Petina (ANAM generative architecture)


FAQ
Q: Is this ready for real data?
A: The method is validated in Phase I. Phase II will test on real consciousness + RNG data.
Q: Why not just use p-values?
A: P-values don't catch artifact, flexible baseline absorption, or out-of-sample failure. CDR's 7 gates do.
Q: Can I apply CDR to my domain?
A: Yes, but you'll need to: (1) define your baseline kernel P₀, (2) implement gates for your state space, (3) pre-register. Phase II protocol template coming soon.
Q: What's the connection to TSQP?
A: TSQP is the hypothesis (consciousness modulates quantum RNG). CDR is the method to test it. Phase II combines both.

Last updated: February 2026
Status: Phase I complete. Phase II planning in progress.