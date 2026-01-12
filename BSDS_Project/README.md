# BSDS for Emo-Film Analysis

**Bayesian Switching Dynamical Systems (BSDS)** applied to emotional film viewing fMRI data.

## Overview

This repository contains:
- **`bsds_complete/`**: Full Python implementation of BSDS with AR dynamics
- **Original MATLAB code**: From Taghia & Cai (2018) Nature Communications
- **CLI Pipeline**: One-command analysis for Emo-Film data
- **Analysis tools**: Visualization and statistical summaries

## Quick Start

### Option 1: Full Pipeline (Recommended)

```bash
# Analyze BigBuckBunny movie for all subjects
python run_emofilm_bsds.py --task BigBuckBunny --subjects all --n-states 5

# Custom parameters
python run_emofilm_bsds.py \
    --task FirstBite \
    --subjects sub-S01 sub-S02 sub-S03 \
    --n-states 8 \
    --max-ldim 15 \
    --n-iter 100 \
    --output ./my_results
```

### Option 2: Python API

```python
from bsds_complete import BSDSModel, BSDSConfig

# Configure model
config = BSDSConfig(
    n_states=5,
    max_ldim=10,
    n_iter=100,
    TR=2.0
)

# Fit model
model = BSDSModel(config)
model.fit(data_list)  # List of (ROI x Time) arrays

# Get results
states = model.get_states()
stats = model.get_summary_statistics()
print(f"Occupancy: {stats['occupancy_group']}")
print(f"Mean lifetime: {stats['mean_lifetime_group']}")
```

## Project Structure

```
BSDS_Project/
├── bsds_complete/               # COMPLETE Python implementation
│   ├── core/
│   │   ├── model.py            # Main BSDSModel class
│   │   └── config.py           # Configuration
│   ├── inference/
│   │   ├── hmm.py              # Forward-Backward, Viterbi
│   │   └── latent.py           # Latent variable inference
│   ├── learning/
│   │   ├── factor_learning.py  # Factor loadings, ARD
│   │   ├── transition_learning.py  # HMM transitions
│   │   └── ar_learning.py      # AR(1) dynamics (KEY!)
│   ├── analysis/
│   │   ├── statistics.py       # Occupancy, lifetime, etc.
│   │   └── visualization.py    # Plots and reports
│   └── utils/
│       ├── math_utils.py       # Numerical utilities
│       └── data_utils.py       # Data preprocessing
├── Taghia_Cai_NatureComm_2018-main/  # Original MATLAB
├── run_emofilm_bsds.py         # CLI pipeline
├── test_bsds_complete.py       # Test suite
└── papers/                     # Reference papers
```

## Implementation Status

| Component | Status | File |
|-----------|--------|------|
| HMM Forward-Backward | ✅ Complete | `inference/hmm.py` |
| Viterbi Decoding | ✅ Complete | `inference/hmm.py` |
| Latent Variable (Q(X)) | ✅ Complete | `inference/latent.py` |
| Factor Loadings (Q(L)) | ✅ Complete | `learning/factor_learning.py` |
| ARD Parameters (Q(nu)) | ✅ Complete | `learning/factor_learning.py` |
| Noise Precision (psi) | ✅ Complete | `learning/factor_learning.py` |
| **AR(1) Dynamics** | ✅ Complete | `learning/ar_learning.py` |
| **VAR M-step** | ✅ Complete | `learning/ar_learning.py` |
| Transition Probs (Q(theta)) | ✅ Complete | `learning/transition_learning.py` |
| Occupancy/Lifetime Stats | ✅ Complete | `analysis/statistics.py` |
| Visualization | ✅ Complete | `analysis/visualization.py` |

## CLI Reference

```
usage: run_emofilm_bsds.py [-h] [--mode {full,extract,fit,analyze}]
                          [--data-dir DATA_DIR]
                          [--task {BigBuckBunny,FirstBite,YouAgain,Rest,all}]
                          [--subjects SUBJECTS [SUBJECTS ...]]
                          [--n-states N_STATES] [--max-ldim MAX_LDIM]
                          [--n-iter N_ITER] [--TR TR]
                          [--output OUTPUT]

Key Arguments:
  --mode          Pipeline mode: full, extract, fit, or analyze
  --task          Movie to analyze (BigBuckBunny, FirstBite, YouAgain, Rest)
  --subjects      Subject IDs or 'all'
  --n-states, -K  Number of brain states (default: 5)
  --max-ldim, -L  Latent dimension (default: 10)
  --n-iter        VB iterations (default: 100)
  --TR            Repetition time in seconds (default: 2.0)
  --output, -o    Output directory
```

## Example Results

After running the pipeline, you get:

```
results/bsds_BigBuckBunny_5states_20251215/
├── bsds_..._model.pkl          # Fitted model
├── bsds_..._config.json        # Configuration
├── bsds_..._report.txt         # Text summary
├── bsds_..._summary.png        # Visualization
├── bsds_..._results.json       # All statistics
├── sub-S01_BigBuckBunny_states.npy  # State sequences
└── ...
```

## Dependencies

```bash
pip install numpy scipy scikit-learn nilearn nibabel matplotlib
```

## Testing

```bash
python test_bsds_complete.py
```

Expected output:
```
==================================================
BSDS Complete - Test Suite
==================================================
  ✓ PASS: Imports
  ✓ PASS: Synthetic Data
  ✓ PASS: Save/Load

Total: 3/3 tests passed
```

## References

1. Taghia, J., Cai, M. B., et al. (2018). Uncovering hidden brain state dynamics that regulate performance and decision-making during cognition. *Nature Communications*, 9, 2505.

2. Schaefer, A., et al. (2018). Local-Global Parcellation of the Human Cerebral Cortex from Intrinsic Functional Connectivity MRI. *Cerebral Cortex*.

## License

- MATLAB code: Original authors' license
- Python implementation: MIT License

---
*Last updated: 2025-12-15*
