# Lab Server Setup Guide - Emo-FiLM HMM Analysis

## Quick Start

### 1. Clone Repository on Lab Server

```bash
ssh your_lab_server

# Navigate to your workspace
cd /your/workspace/

# Clone the repository
git clone https://github.com/kyungjinasusual/anxiety-event-boundary-fmri.git
cd anxiety-event-boundary-fmri
```

### 2. Setup Python Environment

```bash
# Create conda environment
conda create -n emofilm-hmm python=3.9
conda activate emofilm-hmm

# Install required packages
pip install numpy pandas scipy scikit-learn hmmlearn
pip install nibabel nilearn matplotlib seaborn statsmodels
```

### 3. Verify Emo-FiLM Data Access

```bash
# Check if Emo-FiLM data is accessible
ls -lh /storage/bigdata/Emo-FiLM/

# Expected structure:
# /storage/bigdata/Emo-FiLM/
# ├── sub-001/
# │   ├── func/
# │   │   ├── sub-001_task-rest_bold.nii.gz
# │   │   └── ...
# │   └── anat/
# ├── sub-002/
# └── participants.tsv
```

### 4. Run HMM Analysis

```bash
cd analysis/

# Basic usage (all subjects, resting-state)
python run_hmm_emofilm.py \
    --data_root /storage/bigdata/Emo-FiLM \
    --session rest \
    --atlas aal \
    --output_dir ../results/emofilm_hmm_rest

# Test with fewer subjects first
python run_hmm_emofilm.py \
    --data_root /storage/bigdata/Emo-FiLM \
    --session rest \
    --n_subjects 5 \
    --output_dir ../results/test_5subjects
```

---

## Detailed Instructions

### Data Structure Requirements

The script expects BIDS-format data:

```
/storage/bigdata/Emo-FiLM/
├── participants.tsv          # Demographics and questionnaires
├── sub-001/
│   ├── func/
│   │   ├── sub-001_task-rest_bold.nii.gz
│   │   ├── sub-001_task-rest_bold.json
│   │   └── sub-001_task-{filmname}_bold.nii.gz
│   └── anat/
│       └── sub-001_T1w.nii.gz
└── sub-002/
    └── ...
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_root` | `/storage/bigdata/Emo-FiLM` | Path to Emo-FiLM dataset |
| `--session` | `rest` | Session type: 'rest' or film name |
| `--atlas` | `aal` | Brain atlas: 'aal' or 'schaefer' |
| `--n_subjects` | `None` | Number of subjects (None = all) |
| `--output_dir` | `results_emofilm` | Output directory |

### Atlas Options

**AAL (Automated Anatomical Labeling)**
- 116 ROIs
- Anatomically defined regions
- Good for whole-brain analysis

**Schaefer 2018**
- 200 ROIs (7 networks)
- Functionally defined parcels
- Better for network-level analysis

```bash
# Using AAL atlas (default)
python run_hmm_emofilm.py --atlas aal

# Using Schaefer atlas
python run_hmm_emofilm.py --atlas schaefer
```

### Session Types

```bash
# Resting-state data
python run_hmm_emofilm.py --session rest

# Task data (film viewing)
python run_hmm_emofilm.py --session filmname
```

---

## Expected Output

### Files Generated

```
results_emofilm/
├── emofilm_subject_metrics.csv          # Per-subject boundary metrics
├── fig1_anxiety_boundary_correlation.png  # Scatterplot
└── fig2_group_comparison.png             # Group boxplot
```

### CSV Columns

`emofilm_subject_metrics.csv`:
- `subject_id`: Subject identifier
- `n_boundaries`: Total event boundaries detected
- `boundary_rate_per_min`: Boundaries per minute
- `mean_state_duration_sec`: Average state duration
- `anxiety_score`: DASS anxiety or BIS score
- `age`, `sex`: Demographics

### Console Output

Expected console output during analysis:

```
============================================================
HMM EVENT BOUNDARY DETECTION - EMO-FILM DATASET
============================================================
Data root: /storage/bigdata/Emo-FiLM
Session: rest
Atlas: aal

Found 30 subjects

Loading sub-001, session: rest...
  Loading: sub-001_task-rest_bold.nii.gz
  Extracting ROI timeseries using aal atlas...
  ✓ Loaded timeseries: (300, 116)
    Timepoints: 300, ROIs: 116

Analyzing sub-001...
  Selecting optimal number of states...
  Testing 3 states...
  Testing 4 states...
  ...
  ✓ Optimal number of states: 7
  Fitting HMM with 7 states...
  ✓ Model fitted successfully
  ✓ Detected 42 boundaries
    Boundary rate: 6.46 per min

...

✓ Successfully processed 30 subjects

============================================================
GROUP-LEVEL ANALYSIS - EMO-FILM
============================================================

=== H1: Anxiety × Event Boundary Count ===
Pearson r = 0.XXX, p = 0.XXXX
N = 30 subjects
✓ SIGNIFICANT correlation detected

✓ Results saved to results_emofilm/

============================================================
ANALYSIS COMPLETE
============================================================
Anxiety × Boundary correlation: r = 0.XXX
p-value: 0.XXXX
```

---

## Troubleshooting

### Issue: "FileNotFoundError: Functional data not found"

**Solution**: Check BIDS structure and session naming

```bash
# List available functional files
ls /storage/bigdata/Emo-FiLM/sub-001/func/

# If files have different naming, adjust the script
# Example: if files are named differently
# sub-001_rest_bold.nii.gz instead of sub-001_task-rest_bold.nii.gz
```

### Issue: "nilearn import error"

**Solution**: Install neuroimaging packages

```bash
conda activate emofilm-hmm
pip install nibabel nilearn
```

### Issue: "hmmlearn convergence warning"

**Solution**: This is normal - increase iterations or adjust convergence threshold

The script already uses `n_iter=1000`, which should be sufficient for most cases.

### Issue: "Memory error with full dataset"

**Solution**: Process subjects in batches

```bash
# Process first 10 subjects
python run_hmm_emofilm.py --n_subjects 10 --output_dir results_batch1

# Process next 10 subjects
# Modify script to start from subject 11
```

---

## Running in Background (Recommended)

For long-running analyses, use `screen` or `tmux`:

```bash
# Using screen
screen -S hmm_analysis
conda activate emofilm-hmm
cd anxiety-event-boundary-fmri/analysis
python run_hmm_emofilm.py --data_root /storage/bigdata/Emo-FiLM

# Detach: Ctrl+A, then D
# Reattach: screen -r hmm_analysis
```

```bash
# Using tmux
tmux new -s hmm_analysis
conda activate emofilm-hmm
cd anxiety-event-boundary-fmri/analysis
python run_hmm_emofilm.py --data_root /storage/bigdata/Emo-FiLM

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t hmm_analysis
```

---

## Performance Notes

### Computational Requirements

- **RAM**: ~4-8 GB per subject (depends on atlas)
- **Time**: ~1-3 minutes per subject
- **Total time (30 subjects)**: ~30-90 minutes

### Optimization Tips

1. **Use fewer ROIs**: Schaefer atlas (200 ROIs) vs AAL (116 ROIs)
2. **Reduce state range**: `n_states_range=(3, 8)` instead of `(3, 11)`
3. **Skip cross-validation**: Manually set optimal states after first subject

---

## Next Steps After Analysis

### 1. Review Results

```bash
# View subject metrics
cat results_emofilm/emofilm_subject_metrics.csv

# View figures
# Copy to local machine if needed
scp user@server:~/anxiety-event-boundary-fmri/results_emofilm/*.png .
```

### 2. Statistical Analysis

The script automatically performs:
- Pearson correlation (Anxiety × Boundary Count)
- Group comparison (High vs Low anxiety)

### 3. Advanced Analyses

After confirming basic HMM works:
- Try different atlases
- Compare rest vs task sessions
- Implement additional boundary detection methods (GSBS, DFC)

---

## Contact

Issues or questions:
- Check GitHub repo: https://github.com/kyungjinasusual/anxiety-event-boundary-fmri
- Email: castella@snu.ac.kr
