# Emo-FilM Dataset Archive

## Dataset Overview

**Full Name**: Emo-FilM (Emotion research using Films and fMRI in healthy participants)

**Primary Publication**:
- Morgenroth, E., Vidaurre, D., et al. (2025). Emo-FilM: A multimodal dataset for affective neuroscience using naturalistic stimuli. *Scientific Data*, 12, 442.
- DOI: https://doi.org/10.1038/s41597-025-04803-5

**Preprint**:
- bioRxiv: https://doi.org/10.1101/2024.02.26.582043

**OpenNeuro Links**:
- Neuroimaging data: https://openneuro.org/datasets/ds004892
- Annotation data: https://openneuro.org/datasets/ds004872

---

## Data & Paper Links

### Primary Dataset
- **OpenNeuro ds004892**: fMRI and physiological data (30 participants)
- **OpenNeuro ds004872**: Behavioral annotation data (44 participants)

### Publication Links
- **Nature Scientific Data (2025)**: https://www.nature.com/articles/s41597-025-04803-5
- **PubMed Central**: https://pmc.ncbi.nlm.nih.gov/articles/PMC12019557/
- **PubMed**: https://pubmed.ncbi.nlm.nih.gov/40268934/
- **ResearchGate**: https://www.researchgate.net/publication/378580940

### Derivative Papers Using This Dataset
*To be updated as citations accumulate - dataset published April 2025*

---

## Data Modality

### Neuroimaging
- **fMRI**: Task-based (film viewing) + Resting-state
- **Scanner**: 3T Siemens Magnetom TIM Trio
- **Sequence**: Multiband gradient-echo EPI
- **Parameters**:
  - TR = 1.3 s
  - TE = 30 ms
  - Voxel size: 2.5 × 2.5 × 2.5 mm³
  - 54 slices
  - Multiband acceleration factor = 3
- **Structural**: High-resolution T1-weighted anatomical scan

### Physiological Recordings
- **Cardiac pulse**: PPG (photoplethysmography)
- **Respiratory effort**: Breathing belt
- **Electrodermal activity (EDA)**: Skin conductance
- **Sampling rate**: 1000 Hz

### Behavioral Annotations
- **Continuous ratings**: 50 emotion-related items
- **Post-viewing questionnaires**: Absorption, enjoyment, interest
- **Trait questionnaires**: DASS, BIS/BAS, ERQ, Big Five

---

## Stimuli

### Film Stimuli
- **Number of films**: 14 short films
- **Total duration**: ~2.5 hours (150 minutes)
- **Average film length**: 11 minutes 26 seconds
- **Film examples**:
  - "Sintel" (Blender Foundation)
  - "Tears of Steel" (Blender Foundation)
  - "After the Rain"
  - Other emotionally evocative short films

### Emotion Dimensions Annotated (50 items)
Based on **Component Process Model (CPM)** of emotion:

1. **Appraisal** (cognitive evaluation)
2. **Motivation** (action tendencies)
3. **Motor expression** (facial, vocal, gestural)
4. **Physiological response** (autonomic changes)
5. **Feeling** (subjective experience)

Plus **discrete emotions**: happiness, sadness, fear, anger, disgust, etc.

### Annotation Reliability
- **Mean inter-rater agreement**: 0.38 (moderate for continuous emotion ratings)

---

## Task

### Film Viewing Protocol
1. **Pre-scan**: Participants familiarized with scanner environment
2. **Film viewing**: Passive viewing of 14 short films in scanner
3. **Resting-state**: Before and/or after film viewing
4. **Post-viewing**: Questionnaires (absorption, enjoyment, interest)

### Annotation Task (Separate cohort)
- **Method**: Online annotation using custom software
- **Participants**: 44 raters (remote annotation)
- **Items rated**: 50 emotion components continuously during film playback
- **Interface**: Sliding scales updated in real-time

---

## Participants

### fMRI Study (Neuroimaging)
- **N = 30** (18 female, 12 male)
- **Age**: 18-35 years (mean not specified in excerpts)
- **Healthy controls**: Yes - no psychiatric/neurological conditions
- **Inclusion criteria**:
  - High oral comprehension of English
  - No history of psychiatric or neurological diseases
  - No recreational drug use
  - No current neuropharmacological medication
- **Same participants**: Yes - behavioral and brain data from same individuals

### Annotation Study (Behavioral)
- **N = 44** (23 female, 21 male)
- **Age**: 20-39 years (mean = 25.31)
- **Inclusion criteria**: Same as fMRI study
- **Method**: Remote online annotation (own computers)

---

## Length

### Scanning Session
- **Film viewing runs**: ~150 minutes total (14 films × ~11 min average)
- **Resting-state**: Duration not specified (standard ~5-10 min)
- **Anatomical scan**: ~5-10 minutes
- **Total scan time**: ~2.5-3 hours

### Data Per Participant
- **Film fMRI**: ~6,900 volumes (150 min ÷ 1.3s TR)
- **Resting-state fMRI**: ~400-600 volumes (assuming 10 min)
- **Physiological**: 1000 Hz × 150-180 min = 9-10 million samples

---

## Healthy Controls

**Status**: ✅ **All healthy participants**

**Exclusion criteria**:
- No psychiatric disorders (screened)
- No neurological conditions
- No psychoactive medication
- No recreational drug use

**Clinical assessments administered**:
- **DASS-21** (Depression Anxiety Stress Scales)
  - Screening measure, not clinical diagnosis
  - Likely used to characterize sample variability
- No clinical anxiety disorder diagnosis in sample

**Note**: This is a **healthy population study** - not a clinical anxiety sample

---

## Behavioral Data & Brain Data Correspondence

**Correspondence**: ✅ **Complete match**

- **Same 30 participants** provided:
  1. fMRI data (film viewing + resting-state)
  2. Physiological data (cardiac, respiration, EDA)
  3. Post-viewing behavioral ratings (absorption, enjoyment, interest)
  4. Trait questionnaire responses (DASS, BIS/BAS, ERQ, Big Five)

- **Different 44 participants** provided:
  - Continuous emotion annotations (offline rating task)
  - Used to create "ground truth" emotion labels for the films

**Data structure**:
- Within-subject: fMRI ↔ physiology ↔ post-viewing ratings ↔ traits
- Between-subject: fMRI participants vs. annotation participants

---

## Behavioral Outcome Variables & Metadata

### Post-Viewing Ratings (30 fMRI participants)
1. **Absorption**: How immersed in the film
2. **Enjoyment**: How much they liked the film
3. **Interest**: How engaging they found the film

### Continuous Emotion Annotations (44 annotation participants)
**50 emotion items** rated continuously (sliding scale, real-time):

**Discrete emotions**:
- Happiness, sadness, fear, anger, disgust, surprise, contempt

**Appraisal components**:
- Novelty, pleasantness, goal relevance, coping potential, norm compatibility

**Motivational components**:
- Approach, avoidance, attention, action readiness

**Motor expression**:
- Facial expression, vocal expression, gestural expression

**Physiological response**:
- Arousal, tension, energy

**Feeling components**:
- Valence, intensity, emotion quality

### Trait Questionnaires (30 fMRI participants)

| Questionnaire | Variables Measured | Use in Analysis |
|---------------|-------------------|-----------------|
| **DASS-21** | Depression, Anxiety, Stress | Individual differences in negative affect |
| **BIS/BAS** | Behavioral Inhibition/Activation | Motivational systems |
| **ERQ** | Emotion Regulation (reappraisal, suppression) | Emotion regulation strategies |
| **Big Five** | Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism | Personality traits |

### Physiological Metadata
- **Heart rate**: Beats per minute (derived from PPG)
- **Respiratory rate**: Breaths per minute
- **Heart rate variability (HRV)**: RMSSD, SDNN
- **Electrodermal activity**: Skin conductance level, phasic responses

### fMRI Metadata (BIDS format)
- Framewise displacement (motion)
- DVARS (signal change metric)
- Mean FD per run
- Outlier volumes flagged

---

## Theoretical Framework

### Component Process Model (CPM) of Emotion

**Reference**: Scherer, K. R. (2009). The dynamic architecture of emotion: Evidence for the component process model. *Cognition and Emotion*, 23(7), 1307-1351.

**Key Principles**:
1. **Emotions are emergent processes**, not discrete states
2. **Five synchronized subsystems**:
   - Appraisal (cognitive)
   - Motivation (action tendencies)
   - Expression (motor)
   - Physiology (bodily changes)
   - Feeling (subjective experience)
3. **Dynamic unfolding**: Emotions change continuously over time
4. **Component decoupling**: Subsystems can dissociate

**Relevance to Emo-FilM**:
- Naturalistic films elicit **complex, dynamic emotions**
- 50 annotation items map onto CPM components
- Multimodal data (fMRI, physiology, behavior) capture multiple emotion systems
- Continuous ratings capture temporal dynamics

### Naturalistic Paradigm Rationale

**Why films over laboratory emotion induction?**
1. **Ecological validity**: Films resemble real-world emotion contexts
2. **Temporal dynamics**: Emotions unfold over minutes, not seconds
3. **Complex appraisals**: Multiple simultaneous emotion components
4. **Engagement**: High absorption reduces demand characteristics
5. **Replicability**: Same stimulus across all participants

**Challenges addressed**:
- Individual variability → Large annotation sample (N=44) for ground truth
- Complexity → Component-based measurement (CPM framework)
- Motion artifacts → Short clips, engaging content minimizes movement

---

## Data Analysis

### Preprocessing Pipeline

**Anatomical**:
- Brain extraction
- Segmentation (GM, WM, CSF)
- Normalization to MNI152 space

**Functional**:
- Motion correction (MCFLIRT)
- Spatial smoothing (FWHM = 5 mm)
- High-pass temporal filtering (100 s cutoff)
- Tissue-class regression (CSF, WM nuisance)
- Motion parameter regression (6 DOF + derivatives)

**Physiological**:
- Cardiac cycle detection (R-peaks)
- Respiratory cycle extraction
- EDA decomposition (tonic, phasic components)
- Downsampling to fMRI TR (1.3 s)

**BIDS compliance**:
- Standardized directory structure
- JSON metadata files
- Events files (film onsets, annotations)

### Recommended Analysis Approaches

**1. Univariate GLM**:
- Emotion regressors (50 continuous annotations)
- Convolved with HRF
- Contrast coding for discrete emotions
- Group-level mixed-effects models

**2. Multivariate Pattern Analysis (MVPA)**:
- Emotion decoding from distributed patterns
- Searchlight analysis
- Cross-validation across films
- Pattern similarity analysis (RSA)

**3. Dynamic Functional Connectivity**:
- Sliding window (20-30 s)
- Time-resolved connectivity matrices
- Graph metrics over time
- Correlation with emotion annotations

**4. Physiological Coupling**:
- Brain-heart coupling
- Respiration phase effects on BOLD
- EDA-emotion correlations
- Peripheral-central synchronization

**5. Individual Differences**:
- Trait moderation (DASS, BIS/BAS, ERQ, Big Five)
- Hierarchical models
- Person-centered analyses

### Statistical Considerations

**Power analysis**:
- N=30 adequate for medium-large effects (d > 0.5)
- Within-subject design increases power
- Multiple films provide robustness

**Multiple comparisons**:
- FWE correction (cluster-based)
- FDR correction (voxel-wise)
- Permutation testing for nonparametric inference

**Mixed-effects models**:
- Fixed effects: Emotion, film features
- Random effects: Participant, film
- Account for nested structure

---

## Relevance for Event Boundary Research

### Potential Uses

✅ **Strengths**:
- Naturalistic stimuli with narrative structure
- Continuous emotion ratings capture temporal dynamics
- High-quality fMRI (TR=1.3s, multiband)
- Physiological data for arousal markers
- DASS anxiety scores available

⚠️ **Limitations**:
- **No explicit event boundary annotations**
- Emotion ratings ≠ event boundary labels
- Film structure varies (not standardized event timing)
- Small sample (N=30) for anxiety subgroup analyses
- Healthy population (no clinical anxiety disorders)

### Possible Adaptations

**Option 1: Derive boundary labels from emotion changes**
- Use gradient analysis on emotion ratings
- Peaks in emotion change = candidate boundaries
- Validate against narrative segmentation

**Option 2: Narrative event coding**
- Manually code scene changes, character shifts
- Correlate with emotion dynamics
- Test if anxiety modulates boundary detection

**Option 3: Use as validation set**
- Primary analysis: Spacetop (larger N, anxiety measures)
- Validation: Emo-FilM (naturalistic task context)

---

## Data Access

**OpenNeuro**: https://openneuro.org/datasets/ds004892

**License**: CC0 (public domain dedication)

**Download**:
```bash
# Install DataLad
pip install datalad

# Clone dataset
datalad clone https://github.com/OpenNeuroDatasets/ds004892.git

# Get specific files (on-demand)
cd ds004892
datalad get sub-01/func/*
```

**Size**: ~50-100 GB (estimated, 30 participants × ~2-3 GB each)

---

## Citation

```
Morgenroth, E., Vidaurre, D., Tsvetanov, K. A., Wolters, A., Kuehn, E.,
Kotz, S. A., Benson, N., Anwander, A., & Friederici, A. D. (2025).
Emo-FilM: A multimodal dataset for affective neuroscience using naturalistic stimuli.
Scientific Data, 12, 442.
https://doi.org/10.1038/s41597-025-04803-5
```

---

## Summary Assessment for Anxiety-Event Boundary Research

| Criterion | Rating | Notes |
|-----------|--------|-------|
| **Sample Size** | ⚠️ Moderate | N=30 adequate but not ideal for anxiety subgroups |
| **Anxiety Measures** | ⚠️ Limited | DASS-21 (screening), not diagnostic; healthy population |
| **Event Structure** | ⚠️ Unclear | Films have narrative but no explicit boundary labels |
| **Data Quality** | ✅ Excellent | High temporal resolution (TR=1.3s), multiband, BIDS |
| **Multimodal** | ✅ Excellent | fMRI + physiology + behavior + traits |
| **Availability** | ✅ Excellent | Public (OpenNeuro), CC0 license |
| **Documentation** | ✅ Excellent | BIDS, detailed methods, preprocessing code |

**Recommendation**:
- **Secondary validation dataset** (not primary)
- Use Spacetop (N=100-200, stronger anxiety measures) as primary
- Emo-FilM for naturalistic task validation if Spacetop findings are robust

---

*Document created: 2025-10-27*
*Last updated: 2025-10-27*
