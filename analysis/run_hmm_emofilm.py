#!/usr/bin/env python3
"""
HMM Event Boundary Detection - Emo-FiLM Dataset
Real data analysis script for lab server execution

Author: Kyungjin Oh
Date: 2025-11-01
Dataset: Emo-FiLM at /storage/bigdata/Emo-FiLM
"""

import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from hmmlearn import hmm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Import from the test implementation
from test_hmm_boundary_detection import EventBoundaryHMM, create_visualizations


def load_emofilm_data(data_root='/storage/bigdata/Emo-FiLM/brain_data',
                       subject_id='sub-S01',
                       session_id='ses-1',
                       task='Rest',
                       atlas='aal'):
    """
    Load Emo-FiLM fMRI data for a single subject.

    Parameters
    ----------
    data_root : str
        Path to Emo-FiLM dataset root directory (brain_data folder)
    subject_id : str
        Subject identifier (e.g., 'sub-S01')
    session_id : str
        Session identifier (e.g., 'ses-1', 'ses-2', etc.)
    task : str
        Task name: 'Rest' for resting-state, or film name ('BigBuckBunny', etc.)
    atlas : str
        Brain atlas for ROI extraction ('aal', 'schaefer', etc.)

    Returns
    -------
    timeseries : array, shape (n_timepoints, n_rois)
        ROI time series data
    metadata : dict
        Subject and scan metadata
    """
    print(f"Loading {subject_id}, session: {session_id}, task: {task}...")

    data_path = Path(data_root)

    # Construct BIDS path: /storage/bigdata/Emo-FiLM/brain_data/sub-S01/ses-1/func/sub-S01_ses-1_task-Rest_bold.nii.gz
    func_dir = data_path / subject_id / session_id / 'func'
    func_file = func_dir / f'{subject_id}_{session_id}_task-{task}_bold.nii.gz'

    if not func_file.exists():
        raise FileNotFoundError(f"Functional data not found: {func_file}")

    # Load fMRI data
    print(f"  Loading: {func_file.name}")
    func_img = nib.load(func_file)

    # Extract ROI time series using atlas
    print(f"  Extracting ROI timeseries using {atlas} atlas...")

    if atlas == 'aal':
        atlas_data = datasets.fetch_atlas_aal()
        atlas_img = atlas_data.maps
        roi_labels = atlas_data.labels
        n_rois = len(roi_labels)
    elif atlas == 'schaefer':
        atlas_data = datasets.fetch_atlas_schaefer_2018(n_rois=200)
        atlas_img = atlas_data.maps
        roi_labels = atlas_data.labels
        n_rois = 200
    else:
        raise ValueError(f"Unsupported atlas: {atlas}")

    # Apply masker
    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize=True,
        detrend=True,
        low_pass=0.1,
        high_pass=0.01,
        t_r=1.3  # Emo-FiLM TR
    )

    timeseries = masker.fit_transform(func_img)

    print(f"  ✓ Loaded timeseries: {timeseries.shape}")
    print(f"    Timepoints: {timeseries.shape[0]}, ROIs: {timeseries.shape[1]}")

    # Metadata
    metadata = {
        'subject_id': subject_id,
        'session_id': session_id,
        'task': task,
        'atlas': atlas,
        'n_timepoints': timeseries.shape[0],
        'n_rois': timeseries.shape[1],
        'tr': 1.3,
        'duration_min': timeseries.shape[0] * 1.3 / 60
    }

    return timeseries, metadata


def load_emofilm_anxiety_scores(data_root='/storage/bigdata/Emo-FiLM/brain_data'):
    """
    Load anxiety/trait questionnaire data from Emo-FiLM.

    Emo-FiLM includes DASS (anxiety subscale), BIS/BAS, ERQ, Big Five.

    Parameters
    ----------
    data_root : str
        Path to Emo-FiLM brain_data directory

    Returns
    -------
    demographics : DataFrame
        Subject demographics and trait scores
    """
    print("Loading anxiety/trait questionnaire data...")

    # Emo-FiLM BIDS participants.tsv file
    participants_file = Path(data_root) / 'participants.tsv'

    if not participants_file.exists():
        raise FileNotFoundError(f"participants.tsv not found at {participants_file}")

    demographics = pd.read_csv(participants_file, sep='\t')
    print(f"  ✓ Loaded demographics for {len(demographics)} participants")

    # Rename participant_id to subject_id for consistency
    if 'participant_id' in demographics.columns:
        demographics = demographics.rename(columns={'participant_id': 'subject_id'})

    # Check for DASS_anx column (primary anxiety measure in Emo-FiLM)
    if 'DASS_anx' in demographics.columns:
        print("  ✓ Found DASS_anx (anxiety subscale) scores")
        demographics['anxiety_score'] = demographics['DASS_anx']
    elif 'bis' in demographics.columns:
        # Behavioral Inhibition System (related to anxiety)
        print("  ⚠ Warning: DASS_anx not found, using BIS as anxiety proxy")
        demographics['anxiety_score'] = demographics['bis']
    else:
        raise ValueError("No anxiety-related columns found in participants.tsv")

    return demographics


def analyze_single_subject(timeseries, metadata, hmm_detector):
    """
    Run HMM boundary detection on a single subject.

    Parameters
    ----------
    timeseries : array
        ROI timeseries data
    metadata : dict
        Subject metadata
    hmm_detector : EventBoundaryHMM
        Fitted HMM detector

    Returns
    -------
    results : dict
        Boundary detection results and metrics
    """
    subject_id = metadata['subject_id']
    print(f"\nAnalyzing {subject_id}...")

    # Fit HMM (use optimal states if already determined)
    if hmm_detector.optimal_n_states is None:
        print("  Selecting optimal number of states...")
        optimal_n, scores = hmm_detector.select_optimal_states(timeseries)

    hmm_detector.fit(timeseries)

    # Detect boundaries
    boundaries, state_sequence = hmm_detector.detect_boundaries(timeseries)

    # Compute metrics
    metrics = hmm_detector.compute_metrics(
        boundaries,
        state_sequence,
        tr=metadata['tr'],
        duration_min=metadata['duration_min']
    )

    metrics['subject_id'] = subject_id

    print(f"  ✓ Detected {metrics['n_boundaries']} boundaries")
    print(f"    Boundary rate: {metrics['boundary_rate_per_min']:.2f} per min")

    return metrics, boundaries, state_sequence


def run_group_analysis_emofilm(all_metrics, demographics, output_dir):
    """
    Group-level statistical analysis for Emo-FiLM data.

    Parameters
    ----------
    all_metrics : list of dict
        Boundary metrics for all subjects
    demographics : DataFrame
        Subject demographics and anxiety scores
    output_dir : Path
        Output directory
    """
    print("\n" + "="*60)
    print("GROUP-LEVEL ANALYSIS - EMO-FILM")
    print("="*60)

    # Combine metrics with demographics
    df = pd.DataFrame(all_metrics)

    # Merge with demographics (handle different ID column names)
    if 'participant_id' in demographics.columns:
        demographics = demographics.rename(columns={'participant_id': 'subject_id'})

    df = df.merge(demographics, on='subject_id', how='left')

    # Primary analysis: Anxiety × Boundary Count
    print("\n=== H1: Anxiety × Event Boundary Count ===")

    valid_data = df.dropna(subset=['anxiety_score', 'n_boundaries'])

    if len(valid_data) < 10:
        print("⚠ Warning: Insufficient data for correlation analysis")
        return None

    r, p = stats.pearsonr(valid_data['anxiety_score'], valid_data['n_boundaries'])
    print(f"Pearson r = {r:.3f}, p = {p:.4f}")
    print(f"N = {len(valid_data)} subjects")

    if p < 0.05:
        print("✓ SIGNIFICANT correlation detected")
    else:
        print("✗ No significant correlation")

    # Visualizations
    create_visualizations(df, output_dir)

    # Save results
    df.to_csv(output_dir / 'emofilm_subject_metrics.csv', index=False)

    results = {
        'correlation': {'r': r, 'p': p, 'n': len(valid_data)},
        'mean_boundaries': df['n_boundaries'].mean(),
        'std_boundaries': df['n_boundaries'].std()
    }

    print(f"\n✓ Results saved to {output_dir}/")

    return results


def main():
    """
    Main pipeline for Emo-FiLM HMM analysis.
    """
    parser = argparse.ArgumentParser(description='HMM Event Boundary Detection - Emo-FiLM')
    parser.add_argument('--data_root', type=str,
                       default='/storage/bigdata/Emo-FiLM/brain_data',
                       help='Path to Emo-FiLM brain_data directory')
    parser.add_argument('--output_dir', type=str,
                       default='results_emofilm',
                       help='Output directory')
    parser.add_argument('--session_id', type=str,
                       default='ses-1',
                       help='Session ID (ses-1, ses-2, ses-3, or ses-4)')
    parser.add_argument('--task', type=str,
                       default='Rest',
                       help='Task name (Rest, BigBuckBunny, etc.)')
    parser.add_argument('--atlas', type=str,
                       default='aal',
                       choices=['aal', 'schaefer'],
                       help='Brain atlas for ROI extraction')
    parser.add_argument('--n_subjects', type=int,
                       default=None,
                       help='Number of subjects to analyze (None = all)')

    args = parser.parse_args()

    print("="*60)
    print("HMM EVENT BOUNDARY DETECTION - EMO-FILM DATASET")
    print("="*60)
    print(f"Data root: {args.data_root}")
    print(f"Session: {args.session_id}")
    print(f"Task: {args.task}")
    print(f"Atlas: {args.atlas}")

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)

    # Load demographics/anxiety data
    demographics = load_emofilm_anxiety_scores(args.data_root)

    # Get subject list (sub-S01 to sub-S30, excluding missing sub-S12, sub-S18)
    data_root = Path(args.data_root)
    subject_dirs = sorted([d for d in data_root.glob('sub-S*') if d.is_dir()])

    if args.n_subjects is not None:
        subject_dirs = subject_dirs[:args.n_subjects]

    print(f"\nFound {len(subject_dirs)} subjects")

    # Initialize HMM detector
    hmm_detector = EventBoundaryHMM(n_states_range=(3, 11), random_state=42)

    # Process each subject
    all_metrics = []

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name

        try:
            # Load data
            timeseries, metadata = load_emofilm_data(
                data_root=args.data_root,
                subject_id=subject_id,
                session_id=args.session_id,
                task=args.task,
                atlas=args.atlas
            )

            # Analyze
            metrics, boundaries, state_seq = analyze_single_subject(
                timeseries, metadata, hmm_detector
            )

            all_metrics.append(metrics)

        except Exception as e:
            print(f"  ✗ Error processing {subject_id}: {e}")
            continue

    print(f"\n✓ Successfully processed {len(all_metrics)} subjects")

    # Group analysis
    if len(all_metrics) > 0:
        results = run_group_analysis_emofilm(
            all_metrics,
            demographics,
            output_path
        )

        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        if results:
            print(f"Anxiety × Boundary correlation: r = {results['correlation']['r']:.3f}")
            print(f"p-value: {results['correlation']['p']:.4f}")
    else:
        print("\n✗ No subjects successfully processed")


if __name__ == '__main__':
    main()
