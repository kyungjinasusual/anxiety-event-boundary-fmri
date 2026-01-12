#!/usr/bin/env python3
"""
Emo-Film BSDS Analysis Pipeline
================================

Complete pipeline for applying BSDS to Emo-Film fMRI data.

Features:
- ROI time-series extraction (Schaefer Atlas 200/400)
- BSDS model fitting with full AR dynamics
- Optimization presets (fast/balanced/quality)
- ROI cache reuse for faster re-runs
- Auto session detection (ses-1 through ses-5)
- Comprehensive result analysis and visualization

Usage Examples:
    # Scan available tasks in data directory
    python run_emofilm_bsds.py --task scan

    # RECOMMENDED: Balanced preset (publication quality, ~20-30 min)
    python run_emofilm_bsds.py --task BigBuckBunny --subjects all --preset balanced

    # Fast exploratory analysis (~5-10 min)
    python run_emofilm_bsds.py --task BigBuckBunny --subjects all --preset fast

    # High quality comprehensive analysis (~60+ min)
    python run_emofilm_bsds.py --task BigBuckBunny --subjects all --preset quality

    # Reuse cached ROI extraction (saves ~50% time on re-runs)
    python run_emofilm_bsds.py --task BigBuckBunny --preset balanced --skip-extraction

    # Custom parameters
    python run_emofilm_bsds.py --task BigBuckBunny -K 6 -L 8 --n-iter 50 --n-init 2

    # Analyze all movies
    python run_emofilm_bsds.py --task all --preset balanced

Presets:
    fast:       200 ROIs, K=5, L=6, iter=30, init=1  (~5-10 min)
    balanced:   200 ROIs, K=6, L=8, iter=50, init=2  (~20-30 min)
    quality:    400 ROIs, K=8, L=10, iter=100, init=3 (~60+ min)
    event:      64 event ROIs, K=8, L=12, iter=150, init=5 (~30-45 min) [RECOMMENDED]
    event-fast: 64 event ROIs, K=6, L=8, iter=50, init=2 (~10-15 min)

Event Segmentation ROIs:
    V1, A1, STS, AG, PMC (Precuneus), RSC, mPFC, Anterior Insula, Hippocampus, Amygdala
    These are the key regions for event boundary detection in narrative processing.

Author: Kyungjin Oh
Date: 2025-12-15
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import pickle

# Add bsds_complete to path
sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="BSDS Analysis Pipeline for Emo-Film fMRI Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan available tasks
  python run_emofilm_bsds.py --task scan

  # RECOMMENDED: Balanced preset for publication-quality results
  python run_emofilm_bsds.py --task BigBuckBunny --subjects all --preset balanced

  # Fast exploratory analysis
  python run_emofilm_bsds.py --task BigBuckBunny --subjects all --preset fast

  # Reuse cached ROI extraction (much faster on re-runs)
  python run_emofilm_bsds.py --task BigBuckBunny --preset balanced --skip-extraction

  # Or use external ROI cache directory
  python run_emofilm_bsds.py --task BigBuckBunny --preset balanced --roi-cache ./prev_results/

  # Analyze existing model only
  python run_emofilm_bsds.py --mode analyze --model results/bsds_model.pkl

Presets:
  fast:     K=5, L=6, iter=30, init=1, 200 ROIs  (~5-10 min/movie)
  balanced: K=6, L=8, iter=50, init=2, 200 ROIs  (~20-30 min/movie) [RECOMMENDED]
  quality:  K=8, L=10, iter=100, init=3, 400 ROIs (~60+ min/movie)
        """
    )

    # Mode selection
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'extract', 'fit', 'analyze'],
                       help='Pipeline mode (default: full)')

    # Data source
    parser.add_argument('--data-dir', type=str,
                       default='/storage/bigdata/Emo-FilM/brain_data/derivatives/preprocessing',
                       help='Base directory for preprocessed fMRI data')

    # Emo-FilM task list (14 movies + rest)
    # Comedy: BigBuckBunny, Spaceship
    # Neutral: Coral, Paperman
    # Negative: AfterTheRain, BetweenViewings, Chatter, CitLights,
    #           Lesson, Payload, Riding, Sintel, Superhero, YouAgain
    EMOFILM_TASKS = [
        'AfterTheRain', 'BetweenViewings', 'BigBuckBunny', 'Chatter',
        'CitLights', 'Coral', 'Lesson', 'Paperman', 'Payload',
        'Riding', 'Sintel', 'Spaceship', 'Superhero', 'YouAgain',
        'Rest'
    ]
    parser.add_argument('--task', type=str, default='BigBuckBunny',
                       help=f'Movie task to analyze. Available: {", ".join(EMOFILM_TASKS)}, or "all", or "scan" to detect')
    parser.add_argument('--subjects', type=str, nargs='+', default=['all'],
                       help='Subject IDs (e.g., sub-S01 sub-S02) or "all"')

    # Extraction parameters
    parser.add_argument('--atlas', type=str, default='schaefer',
                       choices=['schaefer', 'aal', 'harvard-oxford', 'event'],
                       help='Brain atlas for ROI extraction. "event" uses event segmentation ROIs')
    parser.add_argument('--n-rois', type=int, default=400,
                       help='Number of ROIs (for Schaefer: 100, 200, 400, etc.)')
    parser.add_argument('--standardize', action='store_true', default=True,
                       help='Z-score normalize time series')

    # Event segmentation ROI options
    parser.add_argument('--roi-preset', type=str, default=None,
                       choices=['full', 'cortical', 'dmn', 'memory', 'sensory'],
                       help='''Event segmentation ROI presets:
                         full: All regions (~64 ROIs) - V1,A1,STS,AG,PMC,RSC,mPFC,AntInsula,Hipp,Amyg
                         cortical: Cortical only (~60 ROIs) - without Hippocampus/Amygdala
                         dmn: Default Mode (~30 ROIs) - AG,PMC,RSC,mPFC
                         memory: Memory regions (~28 ROIs) - PMC,RSC,mPFC,Hippocampus
                         sensory: Sensory processing (~24 ROIs) - V1,A1,STS''')
    parser.add_argument('--event-regions', type=str, nargs='+', default=None,
                       help='Custom event ROI regions: V1,A1,STS,AG,PMC,RSC,mPFC,AntInsula,Hippocampus,Amygdala')
    parser.add_argument('--no-subcortical', action='store_true',
                       help='Exclude subcortical regions (Hippocampus, Amygdala)')

    # Optimization presets (overrides individual params if set)
    parser.add_argument('--preset', type=str, default=None,
                       choices=['fast', 'balanced', 'quality', 'event', 'event-fast'],
                       help='''Optimization presets:
                         fast: ~5-10min, 200 ROIs, K=5, L=6, iter=30, init=1 (exploratory)
                         balanced: ~20-30min, 200 ROIs, K=6, L=8, iter=50, init=2 (publication)
                         quality: ~60min+, 400 ROIs, K=8, L=10, iter=100, init=3 (comprehensive)
                         event: ~30-45min, 64 event ROIs, K=8, L=12, iter=150, init=5 (RECOMMENDED)
                         event-fast: ~10-15min, 64 event ROIs, K=6, L=8, iter=50, init=2''')

    # BSDS model parameters (can be overridden by --preset)
    parser.add_argument('--n-states', '-K', type=int, default=None,
                       help='Number of brain states (default: 5, or set by preset)')
    parser.add_argument('--max-ldim', '-L', type=int, default=None,
                       help='Maximum latent dimension (default: 10, or set by preset)')
    parser.add_argument('--n-iter', type=int, default=None,
                       help='Number of VB iterations (default: 100, or set by preset)')
    parser.add_argument('--n-init', type=int, default=None,
                       help='Number of random initializations (default: 5, or set by preset)')
    parser.add_argument('--tol', type=float, default=1e-3,
                       help='Convergence tolerance (default: 1e-3)')
    parser.add_argument('--ar-approach', type=int, default=1, choices=[1, 2, 3],
                       help='AR dynamics approach (default: 1)')
    parser.add_argument('--TR', type=float, default=1.3,
                       help='Repetition time in seconds (default: 1.3 for Emo-FilM)')

    # Input/Output
    parser.add_argument('--input', '-i', type=str, default=None,
                       help='Input directory for extracted time series (for fit mode)')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to saved model (for analyze mode)')
    parser.add_argument('--output', '-o', type=str, default='./results',
                       help='Output directory (default: ./results)')
    parser.add_argument('--prefix', type=str, default=None,
                       help='Output file prefix (default: auto-generated)')

    # Execution options
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--verbose', '-v', action='store_true', default=True,
                       help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress output')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--save-intermediate', action='store_true',
                       help='Save intermediate results')

    # ROI extraction reuse options
    parser.add_argument('--skip-extraction', action='store_true',
                       help='Skip ROI extraction if files already exist (reuse previous results)')
    parser.add_argument('--roi-cache', type=str, default=None,
                       help='Directory containing pre-extracted ROI time series to reuse')

    return parser.parse_args()


# =============================================================================
# Preset Configurations
# =============================================================================

PRESETS = {
    'fast': {
        'n_rois': 200,
        'n_states': 5,
        'max_ldim': 6,
        'n_iter': 30,
        'n_init': 1,
        'description': 'Fast exploratory analysis (~5-10 min per movie)',
    },
    'balanced': {
        'n_rois': 200,
        'n_states': 6,
        'max_ldim': 8,
        'n_iter': 50,
        'n_init': 2,
        'description': 'Publication-quality with reasonable time (~20-30 min)',
    },
    'quality': {
        'n_rois': 400,
        'n_states': 8,
        'max_ldim': 10,
        'n_iter': 100,
        'n_init': 3,
        'description': 'Comprehensive high-quality analysis (~60+ min)',
    },
    # Event segmentation optimized presets (for ~60 ROIs)
    'event': {
        'n_rois': 400,  # Base atlas, actual ROIs selected by roi-preset
        'n_states': 8,
        'max_ldim': 12,
        'n_iter': 150,
        'n_init': 5,
        'roi_preset': 'full',
        'description': 'Event segmentation ROIs (~64 ROIs), high quality (~30-45 min)',
    },
    'event-fast': {
        'n_rois': 400,
        'n_states': 6,
        'max_ldim': 8,
        'n_iter': 50,
        'n_init': 2,
        'roi_preset': 'full',
        'description': 'Event segmentation ROIs, faster (~10-15 min)',
    },
}

# Default values when no preset is specified
DEFAULTS = {
    'n_rois': 400,
    'n_states': 5,
    'max_ldim': 10,
    'n_iter': 100,
    'n_init': 5,
}


def apply_preset(args):
    """Apply preset configuration, with explicit args taking precedence."""
    preset_config = PRESETS.get(args.preset, {}) if args.preset else {}

    # Apply preset or defaults, but explicit args take precedence
    if args.n_rois == 400 and 'n_rois' in preset_config:  # Only override if using default
        args.n_rois = preset_config['n_rois']

    args.n_states = args.n_states if args.n_states is not None else preset_config.get('n_states', DEFAULTS['n_states'])
    args.max_ldim = args.max_ldim if args.max_ldim is not None else preset_config.get('max_ldim', DEFAULTS['max_ldim'])
    args.n_iter = args.n_iter if args.n_iter is not None else preset_config.get('n_iter', DEFAULTS['n_iter'])
    args.n_init = args.n_init if args.n_init is not None else preset_config.get('n_init', DEFAULTS['n_init'])

    # Apply roi_preset from event presets if not explicitly set
    if args.roi_preset is None and 'roi_preset' in preset_config:
        args.roi_preset = preset_config['roi_preset']

    # If using event preset, also set atlas to 'event'
    if args.preset in ['event', 'event-fast'] and args.atlas == 'schaefer':
        args.atlas = 'event'

    if args.preset:
        print(f"\n[Preset: {args.preset}] {preset_config.get('description', '')}")

    return args


def get_subjects(data_dir: str, subjects_arg: list) -> list:
    """Get list of subject IDs."""
    if 'all' in subjects_arg:
        # Scan directory for subjects
        data_path = Path(data_dir)
        if data_path.exists():
            subjects = sorted([d.name for d in data_path.iterdir()
                             if d.is_dir() and d.name.startswith('sub-')])
        else:
            subjects = []
    else:
        subjects = subjects_arg
    return subjects


def scan_available_tasks(data_dir: str, subjects: list = None) -> dict:
    """
    Scan the data directory to find all available tasks.

    Returns:
        Dictionary with task names as keys and count of subjects with that task
    """
    data_path = Path(data_dir)
    task_counts = {}

    if subjects is None:
        subjects = get_subjects(data_dir, ['all'])

    for subj in subjects[:5]:  # Sample first 5 subjects for speed
        subj_dir = data_path / subj
        if not subj_dir.exists():
            continue

        for session_dir in subj_dir.iterdir():
            if not session_dir.is_dir() or not session_dir.name.startswith('ses-'):
                continue

            func_dir = session_dir / 'func'
            if not func_dir.exists():
                continue

            # Find all task files
            for f in func_dir.glob('*task-*_bold.nii.gz'):
                # Extract task name from filename
                fname = f.name
                task_start = fname.find('task-') + 5
                task_end = fname.find('_', task_start)
                if task_start > 4 and task_end > task_start:
                    task_name = fname[task_start:task_end]
                    task_counts[task_name] = task_counts.get(task_name, 0) + 1

    return task_counts


# Emo-FilM known tasks (will be validated against scan)
EMOFILM_TASKS = [
    'AfterTheRain', 'BetweenViewings', 'BigBuckBunny', 'Chatter',
    'CitLights', 'Coral', 'Lesson', 'Paperman', 'Payload',
    'Riding', 'Sintel', 'Spaceship', 'Superhero', 'YouAgain',
    'Rest'
]


def find_session_for_task(data_dir: str, subject: str, task: str) -> str:
    """
    Auto-detect which session contains the given task for a subject.

    Emo-FilM uses pseudo-randomized design where different subjects
    see different movies in different sessions (ses-1 through ses-5).

    Args:
        data_dir: Base data directory
        subject: Subject ID (e.g., 'sub-S01')
        task: Task name (e.g., 'BigBuckBunny')

    Returns:
        Session ID (e.g., 'ses-1', 'ses-2', ..., 'ses-5') or None if not found
    """
    data_path = Path(data_dir)
    subj_dir = data_path / subject

    if not subj_dir.exists():
        return None

    # Dynamically find all sessions for this subject
    sessions = sorted([d.name for d in subj_dir.iterdir()
                      if d.is_dir() and d.name.startswith('ses-')])

    if not sessions:
        return None

    # Search through ALL sessions
    for session in sessions:
        func_dir = subj_dir / session / 'func'
        if func_dir.exists():
            # Look for the task file with multiple naming patterns
            patterns = [
                f"{subject}_{session}_task-{task}_*bold.nii.gz",
                f"{subject}_{session}_task-{task}_space-MNI*bold.nii.gz",
                f"*{session}*task-{task}*bold.nii.gz",
            ]
            for pattern in patterns:
                matches = list(func_dir.glob(pattern))
                if matches:
                    return session

    # Not found in any session
    return None


def find_all_sessions_for_task(data_dir: str, subjects: list, task: str) -> dict:
    """
    Find which session contains the task for all subjects.

    Args:
        data_dir: Base data directory
        subjects: List of subject IDs
        task: Task name

    Returns:
        Dictionary mapping subject -> session
    """
    session_map = {}
    for subj in subjects:
        session_map[subj] = find_session_for_task(data_dir, subj, task)
    return session_map


def load_cached_timeseries(cache_dir: Path, subjects: list, task: str) -> dict:
    """Load pre-extracted ROI time series from cache directory."""
    results = {}
    loaded = 0
    for subj in subjects:
        cache_file = cache_dir / f"{subj}_{task}_timeseries.npy"
        if cache_file.exists():
            ts = np.load(cache_file)
            results[subj] = ts
            loaded += 1
    return results, loaded


def find_existing_cache(output_base: Path, task: str) -> Path:
    """
    Auto-detect existing timeseries cache from previous runs.

    Searches for the most recent results folder containing timeseries files
    for the given task.

    Args:
        output_base: Base output directory (e.g., ./results)
        task: Task name (e.g., 'BigBuckBunny')

    Returns:
        Path to cache directory, or None if not found
    """
    if not output_base.exists():
        return None

    # Look for folders matching pattern: bsds_{task}_*
    candidates = []
    for folder in output_base.iterdir():
        if not folder.is_dir():
            continue

        # Check if folder name contains the task
        if task.lower() in folder.name.lower() or 'bsds_' in folder.name.lower():
            # Check if it contains timeseries files for this task
            ts_files = list(folder.glob(f"*_{task}_timeseries.npy"))
            if ts_files:
                # Get modification time of the folder
                mtime = folder.stat().st_mtime
                candidates.append((mtime, folder, len(ts_files)))

    if not candidates:
        # Also check in output_base directly (flat structure)
        ts_files = list(output_base.glob(f"*_{task}_timeseries.npy"))
        if ts_files:
            return output_base
        return None

    # Sort by modification time (most recent first)
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Return the most recent folder with the most files
    best_folder = candidates[0][1]
    n_files = candidates[0][2]

    return best_folder


def extract_timeseries(args, subjects: list, output_dir: Path) -> dict:
    """Extract ROI time series from fMRI data."""
    print("\n" + "=" * 60)
    print("STEP 1: ROI Time Series Extraction")
    print("=" * 60)

    # Determine tasks
    if args.task == 'all':
        tasks = EMOFILM_TASKS
    else:
        tasks = [args.task]

    # Check for ROI cache reuse
    output_base = Path(args.output)  # Base output directory for auto-detection
    results = {}

    for task in tasks:
        print(f"\nTask: {task}")
        results[task] = {}

        # Determine cache directory
        if args.roi_cache:
            cache_dir = Path(args.roi_cache)
        elif args.skip_extraction:
            # Auto-detect from previous runs
            cache_dir = find_existing_cache(output_base, task)
            if cache_dir:
                print(f"  Auto-detected cache: {cache_dir}")
            else:
                cache_dir = output_dir  # Fallback to current output
        else:
            cache_dir = output_dir

        # Check if we should skip extraction and use cached files
        if args.skip_extraction or args.roi_cache:
            cached_results, n_loaded = load_cached_timeseries(cache_dir, subjects, task)
            if n_loaded > 0:
                print(f"  Loaded {n_loaded} cached ROI time series from {cache_dir}")
                results[task] = cached_results

                # If all subjects loaded from cache, skip extraction for this task
                if n_loaded == len(subjects):
                    print(f"  All {n_loaded} subjects loaded from cache, skipping extraction")
                    continue
                else:
                    # Only extract for subjects not in cache
                    subjects_to_extract = [s for s in subjects if s not in cached_results]
                    print(f"  {len(subjects_to_extract)} subjects need extraction")
            else:
                subjects_to_extract = subjects
        else:
            subjects_to_extract = subjects

        if not subjects_to_extract:
            continue

        # Now do actual extraction for subjects_to_extract
        try:
            from nilearn import datasets
            from nilearn.maskers import NiftiLabelsMasker
        except ImportError:
            print("ERROR: nilearn not installed. Run: pip install nilearn")
            sys.exit(1)

        # Setup atlas/masker based on atlas type
        roi_info = None
        if args.atlas == 'event' or args.roi_preset is not None:
            # Use event segmentation ROIs
            from bsds_complete.utils.roi_selection import (
                create_event_roi_masker, EVENT_ROI_PRESETS, get_region_summary
            )

            # Determine regions
            if args.event_regions:
                regions = args.event_regions
                include_subcortical = not args.no_subcortical
            elif args.roi_preset:
                preset_info = EVENT_ROI_PRESETS.get(args.roi_preset, EVENT_ROI_PRESETS['full'])
                regions = preset_info.get('regions')
                include_subcortical = preset_info.get('include_subcortical', True) and not args.no_subcortical
            else:
                regions = None
                include_subcortical = not args.no_subcortical

            print(f"  Loading event segmentation ROIs (preset: {args.roi_preset or 'custom'})...")
            masker, roi_info = create_event_roi_masker(
                n_rois=args.n_rois,
                include_subcortical=include_subcortical,
                regions=regions,
                standardize=args.standardize
            )
            print(f"  Selected {roi_info['n_selected']} ROIs from regions: {list(roi_info['cortical_regions'].keys())}")
            if roi_info['subcortical_regions']:
                print(f"  + Subcortical: {roi_info['subcortical_regions']}")

        elif args.atlas == 'schaefer':
            print(f"  Loading Schaefer atlas ({args.n_rois} ROIs)...")
            atlas = datasets.fetch_atlas_schaefer_2018(n_rois=args.n_rois, yeo_networks=7)
            atlas_img = atlas.maps
            masker = NiftiLabelsMasker(
                labels_img=atlas_img,
                standardize=args.standardize,
                memory='nilearn_cache',
                verbose=0
            )
        else:
            print(f"  Atlas {args.atlas} not yet implemented. Using Schaefer.")
            atlas = datasets.fetch_atlas_schaefer_2018(n_rois=args.n_rois, yeo_networks=7)
            atlas_img = atlas.maps
            masker = NiftiLabelsMasker(
                labels_img=atlas_img,
                standardize=args.standardize,
                memory='nilearn_cache',
                verbose=0
            )

        # Auto-detect sessions for this task
        session_map = find_all_sessions_for_task(args.data_dir, subjects_to_extract, task)

        # Show session mapping summary
        found_sessions = {s: sess for s, sess in session_map.items() if sess is not None}
        not_found = [s for s, sess in session_map.items() if sess is None]
        print(f"  Session detection: {len(found_sessions)} found, {len(not_found)} not found")
        if found_sessions:
            # Show unique session distribution
            session_counts = {}
            for sess in found_sessions.values():
                session_counts[sess] = session_counts.get(sess, 0) + 1
            print(f"  Session distribution: {session_counts}")

        for subj in subjects_to_extract:
            session = session_map.get(subj)

            # Skip if session not found
            if session is None:
                print(f"  {subj}: Task '{task}' not found in any session, skipping")
                continue

            # Find the actual file
            func_dir = Path(args.data_dir) / subj / session / 'func'
            func_path = None

            # Search for file with multiple naming patterns
            file_patterns = [
                f"{subj}_{session}_task-{task}_space-MNI_desc-ppres_bold.nii.gz",
                f"{subj}_{session}_task-{task}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
                f"{subj}_{session}_task-{task}_bold.nii.gz",
            ]

            for pattern in file_patterns:
                candidate = func_dir / pattern
                if candidate.exists():
                    func_path = candidate
                    break

            # Also try glob patterns
            if func_path is None:
                glob_patterns = [
                    f"{subj}_{session}_task-{task}_*bold.nii.gz",
                    f"*task-{task}*bold.nii.gz",
                ]
                for glob_pat in glob_patterns:
                    matches = list(func_dir.glob(glob_pat))
                    if matches:
                        func_path = matches[0]
                        break

            if func_path is None:
                print(f"  {subj} ({session}): File not found, skipping")
                continue

            try:
                ts = masker.fit_transform(str(func_path))
                results[task][subj] = ts

                # Save individual file
                save_path = output_dir / f"{subj}_{task}_timeseries.npy"
                np.save(save_path, ts)
                print(f"  {subj} ({session}): {ts.shape} saved")

            except Exception as e:
                print(f"  {subj} ({session}): Error - {e}")

    return results


def fit_bsds(args, data_dict: dict, output_dir: Path):
    """Fit BSDS model to extracted time series."""
    print("\n" + "=" * 60)
    print("STEP 2: BSDS Model Fitting")
    print("=" * 60)

    from bsds_complete import BSDSModel, BSDSConfig

    # Prepare data
    data_list = []
    subject_order = []

    for task, subjects_data in data_dict.items():
        for subj, ts in subjects_data.items():
            # Transpose to (ROI x Time) format
            data_list.append(ts.T)
            subject_order.append(f"{subj}_{task}")

    if len(data_list) == 0:
        print("ERROR: No data to fit!")
        sys.exit(1)

    print(f"Data: {len(data_list)} runs, {data_list[0].shape[0]} ROIs")

    # Configure model
    config = BSDSConfig(
        n_states=args.n_states,
        max_ldim=args.max_ldim,
        n_iter=args.n_iter,
        n_init_learning=args.n_init,
        tol=args.tol,
        ar_approach=args.ar_approach,
        TR=args.TR,
        random_seed=args.seed,
        verbose=not args.quiet
    )

    print(f"\nModel Configuration:")
    print(f"  States (K): {config.n_states}")
    print(f"  Latent dim: {config.max_ldim}")
    print(f"  Iterations: {config.n_iter}")
    print(f"  Initializations: {config.n_init_learning}")
    print(f"  TR: {config.TR}s")

    # Fit model
    print("\nFitting BSDS model...")
    model = BSDSModel(config)
    model.fit(data_list)

    # Save model
    model_path = output_dir / f"{args.prefix}_model.pkl"
    model.save(str(model_path))
    print(f"\nModel saved: {model_path}")

    # Save config
    config_path = output_dir / f"{args.prefix}_config.json"
    config.save(str(config_path))

    # Save subject order
    order_path = output_dir / f"{args.prefix}_subjects.json"
    with open(order_path, 'w') as f:
        json.dump(subject_order, f, indent=2)

    return model, subject_order


def analyze_results(args, model, subject_order: list, output_dir: Path):
    """Analyze and visualize BSDS results."""
    print("\n" + "=" * 60)
    print("STEP 3: Results Analysis")
    print("=" * 60)

    from bsds_complete.analysis import create_summary_report

    # Get summary statistics
    stats = model.get_summary_statistics()
    stats['subject_order'] = subject_order

    # Print key results
    print("\n--- Key Results ---")
    print(f"Number of states: {stats['n_states']}")
    print(f"Dominant states: {stats['dominant_states_group']}")

    print("\nState Occupancy:")
    for i, occ in enumerate(stats['occupancy_group']):
        print(f"  State {i}: {occ*100:.1f}%")

    print("\nMean Lifetime (seconds):")
    for i, lt in enumerate(stats['mean_lifetime_group']):
        print(f"  State {i}: {lt:.2f}s")

    print("\nTransition Statistics:")
    ts = stats['transition_stats']
    print(f"  Mean transitions per run: {ts['mean_transitions']:.1f}")

    # Generate report
    if not args.no_plots:
        report_path = create_summary_report(stats, str(output_dir), args.prefix)
        print(f"\nReport generated: {report_path}")

    # Save state sequences
    states = model.get_states()
    for i, (states_i, subj) in enumerate(zip(states, subject_order)):
        state_path = output_dir / f"{subj}_states.npy"
        np.save(state_path, states_i)

    print(f"\nState sequences saved to {output_dir}/")

    return stats


def main():
    """Main pipeline execution."""
    args = parse_args()

    # Apply preset configuration first
    args = apply_preset(args)

    # Handle task scanning
    if args.task == 'scan':
        print("\n" + "=" * 60)
        print("    Scanning Available Tasks")
        print("=" * 60)
        task_counts = scan_available_tasks(args.data_dir)
        print(f"\nFound {len(task_counts)} tasks in {args.data_dir}:")
        for task, count in sorted(task_counts.items()):
            known = "✓" if task in EMOFILM_TASKS else "?"
            print(f"  [{known}] {task}: {count} files (sampled)")
        print("\nUse --task <TaskName> to analyze a specific task")
        return

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.prefix is None:
        args.prefix = f"bsds_{args.task}_{args.n_states}states_{timestamp}"

    output_dir = Path(args.output) / args.prefix
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("    BSDS Analysis Pipeline for Emo-Film")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Preset: {args.preset or 'custom'}")
    print(f"Output: {output_dir}")
    print(f"Parameters: K={args.n_states}, L={args.max_ldim}, iter={args.n_iter}, init={args.n_init}, ROIs={args.n_rois}")

    # Save args
    args_path = output_dir / "args.json"
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Execute based on mode
    if args.mode in ['full', 'extract']:
        subjects = get_subjects(args.data_dir, args.subjects)
        print(f"Subjects: {len(subjects)}")

        if len(subjects) == 0:
            print("ERROR: No subjects found!")
            sys.exit(1)

        data_dict = extract_timeseries(args, subjects, output_dir)

        if args.mode == 'extract':
            print("\nExtraction complete!")
            return

    elif args.mode == 'fit':
        # Load from input directory
        if args.input is None:
            print("ERROR: --input required for fit mode")
            sys.exit(1)

        input_dir = Path(args.input)
        data_dict = {'loaded': {}}

        for f in input_dir.glob("*_timeseries.npy"):
            name = f.stem.replace('_timeseries', '')
            ts = np.load(f)
            data_dict['loaded'][name] = ts

        if len(data_dict['loaded']) == 0:
            print(f"ERROR: No timeseries files found in {input_dir}")
            sys.exit(1)

    elif args.mode == 'analyze':
        if args.model is None:
            print("ERROR: --model required for analyze mode")
            sys.exit(1)

        from bsds_complete import BSDSModel
        model = BSDSModel.load(args.model)

        # Load subject order if available
        model_dir = Path(args.model).parent
        order_file = model_dir / f"{Path(args.model).stem.replace('_model', '')}_subjects.json"
        if order_file.exists():
            with open(order_file) as f:
                subject_order = json.load(f)
        else:
            subject_order = [f"run_{i}" for i in range(len(model.get_states()))]

        analyze_results(args, model, subject_order, output_dir)
        print("\nAnalysis complete!")
        return

    # Fit and analyze
    if args.mode in ['full', 'fit']:
        model, subject_order = fit_bsds(args, data_dict, output_dir)
        analyze_results(args, model, subject_order, output_dir)

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
