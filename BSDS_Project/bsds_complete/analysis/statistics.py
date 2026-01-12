"""
Post-hoc Statistics for BSDS
Implements: compute_occupancy_and_mean_life_*.m, getDominantStateIds*.m, getMean.m, getCovariance.m
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from collections import Counter


def compute_occupancy_group(states_list: List[np.ndarray],
                            n_states: int) -> np.ndarray:
    """
    Compute fractional occupancy of each state across all subjects.
    Corresponds to compute_occupancy_and_mean_life_group_wise.m (occupancy part)

    Fractional occupancy = (time spent in state) / (total time)

    Args:
        states_list: List of state sequences per subject
        n_states: Total number of states

    Returns:
        Fractional occupancy for each state (K,)
    """
    total_counts = np.zeros(n_states)
    total_time = 0

    for states in states_list:
        for s in range(n_states):
            total_counts[s] += np.sum(states == s)
        total_time += len(states)

    occupancy = total_counts / total_time if total_time > 0 else np.zeros(n_states)
    return occupancy


def compute_occupancy_subject(states_list: List[np.ndarray],
                              n_states: int) -> np.ndarray:
    """
    Compute fractional occupancy per subject.
    Corresponds to compute_occupancy_and_mean_life_subject_wise.m (occupancy part)

    Args:
        states_list: List of state sequences per subject
        n_states: Total number of states

    Returns:
        Fractional occupancy per subject (n_subjects x K)
    """
    n_subjects = len(states_list)
    occupancy = np.zeros((n_subjects, n_states))

    for i, states in enumerate(states_list):
        T = len(states)
        for s in range(n_states):
            occupancy[i, s] = np.sum(states == s) / T if T > 0 else 0

    return occupancy


def compute_mean_lifetime_group(states_list: List[np.ndarray],
                                n_states: int,
                                TR: float = 1.0) -> np.ndarray:
    """
    Compute mean lifetime (dwell time) of each state across all subjects.
    Corresponds to compute_occupancy_and_mean_life_group_wise.m (lifetime part)

    Mean lifetime = average duration of contiguous state visits

    Args:
        states_list: List of state sequences per subject
        n_states: Total number of states
        TR: Repetition time in seconds (for conversion to seconds)

    Returns:
        Mean lifetime for each state in seconds (K,)
    """
    state_durations = {s: [] for s in range(n_states)}

    for states in states_list:
        # Find contiguous runs
        runs = _get_runs(states)
        for state, duration in runs:
            state_durations[state].append(duration)

    mean_lifetime = np.zeros(n_states)
    for s in range(n_states):
        if len(state_durations[s]) > 0:
            mean_lifetime[s] = np.mean(state_durations[s]) * TR

    return mean_lifetime


def compute_mean_lifetime_subject(states_list: List[np.ndarray],
                                  n_states: int,
                                  TR: float = 1.0) -> np.ndarray:
    """
    Compute mean lifetime per subject.
    Corresponds to compute_occupancy_and_mean_life_subject_wise.m (lifetime part)

    Args:
        states_list: List of state sequences per subject
        n_states: Total number of states
        TR: Repetition time in seconds

    Returns:
        Mean lifetime per subject (n_subjects x K)
    """
    n_subjects = len(states_list)
    mean_lifetime = np.zeros((n_subjects, n_states))

    for i, states in enumerate(states_list):
        state_durations = {s: [] for s in range(n_states)}
        runs = _get_runs(states)

        for state, duration in runs:
            state_durations[state].append(duration)

        for s in range(n_states):
            if len(state_durations[s]) > 0:
                mean_lifetime[i, s] = np.mean(state_durations[s]) * TR

    return mean_lifetime


def _get_runs(states: np.ndarray) -> List[Tuple[int, int]]:
    """
    Get contiguous runs of states.

    Args:
        states: State sequence

    Returns:
        List of (state, duration) tuples
    """
    if len(states) == 0:
        return []

    runs = []
    current_state = states[0]
    current_duration = 1

    for i in range(1, len(states)):
        if states[i] == current_state:
            current_duration += 1
        else:
            runs.append((current_state, current_duration))
            current_state = states[i]
            current_duration = 1

    runs.append((current_state, current_duration))
    return runs


def get_dominant_states_group(states_list: List[np.ndarray],
                              n_states: int,
                              threshold: float = 0.05) -> List[int]:
    """
    Identify dominant states based on group-level occupancy.
    Corresponds to getDominantStateIdsGroup.m

    A state is dominant if its occupancy exceeds the threshold.

    Args:
        states_list: List of state sequences per subject
        n_states: Total number of states
        threshold: Minimum occupancy to be considered dominant

    Returns:
        List of dominant state indices (0-based)
    """
    occupancy = compute_occupancy_group(states_list, n_states)
    dominant = [i for i in range(n_states) if occupancy[i] > threshold]
    return sorted(dominant, key=lambda x: -occupancy[x])


def get_dominant_states_subject(states_list: List[np.ndarray],
                                n_states: int,
                                threshold: float = 0.05) -> List[List[int]]:
    """
    Identify dominant states for each subject.
    Corresponds to getDominantStateIdsSubject.m

    Args:
        states_list: List of state sequences per subject
        n_states: Total number of states
        threshold: Minimum occupancy to be considered dominant

    Returns:
        List of dominant state indices per subject
    """
    occupancy = compute_occupancy_subject(states_list, n_states)
    dominant_list = []

    for i in range(len(states_list)):
        dominant = [s for s in range(n_states) if occupancy[i, s] > threshold]
        dominant_list.append(sorted(dominant, key=lambda x: -occupancy[i, x]))

    return dominant_list


def compute_state_covariance(Lm: List[np.ndarray],
                             Lcov: List[np.ndarray],
                             psii: np.ndarray) -> List[np.ndarray]:
    """
    Compute observation covariance for each state.
    Corresponds to getCovariance.m

    Cov(Y | state=s) = L_s @ L_s' + diag(1/psii)

    Args:
        Lm: List of factor loading means per state
        Lcov: List of factor loading covariances per state
        psii: Noise precision

    Returns:
        List of covariance matrices per state
    """
    n_states = len(Lm)
    D = Lm[0].shape[0]
    noise_cov = np.diag(1.0 / (psii.flatten() + 1e-10))

    cov_list = []

    for s in range(n_states):
        # Factor contribution
        # E[L @ L'] = Lm @ Lm' + sum_d Lcov[:,:,d]
        factor_cov = Lm[s] @ Lm[s].T
        for d in range(D):
            factor_cov += Lcov[s][:, :, d]

        # Total covariance
        cov_s = factor_cov + noise_cov
        cov_list.append(cov_s)

    return cov_list


def compute_state_mean(Lm: List[np.ndarray]) -> List[np.ndarray]:
    """
    Compute observation mean for each state.
    Corresponds to getMean.m

    E[Y | state=s] = L_s[:, 0] (the bias/mean column)

    Args:
        Lm: List of factor loading means per state

    Returns:
        List of mean vectors per state
    """
    return [L[:, 0] for L in Lm]


def compute_transition_statistics(states_list: List[np.ndarray],
                                  n_states: int) -> Dict[str, Any]:
    """
    Compute transition-related statistics.

    Args:
        states_list: List of state sequences per subject
        n_states: Total number of states

    Returns:
        Dictionary with transition statistics
    """
    # Empirical transition matrix
    trans_counts = np.zeros((n_states, n_states))

    for states in states_list:
        for t in range(len(states) - 1):
            trans_counts[states[t], states[t + 1]] += 1

    # Normalize
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    trans_prob = trans_counts / row_sums

    # Self-transition probability (persistence)
    persistence = np.diag(trans_prob)

    # Number of transitions per subject
    n_transitions = []
    for states in states_list:
        n_trans = np.sum(states[1:] != states[:-1])
        n_transitions.append(n_trans)

    return {
        'transition_counts': trans_counts,
        'transition_prob': trans_prob,
        'persistence': persistence,
        'n_transitions_per_subject': np.array(n_transitions),
        'mean_transitions': np.mean(n_transitions)
    }


def compute_summary_statistics(states_list: List[np.ndarray],
                               n_states: int,
                               TR: float = 1.0) -> Dict[str, Any]:
    """
    Compute comprehensive summary statistics.

    Args:
        states_list: List of state sequences per subject
        n_states: Total number of states
        TR: Repetition time

    Returns:
        Dictionary with all summary statistics
    """
    return {
        'occupancy_group': compute_occupancy_group(states_list, n_states),
        'occupancy_subject': compute_occupancy_subject(states_list, n_states),
        'mean_lifetime_group': compute_mean_lifetime_group(states_list, n_states, TR),
        'mean_lifetime_subject': compute_mean_lifetime_subject(states_list, n_states, TR),
        'dominant_states_group': get_dominant_states_group(states_list, n_states),
        'dominant_states_subject': get_dominant_states_subject(states_list, n_states),
        'transition_stats': compute_transition_statistics(states_list, n_states),
        'n_subjects': len(states_list),
        'n_states': n_states,
        'TR': TR
    }


# =============================================================================
# Transition Timestamp Extraction
# =============================================================================

def extract_transition_timestamps(states: np.ndarray,
                                   TR: float = 1.0,
                                   return_details: bool = False) -> Dict[str, Any]:
    """
    Extract precise timestamps of state transitions (neural event boundaries).

    This function identifies when the brain switches from one state to another,
    which can be interpreted as neural event boundaries.

    Args:
        states: State sequence (T,)
        TR: Repetition time in seconds
        return_details: If True, include detailed transition info

    Returns:
        Dictionary containing:
        - timestamps: Array of transition times in seconds
        - indices: Array of transition indices (TR units)
        - n_transitions: Total number of transitions
        - transition_rate: Transitions per minute
        - details: (optional) List of (time, from_state, to_state) tuples
    """
    T = len(states)

    # Find transition indices
    transition_indices = np.where(np.diff(states) != 0)[0] + 1

    # Convert to timestamps
    timestamps = transition_indices * TR

    # Calculate transition rate (per minute)
    total_time_minutes = (T * TR) / 60.0
    transition_rate = len(transition_indices) / total_time_minutes if total_time_minutes > 0 else 0

    result = {
        'timestamps': timestamps,
        'indices': transition_indices,
        'n_transitions': len(transition_indices),
        'transition_rate': transition_rate,
        'total_duration': T * TR
    }

    if return_details:
        details = []
        for idx in transition_indices:
            from_state = int(states[idx - 1])
            to_state = int(states[idx])
            time_sec = idx * TR
            details.append({
                'time': time_sec,
                'from_state': from_state,
                'to_state': to_state,
                'index': int(idx)
            })
        result['details'] = details

    return result


def extract_transitions_group(states_list: List[np.ndarray],
                               TR: float = 1.0) -> Dict[str, Any]:
    """
    Extract transition timestamps for all subjects.

    Args:
        states_list: List of state sequences
        TR: Repetition time

    Returns:
        Dictionary with group-level transition statistics
    """
    all_timestamps = []
    all_rates = []
    all_n_transitions = []
    subject_details = []

    for i, states in enumerate(states_list):
        trans_info = extract_transition_timestamps(states, TR, return_details=True)
        all_timestamps.append(trans_info['timestamps'])
        all_rates.append(trans_info['transition_rate'])
        all_n_transitions.append(trans_info['n_transitions'])
        subject_details.append(trans_info)

    return {
        'timestamps_per_subject': all_timestamps,
        'transition_rates': np.array(all_rates),
        'n_transitions_per_subject': np.array(all_n_transitions),
        'mean_transition_rate': np.mean(all_rates),
        'std_transition_rate': np.std(all_rates),
        'subject_details': subject_details
    }


# =============================================================================
# Individual Difference Analysis (Anxiety Correlation)
# =============================================================================

def compute_individual_features(states_list: List[np.ndarray],
                                 n_states: int,
                                 TR: float = 1.0) -> Dict[str, np.ndarray]:
    """
    Compute individual-level features for correlation analysis.

    Extracts features relevant to H1 (Hypersegmentation Hypothesis):
    - Higher anxiety -> More frequent neural event boundaries
    - Higher anxiety -> Shorter dwell times
    - Higher anxiety -> More variable state patterns

    Args:
        states_list: List of state sequences per subject
        n_states: Total number of states
        TR: Repetition time

    Returns:
        Dictionary of feature arrays (n_subjects,)
    """
    n_subjects = len(states_list)

    features = {
        # H1: Hypersegmentation measures
        'n_transitions': np.zeros(n_subjects),
        'transition_rate': np.zeros(n_subjects),  # Per minute
        'mean_dwell_time': np.zeros(n_subjects),
        'dwell_time_cv': np.zeros(n_subjects),  # Coefficient of variation

        # State diversity measures
        'n_unique_states': np.zeros(n_subjects),
        'state_entropy': np.zeros(n_subjects),

        # Temporal dynamics
        'dominant_state_occupancy': np.zeros(n_subjects),
        'switching_entropy': np.zeros(n_subjects),
    }

    for i, states in enumerate(states_list):
        T = len(states)
        total_time_sec = T * TR

        # Transition measures
        transitions = np.sum(states[1:] != states[:-1])
        features['n_transitions'][i] = transitions
        features['transition_rate'][i] = (transitions / total_time_sec) * 60  # per minute

        # Dwell time measures
        runs = _get_runs(states)
        dwell_times = [duration * TR for _, duration in runs]
        features['mean_dwell_time'][i] = np.mean(dwell_times) if dwell_times else 0
        features['dwell_time_cv'][i] = (np.std(dwell_times) / np.mean(dwell_times)
                                         if dwell_times and np.mean(dwell_times) > 0 else 0)

        # State diversity
        unique_states = np.unique(states)
        features['n_unique_states'][i] = len(unique_states)

        # State entropy (measure of randomness in state visits)
        state_counts = np.bincount(states.astype(int), minlength=n_states)
        state_probs = state_counts / T
        state_probs = state_probs[state_probs > 0]  # Remove zeros for log
        features['state_entropy'][i] = -np.sum(state_probs * np.log2(state_probs))

        # Dominant state occupancy
        features['dominant_state_occupancy'][i] = np.max(state_counts) / T

        # Switching entropy (entropy of transition types)
        if transitions > 0:
            trans_pairs = [(int(states[t]), int(states[t+1]))
                          for t in range(T-1) if states[t] != states[t+1]]
            trans_counts = Counter(trans_pairs)
            trans_probs = np.array(list(trans_counts.values())) / len(trans_pairs)
            features['switching_entropy'][i] = -np.sum(trans_probs * np.log2(trans_probs))
        else:
            features['switching_entropy'][i] = 0

    return features


def correlate_with_anxiety(features: Dict[str, np.ndarray],
                           anxiety_scores: np.ndarray,
                           method: str = 'pearson') -> Dict[str, Dict[str, float]]:
    """
    Correlate BSDS features with anxiety scores (DASS-21).

    Tests H1: Hypersegmentation Hypothesis
    - Prediction: Higher anxiety -> More transitions, shorter dwell times

    Args:
        features: Dictionary of feature arrays from compute_individual_features
        anxiety_scores: DASS-21 anxiety subscale scores (n_subjects,)
        method: Correlation method ('pearson' or 'spearman')

    Returns:
        Dictionary of correlation results per feature
    """
    from scipy import stats

    results = {}

    for feature_name, feature_values in features.items():
        # Remove NaN values
        valid_mask = ~(np.isnan(feature_values) | np.isnan(anxiety_scores))
        x = feature_values[valid_mask]
        y = anxiety_scores[valid_mask]

        if len(x) < 3:
            results[feature_name] = {'r': np.nan, 'p': np.nan, 'n': len(x)}
            continue

        if method == 'pearson':
            r, p = stats.pearsonr(x, y)
        elif method == 'spearman':
            r, p = stats.spearmanr(x, y)
        else:
            raise ValueError(f"Unknown method: {method}")

        results[feature_name] = {
            'r': r,
            'p': p,
            'n': len(x),
            'significant': p < 0.05,
            'direction': 'positive' if r > 0 else 'negative'
        }

    return results


def test_hypersegmentation_hypothesis(states_list: List[np.ndarray],
                                       anxiety_scores: np.ndarray,
                                       n_states: int,
                                       TR: float = 1.0) -> Dict[str, Any]:
    """
    Comprehensive test of the Hypersegmentation Hypothesis (H1).

    H1: Individuals with higher trait anxiety show:
    - More frequent neural event boundaries (higher transition rate)
    - Shorter dwell times in each state
    - More variable state dynamics

    Args:
        states_list: List of state sequences per subject
        anxiety_scores: DASS-21 anxiety subscale scores
        n_states: Number of states
        TR: Repetition time

    Returns:
        Comprehensive hypothesis test results
    """
    # Compute features
    features = compute_individual_features(states_list, n_states, TR)

    # Test correlations
    correlations = correlate_with_anxiety(features, anxiety_scores)

    # Primary hypothesis tests
    h1_results = {
        'hypothesis': 'H1: Hypersegmentation',
        'prediction': 'Higher anxiety -> More neural event boundaries',
        'primary_measures': {
            'transition_rate': correlations.get('transition_rate', {}),
            'mean_dwell_time': correlations.get('mean_dwell_time', {}),
            'n_transitions': correlations.get('n_transitions', {}),
        },
        'secondary_measures': {
            'state_entropy': correlations.get('state_entropy', {}),
            'dwell_time_cv': correlations.get('dwell_time_cv', {}),
            'switching_entropy': correlations.get('switching_entropy', {}),
        },
        'all_correlations': correlations,
        'features': features,
        'n_subjects': len(states_list),
    }

    # Summary interpretation
    trans_r = correlations.get('transition_rate', {}).get('r', 0)
    trans_p = correlations.get('transition_rate', {}).get('p', 1)

    if trans_p < 0.05 and trans_r > 0:
        h1_results['interpretation'] = (
            f"SUPPORTED: Significant positive correlation between anxiety and "
            f"transition rate (r={trans_r:.3f}, p={trans_p:.3f}). "
            f"Higher anxiety is associated with more frequent neural state transitions."
        )
    elif trans_p < 0.05 and trans_r < 0:
        h1_results['interpretation'] = (
            f"CONTRADICTED: Significant negative correlation (r={trans_r:.3f}, p={trans_p:.3f}). "
            f"Higher anxiety is associated with FEWER transitions."
        )
    else:
        h1_results['interpretation'] = (
            f"NOT SUPPORTED: No significant correlation (r={trans_r:.3f}, p={trans_p:.3f}). "
            f"Anxiety level does not predict transition frequency in this sample."
        )

    return h1_results


# =============================================================================
# Annotation Proxy Analysis (Event Boundary Alignment)
# =============================================================================

def compute_boundary_alignment(state_timestamps: np.ndarray,
                                annotation_timestamps: np.ndarray,
                                tolerance: float = 2.0) -> Dict[str, Any]:
    """
    Compute alignment between neural state transitions and annotation proxies.

    Since direct event boundary annotations are not available in Emo-FilM,
    this function uses annotation proxies (novelty, surprise ratings) to
    assess whether neural state transitions align with perceptual events.

    Args:
        state_timestamps: Neural state transition times (seconds)
        annotation_timestamps: Annotation proxy times (novelty/surprise peaks)
        tolerance: Time window for considering alignment (seconds)

    Returns:
        Dictionary with alignment statistics
    """
    if len(state_timestamps) == 0 or len(annotation_timestamps) == 0:
        return {
            'hit_rate': 0.0,
            'false_alarm_rate': 1.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'mean_alignment_error': np.nan,
            'n_neural_transitions': len(state_timestamps),
            'n_annotation_events': len(annotation_timestamps)
        }

    # For each neural transition, find closest annotation event
    hits = 0
    alignment_errors = []

    for neural_t in state_timestamps:
        distances = np.abs(annotation_timestamps - neural_t)
        min_dist = np.min(distances)
        if min_dist <= tolerance:
            hits += 1
            alignment_errors.append(min_dist)

    # For each annotation event, check if there's a neural transition nearby
    annotation_hits = 0
    for annot_t in annotation_timestamps:
        distances = np.abs(state_timestamps - annot_t)
        if np.min(distances) <= tolerance:
            annotation_hits += 1

    # Compute metrics
    precision = hits / len(state_timestamps) if len(state_timestamps) > 0 else 0
    recall = annotation_hits / len(annotation_timestamps) if len(annotation_timestamps) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return {
        'hit_rate': precision,
        'precision': precision,  # Neural transitions that align with annotations
        'recall': recall,        # Annotations that have nearby neural transitions
        'f1_score': f1,
        'mean_alignment_error': np.mean(alignment_errors) if alignment_errors else np.nan,
        'std_alignment_error': np.std(alignment_errors) if alignment_errors else np.nan,
        'n_neural_transitions': len(state_timestamps),
        'n_annotation_events': len(annotation_timestamps),
        'n_aligned': hits,
        'tolerance_sec': tolerance
    }


def load_emofilm_annotations(annotation_dir: str,
                              task: str,
                              annotation_type: str = 'novelty') -> Dict[str, np.ndarray]:
    """
    Load Emo-FilM annotation data as event boundary proxies.

    Available annotation types (at 1Hz):
    - 'novelty': Novelty ratings (proxy for event boundaries)
    - 'surprise': Surprise ratings (proxy for event boundaries)
    - 'valence': Emotional valence
    - 'arousal': Emotional arousal

    Args:
        annotation_dir: Directory containing annotation files
        task: Movie task name (BigBuckBunny, FirstBite, etc.)
        annotation_type: Type of annotation to load

    Returns:
        Dictionary with annotation data per subject
    """
    from pathlib import Path
    import json

    annot_path = Path(annotation_dir)
    annotations = {}

    # Try to find annotation files
    # Emo-FilM stores annotations in derivatives/annotations/
    for annot_file in annot_path.glob(f"*{task}*{annotation_type}*.json"):
        try:
            with open(annot_file, 'r') as f:
                data = json.load(f)
            subject_id = annot_file.stem.split('_')[0]
            annotations[subject_id] = np.array(data.get('values', []))
        except Exception as e:
            print(f"Warning: Could not load {annot_file}: {e}")

    # Also try TSV format
    for annot_file in annot_path.glob(f"*{task}*{annotation_type}*.tsv"):
        try:
            data = np.loadtxt(annot_file, delimiter='\t', skiprows=1)
            subject_id = annot_file.stem.split('_')[0]
            annotations[subject_id] = data
        except Exception as e:
            print(f"Warning: Could not load {annot_file}: {e}")

    return annotations


def find_annotation_peaks(annotation_timeseries: np.ndarray,
                           sampling_rate: float = 1.0,
                           threshold_percentile: float = 75) -> np.ndarray:
    """
    Find peaks in annotation timeseries as event boundary proxies.

    Args:
        annotation_timeseries: Annotation values over time
        sampling_rate: Sampling rate in Hz (Emo-FilM uses 1Hz)
        threshold_percentile: Percentile threshold for peak detection

    Returns:
        Array of peak timestamps in seconds
    """
    from scipy.signal import find_peaks

    # Normalize
    norm_signal = (annotation_timeseries - np.mean(annotation_timeseries)) / (np.std(annotation_timeseries) + 1e-10)

    # Find peaks above threshold
    threshold = np.percentile(norm_signal, threshold_percentile)
    peaks, _ = find_peaks(norm_signal, height=threshold, distance=3)

    # Convert to timestamps
    timestamps = peaks / sampling_rate

    return timestamps


def analyze_boundary_alignment_group(states_list: List[np.ndarray],
                                      annotations_dict: Dict[str, np.ndarray],
                                      subject_ids: List[str],
                                      TR: float = 1.0,
                                      annotation_sampling_rate: float = 1.0,
                                      tolerance: float = 2.0) -> Dict[str, Any]:
    """
    Analyze neural-annotation alignment across subjects.

    Args:
        states_list: List of state sequences
        annotations_dict: Dictionary mapping subject_id to annotation timeseries
        subject_ids: List of subject identifiers
        TR: fMRI repetition time
        annotation_sampling_rate: Annotation sampling rate (Hz)
        tolerance: Alignment tolerance (seconds)

    Returns:
        Group-level alignment results
    """
    subject_results = []

    for i, (states, subj_id) in enumerate(zip(states_list, subject_ids)):
        # Extract neural transition timestamps
        trans_info = extract_transition_timestamps(states, TR)
        neural_timestamps = trans_info['timestamps']

        # Get annotation data if available
        if subj_id in annotations_dict:
            annotation_ts = annotations_dict[subj_id]
            annot_peaks = find_annotation_peaks(annotation_ts, annotation_sampling_rate)

            # Compute alignment
            alignment = compute_boundary_alignment(neural_timestamps, annot_peaks, tolerance)
            alignment['subject_id'] = subj_id
            subject_results.append(alignment)

    if not subject_results:
        return {
            'warning': 'No matching annotations found for subjects',
            'n_subjects_analyzed': 0
        }

    # Aggregate results
    f1_scores = [r['f1_score'] for r in subject_results]
    precisions = [r['precision'] for r in subject_results]
    recalls = [r['recall'] for r in subject_results]

    return {
        'mean_f1': np.mean(f1_scores),
        'std_f1': np.std(f1_scores),
        'mean_precision': np.mean(precisions),
        'mean_recall': np.mean(recalls),
        'n_subjects_analyzed': len(subject_results),
        'subject_results': subject_results,
        'interpretation': (
            f"Average F1 alignment score: {np.mean(f1_scores):.3f}. "
            f"Higher values indicate neural state transitions align with annotation proxies."
        )
    }
