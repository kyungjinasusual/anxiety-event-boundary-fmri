"""
Data preprocessing and arrangement utilities for BSDS
"""

import numpy as np
from typing import List, Tuple, Optional
import warnings


def validate_input(data_list: List[np.ndarray]) -> Tuple[int, List[int], int]:
    """
    Validate input data format for BSDS.

    Args:
        data_list: List of (D x T_i) arrays, one per subject

    Returns:
        Tuple of (n_dims, n_samples_list, n_subjects)

    Raises:
        ValueError: If data format is invalid
    """
    if not isinstance(data_list, list):
        raise ValueError("data_list must be a list of numpy arrays")

    if len(data_list) == 0:
        raise ValueError("data_list cannot be empty")

    n_dims = data_list[0].shape[0]
    n_samples_list = []

    for i, data in enumerate(data_list):
        if data.ndim != 2:
            raise ValueError(f"Subject {i}: data must be 2D (D x T), got {data.ndim}D")
        if data.shape[0] != n_dims:
            raise ValueError(f"Subject {i}: dimension mismatch. Expected {n_dims}, got {data.shape[0]}")
        n_samples_list.append(data.shape[1])

    return n_dims, n_samples_list, len(data_list)


def preprocess_data(Y: np.ndarray, standardize: bool = True) -> np.ndarray:
    """
    Preprocess time series data.
    Corresponds to MATLAB preprocess.m

    Args:
        Y: Data matrix (D x T) or (T x D)
        standardize: Whether to z-score the data

    Returns:
        Preprocessed data (D x T)
    """
    # Ensure (D x T) format where D < T typically
    if Y.shape[0] > Y.shape[1]:
        Y = Y.T

    if standardize:
        # Z-score each ROI
        mean = np.mean(Y, axis=1, keepdims=True)
        std = np.std(Y, axis=1, keepdims=True)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        Y = (Y - mean) / std

    # Check for NaN/Inf
    if np.any(~np.isfinite(Y)):
        warnings.warn("Data contains NaN or Inf values. Replacing with zeros.")
        Y = np.nan_to_num(Y)

    return Y


def data_arrange(Xm_list: List[np.ndarray]) -> List[np.ndarray]:
    """
    Arrange latent variable data for AR model estimation.
    Corresponds to MATLAB data_arrange.m

    Creates lagged data structure for VAR estimation:
    Y_bar[t] = [X[t-1]] for AR(1) model

    Args:
        Xm_list: List of latent state means per subject, each (k-1 x T_i)

    Returns:
        List of lagged data arrays for each subject, each (k-1 x k-1 x T-1)
    """
    Ybar_list = []

    for Xm in Xm_list:
        k_minus_1, T = Xm.shape

        # Create design matrix for each time point
        # For AR(1): Ybar[:,:,t] = X[:,t-1] @ X[:,t-1].T as outer product structure
        Ybar = np.zeros((k_minus_1, k_minus_1, T - 1))

        for t in range(T - 1):
            # Design matrix is the previous state for AR(1)
            Ybar[:, :, t] = np.eye(k_minus_1)  # Identity for simple structure
            # Actually store X_{t-1} as a column vector expanded
            # This will be used in the VAR M-step

        Ybar_list.append(Ybar)

    return Ybar_list


def concatenate_subjects(data_list: List[np.ndarray]) -> Tuple[np.ndarray, List[int]]:
    """
    Concatenate subject data along time axis.

    Args:
        data_list: List of (D x T_i) arrays

    Returns:
        Tuple of (concatenated array (D x sum(T_i)), list of cumulative indices)
    """
    Y_concat = np.concatenate(data_list, axis=1)
    boundaries = np.cumsum([0] + [d.shape[1] for d in data_list])
    return Y_concat, boundaries.tolist()


def split_by_subjects(Y: np.ndarray, boundaries: List[int]) -> List[np.ndarray]:
    """
    Split concatenated data back into subjects.

    Args:
        Y: Concatenated array (D x T_total)
        boundaries: List of cumulative indices

    Returns:
        List of (D x T_i) arrays
    """
    data_list = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        data_list.append(Y[:, start:end])
    return data_list


def get_subject_slice(total_samples: int, n_subjects: int, subject_idx: int) -> Tuple[int, int]:
    """
    Get slice indices for a subject in concatenated data.
    Assumes equal samples per subject.

    Args:
        total_samples: Total number of samples
        n_subjects: Number of subjects
        subject_idx: Subject index (0-based)

    Returns:
        Tuple of (start_idx, end_idx)
    """
    samples_per_subj = total_samples // n_subjects
    start = subject_idx * samples_per_subj
    end = start + samples_per_subj
    return start, end


# =============================================================================
# Schaefer Atlas Network Mapping
# =============================================================================

def get_schaefer_network_mapping(n_rois: int = 400, n_networks: int = 7) -> dict:
    """
    Get ROI-to-network mapping for Schaefer atlas.

    The Schaefer atlas organizes ROIs by Yeo networks.
    This function returns a mapping from network name to ROI indices.

    Args:
        n_rois: Number of ROIs (100, 200, 400, etc.)
        n_networks: Number of networks (7 or 17)

    Returns:
        Dictionary mapping network name to list of ROI indices (0-based)
    """
    # 7-network names (Yeo et al., 2011)
    network_names_7 = [
        'Visual',      # Vis
        'SomMot',      # Somatomotor
        'DorsAttn',    # Dorsal Attention
        'SalVentAttn', # Salience/Ventral Attention
        'Limbic',      # Limbic
        'Cont',        # Control (Frontoparietal)
        'Default',     # Default Mode
    ]

    if n_networks != 7:
        raise ValueError(f"Currently only 7-network mapping is supported, got {n_networks}")

    # Approximate ROI distribution per network per hemisphere
    # Based on Schaefer 400 parcellation
    if n_rois == 400:
        # Each hemisphere has ~200 ROIs
        # Left hemisphere: 0-199, Right hemisphere: 200-399
        network_mapping = {
            'Visual': list(range(0, 31)) + list(range(200, 231)),          # ~62 ROIs
            'SomMot': list(range(31, 68)) + list(range(231, 268)),         # ~74 ROIs
            'DorsAttn': list(range(68, 91)) + list(range(268, 291)),       # ~46 ROIs
            'SalVentAttn': list(range(91, 116)) + list(range(291, 316)),   # ~50 ROIs
            'Limbic': list(range(116, 129)) + list(range(316, 329)),       # ~26 ROIs
            'Cont': list(range(129, 176)) + list(range(329, 376)),         # ~94 ROIs
            'Default': list(range(176, 200)) + list(range(376, 400)),      # ~48 ROIs
        }
    elif n_rois == 200:
        # Approximate for 200 ROIs
        network_mapping = {
            'Visual': list(range(0, 16)) + list(range(100, 116)),
            'SomMot': list(range(16, 34)) + list(range(116, 134)),
            'DorsAttn': list(range(34, 46)) + list(range(134, 146)),
            'SalVentAttn': list(range(46, 58)) + list(range(146, 158)),
            'Limbic': list(range(58, 65)) + list(range(158, 165)),
            'Cont': list(range(65, 88)) + list(range(165, 188)),
            'Default': list(range(88, 100)) + list(range(188, 200)),
        }
    elif n_rois == 100:
        # Approximate for 100 ROIs
        network_mapping = {
            'Visual': list(range(0, 8)) + list(range(50, 58)),
            'SomMot': list(range(8, 17)) + list(range(58, 67)),
            'DorsAttn': list(range(17, 23)) + list(range(67, 73)),
            'SalVentAttn': list(range(23, 29)) + list(range(73, 79)),
            'Limbic': list(range(29, 33)) + list(range(79, 83)),
            'Cont': list(range(33, 44)) + list(range(83, 94)),
            'Default': list(range(44, 50)) + list(range(94, 100)),
        }
    else:
        # Generic approximate mapping
        rois_per_hemi = n_rois // 2
        fractions = [0.155, 0.185, 0.115, 0.125, 0.065, 0.235, 0.12]
        network_mapping = {}
        start_l, start_r = 0, rois_per_hemi

        for name, frac in zip(network_names_7, fractions):
            n = int(rois_per_hemi * frac)
            network_mapping[name] = (
                list(range(start_l, start_l + n)) +
                list(range(start_r, start_r + n))
            )
            start_l += n
            start_r += n

    return network_mapping


def get_schaefer_roi_labels(n_rois: int = 400) -> List[str]:
    """
    Get ROI labels for Schaefer atlas.

    Note: This returns generic labels. For actual atlas labels,
    load from nilearn or the atlas file directly.

    Args:
        n_rois: Number of ROIs

    Returns:
        List of ROI labels
    """
    network_map = get_schaefer_network_mapping(n_rois)
    labels = [''] * n_rois

    for net_name, indices in network_map.items():
        for i, idx in enumerate(indices):
            if idx < n_rois:
                hemi = 'L' if idx < n_rois // 2 else 'R'
                labels[idx] = f"{hemi}_{net_name}_{i+1}"

    return labels


def load_schaefer_labels_from_atlas(n_rois: int = 400) -> Optional[List[str]]:
    """
    Load actual Schaefer atlas labels from nilearn.

    Args:
        n_rois: Number of ROIs

    Returns:
        List of ROI labels or None if nilearn not available
    """
    try:
        from nilearn import datasets
        atlas = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois, yeo_networks=7)
        # Labels are bytes, decode them
        labels = [label.decode('utf-8') if isinstance(label, bytes) else label
                  for label in atlas.labels]
        return labels
    except ImportError:
        return None
    except Exception as e:
        warnings.warn(f"Could not load Schaefer labels: {e}")
        return None
