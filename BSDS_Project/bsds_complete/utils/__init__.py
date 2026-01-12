"""Utility functions for BSDS"""

from .math_utils import (
    logsumexp,
    normalise,
    kl_dirichlet,
    kl_gamma,
    safe_log,
    safe_cholesky
)
from .data_utils import (
    preprocess_data,
    data_arrange,
    validate_input,
    concatenate_subjects,
    split_by_subjects,
    # Schaefer atlas utilities
    get_schaefer_network_mapping,
    get_schaefer_roi_labels,
    load_schaefer_labels_from_atlas,
)
from .roi_selection import (
    get_event_segmentation_rois,
    create_event_roi_masker,
    extract_event_roi_timeseries,
    get_region_summary,
    EVENT_ROI_PRESETS,
    SCHAEFER400_EVENT_ROIS,
    SUBCORTICAL_EVENT_ROIS,
)

__all__ = [
    # Math utilities
    "logsumexp", "normalise", "kl_dirichlet", "kl_gamma",
    "safe_log", "safe_cholesky",
    # Data utilities
    "preprocess_data", "data_arrange", "validate_input",
    "concatenate_subjects", "split_by_subjects",
    # Schaefer atlas
    "get_schaefer_network_mapping", "get_schaefer_roi_labels",
    "load_schaefer_labels_from_atlas",
    # Event segmentation ROIs
    "get_event_segmentation_rois", "create_event_roi_masker",
    "extract_event_roi_timeseries", "get_region_summary",
    "EVENT_ROI_PRESETS", "SCHAEFER400_EVENT_ROIS", "SUBCORTICAL_EVENT_ROIS",
]
