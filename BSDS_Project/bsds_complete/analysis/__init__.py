"""Analysis module for BSDS results"""

from .statistics import (
    # Basic statistics
    compute_occupancy_group,
    compute_occupancy_subject,
    compute_mean_lifetime_group,
    compute_mean_lifetime_subject,
    get_dominant_states_group,
    get_dominant_states_subject,
    compute_state_covariance,
    compute_state_mean,
    compute_transition_statistics,
    compute_summary_statistics,
    # Transition timestamps
    extract_transition_timestamps,
    extract_transitions_group,
    # Individual difference analysis
    compute_individual_features,
    correlate_with_anxiety,
    test_hypersegmentation_hypothesis,
    # Annotation proxy analysis
    compute_boundary_alignment,
    load_emofilm_annotations,
    find_annotation_peaks,
    analyze_boundary_alignment_group,
)
from .visualization import (
    # Core visualizations
    plot_state_sequence,
    plot_state_sequence_multi,
    plot_transition_matrix,
    plot_occupancy,
    plot_mean_lifetime,
    plot_convergence,
    # Individual difference visualizations
    plot_individual_occupancy,
    plot_anxiety_correlation,
    # Network visualization
    plot_state_network_profile,
    # Report generation
    create_summary_report,
    # Styling utilities
    get_state_colors,
    setup_poster_style,
    PASTEL_COLORS,
    NETWORK_COLORS,
)

__all__ = [
    # Statistics - Basic
    "compute_occupancy_group", "compute_occupancy_subject",
    "compute_mean_lifetime_group", "compute_mean_lifetime_subject",
    "get_dominant_states_group", "get_dominant_states_subject",
    "compute_state_covariance", "compute_state_mean",
    "compute_transition_statistics", "compute_summary_statistics",
    # Statistics - Transitions
    "extract_transition_timestamps", "extract_transitions_group",
    # Statistics - Individual differences
    "compute_individual_features", "correlate_with_anxiety",
    "test_hypersegmentation_hypothesis",
    # Statistics - Annotation proxy
    "compute_boundary_alignment", "load_emofilm_annotations",
    "find_annotation_peaks", "analyze_boundary_alignment_group",
    # Visualization - Core
    "plot_state_sequence", "plot_state_sequence_multi",
    "plot_transition_matrix", "plot_occupancy", "plot_mean_lifetime",
    "plot_convergence",
    # Visualization - Individual differences
    "plot_individual_occupancy", "plot_anxiety_correlation",
    # Visualization - Network
    "plot_state_network_profile",
    # Visualization - Report
    "create_summary_report",
    # Styling
    "get_state_colors", "setup_poster_style",
    "PASTEL_COLORS", "NETWORK_COLORS",
]
