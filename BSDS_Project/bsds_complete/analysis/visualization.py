"""
Visualization functions for BSDS results

Modern, poster-ready visualizations with pastel color schemes.
Designed for academic presentations and publications.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
from pathlib import Path


# =============================================================================
# Color Palettes and Styling
# =============================================================================

# Pastel color palette for states (poster-friendly)
PASTEL_COLORS = [
    '#A8D8EA',  # Light blue
    '#FFB5B5',  # Light coral
    '#B5EAD7',  # Light mint
    '#FFEAA7',  # Light yellow
    '#DDA0DD',  # Plum
    '#98D8C8',  # Light teal
    '#F7DC6F',  # Light gold
    '#BB8FCE',  # Light purple
    '#85C1E9',  # Sky blue
    '#F8B500',  # Amber
]

# Network colors for brain visualization
NETWORK_COLORS = {
    'Visual': '#1f77b4',
    'SomMot': '#ff7f0e',
    'DorsAttn': '#2ca02c',
    'SalVentAttn': '#d62728',
    'Limbic': '#9467bd',
    'Cont': '#8c564b',
    'Default': '#e377c2',
}


def get_state_colors(n_states: int) -> List[str]:
    """Get color palette for states."""
    if n_states <= len(PASTEL_COLORS):
        return PASTEL_COLORS[:n_states]
    else:
        # Generate additional colors if needed
        import colorsys
        colors = PASTEL_COLORS.copy()
        for i in range(n_states - len(PASTEL_COLORS)):
            hue = (i * 0.618033988749895) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.4, 0.9)
            colors.append('#{:02x}{:02x}{:02x}'.format(
                int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)))
        return colors


def setup_poster_style():
    """Configure matplotlib for poster-ready figures."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Set seaborn style with pastel palette
        sns.set_style("whitegrid")
        sns.set_context("talk", font_scale=1.1)

        # Custom RC params for clean poster figures
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.edgecolor': '#333333',
            'axes.linewidth': 1.5,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.color': '#cccccc',
            'font.family': 'sans-serif',
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
        })
        return True
    except ImportError:
        return False


# =============================================================================
# Core Visualization Functions
# =============================================================================

def plot_state_sequence(states: np.ndarray,
                        ax=None,
                        TR: float = 1.0,
                        title: str = "State Sequence",
                        subject_id: str = None,
                        show_transitions: bool = True,
                        save_path: Optional[str] = None):
    """
    Plot state sequence over time (Gantt-style).

    Args:
        states: State sequence (T,)
        ax: Matplotlib axes (optional)
        TR: Repetition time for x-axis scaling
        title: Plot title
        subject_id: Subject identifier for title
        show_transitions: Mark transition points
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib axes object
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available. Skipping plot.")
        return None

    setup_poster_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 3))

    T = len(states)
    time = np.arange(T) * TR
    n_states = int(np.max(states)) + 1
    colors = get_state_colors(n_states)

    # Create Gantt-style visualization
    for t in range(T):
        state = int(states[t])
        ax.axvspan(time[t], time[t] + TR, ymin=0, ymax=1,
                   color=colors[state], alpha=0.8, linewidth=0)

    # Mark transitions
    if show_transitions:
        transitions = np.where(np.diff(states) != 0)[0]
        for t_idx in transitions:
            ax.axvline(x=time[t_idx + 1], color='black',
                       linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xlim(0, time[-1] + TR)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Time (s)', fontweight='bold')
    ax.set_ylabel('')
    ax.set_yticks([])

    if subject_id:
        title = f"{title} - {subject_id}"
    ax.set_title(title, fontweight='bold')

    # Legend
    patches = [mpatches.Patch(color=colors[i], label=f'State {i}')
               for i in range(n_states)]
    ax.legend(handles=patches, loc='upper right', ncol=min(n_states, 5),
              framealpha=0.9)

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return ax


def plot_state_sequence_multi(states_list: List[np.ndarray],
                               subject_ids: List[str] = None,
                               TR: float = 1.0,
                               title: str = "Group State Sequences",
                               save_path: Optional[str] = None):
    """
    Plot multiple state sequences stacked vertically.

    Args:
        states_list: List of state sequences
        subject_ids: List of subject identifiers
        TR: Repetition time
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available. Skipping plot.")
        return None

    setup_poster_style()

    n_subjects = len(states_list)
    fig, axes = plt.subplots(n_subjects, 1, figsize=(14, 2 * n_subjects),
                             sharex=True)

    if n_subjects == 1:
        axes = [axes]

    if subject_ids is None:
        subject_ids = [f"Subject {i+1}" for i in range(n_subjects)]

    n_states = max(int(np.max(s)) + 1 for s in states_list)
    colors = get_state_colors(n_states)

    for i, (states, ax) in enumerate(zip(states_list, axes)):
        T = len(states)
        time = np.arange(T) * TR

        for t in range(T):
            state = int(states[t])
            ax.axvspan(time[t], time[t] + TR, ymin=0, ymax=1,
                       color=colors[state], alpha=0.8, linewidth=0)

        ax.set_xlim(0, time[-1] + TR)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(subject_ids[i], rotation=0, ha='right', va='center',
                      fontsize=9)

        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)

    axes[-1].set_xlabel('Time (s)', fontweight='bold')

    # Common legend
    patches = [mpatches.Patch(color=colors[i], label=f'State {i}')
               for i in range(n_states)]
    fig.legend(handles=patches, loc='upper center', ncol=min(n_states, 8),
               bbox_to_anchor=(0.5, 1.02), framealpha=0.9)

    fig.suptitle(title, fontweight='bold', y=1.05)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_transition_matrix(trans_prob: np.ndarray,
                           ax=None,
                           title: str = "Transition Probability Matrix",
                           annotate: bool = True,
                           save_path: Optional[str] = None):
    """
    Plot transition probability matrix as heatmap (modern style).

    Args:
        trans_prob: Transition probability matrix (K x K)
        ax: Matplotlib axes (optional)
        title: Plot title
        annotate: Show probability values
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib axes object
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available. Skipping plot.")
        return None

    setup_poster_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))

    K = trans_prob.shape[0]

    # Use seaborn heatmap for clean visualization
    sns.heatmap(trans_prob, ax=ax, cmap='Blues', vmin=0, vmax=1,
                annot=annotate, fmt='.2f', annot_kws={'size': 10},
                square=True, linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Probability', 'shrink': 0.8})

    ax.set_xlabel('To State', fontweight='bold')
    ax.set_ylabel('From State', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=15)

    # State labels
    ax.set_xticklabels([f'S{i}' for i in range(K)])
    ax.set_yticklabels([f'S{i}' for i in range(K)])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return ax


def plot_occupancy(occupancy: np.ndarray,
                   ax=None,
                   title: str = "State Occupancy",
                   show_values: bool = True,
                   save_path: Optional[str] = None):
    """
    Plot state occupancy as bar chart (modern pastel style).

    Args:
        occupancy: Fractional occupancy per state (K,)
        ax: Matplotlib axes (optional)
        title: Plot title
        show_values: Display percentage values on bars
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib axes object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Skipping plot.")
        return None

    setup_poster_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    K = len(occupancy)
    colors = get_state_colors(K)

    bars = ax.bar(range(K), occupancy * 100, color=colors, edgecolor='white',
                  linewidth=2, alpha=0.9)

    ax.set_xlabel('State', fontweight='bold')
    ax.set_ylabel('Occupancy (%)', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xticks(range(K))
    ax.set_xticklabels([f'S{i}' for i in range(K)])

    # Add value labels
    if show_values:
        for bar, val in zip(bars, occupancy):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                    f'{val*100:.1f}%', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    ax.set_ylim(0, max(occupancy) * 100 * 1.2)

    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return ax


def plot_mean_lifetime(lifetime: np.ndarray,
                       ax=None,
                       title: str = "Mean Dwell Time",
                       save_path: Optional[str] = None):
    """
    Plot mean lifetime/dwell time per state.

    Args:
        lifetime: Mean lifetime per state in seconds (K,)
        ax: Matplotlib axes
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Skipping plot.")
        return None

    setup_poster_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    K = len(lifetime)
    colors = get_state_colors(K)

    bars = ax.bar(range(K), lifetime, color=colors, edgecolor='white',
                  linewidth=2, alpha=0.9)

    ax.set_xlabel('State', fontweight='bold')
    ax.set_ylabel('Mean Dwell Time (s)', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xticks(range(K))
    ax.set_xticklabels([f'S{i}' for i in range(K)])

    # Value labels
    for bar, val in zip(bars, lifetime):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                f'{val:.1f}s', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    ax.set_ylim(0, max(lifetime) * 1.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return ax


def plot_convergence(log_lik_history: List[float],
                     ax=None,
                     title: str = "Model Convergence",
                     save_path: Optional[str] = None):
    """
    Plot log-likelihood convergence over iterations.

    Args:
        log_lik_history: List of log-likelihood values
        ax: Matplotlib axes (optional)
        title: Plot title
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib axes object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Skipping plot.")
        return None

    setup_poster_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    iterations = range(1, len(log_lik_history) + 1)

    ax.plot(iterations, log_lik_history, 'o-', color='#3498db',
            linewidth=2, markersize=4, markerfacecolor='white',
            markeredgewidth=2)

    ax.fill_between(iterations, log_lik_history, alpha=0.2, color='#3498db')

    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel('Log-Likelihood', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add convergence annotation
    final_ll = log_lik_history[-1]
    ax.annotate(f'Final: {final_ll:.1f}',
                xy=(len(log_lik_history), final_ll),
                xytext=(len(log_lik_history) * 0.7, final_ll),
                fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='gray'))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return ax


# =============================================================================
# Individual Difference Visualizations
# =============================================================================

def plot_individual_occupancy(occupancy_subject: np.ndarray,
                               subject_ids: List[str] = None,
                               trait_values: np.ndarray = None,
                               trait_name: str = "Anxiety",
                               save_path: Optional[str] = None):
    """
    Plot individual occupancy patterns with optional trait coloring.

    Args:
        occupancy_subject: Subject-wise occupancy (n_subjects x K)
        subject_ids: Subject identifiers
        trait_values: Trait scores for coloring (e.g., DASS anxiety)
        trait_name: Name of the trait for legend
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available. Skipping plot.")
        return None

    setup_poster_style()

    n_subjects, K = occupancy_subject.shape

    if subject_ids is None:
        subject_ids = [f"S{i+1}" for i in range(n_subjects)]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Heatmap of individual occupancies
    if trait_values is not None:
        # Sort by trait values
        sort_idx = np.argsort(trait_values)
        occupancy_sorted = occupancy_subject[sort_idx]
        labels_sorted = [subject_ids[i] for i in sort_idx]
    else:
        occupancy_sorted = occupancy_subject
        labels_sorted = subject_ids

    sns.heatmap(occupancy_sorted * 100, ax=ax, cmap='YlOrRd',
                annot=False, cbar_kws={'label': 'Occupancy (%)'})

    ax.set_xlabel('State', fontweight='bold')
    ax.set_ylabel('Subject', fontweight='bold')
    ax.set_xticklabels([f'S{i}' for i in range(K)])
    ax.set_yticklabels(labels_sorted, fontsize=8)

    if trait_values is not None:
        ax.set_title(f'Individual State Occupancy (sorted by {trait_name})',
                     fontweight='bold', pad=15)
    else:
        ax.set_title('Individual State Occupancy', fontweight='bold', pad=15)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_anxiety_correlation(feature_values: np.ndarray,
                              anxiety_scores: np.ndarray,
                              feature_name: str = "Transition Rate",
                              ax=None,
                              save_path: Optional[str] = None):
    """
    Plot correlation between a BSDS feature and anxiety scores.

    Args:
        feature_values: Feature values per subject
        anxiety_scores: DASS-21 anxiety subscale scores
        feature_name: Name of the feature
        ax: Matplotlib axes
        save_path: Path to save figure

    Returns:
        Matplotlib axes and correlation statistics
    """
    try:
        import matplotlib.pyplot as plt
        from scipy import stats
    except ImportError:
        print("matplotlib/scipy not available. Skipping plot.")
        return None

    setup_poster_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Calculate correlation
    r, p = stats.pearsonr(feature_values, anxiety_scores)

    # Scatter plot
    ax.scatter(anxiety_scores, feature_values, s=80, alpha=0.7,
               c='#3498db', edgecolor='white', linewidth=1)

    # Regression line
    z = np.polyfit(anxiety_scores, feature_values, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(min(anxiety_scores), max(anxiety_scores), 100)
    ax.plot(x_line, p_line(x_line), 'r--', linewidth=2, alpha=0.8)

    ax.set_xlabel('DASS-21 Anxiety Score', fontweight='bold')
    ax.set_ylabel(feature_name, fontweight='bold')
    ax.set_title(f'{feature_name} vs Anxiety\n(r={r:.3f}, p={p:.3f})',
                 fontweight='bold', pad=15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return ax, {'r': r, 'p': p}


# =============================================================================
# Network-Level Visualization
# =============================================================================

def plot_state_network_profile(network_activity: Dict[str, float],
                                state_id: int,
                                ax=None,
                                save_path: Optional[str] = None):
    """
    Plot network-level activity profile for a state (radar chart style).

    Args:
        network_activity: Dict mapping network name to activity level
        state_id: State index
        ax: Matplotlib axes
        save_path: Path to save figure

    Returns:
        Matplotlib axes
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available. Skipping plot.")
        return None

    setup_poster_style()

    networks = list(network_activity.keys())
    values = list(network_activity.values())
    N = len(networks)

    # Radar chart
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values = values + [values[0]]  # Close the polygon
    angles = angles + [angles[0]]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.fill(angles, values, color=PASTEL_COLORS[state_id % len(PASTEL_COLORS)],
            alpha=0.5)
    ax.plot(angles, values, 'o-', linewidth=2,
            color=PASTEL_COLORS[state_id % len(PASTEL_COLORS)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(networks, fontsize=10)
    ax.set_title(f'State {state_id} Network Profile', fontweight='bold', pad=20)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return ax


# =============================================================================
# Comprehensive Report Generation
# =============================================================================

def create_summary_report(results: Dict[str, Any],
                          output_dir: str,
                          prefix: str = "bsds") -> str:
    """
    Create comprehensive summary report with modern figures and text.

    Args:
        results: Dictionary containing all BSDS results
        output_dir: Output directory for report
        prefix: Prefix for output files

    Returns:
        Path to the main report file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate plots if matplotlib available
    try:
        import matplotlib.pyplot as plt

        setup_poster_style()

        # Create figure with multiple subplots (2x2 layout)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Occupancy
        if 'occupancy_group' in results:
            plot_occupancy(results['occupancy_group'], ax=axes[0, 0],
                          title="Group-Level State Occupancy")

        # Plot 2: Transition matrix
        if 'transition_prob' in results:
            plot_transition_matrix(results['transition_prob'], ax=axes[0, 1],
                                  title="Transition Probability Matrix")

        # Plot 3: Convergence
        if 'log_lik_history' in results and len(results['log_lik_history']) > 0:
            plot_convergence(results['log_lik_history'], ax=axes[1, 0],
                           title="Model Convergence")

        # Plot 4: Mean lifetime
        if 'mean_lifetime_group' in results:
            plot_mean_lifetime(results['mean_lifetime_group'], ax=axes[1, 1],
                              title='Mean Dwell Time per State')

        plt.tight_layout()
        fig_path = output_path / f"{prefix}_summary.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    except ImportError:
        fig_path = None

    # Create text report
    report_lines = [
        "=" * 70,
        "BSDS Analysis Summary Report",
        "=" * 70,
        "",
        f"Number of subjects: {results.get('n_subjects', 'N/A')}",
        f"Number of states (K_max): {results.get('n_states', 'N/A')}",
        f"Effective states: {results.get('effective_n_states', 'N/A')}",
        f"Active states: {results.get('active_states', 'N/A')}",
        f"TR (seconds): {results.get('TR', 'N/A')}",
        "",
        "-" * 50,
        "State Occupancy (Group-Level)",
        "-" * 50,
    ]

    if 'occupancy_group' in results:
        for i, occ in enumerate(results['occupancy_group']):
            marker = "★" if occ >= 0.1 else " "
            report_lines.append(f"  {marker} State {i}: {occ*100:.2f}%")

    report_lines.extend([
        "",
        "-" * 50,
        "Mean Lifetime (Group-Level)",
        "-" * 50,
    ])

    if 'mean_lifetime_group' in results:
        for i, lt in enumerate(results['mean_lifetime_group']):
            report_lines.append(f"  State {i}: {lt:.2f} seconds")

    report_lines.extend([
        "",
        "-" * 50,
        "Dominant States",
        "-" * 50,
    ])

    if 'dominant_states_group' in results:
        report_lines.append(f"  {results['dominant_states_group']}")

    report_lines.extend([
        "",
        "-" * 50,
        "Transition Statistics",
        "-" * 50,
    ])

    if 'transition_stats' in results:
        ts = results['transition_stats']
        report_lines.append(f"  Mean transitions per subject: {ts.get('mean_transitions', 'N/A'):.1f}")
        if 'persistence' in ts:
            report_lines.append("  State persistence (self-transition):")
            for i, p in enumerate(ts['persistence']):
                report_lines.append(f"    State {i}: {p*100:.1f}%")

    if 'log_lik_history' in results and len(results['log_lik_history']) > 0:
        report_lines.extend([
            "",
            "-" * 50,
            "Convergence",
            "-" * 50,
            f"  Final log-likelihood: {results['log_lik_history'][-1]:.2f}",
            f"  Number of iterations: {len(results['log_lik_history'])}",
        ])

    # Add ARD pruning note
    if 'ard_relevance' in results:
        report_lines.extend([
            "",
            "-" * 50,
            "ARD Relevance (per state)",
            "-" * 50,
        ])
        for i, rel in enumerate(results['ard_relevance']):
            report_lines.append(f"  State {i}: {rel:.4f}")
        report_lines.extend([
            "",
            "NOTE: --n-states is K_max (upper bound). Effective states are determined",
            "by occupancy-based pruning. States with <1% occupancy are inactive.",
        ])

    report_lines.extend([
        "",
        "=" * 70,
        "End of Report",
        "=" * 70,
    ])

    # Write text report
    report_path = output_path / f"{prefix}_report.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    # Save results as JSON (numpy arrays converted to lists)
    json_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            json_results[key] = value.tolist()
        elif isinstance(value, dict):
            json_results[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v
                                 for k, v in value.items()}
        else:
            json_results[key] = value

    json_path = output_path / f"{prefix}_results.json"
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"Report saved to: {report_path}")
    if fig_path:
        print(f"Figure saved to: {fig_path}")
    print(f"JSON results saved to: {json_path}")

    return str(report_path)
