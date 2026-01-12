"""
Complete BSDS Model Implementation
Bayesian Switching Dynamical Systems

Based on: Taghia & Cai (2018) Nature Communications
Full Python port with all missing functions implemented.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.cluster import KMeans
from scipy.special import digamma as psi

from .config import BSDSConfig
from ..utils.math_utils import EPS, safe_log, logsumexp
from ..utils.data_utils import validate_input, concatenate_subjects, preprocess_data
from ..inference.hmm import vbhmm_estep, viterbi_decode_all
from ..inference.latent import infer_qx, compute_log_out_probs
from ..learning.factor_learning import infer_ql, infer_qnu, infer_psii, infer_mcl
from ..learning.transition_learning import infer_qtheta, update_transition_probs
from ..learning.ar_learning import infer_ar, set_ar_hyperpriors
from ..analysis.statistics import compute_summary_statistics


class BSDSModel:
    """
    Bayesian Switching Dynamical Systems Model.

    This model combines:
    - Hidden Markov Model (HMM) for discrete state switching
    - Factor Analysis (FA) for dimensionality reduction within each state
    - Autoregressive (AR) dynamics for temporal structure of latent variables

    Attributes:
        config: Model configuration
        Lm: Factor loading means per state
        Lcov: Factor loading covariances per state
        Xm: Latent variable means per state
        Xcov: Latent variable covariances per state
        psii: Observation noise precision
        stran: State transition probabilities
        sprior: Initial state probabilities
        Wa, Wpi: Dirichlet posterior parameters
        a, b: ARD parameters
        ar_post: AR model posteriors
        log_lik_history: Training log-likelihood history
        states_: Fitted state sequences
    """

    def __init__(self, config: Optional[BSDSConfig] = None, **kwargs):
        """
        Initialize BSDS model.

        Args:
            config: BSDSConfig object (optional)
            **kwargs: Override config parameters
        """
        if config is None:
            config = BSDSConfig(**kwargs)
        else:
            # Apply any overrides
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        self.config = config

        # Model parameters (to be fitted)
        self.Lm = None
        self.Lcov = None
        self.Xm = None
        self.Xcov = None
        self.psii = None
        self.stran = None
        self.sprior = None
        self.Wa = None
        self.Wpi = None
        self.a = None
        self.b = None
        self.mean_mcl = None
        self.nu_mcl = None
        self.ar_post = None

        # Training history
        self.log_lik_history = []
        self.states_ = None
        self.gamma_list_ = None

        # Data info
        self.n_dims_ = None
        self.n_subjects_ = None
        self.n_samples_list_ = None

        # Random state
        if config.random_seed is not None:
            np.random.seed(config.random_seed)

    def fit(self, data_list: List[np.ndarray]) -> 'BSDSModel':
        """
        Fit BSDS model to data.

        Args:
            data_list: List of (D x T_i) arrays, one per subject
                       D = number of ROIs/dimensions
                       T_i = number of time points for subject i

        Returns:
            self
        """
        # Validate and preprocess
        self.n_dims_, self.n_samples_list_, self.n_subjects_ = validate_input(data_list)

        # Preprocess each subject's data
        data_list = [preprocess_data(d) for d in data_list]

        # Concatenate for some operations
        Y_concat, boundaries = concatenate_subjects(data_list)
        n_total = Y_concat.shape[1]

        if self.config.verbose:
            print(f"[BSDS] Data: {self.n_subjects_} subjects, {self.n_dims_} dims, "
                  f"{n_total} total samples")

        # Run multiple initializations and keep best
        best_model = None
        best_ll = -np.inf

        for init_run in range(self.config.n_init_learning):
            if self.config.verbose:
                print(f"\n[BSDS] Initialization {init_run + 1}/{self.config.n_init_learning}")

            # Initialize parameters
            self._initialize(data_list, Y_concat, boundaries)

            # Run initial learning (fewer iterations)
            self._run_vb_loop(data_list, Y_concat, boundaries,
                             n_iter=self.config.n_init_iter,
                             verbose=False)

            final_ll = self.log_lik_history[-1] if self.log_lik_history else -np.inf

            if final_ll > best_ll:
                best_ll = final_ll
                best_model = self._get_state()

        # Restore best model
        if best_model is not None:
            self._set_state(best_model)

        if self.config.verbose:
            print(f"\n[BSDS] Best initialization: LL = {best_ll:.2f}")
            print(f"[BSDS] Running final optimization...")

        # Clear history for final run
        self.log_lik_history = []

        # Final optimization with full iterations
        self._run_vb_loop(data_list, Y_concat, boundaries,
                         n_iter=self.config.n_iter,
                         verbose=self.config.verbose)

        # Compute final state sequences
        log_emissions = compute_log_out_probs(
            data_list, self.Lm, self.Lcov, self.psii,
            self.Xm, self.Xcov, self.config.n_states, boundaries
        )
        log_trans = safe_log(self.stran)
        log_start = safe_log(self.sprior)

        self.states_ = viterbi_decode_all(log_emissions, log_trans, log_start)

        if self.config.verbose:
            print(f"\n[BSDS] Training complete!")
            print(f"[BSDS] Final LL: {self.log_lik_history[-1]:.2f}")

        return self

    def _initialize(self, data_list: List[np.ndarray],
                   Y_concat: np.ndarray,
                   boundaries: List[int]):
        """Initialize all model parameters."""
        D = self.n_dims_
        K = self.config.n_states
        k = self.config.max_ldim + 1  # +1 for bias term
        n_total = Y_concat.shape[1]

        # Initialize noise precision
        self.psii = np.ones((D, 1)) * 10.0

        # Initialize factor loadings
        self.Lm = [np.random.randn(D, k) * 0.1 for _ in range(K)]
        self.Lcov = [np.tile(np.eye(k)[:, :, np.newaxis], (1, 1, D)) for _ in range(K)]

        # Initialize latent variables
        self.Xm = [np.zeros((k, n_total)) for _ in range(K)]
        for s in range(K):
            self.Xm[s][0, :] = 1.0  # Bias term

        self.Xcov = [np.eye(k) for _ in range(K)]

        # Initialize mean/precision priors
        self.mean_mcl = np.mean(Y_concat, axis=1)
        self.nu_mcl = 1.0 / (np.var(Y_concat, axis=1) + EPS)

        # Set first column of Lm to data mean
        for s in range(K):
            self.Lm[s][:, 0] = self.mean_mcl

        # Initialize ARD parameters
        self.a = 1.0
        self.b = [np.ones(k - 1) for _ in range(K)]

        # Initialize state probabilities using K-Means
        self._init_states_kmeans(data_list, Y_concat, boundaries)

        # Initialize AR posteriors
        self.ar_post = None

    def _init_states_kmeans(self, data_list: List[np.ndarray],
                           Y_concat: np.ndarray,
                           boundaries: List[int]):
        """Initialize state assignments using K-Means."""
        K = self.config.n_states

        # Run K-Means on concatenated data (transposed for sklearn)
        kmeans = KMeans(n_clusters=K, n_init=10, random_state=self.config.random_seed)
        labels = kmeans.fit_predict(Y_concat.T)

        # Create one-hot state assignments
        n_total = len(labels)
        Qns = np.zeros((n_total, K))
        Qns[np.arange(n_total), labels] = 1.0

        # Split by subject
        self.gamma_list_ = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            self.gamma_list_.append(Qns[start:end, :])

        # Initialize transition counts
        self.Wa = np.ones((K, K)) * (self.config.alpha_a / K)
        self.Wpi = np.ones(K) * (self.config.alpha_pi / K)

        for i in range(len(labels) - 1):
            if labels[i] != labels[i + 1] or True:  # Count all transitions
                self.Wa[labels[i], labels[i + 1]] += 1

        self.Wpi += Qns[0, :]  # Initial state

        # Compute expected probabilities
        self.stran, self.sprior = update_transition_probs(self.Wa, self.Wpi)

    def _run_vb_loop(self, data_list: List[np.ndarray],
                    Y_concat: np.ndarray,
                    boundaries: List[int],
                    n_iter: int,
                    verbose: bool = True):
        """Run variational Bayes optimization loop."""
        K = self.config.n_states
        prev_weights = np.zeros(K)

        for iteration in range(n_iter):
            # Combine gamma for global Qns
            Qns = np.vstack(self.gamma_list_)

            # === E-Step: Infer latent variables ===

            # Update ARD parameters
            self.a, self.b = infer_qnu(self.Lm, self.Lcov, self.config.pa, self.config.pb, K)

            # Update latent variables X
            self.Xm, self.Xcov = infer_qx(Y_concat, self.Lm, self.Lcov, self.psii, K)

            # Update AR dynamics (core BSDS component)
            self.Xm, self.Xcov, self.ar_post = infer_ar(
                Y_concat, self.Xm, self.Xcov, self.gamma_list_, Qns,
                K, self.n_subjects_, self.ar_post, self.config.ar_approach
            )

            # === M-Step: Update parameters ===

            # Update factor loadings
            self.Lm, self.Lcov = infer_ql(
                Y_concat, self.Xm, self.Xcov, Qns, self.psii,
                self.a, self.b, self.mean_mcl, self.nu_mcl, K
            )

            # Update noise precision
            self.psii = infer_psii(
                Y_concat, self.Lm, self.Lcov, self.Xm, self.Xcov,
                Qns, K, self.config.noise_type
            )

            # Update mean prior
            self.mean_mcl, self.nu_mcl = infer_mcl(Y_concat, self.Xm, Qns, self.Lm, K)

            # Compute log emission probabilities
            log_emissions = compute_log_out_probs(
                data_list, self.Lm, self.Lcov, self.psii,
                self.Xm, self.Xcov, K, boundaries
            )

            # HMM E-step: Update state posteriors
            log_trans = safe_log(self.stran)
            log_start = safe_log(self.sprior)

            log_lik, self.gamma_list_, xi_list = vbhmm_estep(
                data_list, log_emissions, log_trans, log_start
            )

            # Update transition probabilities
            self.Wa, self.Wpi, self.stran, self.sprior = infer_qtheta(
                self.gamma_list_, xi_list, K,
                self.config.alpha_a, self.config.alpha_pi
            )

            # Record log-likelihood
            self.log_lik_history.append(log_lik)

            # Check convergence
            Qns_new = np.vstack(self.gamma_list_)
            current_weights = np.sum(Qns_new, axis=0)
            improvement = np.sum(np.abs(current_weights - prev_weights))
            prev_weights = current_weights.copy()

            if verbose and (iteration + 1) % 10 == 0:
                print(f"  Iter {iteration + 1}/{n_iter}: LL = {log_lik:.2f}, "
                      f"Improvement = {improvement:.4f}")

            if improvement < self.config.tol and iteration > 0:
                if verbose:
                    print(f"  Converged at iteration {iteration + 1}")
                break

    def _get_state(self) -> Dict[str, Any]:
        """Get current model state for saving."""
        return {
            'Lm': [L.copy() for L in self.Lm],
            'Lcov': [L.copy() for L in self.Lcov],
            'Xm': [X.copy() for X in self.Xm],
            'Xcov': [X.copy() for X in self.Xcov],
            'psii': self.psii.copy(),
            'stran': self.stran.copy(),
            'sprior': self.sprior.copy(),
            'Wa': self.Wa.copy(),
            'Wpi': self.Wpi.copy(),
            'a': self.a,
            'b': [b.copy() for b in self.b],
            'mean_mcl': self.mean_mcl.copy(),
            'nu_mcl': self.nu_mcl.copy(),
            'gamma_list': [g.copy() for g in self.gamma_list_],
            'log_lik_history': self.log_lik_history.copy()
        }

    def _set_state(self, state: Dict[str, Any]):
        """Restore model state."""
        self.Lm = state['Lm']
        self.Lcov = state['Lcov']
        self.Xm = state['Xm']
        self.Xcov = state['Xcov']
        self.psii = state['psii']
        self.stran = state['stran']
        self.sprior = state['sprior']
        self.Wa = state['Wa']
        self.Wpi = state['Wpi']
        self.a = state['a']
        self.b = state['b']
        self.mean_mcl = state['mean_mcl']
        self.nu_mcl = state['nu_mcl']
        self.gamma_list_ = state['gamma_list']
        self.log_lik_history = state['log_lik_history']

    def get_states(self) -> List[np.ndarray]:
        """Get fitted state sequences."""
        if self.states_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.states_

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics from fitted model."""
        if self.states_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        stats = compute_summary_statistics(
            self.states_, self.config.n_states, self.config.TR
        )
        stats['log_lik_history'] = self.log_lik_history
        stats['transition_prob'] = self.stran

        # Add effective states information
        effective_info = self.get_effective_n_states()
        stats['effective_n_states'] = effective_info['effective_n_states']
        stats['active_states'] = effective_info['active_states']
        stats['ard_relevance'] = effective_info['ard_relevance']

        return stats

    def get_effective_n_states(self, threshold: float = 0.01) -> Dict[str, Any]:
        """
        Compute effective number of states using ARD pruning.

        IMPORTANT: This differs from original MATLAB implementation.
        The original BSDS uses ARD within factor loadings (Q(nu)) to prune
        latent dimensions, not states directly. This method provides a
        complementary state-level pruning based on occupancy and ARD weights.

        The --n-states parameter is K_max (upper bound). Effective states
        are determined post-hoc by:
        1. Occupancy-based pruning: States with occupancy < threshold are inactive
        2. ARD relevance: Average ARD weight across latent dimensions per state

        Args:
            threshold: Minimum occupancy to consider a state as active (default: 0.01 = 1%)

        Returns:
            Dictionary containing:
            - effective_n_states: Number of active states
            - active_states: List of active state indices
            - inactive_states: List of pruned state indices
            - ard_relevance: ARD relevance score per state
            - occupancy: State occupancy values
            - pruning_note: Explanation of difference from original
        """
        if self.states_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        K = self.config.n_states

        # Method 1: Occupancy-based pruning
        from ..analysis.statistics import compute_occupancy_group
        occupancy = compute_occupancy_group(self.states_, K)

        # Method 2: ARD relevance per state
        # ARD parameters (self.a, self.b) control latent dimension relevance
        # Higher b values indicate less relevant dimensions
        ard_relevance = np.zeros(K)
        if self.b is not None:
            for s in range(K):
                # Expected precision: E[nu] = a / b
                # Lower precision = more relevant dimension
                expected_precision = self.a / (self.b[s] + 1e-10)
                # Convert to relevance: low precision = high relevance
                ard_relevance[s] = np.mean(1.0 / (expected_precision + 1e-10))

        # Determine active states
        active_states = [s for s in range(K) if occupancy[s] >= threshold]
        inactive_states = [s for s in range(K) if occupancy[s] < threshold]

        return {
            'effective_n_states': len(active_states),
            'active_states': active_states,
            'inactive_states': inactive_states,
            'ard_relevance': ard_relevance,
            'occupancy': occupancy,
            'pruning_note': (
                "NOTE: Original BSDS uses ARD for latent dimension pruning, not state pruning. "
                "This effective_n_states is computed post-hoc based on state occupancy. "
                "--n-states is K_max (upper bound); set it higher than expected to allow "
                "the model to find the optimal number of active states."
            )
        }

    def get_state_interpretation(self, roi_labels: Optional[List[str]] = None,
                                  network_mapping: Optional[Dict[str, List[int]]] = None) -> Dict[str, Any]:
        """
        Interpret states based on factor loadings (Lm).

        Analyzes the factor loading matrix to understand what each state represents
        in terms of brain activity patterns.

        Args:
            roi_labels: Optional list of ROI names (length D)
            network_mapping: Optional dict mapping network name to ROI indices
                           e.g., {'DMN': [0,1,2,...], 'Visual': [100,101,...]}

        Returns:
            Dictionary with state interpretations
        """
        if self.Lm is None:
            raise ValueError("Model not fitted. Call fit() first.")

        K = self.config.n_states
        D = self.n_dims_

        interpretations = {}

        for s in range(K):
            Ls = self.Lm[s]  # (D x k) factor loadings for state s

            # Bias term (column 0) = state-specific mean activity
            state_mean = Ls[:, 0]

            # Factor loadings (columns 1:) = activity patterns
            factor_loadings = Ls[:, 1:] if Ls.shape[1] > 1 else None

            # Top contributing ROIs (by absolute mean loading)
            if factor_loadings is not None:
                roi_importance = np.mean(np.abs(factor_loadings), axis=1)
            else:
                roi_importance = np.abs(state_mean)

            top_roi_indices = np.argsort(roi_importance)[::-1][:20]

            state_info = {
                'mean_activity': state_mean,
                'factor_loadings': factor_loadings,
                'roi_importance': roi_importance,
                'top_rois': top_roi_indices.tolist()
            }

            # Add ROI names if provided
            if roi_labels is not None:
                state_info['top_roi_names'] = [roi_labels[i] for i in top_roi_indices]

            # Network-level aggregation if mapping provided
            if network_mapping is not None:
                network_activity = {}
                for net_name, roi_indices in network_mapping.items():
                    valid_indices = [i for i in roi_indices if i < D]
                    if valid_indices:
                        network_activity[net_name] = float(np.mean(roi_importance[valid_indices]))
                state_info['network_activity'] = network_activity

            interpretations[f'state_{s}'] = state_info

        return interpretations

    def save(self, path: str):
        """Save fitted model to file."""
        import pickle
        state = {
            'config': self.config.to_dict(),
            'model_state': self._get_state(),
            'states': self.states_,
            'n_dims': self.n_dims_,
            'n_subjects': self.n_subjects_,
            'n_samples_list': self.n_samples_list_
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> 'BSDSModel':
        """Load fitted model from file."""
        import pickle
        with open(path, 'rb') as f:
            state = pickle.load(f)

        config = BSDSConfig.from_dict(state['config'])
        model = cls(config)
        model._set_state(state['model_state'])
        model.states_ = state['states']
        model.n_dims_ = state['n_dims']
        model.n_subjects_ = state['n_subjects']
        model.n_samples_list_ = state['n_samples_list']

        return model
