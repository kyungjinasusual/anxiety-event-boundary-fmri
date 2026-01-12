"""
Latent Variable Inference for BSDS
Implements: inferQX.m, computeLogOutProbs.m
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from ..utils.math_utils import safe_cholesky, logdet_chol, EPS


def infer_qx(Y: np.ndarray,
             Lm: List[np.ndarray],
             Lcov: List[np.ndarray],
             psii: np.ndarray,
             n_states: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Infer posterior over latent variables X for each state.
    Corresponds to inferQX.m

    The model is: Y = L @ X + noise
    where noise ~ N(0, diag(1/psii))

    For each state s:
        X ~ N(Xm[s], Xcov[s])

    Args:
        Y: Observed data (D x T)
        Lm: List of factor loading means per state, each (D x k)
        Lcov: List of factor loading covariances per state, each (k x k x D)
        psii: Noise precision for each dimension (D,)
        n_states: Number of states

    Returns:
        Tuple of:
        - Xm_list: List of latent means per state, each (k x T)
        - Xcov_list: List of latent covariances per state, each (k x k)
    """
    D, T = Y.shape
    Xm_list = []
    Xcov_list = []

    psii_flat = psii.flatten()
    psi_diag = np.diag(psii_flat + EPS)

    for s in range(n_states):
        k = Lm[s].shape[1]  # Latent dimension (includes bias term)

        # For factor analysis model:
        # p(X|Y,L,psi) propto p(Y|X,L,psi) p(X)
        # where p(X) = N(0, I) for dimensions 2:k (first is bias=1)

        # Posterior covariance for latent factors (excluding bias)
        # Xcov = (I + L'PsiL)^{-1}
        # where L is the factor loading (excluding bias column)

        L_nobias = Lm[s][:, 1:]  # (D x k-1), excluding bias

        # Include uncertainty in L
        # E[L'PsiL] = L'PsiL + sum_d psi_d * Lcov[:,:,d]
        LtPsiL = L_nobias.T @ psi_diag @ L_nobias

        # Add contribution from Lcov
        for d in range(D):
            LtPsiL += psii_flat[d] * Lcov[s][1:, 1:, d]

        # Posterior covariance with robust inversion
        precision_matrix = np.eye(k - 1) + LtPsiL
        reg = max(EPS, 1e-6) * max(1.0, np.trace(precision_matrix) / (k - 1))
        try:
            Xcov_nobias = np.linalg.inv(precision_matrix + reg * np.eye(k - 1))
        except np.linalg.LinAlgError:
            Xcov_nobias = np.eye(k - 1) * 0.1

        # Check for NaN/Inf
        if not np.all(np.isfinite(Xcov_nobias)):
            Xcov_nobias = np.eye(k - 1) * 0.1

        # Posterior mean: Xm = Xcov @ L' @ Psi @ (Y - L[:,0])
        # where L[:,0] is the bias/mean term
        Y_centered = Y - Lm[s][:, 0:1]  # Subtract mean (D x T)
        Xm_nobias = Xcov_nobias @ L_nobias.T @ psi_diag @ Y_centered  # (k-1 x T)

        # Full latent variable (with bias term = 1)
        Xm_full = np.vstack([np.ones((1, T)), Xm_nobias])  # (k x T)

        # Full covariance (bias has zero variance)
        Xcov_full = np.zeros((k, k))
        Xcov_full[1:, 1:] = Xcov_nobias

        Xm_list.append(Xm_full)
        Xcov_list.append(Xcov_full)

    return Xm_list, Xcov_list


def compute_log_out_probs(data_list: List[np.ndarray],
                          Lm: List[np.ndarray],
                          Lcov: List[np.ndarray],
                          psii: np.ndarray,
                          Xm: List[np.ndarray],
                          Xcov: List[np.ndarray],
                          n_states: int,
                          boundaries: List[int] = None) -> List[np.ndarray]:
    """
    Compute log emission probabilities for each state at each time point.
    Corresponds to computeLogOutProbs.m

    This computes log p(Y_t | s_t = k) for all k and t.

    Args:
        data_list: List of data arrays per subject, each (D x T_i)
        Lm: List of factor loading means per state
        Lcov: List of factor loading covariances per state
        psii: Noise precision (D,)
        Xm: List of latent means per state
        Xcov: List of latent covariances per state
        n_states: Number of states
        boundaries: Optional list of cumulative sample boundaries

    Returns:
        List of log emission arrays per subject, each (K x T_i)
    """
    n_subjects = len(data_list)
    psii_flat = psii.flatten()
    psi_diag = np.diag(psii_flat + EPS)
    D = data_list[0].shape[0]

    log_out_probs_list = []

    # Calculate boundaries if not provided
    if boundaries is None:
        boundaries = [0]
        for data in data_list:
            boundaries.append(boundaries[-1] + data.shape[1])

    for ns in range(n_subjects):
        Y = data_list[ns]
        T_subj = Y.shape[1]
        log_qns = np.zeros((n_states, T_subj))

        start_col = boundaries[ns]
        end_col = boundaries[ns + 1]

        for s in range(n_states):
            k = Lm[s].shape[1]

            # Pre-compute terms
            # LmPsiLm = Lm' @ Psi @ Lm
            LmPsiLm = Lm[s].T @ psi_diag @ Lm[s]

            # Lcov contribution: sum_d psi_d * Lcov[:,:,d]
            Lcov_term = np.zeros((k, k))
            for d in range(D):
                Lcov_term += psii_flat[d] * Lcov[s][:, :, d]

            temp_mat = LmPsiLm + Lcov_term

            # Get subject-specific latent variables
            Xm_subj = Xm[s][:, start_col:end_col]
            Xcov_s = Xcov[s]

            # Compute log probability terms
            # Term A: Y' @ Psi @ (Y - 2*L*X)
            recon = Lm[s] @ Xm_subj  # (D x T)
            diff = Y - 2 * recon
            term_a = np.sum(Y * (psi_diag @ diff), axis=0)  # (T,)

            # Term B: tr(temp_mat @ Xcov)
            term_b = np.trace(temp_mat @ Xcov_s) * np.ones(T_subj)

            # Term C: X' @ temp_mat @ X
            term_c = np.sum(Xm_subj * (temp_mat @ Xm_subj), axis=0)  # (T,)

            # Term D: tr(Xcov[2:,2:]) - prior on latent factors
            term_d = np.trace(Xcov_s[1:, 1:]) * np.ones(T_subj)

            # Term E: sum(X[2:]^2) - prior on latent factors
            term_e = np.sum(Xm_subj[1:, :] ** 2, axis=0)  # (T,)

            # Term F: -log det(Xcov[2:,2:]) - entropy term
            try:
                L_chol = safe_cholesky(Xcov_s[1:, 1:] + EPS * np.eye(k - 1))
                term_f = -logdet_chol(L_chol) * np.ones(T_subj)
            except:
                term_f = np.zeros(T_subj)

            # Combine: log q(s) = -0.5 * (sum of terms)
            log_qns[s, :] = -0.5 * (term_a + term_b + term_c + term_d + term_e + term_f)

        log_out_probs_list.append(log_qns)

    return log_out_probs_list


def compute_log_out_probs_fast(Y: np.ndarray,
                               Lm: List[np.ndarray],
                               psii: np.ndarray,
                               n_states: int) -> np.ndarray:
    """
    Fast computation of log emission probabilities (simplified version).
    Uses point estimates only, ignoring covariance terms.

    Args:
        Y: Observed data (D x T)
        Lm: List of factor loading means per state
        psii: Noise precision (D,)
        n_states: Number of states

    Returns:
        Log emission probabilities (K x T)
    """
    D, T = Y.shape
    psii_flat = psii.flatten()

    log_probs = np.zeros((n_states, T))

    for s in range(n_states):
        # Get mean prediction
        mean = Lm[s][:, 0:1]  # Bias term (D x 1)

        # Residual
        residual = Y - mean  # (D x T)

        # Weighted squared error
        weighted_sq = residual ** 2 * psii_flat.reshape(-1, 1)
        log_probs[s, :] = -0.5 * np.sum(weighted_sq, axis=0)

        # Add log determinant term (constant across time)
        log_probs[s, :] += 0.5 * np.sum(np.log(psii_flat + EPS))

    return log_probs
