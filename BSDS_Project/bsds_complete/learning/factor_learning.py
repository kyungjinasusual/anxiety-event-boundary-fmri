"""
Factor Analysis Learning for BSDS
Implements: inferQL.m, inferQnu.m, inferpsii2.m, infermcl.m
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from scipy.special import digamma as psi
from ..utils.math_utils import EPS


def infer_ql(Y: np.ndarray,
             Xm: List[np.ndarray],
             Xcov: List[np.ndarray],
             Qns: np.ndarray,
             psii: np.ndarray,
             a: float,
             b: List[np.ndarray],
             mean_mcl: np.ndarray,
             nu_mcl: np.ndarray,
             n_states: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Update posterior over factor loadings L for each state.
    Corresponds to inferQL.m

    Model: Y = L @ X + noise
    Prior: L[d,j] ~ N(mean_Lambda[d,j], 1/nu[d,j])
           where nu comes from ARD prior

    Args:
        Y: Observed data (D x T)
        Xm: List of latent means per state, each (k x T)
        Xcov: List of latent covariances per state, each (k x k)
        Qns: State responsibilities (T x K)
        psii: Noise precision (D,)
        a: ARD shape parameter
        b: List of ARD rate parameters per state
        mean_mcl: Prior mean for first column (D,)
        nu_mcl: Prior precision for first column (D,)
        n_states: Number of states

    Returns:
        Tuple of:
        - Lm: Updated factor loading means per state
        - Lcov: Updated factor loading covariances per state
    """
    D, T = Y.shape
    psii_flat = psii.flatten()
    psi_diag = np.diag(psii_flat + EPS)

    Lm_new = []
    Lcov_new = []

    for s in range(n_states):
        k = Xm[s].shape[0]

        # Prior precision: first column uses mean_mcl prior, rest use ARD
        # num[d,:] = [nu_mcl[d], a/b[s][0], a/b[s][1], ...]
        num = np.zeros((D, k))
        num[:, 0] = nu_mcl.flatten()
        for j in range(1, k):
            num[:, j] = a / (b[s][j - 1] + EPS)

        # Mean of prior
        mean_Lambda = np.zeros((D, k))
        mean_Lambda[:, 0] = mean_mcl.flatten()  # First column is the mean

        # Weighted sufficient statistics
        # temp = X @ diag(Qns[:,s]) @ X' = weighted outer product sum
        gamma_s = Qns[:, s]  # (T,)
        Xm_s = Xm[s]  # (k x T)

        # T2 = Xcov * sum(gamma) + Xm @ diag(gamma) @ Xm'
        T2 = Xcov[s] * np.sum(gamma_s) + Xm_s @ np.diag(gamma_s) @ Xm_s.T

        # T3 = Psi @ Y @ diag(gamma) @ X'
        T3 = psi_diag @ Y @ np.diag(gamma_s) @ Xm_s.T

        # Update each dimension independently
        Lm_s = np.zeros((D, k))
        Lcov_s = np.zeros((k, k, D))

        for d in range(D):
            # Posterior covariance with robust inversion
            precision_d = np.diag(num[d, :]) + psii_flat[d] * T2
            reg = max(EPS, 1e-6) * max(1.0, np.trace(precision_d) / k)
            try:
                Lcov_s[:, :, d] = np.linalg.inv(precision_d + reg * np.eye(k))
            except np.linalg.LinAlgError:
                Lcov_s[:, :, d] = np.eye(k) * 0.1

            # Check for NaN/Inf
            if not np.all(np.isfinite(Lcov_s[:, :, d])):
                Lcov_s[:, :, d] = np.eye(k) * 0.1

            # Posterior mean
            Lm_s[d, :] = (T3[d, :] + mean_Lambda[d, :] * num[d, :]) @ Lcov_s[:, :, d]

            # Check Lm for NaN/Inf
            if not np.all(np.isfinite(Lm_s[d, :])):
                Lm_s[d, :] = 0.0

        Lm_new.append(Lm_s)
        Lcov_new.append(Lcov_s)

    return Lm_new, Lcov_new


def infer_qnu(Lm: List[np.ndarray],
              Lcov: List[np.ndarray],
              pa: float,
              pb: float,
              n_states: int) -> Tuple[float, List[np.ndarray]]:
    """
    Update ARD (Automatic Relevance Determination) parameters.
    Corresponds to inferQnu.m

    ARD prior: L[:,j] ~ N(0, 1/alpha_j) where alpha_j ~ Gamma(a, b_j)
    This implements the E-step update for the Gamma posterior.

    Args:
        Lm: List of factor loading means per state
        Lcov: List of factor loading covariances per state
        pa: Prior shape for ARD
        pb: Prior rate for ARD
        n_states: Number of states

    Returns:
        Tuple of:
        - a: Updated shape parameter (shared across states)
        - b: List of updated rate parameters per state
    """
    D = Lm[0].shape[0]
    k = Lm[0].shape[1]

    # Shape parameter (same for all)
    a = pa + 0.5 * D

    # Rate parameter per state and per latent dimension
    b_new = []

    for s in range(n_states):
        k_s = Lm[s].shape[1]

        # b[j] = pb + 0.5 * (sum_d Lcov[j,j,d] + sum_d Lm[d,j]^2)
        # Only for j >= 1 (not the bias/mean term)
        b_s = np.zeros(k_s - 1)

        for j in range(1, k_s):
            # Variance from Lcov
            var_term = np.sum([Lcov[s][j, j, d] for d in range(D)])
            # Squared mean
            sq_mean = np.sum(Lm[s][:, j] ** 2)
            b_s[j - 1] = pb + 0.5 * (var_term + sq_mean)

        b_new.append(b_s)

    return a, b_new


def infer_psii(Y: np.ndarray,
               Lm: List[np.ndarray],
               Lcov: List[np.ndarray],
               Xm: List[np.ndarray],
               Xcov: List[np.ndarray],
               Qns: np.ndarray,
               n_states: int,
               noise_type: int = 0) -> np.ndarray:
    """
    Update noise precision parameters.
    Corresponds to inferpsii2.m

    Args:
        Y: Observed data (D x T)
        Lm: List of factor loading means
        Lcov: List of factor loading covariances
        Xm: List of latent means
        Xcov: List of latent covariances
        Qns: State responsibilities (T x K)
        n_states: Number of states
        noise_type: 0 = dimension-specific, 1 = shared across dimensions

    Returns:
        Updated noise precision (D,)
    """
    D, T = Y.shape

    # Expected squared residual for each dimension
    expected_sq_residual = np.zeros(D)

    for s in range(n_states):
        gamma_s = Qns[:, s]  # (T,)
        Xm_s = Xm[s]  # (k x T)
        Xcov_s = Xcov[s]  # (k x k)
        Lm_s = Lm[s]  # (D x k)
        Lcov_s = Lcov[s]  # (k x k x D)

        for d in range(D):
            # E[(Y_d - L_d @ X)^2]
            # = Y_d^2 - 2*Y_d*E[L_d]*E[X] + E[L_d @ X @ X' @ L_d']

            # Term 1: Y_d^2 weighted by gamma
            term1 = np.sum(gamma_s * Y[d, :] ** 2)

            # Term 2: -2 * Y_d * E[L_d] @ E[X]
            pred = Lm_s[d, :] @ Xm_s  # (T,)
            term2 = -2 * np.sum(gamma_s * Y[d, :] * pred)

            # Term 3: E[L_d @ X @ X' @ L_d']
            # = Lm_d @ E[X @ X'] @ Lm_d' + tr(Lcov_d @ E[X @ X'])
            ExxT = Xcov_s * np.sum(gamma_s) + Xm_s @ np.diag(gamma_s) @ Xm_s.T
            term3_a = Lm_s[d, :] @ ExxT @ Lm_s[d, :].T
            term3_b = np.trace(Lcov_s[:, :, d] @ ExxT)
            term3 = term3_a + term3_b

            expected_sq_residual[d] += term1 + term2 + term3

    # Update precision (use Gamma posterior mode)
    # With improper prior, psii[d] = T / expected_sq_residual[d]
    psii_new = T / (expected_sq_residual + EPS)

    if noise_type == 1:
        # Shared across dimensions
        psii_new = np.mean(psii_new) * np.ones(D)

    # Clamp to reasonable range
    psii_new = np.clip(psii_new, 1e-6, 1e6)

    return psii_new


def infer_mcl(Y: np.ndarray,
              Xm: List[np.ndarray],
              Qns: np.ndarray,
              Lm: List[np.ndarray],
              n_states: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update mean and precision for the first column of L (the mean term).
    Corresponds to infermcl.m

    Args:
        Y: Observed data (D x T)
        Xm: List of latent means
        Qns: State responsibilities
        Lm: Current factor loading means
        n_states: Number of states

    Returns:
        Tuple of (mean_mcl, nu_mcl) - prior mean and precision
    """
    D, T = Y.shape

    # Weighted mean of observations
    total_weight = np.sum(Qns)
    weighted_sum = np.zeros(D)

    for s in range(n_states):
        # Subtract contribution from latent factors (except mean)
        if Xm[s].shape[0] > 1:
            residual = Y - Lm[s][:, 1:] @ Xm[s][1:, :]
        else:
            residual = Y
        weighted_sum += residual @ Qns[:, s]

    mean_mcl = weighted_sum / (total_weight + EPS)

    # Precision: inverse of weighted variance
    weighted_sq_sum = np.zeros(D)
    for s in range(n_states):
        if Xm[s].shape[0] > 1:
            residual = Y - Lm[s][:, 1:] @ Xm[s][1:, :]
        else:
            residual = Y
        weighted_sq_sum += (residual - mean_mcl.reshape(-1, 1)) ** 2 @ Qns[:, s]

    variance = weighted_sq_sum / (total_weight + EPS)
    nu_mcl = 1.0 / (variance + EPS)

    return mean_mcl, nu_mcl
