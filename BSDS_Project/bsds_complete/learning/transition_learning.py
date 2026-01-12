"""
Transition Probability Learning for BSDS
Implements: inferQtheta.m, transition updates
"""

import numpy as np
from typing import List, Tuple
from scipy.special import digamma as psi
from ..utils.math_utils import EPS


def infer_qtheta(gamma_list: List[np.ndarray],
                 xi_list: List[np.ndarray],
                 n_states: int,
                 alpha_a: float = 1.0,
                 alpha_pi: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Update transition and initial state probability posteriors.
    Corresponds to inferQtheta.m

    Prior: A[i,:] ~ Dirichlet(alpha_a * ones / K)
           pi ~ Dirichlet(alpha_pi * ones / K)

    Args:
        gamma_list: List of marginal state posteriors per subject, each (T x K)
        xi_list: List of pairwise posteriors per subject, each (K x K x T-1)
        n_states: Number of states
        alpha_a: Dirichlet concentration for transitions
        alpha_pi: Dirichlet concentration for initial

    Returns:
        Tuple of (Wa, Wpi, stran, sprior):
        - Wa: Posterior Dirichlet parameters for transitions (K x K)
        - Wpi: Posterior Dirichlet parameters for initial (K,)
        - stran: Expected transition probabilities (K x K)
        - sprior: Expected initial probabilities (K,)
    """
    n_subjects = len(gamma_list)

    # Prior parameters
    ua = np.ones((n_states, n_states)) * (alpha_a / n_states)
    upi = np.ones(n_states) * (alpha_pi / n_states)

    # Accumulate sufficient statistics
    wa = np.zeros((n_states, n_states))
    wpi = np.zeros(n_states)

    for ns in range(n_subjects):
        gamma = gamma_list[ns]  # (T x K)
        xi = xi_list[ns]  # (K x K x T-1)

        # Transition counts: sum over time of xi
        wa += np.sum(xi, axis=2)

        # Initial state counts: gamma at t=0
        wpi += gamma[0, :]

    # Posterior parameters
    Wa = wa + ua
    Wpi = wpi + upi

    # Expected probabilities under Dirichlet posterior
    # E[log A[i,j]] = psi(Wa[i,j]) - psi(sum_j Wa[i,j])
    log_stran = psi(Wa) - psi(np.sum(Wa, axis=1, keepdims=True))
    stran = np.exp(log_stran)

    log_sprior = psi(Wpi) - psi(np.sum(Wpi))
    sprior = np.exp(log_sprior)

    # Normalize to ensure proper probabilities
    stran = stran / np.sum(stran, axis=1, keepdims=True)
    sprior = sprior / np.sum(sprior)

    return Wa, Wpi, stran, sprior


def update_transition_probs(Wa: np.ndarray, Wpi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute expected transition probabilities from Dirichlet posteriors.

    Args:
        Wa: Posterior Dirichlet parameters for transitions (K x K)
        Wpi: Posterior Dirichlet parameters for initial (K,)

    Returns:
        Tuple of (stran, sprior)
    """
    # Expected log probabilities
    log_stran = psi(Wa) - psi(np.sum(Wa, axis=1, keepdims=True))
    log_sprior = psi(Wpi) - psi(np.sum(Wpi))

    stran = np.exp(log_stran)
    sprior = np.exp(log_sprior)

    # Normalize
    stran = stran / (np.sum(stran, axis=1, keepdims=True) + EPS)
    sprior = sprior / (np.sum(sprior) + EPS)

    return stran, sprior


def compute_transition_kl(Wa: np.ndarray,
                          Wpi: np.ndarray,
                          alpha_a: float = 1.0,
                          alpha_pi: float = 1.0) -> float:
    """
    Compute KL divergence for transition parameters.

    Args:
        Wa: Posterior transition parameters
        Wpi: Posterior initial parameters
        alpha_a: Prior concentration for transitions
        alpha_pi: Prior concentration for initial

    Returns:
        Total KL divergence
    """
    from scipy.special import gammaln

    K = Wa.shape[0]

    # Prior parameters
    ua = np.ones(K) * (alpha_a / K)
    upi = np.ones(K) * (alpha_pi / K)

    kl = 0.0

    # KL for each row of transition matrix
    for i in range(K):
        kl += _kl_dirichlet(Wa[i, :], ua)

    # KL for initial distribution
    kl += _kl_dirichlet(Wpi, upi)

    return kl


def _kl_dirichlet(alpha_p: np.ndarray, alpha_q: np.ndarray) -> float:
    """KL divergence between Dirichlet distributions."""
    from scipy.special import gammaln

    sum_p = np.sum(alpha_p)
    sum_q = np.sum(alpha_q)

    kl = (gammaln(sum_p) - gammaln(sum_q) -
          np.sum(gammaln(alpha_p)) + np.sum(gammaln(alpha_q)) +
          np.sum((alpha_p - alpha_q) * (psi(alpha_p) - psi(sum_p))))

    return kl
