"""
HMM Inference for BSDS
Implements Forward-Backward algorithm, smoothing, and Viterbi decoding
Corresponds to: VBHMMforward.m, VBHMMbackward.m, VBHMMsmooth.m, vbhmmEstep.m
"""

import numpy as np
from typing import List, Tuple, Optional
from ..utils.math_utils import logsumexp, safe_log, EPS


def vbhmm_forward(log_emissions: np.ndarray,
                  log_trans: np.ndarray,
                  log_start: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Forward pass of HMM in log space.
    Corresponds to VBHMMforward.m

    Args:
        log_emissions: Log emission probabilities (K x T)
        log_trans: Log transition matrix (K x K), log_trans[i,j] = log P(s_t=j | s_{t-1}=i)
        log_start: Log initial state probabilities (K,)

    Returns:
        Tuple of (log_alpha (K x T), log_likelihood)
    """
    K, T = log_emissions.shape
    log_alpha = np.zeros((K, T))

    # Initialize
    log_alpha[:, 0] = log_start + log_emissions[:, 0]

    # Forward recursion
    for t in range(1, T):
        for j in range(K):
            # log_alpha[j,t] = log sum_i exp(log_alpha[i,t-1] + log_trans[i,j]) + log_emit[j,t]
            log_alpha[j, t] = logsumexp(log_alpha[:, t-1] + log_trans[:, j]) + log_emissions[j, t]

    # Log likelihood
    log_lik = logsumexp(log_alpha[:, -1])

    return log_alpha, log_lik


def vbhmm_backward(log_emissions: np.ndarray,
                   log_trans: np.ndarray) -> np.ndarray:
    """
    Backward pass of HMM in log space.
    Corresponds to VBHMMbackward.m

    Args:
        log_emissions: Log emission probabilities (K x T)
        log_trans: Log transition matrix (K x K)

    Returns:
        log_beta (K x T)
    """
    K, T = log_emissions.shape
    log_beta = np.zeros((K, T))

    # Initialize: log_beta[:, T-1] = 0 (i.e., beta = 1)

    # Backward recursion
    for t in range(T - 2, -1, -1):
        for i in range(K):
            # log_beta[i,t] = log sum_j exp(log_trans[i,j] + log_emit[j,t+1] + log_beta[j,t+1])
            terms = log_trans[i, :] + log_emissions[:, t+1] + log_beta[:, t+1]
            log_beta[i, t] = logsumexp(terms)

    return log_beta


def vbhmm_smooth(log_alpha: np.ndarray,
                 log_beta: np.ndarray,
                 log_emissions: np.ndarray,
                 log_trans: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute smoothed posteriors from forward-backward messages.
    Corresponds to VBHMMsmooth.m

    Args:
        log_alpha: Forward messages (K x T)
        log_beta: Backward messages (K x T)
        log_emissions: Log emission probabilities (K x T)
        log_trans: Log transition matrix (K x K)

    Returns:
        Tuple of:
        - gamma: Marginal state posteriors (K x T), P(s_t | Y_{1:T})
        - xi: Pairwise state posteriors (K x K x T-1), P(s_{t-1}, s_t | Y_{1:T})
    """
    K, T = log_alpha.shape

    # Marginal posteriors: gamma[k,t] = P(s_t = k | Y)
    log_gamma = log_alpha + log_beta
    log_gamma -= logsumexp(log_gamma, axis=0)  # Normalize
    gamma = np.exp(log_gamma)

    # Pairwise posteriors: xi[i,j,t] = P(s_{t-1}=i, s_t=j | Y)
    xi = np.zeros((K, K, T - 1))

    for t in range(T - 1):
        for i in range(K):
            for j in range(K):
                xi[i, j, t] = (log_alpha[i, t] +
                               log_trans[i, j] +
                               log_emissions[j, t+1] +
                               log_beta[j, t+1])

        # Normalize
        xi[:, :, t] -= logsumexp(xi[:, :, t].flatten())

    xi = np.exp(xi)

    return gamma, xi


def vbhmm_estep(data_list: List[np.ndarray],
                log_emissions_list: List[np.ndarray],
                log_trans: np.ndarray,
                log_start: np.ndarray) -> Tuple[float, List[np.ndarray], List[np.ndarray]]:
    """
    Complete E-step for HMM across all subjects.
    Corresponds to vbhmmEstep.m

    Args:
        data_list: List of data arrays (not used here, for interface consistency)
        log_emissions_list: List of log emission arrays per subject
        log_trans: Log transition matrix (K x K)
        log_start: Log initial probabilities (K,)

    Returns:
        Tuple of:
        - total_log_lik: Total log likelihood across subjects
        - gamma_list: List of marginal posteriors per subject
        - xi_list: List of pairwise posteriors per subject
    """
    n_subjects = len(log_emissions_list)
    gamma_list = []
    xi_list = []
    total_log_lik = 0.0

    for ns in range(n_subjects):
        log_emit = log_emissions_list[ns]

        # Forward-Backward
        log_alpha, log_lik = vbhmm_forward(log_emit, log_trans, log_start)
        log_beta = vbhmm_backward(log_emit, log_trans)

        # Smoothing
        gamma, xi = vbhmm_smooth(log_alpha, log_beta, log_emit, log_trans)

        gamma_list.append(gamma.T)  # (T x K) for consistency
        xi_list.append(xi)  # (K x K x T-1)
        total_log_lik += log_lik

    return total_log_lik, gamma_list, xi_list


def viterbi_decode(log_emissions: np.ndarray,
                   log_trans: np.ndarray,
                   log_start: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Viterbi algorithm for finding most likely state sequence.
    Corresponds to viterbi_path.m and estimateStatesByVitterbi.m

    Args:
        log_emissions: Log emission probabilities (K x T)
        log_trans: Log transition matrix (K x K)
        log_start: Log initial probabilities (K,)

    Returns:
        Tuple of (state_sequence (T,), log_probability)
    """
    K, T = log_emissions.shape

    # Viterbi trellis
    V = np.zeros((K, T))
    backpointer = np.zeros((K, T), dtype=int)

    # Initialize
    V[:, 0] = log_start + log_emissions[:, 0]

    # Forward pass
    for t in range(1, T):
        for j in range(K):
            scores = V[:, t-1] + log_trans[:, j]
            backpointer[j, t] = np.argmax(scores)
            V[j, t] = scores[backpointer[j, t]] + log_emissions[j, t]

    # Backtrack
    states = np.zeros(T, dtype=int)
    states[-1] = np.argmax(V[:, -1])
    log_prob = V[states[-1], -1]

    for t in range(T - 2, -1, -1):
        states[t] = backpointer[states[t + 1], t + 1]

    return states, log_prob


def viterbi_decode_all(log_emissions_list: List[np.ndarray],
                       log_trans: np.ndarray,
                       log_start: np.ndarray) -> List[np.ndarray]:
    """
    Viterbi decoding for all subjects.

    Args:
        log_emissions_list: List of log emission arrays
        log_trans: Log transition matrix
        log_start: Log initial probabilities

    Returns:
        List of state sequences per subject
    """
    states_list = []
    for log_emit in log_emissions_list:
        states, _ = viterbi_decode(log_emit, log_trans, log_start)
        states_list.append(states)
    return states_list
