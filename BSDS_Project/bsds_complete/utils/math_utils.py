"""
Mathematical utility functions for BSDS
Implements: logsumexp, normalise, KL divergences, etc.
"""

import numpy as np
from scipy.special import gammaln, digamma as psi
from scipy.linalg import cholesky, LinAlgError
from typing import Union, Tuple

EPS = 1e-10


def logsumexp(x: np.ndarray, axis: int = None) -> np.ndarray:
    """
    Numerically stable log-sum-exp.

    Args:
        x: Input array
        axis: Axis along which to compute

    Returns:
        log(sum(exp(x))) computed stably
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    result = x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))
    if axis is not None:
        return np.squeeze(result, axis=axis)
    return np.squeeze(result)


def normalise(x: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, float]:
    """
    Normalize array to sum to 1 along axis.

    Args:
        x: Input array
        axis: Axis along which to normalize

    Returns:
        Tuple of (normalized array, normalization constant)
    """
    z = np.sum(x, axis=axis, keepdims=True)
    z = np.where(z == 0, 1, z)  # Avoid division by zero
    return x / z, np.squeeze(z)


def safe_log(x: np.ndarray) -> np.ndarray:
    """Safe logarithm that handles zeros."""
    return np.log(np.maximum(x, EPS))


def safe_cholesky(A: np.ndarray, jitter: float = 1e-6, max_tries: int = 10) -> np.ndarray:
    """
    Compute Cholesky decomposition with automatic jitter for numerical stability.

    Args:
        A: Positive definite matrix
        jitter: Initial jitter to add to diagonal
        max_tries: Maximum number of attempts

    Returns:
        Lower triangular Cholesky factor L such that A = L @ L.T
    """
    n = A.shape[0]
    jitter_val = jitter

    for _ in range(max_tries):
        try:
            L = cholesky(A + jitter_val * np.eye(n), lower=True)
            return L
        except LinAlgError:
            jitter_val *= 10

    # Last resort: use eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, EPS)
    return eigvecs @ np.diag(np.sqrt(eigvals))


def kl_dirichlet(alpha_p: np.ndarray, alpha_q: np.ndarray) -> float:
    """
    KL divergence between two Dirichlet distributions.
    KL(Dir(alpha_p) || Dir(alpha_q))

    Args:
        alpha_p: Parameters of first Dirichlet
        alpha_q: Parameters of second Dirichlet

    Returns:
        KL divergence value
    """
    alpha_p = np.asarray(alpha_p).flatten()
    alpha_q = np.asarray(alpha_q).flatten()

    sum_p = np.sum(alpha_p)
    sum_q = np.sum(alpha_q)

    kl = (gammaln(sum_p) - gammaln(sum_q) -
          np.sum(gammaln(alpha_p)) + np.sum(gammaln(alpha_q)) +
          np.sum((alpha_p - alpha_q) * (psi(alpha_p) - psi(sum_p))))

    return kl


def kl_gamma(a_p: float, b_p: float, a_q: float, b_q: float) -> float:
    """
    KL divergence between two Gamma distributions.
    KL(Gamma(a_p, b_p) || Gamma(a_q, b_q))

    Args:
        a_p, b_p: Shape and rate of first Gamma
        a_q, b_q: Shape and rate of second Gamma

    Returns:
        KL divergence value
    """
    kl = ((a_p - a_q) * psi(a_p) - gammaln(a_p) + gammaln(a_q) +
          a_q * (np.log(b_p) - np.log(b_q)) + a_p * (b_q - b_p) / b_p)

    return kl


def logdet_chol(L: np.ndarray) -> float:
    """Log determinant from Cholesky factor with robust handling."""
    diag_L = np.diag(L)
    # Ensure positive values for log
    diag_L = np.maximum(np.abs(diag_L), 1e-10)
    return 2.0 * np.sum(np.log(diag_L))


def solve_chol(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve L @ L.T @ x = b using Cholesky factor."""
    from scipy.linalg import solve_triangular
    y = solve_triangular(L, b, lower=True)
    x = solve_triangular(L.T, y, lower=False)
    return x


def expected_log_dirichlet(alpha: np.ndarray) -> np.ndarray:
    """
    E[log(pi)] where pi ~ Dirichlet(alpha)

    Args:
        alpha: Dirichlet parameters

    Returns:
        Expected log probabilities
    """
    return psi(alpha) - psi(np.sum(alpha))


def wishart_entropy(nu: float, W: np.ndarray) -> float:
    """
    Entropy of Wishart distribution.

    Args:
        nu: Degrees of freedom
        W: Scale matrix

    Returns:
        Entropy value
    """
    d = W.shape[0]
    L = safe_cholesky(W)
    logdet = logdet_chol(L)

    # Multivariate digamma
    h = 0.5 * (d + 1) * logdet
    for i in range(d):
        h += psi(0.5 * (nu - i))
    h *= 0.5

    return h
