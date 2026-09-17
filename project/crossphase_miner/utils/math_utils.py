"""
Mathematical utilities for CrossPhaseMiner.

Provides only the functions actually used by the pipeline:
- KST local hour-of-day conversion
- Gaussian product (Bayesian update)
- UKF sigma-point and weight generation
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# KST timezone offset in hours (Seoul, South Korea)
KST_OFFSET_HOURS = 9.0


def local_hour_of_day(timestamp: float, offset_hours: float = KST_OFFSET_HOURS) -> float:
    """Convert a Unix timestamp to local hour-of-day [0.0, 24.0)."""
    return ((timestamp + offset_hours * 3600.0) % 86400.0) / 3600.0


def gaussian_product(mu1: float, sigma1: float, mu2: float, sigma2: float) -> Tuple[float, float]:
    """
    Product of two Gaussians N(mu1, sigma1^2) * N(mu2, sigma2^2).

    Returns:
        (mu_posterior, sigma_posterior)
    """
    if sigma1 <= 0 and sigma2 <= 0:
        return mu1, 1e-6
    if sigma1 <= 0:
        return mu1, sigma2
    if sigma2 <= 0:
        return mu2, sigma1

    var1 = sigma1**2
    var2 = sigma2**2
    var_post = 1.0 / (1.0 / var1 + 1.0 / var2)
    mu_post = var_post * (mu1 / var1 + mu2 / var2)
    sigma_post = np.sqrt(var_post)
    return mu_post, sigma_post


def ukf_sigma_points(x: np.ndarray, P: np.ndarray, kappa: float = 0.0) -> np.ndarray:
    """
    Generate sigma points for the Unscented Kalman Filter.

    Args:
        x: State vector (n,).
        P: Covariance matrix (n, n).
        kappa: Scaling parameter.

    Returns:
        Sigma points array of shape (2n+1, n).
    """
    n = len(x)
    sigma_points = np.zeros((2 * n + 1, n))
    sigma_points[0] = x

    try:
        sqrt_P = np.linalg.cholesky((n + kappa) * P)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(P)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        sqrt_P = eigenvectors @ np.diag(np.sqrt(eigenvalues))
        sqrt_P = sqrt_P * np.sqrt(n + kappa)

    for i in range(n):
        sigma_points[i + 1] = x + sqrt_P[:, i]
        sigma_points[i + 1 + n] = x - sqrt_P[:, i]

    return sigma_points


def ukf_weights(
    n: int, kappa: float = 0.0, alpha: float = 1.0, beta: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute UKF mean/covariance weights.

    Returns:
        (Wm, Wc, lambda_param)
    """
    lam = alpha**2 * (n + kappa) - n
    Wm = np.zeros(2 * n + 1)
    Wc = np.zeros(2 * n + 1)
    Wm[0] = lam / (n + lam)
    Wc[0] = lam / (n + lam) + (1 - alpha**2 + beta)
    Wm[1:] = 1.0 / (2 * (n + lam))
    Wc[1:] = 1.0 / (2 * (n + lam))
    return Wm, Wc, lam
