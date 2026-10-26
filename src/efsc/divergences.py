"""Kullback--Leibler divergence utilities for EFSC models.

This module evaluates family-specific KL divergences from natural parameters
using operations that support both NumPy arrays and PyTorch tensors.

Author:
    Hector Rodriguez-Deniz

Affiliation:
    Data Science Institute, Columbia University

Date:
    August 2026

License:
    MIT
"""

import numpy as np

from ._families import one_parameter_log_partition, one_parameter_mean_from_eta


def _is_torch_tensor(x):
    """Return whether ``x`` is a PyTorch tensor without importing PyTorch."""

    return type(x).__module__.startswith("torch")


def _clamp_min(x, eps):
    """Clamp NumPy or PyTorch values below by ``eps``."""

    if _is_torch_tensor(x):
        return x.clamp_min(eps)
    return np.clip(x, a_min=eps, a_max=None)


def _log(x):
    """Evaluate a backend-preserving natural logarithm."""

    if _is_torch_tensor(x):
        return x.log()
    return np.log(x)


def _sum(x, axis):
    """Sum values along an axis using the input backend."""

    if _is_torch_tensor(x):
        return x.sum(dim=axis)
    return np.sum(x, axis=axis)


def _reduce(x, reduction):
    """Apply an optional mean or sum reduction."""

    if reduction is None:
        return x
    if reduction == "mean":
        return x.mean()
    if reduction == "sum":
        return x.sum()
    raise ValueError("reduction must be None, 'mean', or 'sum'.")


def kl_one_parameter_from_eta(family, eta_p, eta_q, reduction=None):
    """
    KL divergence KL(p(y; eta_p) || p(y; eta_q)).

    For a one-parameter exponential family with log-partition A, the KL is

        (eta_p - eta_q) * A'(eta_p) - (A(eta_p) - A(eta_q)).

    Inputs can be scalars, NumPy arrays, or torch tensors with broadcastable
    shapes.
    """

    kl = (
        (eta_p - eta_q) * one_parameter_mean_from_eta(family, eta_p)
        - (
            one_parameter_log_partition(family, eta_p)
            - one_parameter_log_partition(family, eta_q)
        )
    )
    return _reduce(kl, reduction)


def kl_gaussian_univariate(mu_p, var_p, mu_q, var_q, eps=1e-8, reduction=None):
    """
    KL divergence KL(N(mu_p, var_p) || N(mu_q, var_q)).

    Parameters can be scalars, NumPy arrays, or torch tensors with broadcastable
    shapes. Variances are clamped below by eps for numerical stability.
    """

    var_p = _clamp_min(var_p, eps)
    var_q = _clamp_min(var_q, eps)

    kl = 0.5 * (
        _log(var_q / var_p)
        + (var_p + (mu_p - mu_q) ** 2) / var_q
        - 1
    )
    return _reduce(kl, reduction)


def kl_gaussian_univariate_from_theta(theta_p, theta_q, eps=1e-8, reduction=None):
    """
    KL divergence for theta arrays whose last axis is (mu, var).
    """

    mu_p = theta_p[..., 0]
    var_p = theta_p[..., 1]
    mu_q = theta_q[..., 0]
    var_q = theta_q[..., 1]

    return kl_gaussian_univariate(
        mu_p=mu_p,
        var_p=var_p,
        mu_q=mu_q,
        var_q=var_q,
        eps=eps,
        reduction=reduction,
    )


def kl_multinomial(counts_p, probs_p, counts_q, probs_q, eps=1e-12, reduction=None):
    """
    KL divergence KL(Mult(m, probs_p) || Mult(m, probs_q)).

    The last axis indexes categories. The count arrays are used to infer the
    multinomial size m for each cell and to check that both samples have the
    same cell sizes. The multinomial coefficient cancels in the KL, so the
    result is

        m * sum_c probs_p[c] * log(probs_p[c] / probs_q[c]).

    Inputs can be NumPy arrays or torch tensors with broadcastable shapes.
    """

    m_p = _sum(counts_p, axis=-1)
    m_q = _sum(counts_q, axis=-1)
    if _is_torch_tensor(m_p):
        if not (m_p == m_q).all():
            raise ValueError("counts_p and counts_q must have the same cell totals.")
    elif not np.all(m_p == m_q):
        raise ValueError("counts_p and counts_q must have the same cell totals.")

    probs_p = _clamp_min(probs_p, eps)
    probs_q = _clamp_min(probs_q, eps)

    kl = m_p * _sum(probs_p * (_log(probs_p) - _log(probs_q)), axis=-1)
    return _reduce(kl, reduction)


def kl_categorical_probabilities(probs_p, probs_q, eps=1e-12, reduction=None):
    """Return probability-vector KL without multinomial cell-size scaling.

    The last axis indexes categories. This is the ECD convention used by the
    manuscript's Medicaid analysis:

        sum_c probs_p[c] * log(probs_p[c] / probs_q[c]).

    It is intentionally separate from :func:`kl_multinomial`, whose result is
    multiplied by the multinomial cell size.
    """

    probs_p = _clamp_min(probs_p, eps)
    probs_q = _clamp_min(probs_q, eps)
    kl = _sum(probs_p * (_log(probs_p) - _log(probs_q)), axis=-1)
    return _reduce(kl, reduction)
