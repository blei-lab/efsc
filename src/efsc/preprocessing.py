"""Sufficient-statistic preprocessing utilities for EFSC panel models.

This module converts dense, ragged, and long-format panel observations into
the flattened cell indices, sizes, sufficient statistics, and carrier terms
consumed by the family-specific EFSC log-joint functions.

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
import torch


def _one_parameter_sufficient_statistics_flat(
    y,
    mask,
    statistic_fn,
    carrier_fn,
    m=None,
    device="cpu",
):
    """Prepare flattened cell statistics for a one-parameter EF model."""

    y_is_ragged = isinstance(y, np.ndarray) and y.dtype == object

    if y_is_ragged:
        n, T = y.shape
        Tx = torch.zeros((n, T), device=device)
        log_carrier = torch.zeros((n, T), device=device)
        m_panel = np.zeros((n, T), dtype=int)
        for i in range(n):
            for j in range(T):
                y_ij = torch.as_tensor(y[i, j], device=device, dtype=torch.float32)
                m_panel[i, j] = len(y_ij)
                Tx[i, j] = statistic_fn(y_ij)
                log_carrier[i, j] = carrier_fn(y_ij)
    else:
        y = torch.as_tensor(y, device=device, dtype=torch.float32)
        n, T, m_dense = y.shape
        Tx = statistic_fn(y, dim=2)
        log_carrier = carrier_fn(y, dim=2)
        if m is None:
            m = m_dense

    if m is None:
        m = m_panel

    mask_np = np.asarray(mask).astype(bool)
    unit_idx_np, time_idx_np = np.where(mask_np)

    unit_idx = torch.tensor(
        unit_idx_np,
        device=device,
        dtype=torch.long,
    )
    time_idx = torch.tensor(
        time_idx_np,
        device=device,
        dtype=torch.long,
    )

    Tx_vec = Tx[unit_idx, time_idx]
    log_carrier_vec = log_carrier[unit_idx, time_idx]

    m = np.asarray(m)
    if m.ndim == 0:
        m_vec = torch.full(
            (len(unit_idx),),
            fill_value=int(m),
            device=device,
            dtype=torch.float32,
        )
    else:
        if m.shape != (n, T):
            raise ValueError("m must be a scalar or have shape (n, T).")
        m_vec = torch.as_tensor(
            m[unit_idx_np, time_idx_np],
            device=device,
            dtype=torch.float32,
        )

    return (
        unit_idx,
        time_idx,
        m_vec,
        Tx_vec,
        log_carrier_vec,
    )


def gaussian_sufficient_statistics_flat(y, mask, m=None, device="cpu"):
    """
    Prepare flattened sufficient statistics for the Gaussian EF log-joint.

    Parameters
    ----------
    y : torch.Tensor or np.ndarray
        Dense tensor of shape (n, T, m) or ragged object array of shape (n, T).
    mask : np.ndarray or torch.Tensor
        Binary observation mask of shape (n, T).
    m : int or np.ndarray, optional
        Number of observations per cell. If omitted, inferred from y.
    device : torch.device
        Torch device.

    Returns
    -------
    unit_idx : torch.Tensor
    time_idx : torch.Tensor
    m_vec : torch.Tensor
    Tx0_vec : torch.Tensor
    Tx1_vec : torch.Tensor
    lhx : torch.Tensor
    """

    y_is_ragged = isinstance(y, np.ndarray) and y.dtype == object

    if y_is_ragged:
        n, T = y.shape
        Tx = torch.zeros((n, T, 2), device=device)
        m_panel = np.zeros((n, T), dtype=int)
        for i in range(n):
            for j in range(T):
                y_ij = torch.as_tensor(y[i, j], device=device, dtype=torch.float32)
                m_panel[i, j] = len(y_ij)
                Tx[i, j, 0] = torch.sum(y_ij)
                Tx[i, j, 1] = torch.sum(y_ij**2)
    else:
        y = torch.as_tensor(y, device=device, dtype=torch.float32)
        n, T, m_dense = y.shape
        Tx = torch.zeros((n, T, 2), device=device)
        Tx[:, :, 0] = torch.sum(y, dim=2)
        Tx[:, :, 1] = torch.sum(y**2, dim=2)
        if m is None:
            m = m_dense

    if m is None:
        m = m_panel

    # Convert mask to boolean numpy array
    mask_np = np.asarray(mask).astype(bool)

    # Indices of observed cells
    unit_idx_np, time_idx_np = np.where(mask_np)

    unit_idx = torch.tensor(
        unit_idx_np,
        device=device,
        dtype=torch.long,
    )

    time_idx = torch.tensor(
        time_idx_np,
        device=device,
        dtype=torch.long,
    )

    # Flatten sufficient statistics
    Tx0_vec = Tx[unit_idx, time_idx, 0]
    Tx1_vec = Tx[unit_idx, time_idx, 1]

    # Number of observations per cell
    m = np.asarray(m)
    if m.ndim == 0:
        m_vec = torch.full(
            (len(unit_idx),),
            fill_value=int(m),
            device=device,
            dtype=torch.float32,
        )
    else:
        if m.shape != (n, T):
            raise ValueError("m must be a scalar or have shape (n, T).")
        m_vec = torch.as_tensor(
            m[unit_idx_np, time_idx_np],
            device=device,
            dtype=torch.float32,
        )

    # Gaussian carrier/base measure term
    pi2 = torch.tensor(2 * np.pi, device=device)
    lhx = torch.log(1 / torch.sqrt(pi2))

    return (
        unit_idx,
        time_idx,
        m_vec,
        Tx0_vec,
        Tx1_vec,
        lhx,
    )


def poisson_sufficient_statistics_flat(y, mask, m=None, device="cpu"):
    """
    Prepare flattened sufficient statistics for the Poisson EF log-joint.

    Parameters
    ----------
    y : torch.Tensor or np.ndarray
        Dense tensor of shape (n, T, m) or ragged object array of shape (n, T).
    mask : np.ndarray or torch.Tensor
        Binary observation mask of shape (n, T).
    m : int or np.ndarray, optional
        Number of observations per cell. If omitted, inferred from y.
    device : torch.device
        Torch device.

    Returns
    -------
    unit_idx : torch.Tensor
    time_idx : torch.Tensor
    m_vec : torch.Tensor
    Tx_vec : torch.Tensor
    log_carrier_vec : torch.Tensor
    """

    return _one_parameter_sufficient_statistics_flat(
        y=y,
        mask=mask,
        statistic_fn=lambda y, dim=None: torch.sum(y, dim=dim),
        carrier_fn=lambda y, dim=None: -torch.sum(torch.lgamma(1 + y), dim=dim),
        m=m,
        device=device,
    )


def bernoulli_sufficient_statistics_flat(y, mask, m=None, device="cpu"):
    """
    Prepare flattened sufficient statistics for the Bernoulli EF log-joint.
    """

    return _one_parameter_sufficient_statistics_flat(
        y=y,
        mask=mask,
        statistic_fn=lambda y, dim=None: torch.sum(y, dim=dim),
        carrier_fn=lambda y, dim=None: torch.zeros_like(torch.sum(y, dim=dim)),
        m=m,
        device=device,
    )


def exponential_sufficient_statistics_flat(y, mask, m=None, device="cpu"):
    """
    Prepare flattened sufficient statistics for the Exponential EF log-joint.
    """

    return _one_parameter_sufficient_statistics_flat(
        y=y,
        mask=mask,
        statistic_fn=lambda y, dim=None: torch.sum(y, dim=dim),
        carrier_fn=lambda y, dim=None: torch.zeros_like(torch.sum(y, dim=dim)),
        m=m,
        device=device,
    )


def laplace_sufficient_statistics_flat(y, mask, mu, m=None, device="cpu"):
    """
    Prepare flattened sufficient statistics for the Laplace-known-location EF log-joint.
    """

    return _one_parameter_sufficient_statistics_flat(
        y=y,
        mask=mask,
        statistic_fn=lambda y, dim=None: torch.sum(torch.abs(y - mu), dim=dim),
        carrier_fn=lambda y, dim=None: torch.zeros_like(torch.sum(y, dim=dim)),
        m=m,
        device=device,
    )


def chi_square_sufficient_statistics_flat(y, mask, m=None, device="cpu"):
    """
    Prepare flattened sufficient statistics for the Chi-square EF log-joint.
    """

    return _one_parameter_sufficient_statistics_flat(
        y=y,
        mask=mask,
        statistic_fn=lambda y, dim=None: torch.sum(torch.log(y), dim=dim),
        carrier_fn=lambda y, dim=None: -torch.sum(y / 2, dim=dim),
        m=m,
        device=device,
    )


def gaussian_known_sigma_sufficient_statistics_flat(
    y,
    mask,
    sigma,
    m=None,
    device="cpu",
):
    """
    Prepare flattened sufficient statistics for the Gaussian-known-scale EF log-joint.
    """

    log_scale = torch.as_tensor(2 * np.pi * sigma**2, device=device, dtype=torch.float32)

    return _one_parameter_sufficient_statistics_flat(
        y=y,
        mask=mask,
        statistic_fn=lambda y, dim=None: torch.sum(y / sigma, dim=dim),
        carrier_fn=lambda y, dim=None: -torch.sum(
            (y / sigma) ** 2 + torch.log(log_scale),
            dim=dim,
        ) / 2,
        m=m,
        device=device,
    )


def multinomial_sufficient_statistics_flat(counts, mask, device="cpu"):
    """
    Prepare flattened sufficient statistics for the Multinomial EF log-joint.

    The last category is treated as the reference category. If there are C
    categories, the sufficient statistic has C - 1 entries per observed cell.
    """

    counts = torch.as_tensor(counts, device=device, dtype=torch.float32)
    if counts.ndim != 3:
        raise ValueError("counts must have shape (n, T, C).")

    n, T, C = counts.shape
    if C < 2:
        raise ValueError("counts must contain at least two categories.")
    if torch.any(counts < 0):
        raise ValueError("counts must be nonnegative.")
    if not torch.allclose(counts, torch.round(counts)):
        raise ValueError("counts must contain integer-valued category counts.")

    mask_np = np.asarray(mask).astype(bool)
    if mask_np.shape != (n, T):
        raise ValueError("mask must have shape (n, T).")

    unit_idx_np, time_idx_np = np.where(mask_np)
    unit_idx = torch.tensor(unit_idx_np, device=device, dtype=torch.long)
    time_idx = torch.tensor(time_idx_np, device=device, dtype=torch.long)

    counts_vec = counts[unit_idx, time_idx]
    m_vec = torch.sum(counts_vec, dim=1)
    Tx_vec = counts_vec[:, : C - 1]
    log_carrier_vec = (
        torch.lgamma(1 + m_vec)
        - torch.sum(torch.lgamma(1 + counts_vec), dim=1)
    )

    return (
        unit_idx,
        time_idx,
        m_vec,
        Tx_vec,
        log_carrier_vec,
    )
