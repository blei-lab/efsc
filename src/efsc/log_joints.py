"""Exponential-family log-joint functions for EFSC factorization models.

This module evaluates family-specific log-joint densities from flattened panel
sufficient statistics. It provides the likelihood and prior calculations used
by the shared mean-field variational fitting engine.

Author:
    Hector Rodriguez-Deniz

Affiliation:
    Data Science Institute, Columbia University

Date:
    August 2026

License:
    MIT
"""

import torch

from ._families import constrain_raw_component, one_parameter_log_partition


def _one_parameter_log_joint_flat(
    theta,
    unit_idx,
    time_idx,
    m_vec,
    Tx_vec,
    log_carrier_vec,
    n,
    T,
    r,
    prior,
    eta_transform,
    log_partition,
):
    """Evaluate a one-parameter EF log-joint over flattened panel cells."""

    alpha, gamma, t, b = torch.tensor_split(
        theta, [n, n + T, n + T + n * r]
    )
    theta_latent = t.reshape(n, r)
    beta = b.reshape(T, r)

    eta_raw = (
        alpha[unit_idx]
        + gamma[time_idx]
        + torch.sum(theta_latent[unit_idx] * beta[time_idx], dim=1)
    )
    eta = eta_transform(eta_raw)
    A = log_partition(eta)

    ll = torch.sum(
        log_carrier_vec
        + Tx_vec * eta
        - m_vec * A
    )

    lp = prior.log_prob(theta).sum()

    return ll + lp


def gaussian_univariate_log_joint_flat(
    theta,
    unit_idx,
    time_idx,
    m_vec,
    Tx0_vec,
    Tx1_vec,
    n,
    T,
    r,
    prior,
    lhx,
):
    """
    Log joint for univariate Gaussian EF model using flattened observed cells.
    """

    theta_eta0, theta_eta1 = torch.tensor_split(theta, 2)

    alpha0, gamma0, t0, b0 = torch.tensor_split(
        theta_eta0, [n, n + T, n + T + n * r]
    )
    theta0 = t0.reshape(n, r)
    beta0 = b0.reshape(T, r)

    alpha1, gamma1, t1, b1 = torch.tensor_split(
        theta_eta1, [n, n + T, n + T + n * r]
    )
    theta1 = t1.reshape(n, r)
    beta1 = b1.reshape(T, r)

    eta0 = (
        alpha0[unit_idx]
        + gamma0[time_idx]
        + torch.sum(theta0[unit_idx] * beta0[time_idx], dim=1)
    )

    eta1_raw = (
        alpha1[unit_idx]
        + gamma1[time_idx]
        + torch.sum(theta1[unit_idx] * beta1[time_idx], dim=1)
    )

    eta1 = constrain_raw_component(
        eta1_raw,
        family="gaussian",
        component=1,
    )

    A = (
        -0.25 * eta0.pow(2) / eta1
        - 0.5 * torch.log(-2 * eta1)
    )

    ll = torch.sum(
        m_vec * (lhx - A)
        + Tx0_vec * eta0
        + Tx1_vec * eta1
    )

    lp = prior.log_prob(theta).sum()

    return ll + lp


def poisson_log_joint_flat(
    theta,
    unit_idx,
    time_idx,
    m_vec,
    Tx_vec,
    log_carrier_vec,
    n,
    T,
    r,
    prior,
):
    """
    Log joint for Poisson EF model using flattened observed cells.
    """

    return _one_parameter_log_joint_flat(
        theta=theta,
        unit_idx=unit_idx,
        time_idx=time_idx,
        m_vec=m_vec,
        Tx_vec=Tx_vec,
        log_carrier_vec=log_carrier_vec,
        n=n,
        T=T,
        r=r,
        prior=prior,
        eta_transform=lambda eta: constrain_raw_component(
            eta,
            family="poisson",
        ),
        log_partition=lambda eta: one_parameter_log_partition("poisson", eta),
    )


def bernoulli_log_joint_flat(
    theta,
    unit_idx,
    time_idx,
    m_vec,
    Tx_vec,
    log_carrier_vec,
    n,
    T,
    r,
    prior,
):
    """
    Log joint for Bernoulli EF model using flattened observed cells.
    """

    return _one_parameter_log_joint_flat(
        theta=theta,
        unit_idx=unit_idx,
        time_idx=time_idx,
        m_vec=m_vec,
        Tx_vec=Tx_vec,
        log_carrier_vec=log_carrier_vec,
        n=n,
        T=T,
        r=r,
        prior=prior,
        eta_transform=lambda eta: constrain_raw_component(
            eta,
            family="bernoulli",
        ),
        log_partition=lambda eta: one_parameter_log_partition("bernoulli", eta),
    )


def exponential_log_joint_flat(
    theta,
    unit_idx,
    time_idx,
    m_vec,
    Tx_vec,
    log_carrier_vec,
    n,
    T,
    r,
    prior,
):
    """
    Log joint for Exponential EF model using flattened observed cells.
    """

    return _one_parameter_log_joint_flat(
        theta=theta,
        unit_idx=unit_idx,
        time_idx=time_idx,
        m_vec=m_vec,
        Tx_vec=Tx_vec,
        log_carrier_vec=log_carrier_vec,
        n=n,
        T=T,
        r=r,
        prior=prior,
        eta_transform=lambda eta: constrain_raw_component(
            eta,
            family="exponential",
        ),
        log_partition=lambda eta: one_parameter_log_partition("exponential", eta),
    )


def laplace_log_joint_flat(
    theta,
    unit_idx,
    time_idx,
    m_vec,
    Tx_vec,
    log_carrier_vec,
    n,
    T,
    r,
    prior,
):
    """
    Log joint for Laplace-known-location EF model using flattened observed cells.
    """

    return _one_parameter_log_joint_flat(
        theta=theta,
        unit_idx=unit_idx,
        time_idx=time_idx,
        m_vec=m_vec,
        Tx_vec=Tx_vec,
        log_carrier_vec=log_carrier_vec,
        n=n,
        T=T,
        r=r,
        prior=prior,
        eta_transform=lambda eta: constrain_raw_component(
            eta,
            family="laplace_known_mu",
        ),
        log_partition=lambda eta: one_parameter_log_partition(
            "laplace_known_mu",
            eta,
        ),
    )


def laplace_known_mu_log_joint_flat(*args, **kwargs):
    """
    Alias for Laplace-known-location EF log joint.
    """

    return laplace_log_joint_flat(*args, **kwargs)


def chi_square_log_joint_flat(
    theta,
    unit_idx,
    time_idx,
    m_vec,
    Tx_vec,
    log_carrier_vec,
    n,
    T,
    r,
    prior,
):
    """
    Log joint for Chi-square EF model using flattened observed cells.
    """

    return _one_parameter_log_joint_flat(
        theta=theta,
        unit_idx=unit_idx,
        time_idx=time_idx,
        m_vec=m_vec,
        Tx_vec=Tx_vec,
        log_carrier_vec=log_carrier_vec,
        n=n,
        T=T,
        r=r,
        prior=prior,
        eta_transform=lambda eta: constrain_raw_component(
            eta,
            family="chi_square",
        ),
        log_partition=lambda eta: one_parameter_log_partition("chi_square", eta),
    )


def gaussian_known_sigma_log_joint_flat(
    theta,
    unit_idx,
    time_idx,
    m_vec,
    Tx_vec,
    log_carrier_vec,
    n,
    T,
    r,
    prior,
):
    """
    Log joint for Gaussian-known-scale EF model using flattened observed cells.
    """

    return _one_parameter_log_joint_flat(
        theta=theta,
        unit_idx=unit_idx,
        time_idx=time_idx,
        m_vec=m_vec,
        Tx_vec=Tx_vec,
        log_carrier_vec=log_carrier_vec,
        n=n,
        T=T,
        r=r,
        prior=prior,
        eta_transform=lambda eta: constrain_raw_component(
            eta,
            family="gaussian_known_sigma",
        ),
        log_partition=lambda eta: one_parameter_log_partition(
            "gaussian_known_sigma",
            eta,
        ),
    )


def multinomial_log_joint_flat(
    theta,
    unit_idx,
    time_idx,
    m_vec,
    Tx_vec,
    log_carrier_vec,
    n,
    T,
    r,
    prior,
):
    """
    Log joint for Multinomial EF model using flattened observed cells.

    The last category is the reference category. With C categories, Tx_vec has
    C - 1 columns and theta contains one latent PMF block per free logit.
    """

    p = Tx_vec.shape[1]
    d_latent = (n + T) * (r + 1)
    if theta.numel() != p * d_latent:
        raise ValueError(
            "theta has incompatible length for multinomial log-joint."
        )

    eta_cols = []
    for theta_k in torch.split(theta, d_latent):
        alpha, gamma, t, b = torch.tensor_split(
            theta_k,
            [n, n + T, n + T + n * r],
        )
        theta_latent = t.reshape(n, r)
        beta = b.reshape(T, r)
        eta_k = (
            alpha[unit_idx]
            + gamma[time_idx]
            + torch.sum(theta_latent[unit_idx] * beta[time_idx], dim=1)
        )
        eta_cols.append(eta_k)

    eta = torch.stack(eta_cols, dim=1)
    reference_eta = torch.zeros(
        (eta.shape[0], 1),
        device=eta.device,
        dtype=eta.dtype,
    )
    A = torch.logsumexp(torch.cat([eta, reference_eta], dim=1), dim=1)

    ll = torch.sum(
        log_carrier_vec
        + torch.sum(Tx_vec * eta, dim=1)
        - m_vec * A
    )

    lp = prior.log_prob(theta).sum()

    return ll + lp
