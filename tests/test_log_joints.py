import pytest
import torch

from efsc.log_joints import (
    bernoulli_log_joint_flat,
    chi_square_log_joint_flat,
    exponential_log_joint_flat,
    gaussian_known_sigma_log_joint_flat,
    gaussian_univariate_log_joint_flat,
    laplace_known_mu_log_joint_flat,
    laplace_log_joint_flat,
    multinomial_log_joint_flat,
    poisson_log_joint_flat,
)


DTYPE = torch.float64


def _raw_eta(theta, unit_idx, time_idx, n=2, T=2, r=1):
    alpha, gamma, unit_factors, time_factors = torch.tensor_split(
        theta,
        [n, n + T, n + T + n * r],
    )
    unit_factors = unit_factors.reshape(n, r)
    time_factors = time_factors.reshape(T, r)
    return (
        alpha[unit_idx]
        + gamma[time_idx]
        + torch.sum(unit_factors[unit_idx] * time_factors[time_idx], dim=1)
    )


ONE_PARAMETER_CASES = [
    (
        "poisson",
        poisson_log_joint_flat,
        lambda raw: raw,
        lambda eta: torch.exp(eta),
    ),
    (
        "bernoulli",
        bernoulli_log_joint_flat,
        lambda raw: raw,
        lambda eta: torch.logaddexp(torch.zeros_like(eta), eta),
    ),
    (
        "exponential",
        exponential_log_joint_flat,
        lambda raw: -torch.exp(raw),
        lambda eta: -torch.log(-eta),
    ),
    (
        "laplace",
        laplace_log_joint_flat,
        lambda raw: -torch.exp(raw),
        lambda eta: torch.log(-2.0 / eta),
    ),
    (
        "chi_square",
        chi_square_log_joint_flat,
        lambda raw: torch.exp(raw) - 1,
        lambda eta: torch.lgamma(1 + eta)
        + (eta + 1) * torch.log(torch.tensor(2.0, dtype=DTYPE)),
    ),
    (
        "gaussian_known_sigma",
        gaussian_known_sigma_log_joint_flat,
        lambda raw: raw,
        lambda eta: 0.5 * eta**2,
    ),
]


@pytest.mark.parametrize(
    ("name", "log_joint", "transform", "log_partition"),
    ONE_PARAMETER_CASES,
)
def test_one_parameter_log_joints_match_explicit_formula(
    name,
    log_joint,
    transform,
    log_partition,
):
    del name
    n, T, r = 2, 2, 1
    theta = torch.tensor(
        [0.2, -0.1, 0.3, -0.2, 0.4, -0.5, 0.6, 0.25],
        dtype=DTYPE,
    )
    unit_idx = torch.tensor([0, 1, 1])
    time_idx = torch.tensor([0, 0, 1])
    m_vec = torch.tensor([2.0, 3.0, 1.0], dtype=DTYPE)
    tx_vec = torch.tensor([1.25, 0.75, 2.5], dtype=DTYPE)
    carrier = torch.tensor([-0.2, 0.0, -0.4], dtype=DTYPE)
    prior = torch.distributions.Normal(
        torch.zeros(theta.numel(), dtype=DTYPE),
        torch.full((theta.numel(),), 1.7, dtype=DTYPE),
    )

    actual = log_joint(
        theta=theta,
        unit_idx=unit_idx,
        time_idx=time_idx,
        m_vec=m_vec,
        Tx_vec=tx_vec,
        log_carrier_vec=carrier,
        n=n,
        T=T,
        r=r,
        prior=prior,
    )

    eta = transform(_raw_eta(theta, unit_idx, time_idx, n=n, T=T, r=r))
    expected = torch.sum(carrier + tx_vec * eta - m_vec * log_partition(eta))
    expected = expected + prior.log_prob(theta).sum()

    torch.testing.assert_close(actual, expected)


def test_laplace_known_mu_alias_matches_laplace_log_joint():
    kwargs = {
        "theta": torch.zeros(8, dtype=DTYPE),
        "unit_idx": torch.tensor([0]),
        "time_idx": torch.tensor([0]),
        "m_vec": torch.tensor([2.0], dtype=DTYPE),
        "Tx_vec": torch.tensor([1.5], dtype=DTYPE),
        "log_carrier_vec": torch.tensor([0.0], dtype=DTYPE),
        "n": 2,
        "T": 2,
        "r": 1,
        "prior": torch.distributions.Normal(
            torch.zeros(8, dtype=DTYPE),
            torch.ones(8, dtype=DTYPE),
        ),
    }

    torch.testing.assert_close(
        laplace_known_mu_log_joint_flat(**kwargs),
        laplace_log_joint_flat(**kwargs),
    )


def test_gaussian_log_joint_matches_explicit_formula():
    n, T, r = 2, 2, 1
    block0 = torch.tensor(
        [0.2, -0.1, 0.3, -0.2, 0.4, -0.5, 0.6, 0.25],
        dtype=DTYPE,
    )
    block1 = torch.tensor(
        [-0.3, 0.1, 0.2, -0.4, 0.15, -0.2, 0.3, -0.1],
        dtype=DTYPE,
    )
    theta = torch.cat([block0, block1])
    unit_idx = torch.tensor([0, 1, 1])
    time_idx = torch.tensor([0, 0, 1])
    m_vec = torch.tensor([2.0, 3.0, 1.0], dtype=DTYPE)
    tx0 = torch.tensor([1.0, -0.5, 2.0], dtype=DTYPE)
    tx1 = torch.tensor([2.0, 1.25, 4.0], dtype=DTYPE)
    log_carrier = -0.5 * torch.log(torch.tensor(2 * torch.pi, dtype=DTYPE))
    prior = torch.distributions.Normal(
        torch.zeros(theta.numel(), dtype=DTYPE),
        torch.full((theta.numel(),), 1.3, dtype=DTYPE),
    )

    actual = gaussian_univariate_log_joint_flat(
        theta=theta,
        unit_idx=unit_idx,
        time_idx=time_idx,
        m_vec=m_vec,
        Tx0_vec=tx0,
        Tx1_vec=tx1,
        n=n,
        T=T,
        r=r,
        prior=prior,
        lhx=log_carrier,
    )

    eta0 = _raw_eta(block0, unit_idx, time_idx, n=n, T=T, r=r)
    eta1 = -torch.exp(_raw_eta(block1, unit_idx, time_idx, n=n, T=T, r=r))
    partition = -0.25 * eta0**2 / eta1 - 0.5 * torch.log(-2 * eta1)
    expected = torch.sum(
        m_vec * (log_carrier - partition) + tx0 * eta0 + tx1 * eta1
    )
    expected = expected + prior.log_prob(theta).sum()

    torch.testing.assert_close(actual, expected)


def test_multinomial_log_joint_matches_reference_logit_formula():
    n, T, r, p = 2, 2, 1, 2
    block0 = torch.tensor(
        [0.2, -0.1, 0.3, -0.2, 0.4, -0.5, 0.6, 0.25],
        dtype=DTYPE,
    )
    block1 = torch.tensor(
        [-0.3, 0.1, 0.2, -0.4, 0.15, -0.2, 0.3, -0.1],
        dtype=DTYPE,
    )
    theta = torch.cat([block0, block1])
    unit_idx = torch.tensor([0, 1, 1])
    time_idx = torch.tensor([0, 0, 1])
    m_vec = torch.tensor([10.0, 5.0, 8.0], dtype=DTYPE)
    tx_vec = torch.tensor([[2.0, 3.0], [1.0, 0.0], [3.0, 3.0]], dtype=DTYPE)
    carrier = torch.tensor([7.0, 2.0, 4.0], dtype=DTYPE)
    prior = torch.distributions.Normal(
        torch.zeros(theta.numel(), dtype=DTYPE),
        torch.full((theta.numel(),), 1.3, dtype=DTYPE),
    )

    actual = multinomial_log_joint_flat(
        theta=theta,
        unit_idx=unit_idx,
        time_idx=time_idx,
        m_vec=m_vec,
        Tx_vec=tx_vec,
        log_carrier_vec=carrier,
        n=n,
        T=T,
        r=r,
        prior=prior,
    )

    eta = torch.stack(
        [
            _raw_eta(block0, unit_idx, time_idx, n=n, T=T, r=r),
            _raw_eta(block1, unit_idx, time_idx, n=n, T=T, r=r),
        ],
        dim=1,
    )
    logits = torch.cat([eta, torch.zeros((eta.shape[0], 1), dtype=DTYPE)], dim=1)
    expected = torch.sum(
        carrier + torch.sum(tx_vec * eta, dim=1) - m_vec * torch.logsumexp(logits, dim=1)
    )
    expected = expected + prior.log_prob(theta).sum()

    torch.testing.assert_close(actual, expected)


def test_multinomial_rejects_incompatible_latent_vector_length():
    with pytest.raises(ValueError, match="incompatible length"):
        multinomial_log_joint_flat(
            theta=torch.zeros(7),
            unit_idx=torch.tensor([0]),
            time_idx=torch.tensor([0]),
            m_vec=torch.tensor([1.0]),
            Tx_vec=torch.tensor([[1.0, 0.0]]),
            log_carrier_vec=torch.tensor([0.0]),
            n=2,
            T=2,
            r=1,
            prior=torch.distributions.Normal(torch.zeros(7), torch.ones(7)),
        )
