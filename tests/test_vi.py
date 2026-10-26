import numpy as np
import torch

from efsc.log_joints import (
    exponential_log_joint_flat,
    gaussian_univariate_log_joint_flat,
    multinomial_log_joint_flat,
)
from efsc.preprocessing import (
    exponential_sufficient_statistics_flat,
    gaussian_sufficient_statistics_flat,
    multinomial_sufficient_statistics_flat,
)
from efsc.vi import fit_vi, make_negative_elbo


DTYPE = torch.float32
N = 3
T = 4
RANK = 1
D_ONE = (N + T) * (RANK + 1)
MASK = np.array(
    [
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [1, 1, 1, 0],
    ],
    dtype=bool,
)


def _fit_fixed_seed(log_joint, dim, seed, learning_rate, epochs):
    torch.manual_seed(seed)
    negative_elbo = make_negative_elbo(
        log_joint=log_joint,
        dim=dim,
        device="cpu",
        n_samples=2,
    )
    init_mean = 0.01 * torch.randn(dim, dtype=DTYPE)
    init_log_var = torch.full((dim,), -6.0, dtype=DTYPE)
    init_params = torch.cat([init_mean, init_log_var])

    params, loss_history, elapsed = fit_vi(
        negative_elbo=negative_elbo,
        init_params=init_params,
        lr=learning_rate,
        num_epochs=epochs,
        print_every=None,
    )
    return params, np.asarray(loss_history), elapsed


def _raw_eta_from_mean(mean, n=N, T_=T, rank=RANK):
    alpha, gamma, unit_flat, time_flat = torch.tensor_split(
        mean,
        [n, n + T_, n + T_ + n * rank],
    )
    unit_factors = unit_flat.reshape(n, rank)
    time_factors = time_flat.reshape(T_, rank)
    return (
        alpha[:, None]
        + gamma[None, :]
        + unit_factors @ time_factors.T
    )


def _assert_healthy_fit(params, losses, elapsed, epochs):
    assert params.shape[0] % 2 == 0
    assert len(losses) == epochs
    assert elapsed >= 0
    assert torch.isfinite(params).all()
    assert np.isfinite(losses).all()
    window = min(20, epochs // 3)
    assert losses[-window:].mean() < losses[:window].mean()


def test_make_negative_elbo_and_fit_vi_are_repeatable_with_fixed_seed():
    dim = 4
    prior = torch.distributions.Normal(
        torch.zeros(dim, dtype=DTYPE),
        torch.ones(dim, dtype=DTYPE),
    )

    def log_joint(theta):
        return prior.log_prob(theta).sum()

    first_params, first_losses, _ = _fit_fixed_seed(
        log_joint=log_joint,
        dim=dim,
        seed=31415,
        learning_rate=0.05,
        epochs=40,
    )
    second_params, second_losses, _ = _fit_fixed_seed(
        log_joint=log_joint,
        dim=dim,
        seed=31415,
        learning_rate=0.05,
        epochs=40,
    )

    torch.testing.assert_close(first_params, second_params, rtol=0, atol=0)
    np.testing.assert_array_equal(first_losses, second_losses)


def test_gaussian_vi_smoke_fit_is_finite_and_respects_variance_domain():
    values = np.linspace(-1.5, 2.0, N * T * 3, dtype=np.float32)
    y = values.reshape(N, T, 3)
    unit_idx, time_idx, m_vec, tx0, tx1, log_carrier = (
        gaussian_sufficient_statistics_flat(y=y, mask=MASK)
    )
    dim = 2 * D_ONE
    prior = torch.distributions.Normal(
        torch.zeros(dim, dtype=DTYPE),
        torch.full((dim,), 2.0, dtype=DTYPE),
    )

    def log_joint(theta):
        return gaussian_univariate_log_joint_flat(
            theta=theta,
            unit_idx=unit_idx,
            time_idx=time_idx,
            m_vec=m_vec,
            Tx0_vec=tx0,
            Tx1_vec=tx1,
            n=N,
            T=T,
            r=RANK,
            prior=prior,
            lhx=log_carrier,
        )

    epochs = 120
    params, losses, elapsed = _fit_fixed_seed(
        log_joint=log_joint,
        dim=dim,
        seed=101,
        learning_rate=0.1,
        epochs=epochs,
    )

    _assert_healthy_fit(params, losses, elapsed, epochs)
    posterior_mean = params[:dim]
    eta0 = _raw_eta_from_mean(posterior_mean[:D_ONE])
    eta1 = -torch.exp(_raw_eta_from_mean(posterior_mean[D_ONE:]))
    variance = -0.5 / eta1
    mean = -0.5 * eta0 / eta1
    assert torch.isfinite(mean).all()
    assert torch.isfinite(variance).all()
    assert torch.all(eta1 < 0)
    assert torch.all(variance > 0)


def test_exponential_vi_smoke_fit_is_finite_and_respects_rate_domain():
    values = np.linspace(0.2, 3.0, N * T * 3, dtype=np.float32)
    y = values.reshape(N, T, 3)
    unit_idx, time_idx, m_vec, tx_vec, carrier = (
        exponential_sufficient_statistics_flat(y=y, mask=MASK)
    )
    dim = D_ONE
    prior = torch.distributions.Normal(
        torch.zeros(dim, dtype=DTYPE),
        torch.full((dim,), 2.0, dtype=DTYPE),
    )

    def log_joint(theta):
        return exponential_log_joint_flat(
            theta=theta,
            unit_idx=unit_idx,
            time_idx=time_idx,
            m_vec=m_vec,
            Tx_vec=tx_vec,
            log_carrier_vec=carrier,
            n=N,
            T=T,
            r=RANK,
            prior=prior,
        )

    epochs = 140
    params, losses, elapsed = _fit_fixed_seed(
        log_joint=log_joint,
        dim=dim,
        seed=202,
        learning_rate=0.005,
        epochs=epochs,
    )

    _assert_healthy_fit(params, losses, elapsed, epochs)
    raw_eta = _raw_eta_from_mean(params[:dim])
    eta = -torch.exp(raw_eta)
    rate = -eta
    assert torch.isfinite(eta).all()
    assert torch.all(eta < 0)
    assert torch.all(rate > 0)


def test_multinomial_vi_smoke_fit_is_finite_and_returns_probabilities():
    counts = np.empty((N, T, 3), dtype=np.int64)
    for i in range(N):
        for j in range(T):
            counts[i, j] = [2 + i, 3 + j, 4 + ((i + j) % 2)]

    unit_idx, time_idx, m_vec, tx_vec, carrier = (
        multinomial_sufficient_statistics_flat(counts=counts, mask=MASK)
    )
    n_free_categories = counts.shape[2] - 1
    dim = n_free_categories * D_ONE
    prior = torch.distributions.Normal(
        torch.zeros(dim, dtype=DTYPE),
        torch.full((dim,), 2.0, dtype=DTYPE),
    )

    def log_joint(theta):
        return multinomial_log_joint_flat(
            theta=theta,
            unit_idx=unit_idx,
            time_idx=time_idx,
            m_vec=m_vec,
            Tx_vec=tx_vec,
            log_carrier_vec=carrier,
            n=N,
            T=T,
            r=RANK,
            prior=prior,
        )

    epochs = 140
    params, losses, elapsed = _fit_fixed_seed(
        log_joint=log_joint,
        dim=dim,
        seed=303,
        learning_rate=0.01,
        epochs=epochs,
    )

    _assert_healthy_fit(params, losses, elapsed, epochs)
    posterior_mean = params[:dim]
    eta = torch.stack(
        [
            _raw_eta_from_mean(block)
            for block in torch.split(posterior_mean, D_ONE)
        ],
        dim=2,
    )
    reference = torch.zeros((N, T, 1), dtype=DTYPE)
    probabilities = torch.softmax(torch.cat([eta, reference], dim=2), dim=2)
    assert torch.isfinite(probabilities).all()
    assert torch.all(probabilities > 0)
    torch.testing.assert_close(
        probabilities.sum(dim=2),
        torch.ones((N, T), dtype=DTYPE),
    )
