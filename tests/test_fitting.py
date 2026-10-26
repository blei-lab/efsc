import numpy as np
import pandas as pd
import pytest
import torch

from efsc.fitting import (
    InferenceConfig,
    fit_factorization,
    prepare_sufficient_statistics,
    reconstruct_natural_parameters,
)
from efsc.log_joints import (
    exponential_log_joint_flat,
    gaussian_univariate_log_joint_flat,
    multinomial_log_joint_flat,
)
from efsc.panel import prepare_panel_data
from efsc.preprocessing import (
    exponential_sufficient_statistics_flat,
    gaussian_sufficient_statistics_flat,
    multinomial_sufficient_statistics_flat,
)
from efsc.vi import fit_vi, make_negative_elbo


N = 2
T = 3
RANK = 1
MASK = np.array([[True, True, False], [True, True, True]])


def _frame(values):
    rows = []
    for i, unit in enumerate(["A", "B"]):
        for j, time in enumerate([1, 2, 3]):
            for value in values[i][j]:
                rows.append({"unit": unit, "time": time, "y": value})
    return pd.DataFrame(rows)


def _panel(family, **kwargs):
    if family == "multinomial":
        values = [
            [["a", "a", "b"], ["b", "c"], ["a", "c", "c"]],
            [["a", "b", "c"], ["b", "b", "c"], ["a", "a", "c"]],
        ]
    elif family == "bernoulli":
        values = [
            [[0, 1, 1], [0, 1], [1, 1, 0]],
            [[0, 0, 1], [1, 0, 1], [1, 1, 0]],
        ]
    elif family == "poisson":
        values = [
            [[0, 1, 2], [1, 3], [2, 2, 4]],
            [[1, 2, 3], [0, 1, 1], [2, 3, 5]],
        ]
    elif family in {"exponential", "chi_square"}:
        values = [
            [[0.5, 1.0, 2.0], [0.8, 1.2], [1.1, 1.5, 2.5]],
            [[0.7, 1.4, 2.1], [0.4, 0.9, 1.8], [1.2, 1.7, 2.2]],
        ]
    else:
        values = [
            [[-1.0, 0.5, 1.5], [0.2, 0.8], [-0.5, 1.0, 2.0]],
            [[-0.7, 0.4, 1.8], [0.1, 0.9, 1.4], [-0.2, 1.1, 2.1]],
        ]

    options = dict(kwargs)
    if family == "multinomial":
        options["categories"] = ["a", "b", "c"]
    if family == "gaussian_known_sigma":
        options["sigma"] = 1.5
    if family == "laplace_known_mu":
        options["mu"] = 0.25
    return prepare_panel_data(
        _frame(values),
        outcome="y",
        unit="unit",
        time="time",
        family=family,
        **options,
    )


def _direct_fit(log_joint, dim, config, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    objective = make_negative_elbo(
        log_joint=log_joint,
        dim=dim,
        device="cpu",
        n_samples=config.elbo_samples,
    )
    init_mean = config.init_scale * torch.randn(dim)
    init_log_variance = torch.full((dim,), config.init_log_variance)
    params = torch.cat([init_mean, init_log_variance])
    return fit_vi(
        negative_elbo=objective,
        init_params=params,
        lr=config.learning_rate,
        num_epochs=config.epochs,
        print_every=None,
    )


def test_inference_configuration_matches_approved_defaults():
    config = InferenceConfig()
    assert config.epochs == 2000
    assert config.learning_rate is None
    assert config.elbo_samples == 2
    assert config.prior_scale == 2.0
    assert config.init_scale == 0.01
    assert config.init_log_variance == -6.0
    assert config.device == "cpu"
    assert config.print_every is None


def test_prepared_gaussian_statistics_match_existing_preprocessor():
    panel = _panel("gaussian")
    prepared = prepare_sufficient_statistics(panel, MASK)
    direct = gaussian_sufficient_statistics_flat(panel.observations, MASK)

    torch.testing.assert_close(prepared.unit_index, direct[0])
    torch.testing.assert_close(prepared.time_index, direct[1])
    torch.testing.assert_close(prepared.cell_size, direct[2])
    torch.testing.assert_close(prepared.statistics[:, 0], direct[3])
    torch.testing.assert_close(prepared.statistics[:, 1], direct[4])
    torch.testing.assert_close(prepared.log_carrier, direct[5])


def test_prepared_one_parameter_statistics_match_existing_preprocessor():
    panel = _panel("exponential")
    prepared = prepare_sufficient_statistics(panel, MASK)
    direct = exponential_sufficient_statistics_flat(panel.observations, MASK)

    torch.testing.assert_close(prepared.unit_index, direct[0])
    torch.testing.assert_close(prepared.time_index, direct[1])
    torch.testing.assert_close(prepared.cell_size, direct[2])
    torch.testing.assert_close(prepared.statistics[:, 0], direct[3])
    torch.testing.assert_close(prepared.log_carrier, direct[4])


def test_prepared_multinomial_statistics_match_existing_preprocessor():
    panel = _panel("multinomial")
    prepared = prepare_sufficient_statistics(panel, MASK)
    counts = np.array(
        [
            [[2, 1, 0], [0, 1, 1], [1, 0, 2]],
            [[1, 1, 1], [0, 2, 1], [2, 0, 1]],
        ]
    )
    direct = multinomial_sufficient_statistics_flat(counts, MASK)

    torch.testing.assert_close(prepared.unit_index, direct[0])
    torch.testing.assert_close(prepared.time_index, direct[1])
    torch.testing.assert_close(prepared.cell_size, direct[2])
    torch.testing.assert_close(prepared.statistics, direct[3])
    torch.testing.assert_close(prepared.log_carrier, direct[4])


@pytest.mark.parametrize(
    ("family", "blocks", "domain_check"),
    [
        ("gaussian", 2, lambda eta: torch.all(eta[..., 1] < 0)),
        ("gaussian_known_sigma", 1, lambda eta: torch.isfinite(eta).all()),
        ("poisson", 1, lambda eta: torch.isfinite(eta).all()),
        ("bernoulli", 1, lambda eta: torch.isfinite(eta).all()),
        ("exponential", 1, lambda eta: torch.all(eta < 0)),
        ("laplace_known_mu", 1, lambda eta: torch.all(eta < 0)),
        ("chi_square", 1, lambda eta: torch.all(eta > -1)),
        ("multinomial", 2, lambda eta: torch.isfinite(eta).all()),
    ],
)
def test_reconstruction_uses_separate_blocks_and_family_domains(
    family,
    blocks,
    domain_check,
):
    block_size = (N + T) * (RANK + 1)
    posterior_mean = torch.linspace(-0.4, 0.5, blocks * block_size)
    eta = reconstruct_natural_parameters(
        posterior_mean,
        family=family,
        n=N,
        T=T,
        rank=RANK,
        n_categories=3 if family == "multinomial" else None,
    )

    assert eta.shape == (N, T, blocks)
    assert domain_check(eta)
    if blocks == 2:
        assert not torch.equal(eta[..., 0], eta[..., 1])


@pytest.mark.parametrize("rank", [0, -1, 1.5, True, 4])
def test_fit_rejects_invalid_rank(rank):
    with pytest.raises(ValueError, match="rank"):
        fit_factorization(
            _panel("gaussian"),
            MASK,
            rank=rank,
            config=InferenceConfig(epochs=1),
        )


def test_fit_rejects_masked_missing_cells():
    panel = _panel("gaussian")
    bad_mask = MASK.copy()
    panel.observed_mask[0, 2] = False
    bad_mask[0, 2] = True

    with pytest.raises(ValueError, match="missing cells"):
        fit_factorization(
            panel,
            bad_mask,
            rank=1,
            config=InferenceConfig(epochs=1),
        )


def test_gaussian_engine_is_exactly_equivalent_to_direct_wiring():
    panel = _panel("gaussian")
    config = InferenceConfig(epochs=30, learning_rate=0.03)
    seed = 1701
    fit = fit_factorization(panel, MASK, rank=RANK, config=config, seed=seed)

    stats = gaussian_sufficient_statistics_flat(panel.observations, MASK)
    dim = 2 * (N + T) * (RANK + 1)
    prior = torch.distributions.Normal(torch.zeros(dim), torch.full((dim,), 2.0))

    def log_joint(theta):
        return gaussian_univariate_log_joint_flat(
            theta=theta,
            unit_idx=stats[0],
            time_idx=stats[1],
            m_vec=stats[2],
            Tx0_vec=stats[3],
            Tx1_vec=stats[4],
            n=N,
            T=T,
            r=RANK,
            prior=prior,
            lhx=stats[5],
        )

    direct, losses, _ = _direct_fit(log_joint, dim, config, seed)
    torch.testing.assert_close(fit.variational_parameters, direct, rtol=0, atol=0)
    np.testing.assert_array_equal(fit.loss_history, losses)


def test_one_parameter_engine_is_exactly_equivalent_to_direct_wiring():
    panel = _panel("exponential")
    config = InferenceConfig(epochs=30, learning_rate=0.005)
    seed = 1702
    fit = fit_factorization(panel, MASK, rank=RANK, config=config, seed=seed)

    stats = exponential_sufficient_statistics_flat(panel.observations, MASK)
    dim = (N + T) * (RANK + 1)
    prior = torch.distributions.Normal(torch.zeros(dim), torch.full((dim,), 2.0))

    def log_joint(theta):
        return exponential_log_joint_flat(
            theta=theta,
            unit_idx=stats[0],
            time_idx=stats[1],
            m_vec=stats[2],
            Tx_vec=stats[3],
            log_carrier_vec=stats[4],
            n=N,
            T=T,
            r=RANK,
            prior=prior,
        )

    direct, losses, _ = _direct_fit(log_joint, dim, config, seed)
    torch.testing.assert_close(fit.variational_parameters, direct, rtol=0, atol=0)
    np.testing.assert_array_equal(fit.loss_history, losses)


def test_multinomial_engine_is_exactly_equivalent_to_direct_wiring():
    panel = _panel("multinomial")
    config = InferenceConfig(epochs=30, learning_rate=0.01)
    seed = 1703
    fit = fit_factorization(panel, MASK, rank=RANK, config=config, seed=seed)

    counts = np.array(
        [
            [[2, 1, 0], [0, 1, 1], [1, 0, 2]],
            [[1, 1, 1], [0, 2, 1], [2, 0, 1]],
        ]
    )
    stats = multinomial_sufficient_statistics_flat(counts, MASK)
    dim = 2 * (N + T) * (RANK + 1)
    prior = torch.distributions.Normal(torch.zeros(dim), torch.full((dim,), 2.0))

    def log_joint(theta):
        return multinomial_log_joint_flat(
            theta=theta,
            unit_idx=stats[0],
            time_idx=stats[1],
            m_vec=stats[2],
            Tx_vec=stats[3],
            log_carrier_vec=stats[4],
            n=N,
            T=T,
            r=RANK,
            prior=prior,
        )

    direct, losses, _ = _direct_fit(log_joint, dim, config, seed)
    torch.testing.assert_close(fit.variational_parameters, direct, rtol=0, atol=0)
    np.testing.assert_array_equal(fit.loss_history, losses)


@pytest.mark.parametrize(
    ("family", "expected_learning_rate"),
    [
        ("gaussian", 0.1),
        ("gaussian_known_sigma", 0.05),
        ("poisson", 0.1),
        ("bernoulli", 0.05),
        ("exponential", 0.005),
        ("laplace_known_mu", 0.005),
        ("chi_square", 0.002),
        ("multinomial", 0.01),
    ],
)
def test_all_supported_families_have_finite_fixed_seed_smoke_fits(
    family,
    expected_learning_rate,
):
    panel = _panel(family)
    fit = fit_factorization(
        panel,
        MASK,
        rank=RANK,
        config=InferenceConfig(epochs=15),
        seed=99,
    )

    assert fit.family == family
    assert fit.rank == RANK
    assert fit.seed == 99
    assert fit.learning_rate == expected_learning_rate
    assert len(fit.loss_history) == 15
    assert np.isfinite(fit.loss_history).all()
    assert torch.isfinite(fit.variational_parameters).all()
    assert torch.isfinite(fit.natural_parameters).all()
