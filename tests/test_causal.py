import numpy as np
import pytest
import torch
from scipy.special import digamma, gammaln

from efsc.causal import plugin_ecd_from_eta, plugin_ece_from_eta


def test_plugin_ece_is_treated_minus_counterfactual_for_vector_parameters():
    treated = np.array([[[1.2, -0.4], [0.7, -0.8]]])
    counterfactual = np.array([[[0.5, -0.6], [0.9, -0.5]]])

    actual = plugin_ece_from_eta(treated, counterfactual)

    np.testing.assert_allclose(actual, treated - counterfactual)
    assert actual.shape == treated.shape


@pytest.mark.parametrize(
    ("family", "treated", "counterfactual", "expected"),
    [
        (
            "poisson",
            np.log(2.0),
            np.log(5.0),
            2.0 * np.log(2.0 / 5.0) + 5.0 - 2.0,
        ),
        (
            "bernoulli",
            np.log(3.0),
            np.log(0.5),
            0.75 * np.log(0.75 / (1.0 / 3.0))
            + 0.25 * np.log(0.25 / (2.0 / 3.0)),
        ),
        (
            "exponential",
            -2.0,
            -0.5,
            np.log(2.0 / 0.5) + 0.5 / 2.0 - 1.0,
        ),
        (
            "laplace_known_mu",
            -0.5,
            -2.0,
            np.log(0.5 / 2.0) + 2.0 / 0.5 - 1.0,
        ),
        (
            "chi_square",
            1.5,
            0.25,
            (1.5 - 0.25) * (digamma(2.5) + np.log(2.0))
            - (
                gammaln(2.5)
                + 2.5 * np.log(2.0)
                - gammaln(1.25)
                - 1.25 * np.log(2.0)
            ),
        ),
        (
            "gaussian_known_sigma",
            1.5,
            -0.5,
            0.5 * (1.5 - (-0.5)) ** 2,
        ),
    ],
)
def test_one_parameter_plugin_ecd_matches_independent_formulas(
    family,
    treated,
    counterfactual,
    expected,
):
    treated_eta = np.array([[treated]], dtype=float)
    counterfactual_eta = np.array([[counterfactual]], dtype=float)

    actual = plugin_ecd_from_eta(family, treated_eta, counterfactual_eta)

    np.testing.assert_allclose(actual, expected)


def test_gaussian_plugin_ecd_matches_distribution_parameter_formula():
    treated_mean, treated_variance = 1.25, 0.8
    counterfactual_mean, counterfactual_variance = -0.4, 1.7
    treated = np.array(
        [[treated_mean / treated_variance, -0.5 / treated_variance]]
    )
    counterfactual = np.array(
        [
            [
                counterfactual_mean / counterfactual_variance,
                -0.5 / counterfactual_variance,
            ]
        ]
    )
    expected = 0.5 * (
        np.log(counterfactual_variance / treated_variance)
        + (
            treated_variance
            + (treated_mean - counterfactual_mean) ** 2
        )
        / counterfactual_variance
        - 1.0
    )

    actual = plugin_ecd_from_eta("gaussian", treated, counterfactual)

    np.testing.assert_allclose(actual, expected)


def _probabilities_from_free_logits(eta):
    eta_with_reference = np.concatenate(
        [eta, np.zeros((*eta.shape[:-1], 1))],
        axis=-1,
    )
    weights = np.exp(
        eta_with_reference
        - eta_with_reference.max(axis=-1, keepdims=True)
    )
    return weights / weights.sum(axis=-1, keepdims=True)


def test_multinomial_plugin_ecd_is_unscaled_probability_vector_kl():
    treated = np.array([[1.2, -0.3]])
    counterfactual = np.array([[-0.4, 0.6]])
    treated_probabilities = _probabilities_from_free_logits(treated)
    counterfactual_probabilities = _probabilities_from_free_logits(counterfactual)
    expected = np.sum(
        treated_probabilities
        * np.log(treated_probabilities / counterfactual_probabilities),
        axis=-1,
    )

    actual = plugin_ecd_from_eta("multinomial", treated, counterfactual)

    np.testing.assert_allclose(actual, expected)
    assert not np.allclose(actual, 1000 * expected)


def test_plugin_ecd_direction_is_treated_relative_to_counterfactual():
    treated = np.array([[np.log(2.0)]])
    counterfactual = np.array([[np.log(5.0)]])

    forward = plugin_ecd_from_eta("poisson", treated, counterfactual)
    reverse = plugin_ecd_from_eta("poisson", counterfactual, treated)

    np.testing.assert_allclose(forward, 2.0 * np.log(2.0 / 5.0) + 3.0)
    assert not np.allclose(forward, reverse)


def test_plugin_estimands_preserve_torch_and_support_reduction():
    treated = torch.tensor([[[0.2]], [[-0.4]]], dtype=torch.float64)
    counterfactual = torch.tensor([[[-0.1]], [[0.5]]], dtype=torch.float64)

    ece = plugin_ece_from_eta(treated, counterfactual)
    ecd = plugin_ecd_from_eta("bernoulli", treated, counterfactual)

    assert isinstance(ece, torch.Tensor)
    assert isinstance(ecd, torch.Tensor)
    torch.testing.assert_close(ece, treated - counterfactual)
    torch.testing.assert_close(
        plugin_ecd_from_eta(
            "bernoulli",
            treated,
            counterfactual,
            reduction="mean",
        ),
        ecd.mean(),
    )


@pytest.mark.parametrize(
    ("family", "treated", "counterfactual", "message"),
    [
        ("gaussian", [[0.2]], [[0.1]], "2 natural"),
        ("poisson", [[0.2, 0.3]], [[0.1, 0.4]], "1 natural"),
        ("gaussian", [[0.2, 0.1]], [[0.1, -0.5]], "must be negative"),
        ("exponential", [[0.2]], [[-0.5]], "must be negative"),
        ("laplace_known_mu", [[-0.2]], [[0.5]], "must be negative"),
        ("chi_square", [[-1.0]], [[0.5]], "greater than -1"),
    ],
)
def test_plugin_ecd_rejects_invalid_natural_parameters(
    family,
    treated,
    counterfactual,
    message,
):
    with pytest.raises(ValueError, match=message):
        plugin_ecd_from_eta(
            family,
            np.asarray(treated, dtype=float),
            np.asarray(counterfactual, dtype=float),
        )
