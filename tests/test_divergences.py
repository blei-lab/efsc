import numpy as np
import pytest
import torch
from scipy.special import digamma, gammaln

from efsc.divergences import (
    kl_categorical_probabilities,
    kl_gaussian_univariate,
    kl_gaussian_univariate_from_theta,
    kl_multinomial,
    kl_one_parameter_from_eta,
    one_parameter_log_partition,
    one_parameter_mean_from_eta,
)


@pytest.mark.parametrize(
    ("family", "eta", "expected_partition", "expected_mean"),
    [
        ("poisson", np.log(2.0), 2.0, 2.0),
        ("bernoulli", np.log(3.0), np.log(4.0), 0.75),
        ("exponential", -0.5, np.log(2.0), 2.0),
        ("laplace_known_mu", -0.5, np.log(4.0), 2.0),
        (
            "chi_square",
            1.5,
            gammaln(2.5) + 2.5 * np.log(2.0),
            digamma(2.5) + np.log(2.0),
        ),
        ("gauss_univariate_known_sigma", 1.5, 1.125, 1.5),
    ],
)
def test_one_parameter_partition_and_mean(
    family,
    eta,
    expected_partition,
    expected_mean,
):
    np.testing.assert_allclose(
        one_parameter_log_partition(family, eta),
        expected_partition,
    )
    np.testing.assert_allclose(
        one_parameter_mean_from_eta(family, eta),
        expected_mean,
    )


@pytest.mark.parametrize(
    ("family", "eta", "expected_partition", "expected_mean"),
    [
        (
            "poisson",
            [-1.2, 0.4],
            np.exp([-1.2, 0.4]),
            np.exp([-1.2, 0.4]),
        ),
        (
            "bernoulli",
            [-1.2, 0.4],
            np.logaddexp(0.0, [-1.2, 0.4]),
            1.0 / (1.0 + np.exp(-np.asarray([-1.2, 0.4]))),
        ),
        (
            "exponential",
            [-2.0, -0.4],
            -np.log([2.0, 0.4]),
            [0.5, 2.5],
        ),
        (
            "laplace_known_mu",
            [-2.0, -0.4],
            np.log([1.0, 5.0]),
            [0.5, 2.5],
        ),
        (
            "chi_square",
            [-0.4, 1.2],
            gammaln([0.6, 2.2]) + np.asarray([0.6, 2.2]) * np.log(2.0),
            digamma([0.6, 2.2]) + np.log(2.0),
        ),
        (
            "gauss_univariate_known_sigma",
            [-1.2, 0.4],
            0.5 * np.asarray([-1.2, 0.4]) ** 2,
            [-1.2, 0.4],
        ),
    ],
)
def test_partition_and_mean_preserve_numpy_backend_and_vector_values(
    family,
    eta,
    expected_partition,
    expected_mean,
):
    """Lock vectorized NumPy values before sharing the family formulas."""

    eta = np.asarray(eta, dtype=np.float64)
    partition = one_parameter_log_partition(family, eta)
    mean = one_parameter_mean_from_eta(family, eta)

    assert isinstance(partition, np.ndarray)
    assert isinstance(mean, np.ndarray)
    np.testing.assert_allclose(partition, expected_partition)
    np.testing.assert_allclose(mean, expected_mean)


@pytest.mark.parametrize(
    ("family", "eta"),
    [
        ("poisson", [-1.2, 0.4]),
        ("bernoulli", [-1.2, 0.4]),
        ("exponential", [-2.0, -0.4]),
        ("laplace_known_mu", [-2.0, -0.4]),
        ("chi_square", [-0.4, 1.2]),
        ("gauss_univariate_known_sigma", [-1.2, 0.4]),
    ],
)
def test_partition_gradient_equals_mean_for_torch_backend(family, eta):
    """Lock PyTorch values, dtype, and the identity A'(eta)=E[T(Y)]."""

    eta = torch.tensor(eta, dtype=torch.float64, requires_grad=True)
    partition = one_parameter_log_partition(family, eta)
    mean = one_parameter_mean_from_eta(family, eta)
    partition.sum().backward()

    assert torch.is_tensor(partition)
    assert torch.is_tensor(mean)
    assert partition.dtype == eta.dtype
    assert mean.dtype == eta.dtype
    numpy_eta = eta.detach().cpu().numpy()
    np.testing.assert_allclose(
        partition.detach().cpu().numpy(),
        one_parameter_log_partition(family, numpy_eta),
    )
    np.testing.assert_allclose(
        mean.detach().cpu().numpy(),
        one_parameter_mean_from_eta(family, numpy_eta),
    )
    torch.testing.assert_close(eta.grad, mean)


@pytest.mark.parametrize(
    ("family", "eta_p", "eta_q", "expected"),
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
            0.75 * np.log(0.75 / (1 / 3))
            + 0.25 * np.log(0.25 / (2 / 3)),
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
            "gauss_univariate_known_sigma",
            1.5,
            -0.5,
            0.5 * (1.5 - (-0.5)) ** 2,
        ),
    ],
)
def test_one_parameter_kl_matches_distribution_specific_formula(
    family,
    eta_p,
    eta_q,
    expected,
):
    np.testing.assert_allclose(
        kl_one_parameter_from_eta(family, eta_p, eta_q),
        expected,
    )


def test_chi_square_kl_matches_exponential_family_identity():
    eta_p = 1.5
    eta_q = 0.25
    mean_p = digamma(1 + eta_p) + np.log(2.0)
    partition_p = gammaln(1 + eta_p) + (eta_p + 1) * np.log(2.0)
    partition_q = gammaln(1 + eta_q) + (eta_q + 1) * np.log(2.0)
    expected = (eta_p - eta_q) * mean_p - (partition_p - partition_q)

    np.testing.assert_allclose(
        kl_one_parameter_from_eta("chi_square", eta_p, eta_q),
        expected,
    )


@pytest.mark.parametrize(
    ("mu_p", "var_p", "mu_q", "var_q"),
    [(1.0, 2.0, -0.5, 3.0), (0.0, 1.0, 0.0, 1.0)],
)
def test_gaussian_kl_matches_closed_form(mu_p, var_p, mu_q, var_q):
    expected = 0.5 * (
        np.log(var_q / var_p)
        + (var_p + (mu_p - mu_q) ** 2) / var_q
        - 1
    )

    np.testing.assert_allclose(
        kl_gaussian_univariate(mu_p, var_p, mu_q, var_q),
        expected,
    )
    np.testing.assert_allclose(
        kl_gaussian_univariate_from_theta(
            np.array([mu_p, var_p]),
            np.array([mu_q, var_q]),
        ),
        expected,
    )


def test_multinomial_kl_characterizes_current_count_scaled_convention():
    counts = np.array([[2, 3, 5], [1, 1, 2]])
    probs_p = np.array([[0.2, 0.3, 0.5], [0.25, 0.25, 0.5]])
    probs_q = np.array([[0.1, 0.4, 0.5], [0.4, 0.2, 0.4]])
    cell_sizes = counts.sum(axis=1)
    categorical_kl = np.sum(probs_p * np.log(probs_p / probs_q), axis=1)

    actual = kl_multinomial(counts, probs_p, counts, probs_q)

    np.testing.assert_allclose(actual, cell_sizes * categorical_kl)
    np.testing.assert_allclose(
        kl_multinomial(counts, probs_p, counts, probs_q, reduction="mean"),
        np.mean(cell_sizes * categorical_kl),
    )

    np.testing.assert_allclose(
        kl_categorical_probabilities(probs_p, probs_q),
        categorical_kl,
    )


def test_multinomial_kl_rejects_different_cell_sizes():
    with pytest.raises(ValueError, match="same cell totals"):
        kl_multinomial(
            np.array([[2, 3]]),
            np.array([[0.4, 0.6]]),
            np.array([[2, 4]]),
            np.array([[0.4, 0.6]]),
        )


def test_divergence_helpers_support_torch_and_reductions():
    eta_p = torch.tensor([0.2, -0.4], dtype=torch.float64)
    eta_q = torch.tensor([-0.1, 0.5], dtype=torch.float64)

    values = kl_one_parameter_from_eta("bernoulli", eta_p, eta_q)

    assert isinstance(values, torch.Tensor)
    torch.testing.assert_close(
        kl_one_parameter_from_eta("bernoulli", eta_p, eta_q, reduction="sum"),
        values.sum(),
    )


def test_invalid_reduction_is_rejected():
    with pytest.raises(ValueError, match="reduction"):
        kl_gaussian_univariate(0.0, 1.0, 1.0, 1.0, reduction="median")
