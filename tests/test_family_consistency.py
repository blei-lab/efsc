import numpy as np
import pytest
import torch
from scipy import stats

from efsc.causal import plugin_ecd_from_eta
from efsc.fitting import reconstruct_natural_parameters
from efsc.log_joints import (
    bernoulli_log_joint_flat,
    chi_square_log_joint_flat,
    exponential_log_joint_flat,
    gaussian_known_sigma_log_joint_flat,
    gaussian_univariate_log_joint_flat,
    laplace_known_mu_log_joint_flat,
    multinomial_log_joint_flat,
    poisson_log_joint_flat,
)
from efsc.preprocessing import (
    bernoulli_sufficient_statistics_flat,
    chi_square_sufficient_statistics_flat,
    exponential_sufficient_statistics_flat,
    gaussian_known_sigma_sufficient_statistics_flat,
    gaussian_sufficient_statistics_flat,
    laplace_sufficient_statistics_flat,
    multinomial_sufficient_statistics_flat,
    poisson_sufficient_statistics_flat,
)
from efsc.results import distribution_parameters_from_eta


def _one_cell_parameter_vector(*raw_predictors, dtype=torch.float64):
    """Encode one rank-one PMF block per predictor with only alpha nonzero."""

    return torch.tensor(
        [
            value
            for raw_predictor in raw_predictors
            for value in (raw_predictor, 0.0, 0.0, 0.0)
        ],
        dtype=dtype,
    )


@pytest.mark.parametrize(
    (
        "family",
        "raw_predictors",
        "conversion_options",
        "expected_eta",
        "expected_parameters",
        "n_categories",
    ),
    [
        (
            "gaussian",
            (0.7, -0.2),
            {},
            (0.7, -np.exp(-0.2)),
            {
                "mean": -0.5 * 0.7 / -np.exp(-0.2),
                "variance": -0.5 / -np.exp(-0.2),
            },
            None,
        ),
        (
            "gaussian_known_sigma",
            (-0.4,),
            {"sigma": 1.5},
            (-0.4,),
            {"mean": -0.6, "variance": 2.25},
            None,
        ),
        (
            "poisson",
            (0.3,),
            {},
            (0.3,),
            {"rate": np.exp(0.3)},
            None,
        ),
        (
            "bernoulli",
            (0.3,),
            {},
            (0.3,),
            {"probability": 1.0 / (1.0 + np.exp(-0.3))},
            None,
        ),
        (
            "exponential",
            (0.3,),
            {},
            (-np.exp(0.3),),
            {"rate": np.exp(0.3), "mean": np.exp(-0.3)},
            None,
        ),
        (
            "laplace_known_mu",
            (0.3,),
            {"mu": 0.8},
            (-np.exp(0.3),),
            {"location": 0.8, "scale": np.exp(-0.3)},
            None,
        ),
        (
            "chi_square",
            (0.3,),
            {},
            (np.exp(0.3) - 1.0,),
            {"degrees_of_freedom": 2.0 * np.exp(0.3)},
            None,
        ),
        (
            "multinomial",
            (0.3, -0.4),
            {"categories": ("first", "second", "reference")},
            (0.3, -0.4),
            {
                "probabilities": np.exp([0.3, -0.4, 0.0])
                / np.exp([0.3, -0.4, 0.0]).sum()
            },
            3,
        ),
    ],
)
def test_reconstruction_and_distribution_conversion_agree_across_families(
    family,
    raw_predictors,
    conversion_options,
    expected_eta,
    expected_parameters,
    n_categories,
):
    """Lock the predictor transform and inverse link into one cross-layer test."""

    eta = reconstruct_natural_parameters(
        _one_cell_parameter_vector(*raw_predictors),
        family=family,
        n=1,
        T=1,
        rank=1,
        n_categories=n_categories,
    )
    torch.testing.assert_close(
        eta,
        torch.tensor(expected_eta, dtype=eta.dtype).reshape(1, 1, -1),
    )

    converted = distribution_parameters_from_eta(
        eta,
        family=family,
        **conversion_options,
    )
    assert converted.parameters.keys() == expected_parameters.keys()
    for name, expected in expected_parameters.items():
        actual = converted.parameters[name]
        torch.testing.assert_close(
            actual,
            torch.as_tensor(expected, dtype=actual.dtype).expand_as(actual),
        )


def _active_log_likelihood(family, raw_predictors, observations, options):
    """Evaluate the active flattened likelihood while removing its test prior."""

    theta = _one_cell_parameter_vector(*raw_predictors, dtype=torch.float32)
    mask = np.ones((1, 1), dtype=bool)
    prior = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))

    if family == "gaussian":
        prepared = gaussian_sufficient_statistics_flat(observations, mask)
        log_joint = gaussian_univariate_log_joint_flat(
            theta,
            *prepared[:5],
            1,
            1,
            1,
            prior,
            prepared[5],
        )
    elif family == "multinomial":
        prepared = multinomial_sufficient_statistics_flat(observations, mask)
        log_joint = multinomial_log_joint_flat(
            theta,
            *prepared,
            1,
            1,
            1,
            prior,
        )
    else:
        preprocessor, log_joint_function = {
            "gaussian_known_sigma": (
                gaussian_known_sigma_sufficient_statistics_flat,
                gaussian_known_sigma_log_joint_flat,
            ),
            "poisson": (
                poisson_sufficient_statistics_flat,
                poisson_log_joint_flat,
            ),
            "bernoulli": (
                bernoulli_sufficient_statistics_flat,
                bernoulli_log_joint_flat,
            ),
            "exponential": (
                exponential_sufficient_statistics_flat,
                exponential_log_joint_flat,
            ),
            "laplace_known_mu": (
                laplace_sufficient_statistics_flat,
                laplace_known_mu_log_joint_flat,
            ),
            "chi_square": (
                chi_square_sufficient_statistics_flat,
                chi_square_log_joint_flat,
            ),
        }[family]
        prepared = preprocessor(observations, mask, **options)
        log_joint = log_joint_function(
            theta,
            *prepared,
            1,
            1,
            1,
            prior,
        )

    return (log_joint - prior.log_prob(theta).sum()).item()


@pytest.mark.parametrize(
    ("family", "raw_predictors", "observations", "options"),
    [
        ("gaussian", (0.37, -0.61), [[[-0.2, 0.9, 2.1]]], {}),
        (
            "gaussian_known_sigma",
            (0.37,),
            [[[-1.1, 0.3, 2.2]]],
            {"sigma": 1.7},
        ),
        ("poisson", (0.37,), [[[0, 2, 5]]], {}),
        ("bernoulli", (0.37,), [[[0, 1, 1]]], {}),
        ("exponential", (0.37,), [[[0.1, 0.7, 1.4]]], {}),
        ("laplace_known_mu", (0.37,), [[[-0.2, 0.7, 2.0]]], {"mu": 0.4}),
        ("chi_square", (0.37,), [[[0.2, 1.7, 4.1]]], {}),
        ("multinomial", (0.37, -0.61), [[[2, 4, 3]]], {}),
    ],
)
def test_active_likelihood_matches_reference_distribution(
    family,
    raw_predictors,
    observations,
    options,
):
    """Compare each complete active likelihood path with SciPy's distribution."""

    observations = np.asarray(observations)
    if family == "gaussian":
        eta0 = raw_predictors[0]
        eta1 = -np.exp(raw_predictors[1])
        variance = -0.5 / eta1
        mean = -0.5 * eta0 / eta1
        reference = stats.norm.logpdf(
            observations.reshape(-1),
            loc=mean,
            scale=np.sqrt(variance),
        ).sum()
    elif family == "gaussian_known_sigma":
        sigma = options["sigma"]
        reference = stats.norm.logpdf(
            observations.reshape(-1),
            loc=sigma * raw_predictors[0],
            scale=sigma,
        ).sum()
    elif family == "poisson":
        reference = stats.poisson.logpmf(
            observations.reshape(-1),
            mu=np.exp(raw_predictors[0]),
        ).sum()
    elif family == "bernoulli":
        probability = 1.0 / (1.0 + np.exp(-raw_predictors[0]))
        reference = stats.bernoulli.logpmf(
            observations.reshape(-1),
            p=probability,
        ).sum()
    elif family == "exponential":
        reference = stats.expon.logpdf(
            observations.reshape(-1),
            scale=np.exp(-raw_predictors[0]),
        ).sum()
    elif family == "laplace_known_mu":
        reference = stats.laplace.logpdf(
            observations.reshape(-1),
            loc=options["mu"],
            scale=np.exp(-raw_predictors[0]),
        ).sum()
    elif family == "chi_square":
        reference = stats.chi2.logpdf(
            observations.reshape(-1),
            df=2.0 * np.exp(raw_predictors[0]),
        ).sum()
    else:
        logits = np.asarray([*raw_predictors, 0.0])
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        counts = observations.reshape(-1)
        reference = stats.multinomial.logpmf(
            counts,
            n=int(counts.sum()),
            p=probabilities,
        )

    active = _active_log_likelihood(
        family,
        raw_predictors,
        observations,
        options,
    )
    assert active == pytest.approx(reference, abs=1e-6)


@pytest.mark.parametrize(
    ("family", "eta", "options"),
    [
        ("gaussian", [[[0.7, -0.8]]], {}),
        ("gaussian_known_sigma", [[[0.4]]], {"sigma": 1.7}),
        ("poisson", [[[0.4]]], {}),
        ("bernoulli", [[[0.4]]], {}),
        ("exponential", [[[-0.7]]], {}),
        ("laplace_known_mu", [[[-0.7]]], {"mu": 0.25}),
        ("chi_square", [[[-0.2]]], {}),
        (
            "multinomial",
            [[[0.4, -0.2]]],
            {"categories": ("first", "second", "reference")},
        ),
    ],
)
def test_distribution_conversion_agrees_for_numpy_and_torch_inputs(
    family,
    eta,
    options,
):
    """Lock inverse-link values independently of the accepted input backend."""

    numpy_result = distribution_parameters_from_eta(
        np.asarray(eta, dtype=np.float64),
        family=family,
        **options,
    )
    torch_result = distribution_parameters_from_eta(
        torch.tensor(eta, dtype=torch.float64),
        family=family,
        **options,
    )

    assert numpy_result.family == torch_result.family == family
    assert numpy_result.parameters.keys() == torch_result.parameters.keys()
    for name in numpy_result.parameters:
        torch.testing.assert_close(
            numpy_result[name],
            torch_result[name],
        )


@pytest.mark.parametrize(
    ("family", "wrong_count"),
    [
        ("gaussian", 1),
        ("gaussian_known_sigma", 2),
        ("poisson", 2),
        ("bernoulli", 2),
        ("exponential", 2),
        ("laplace_known_mu", 2),
        ("chi_square", 2),
    ],
)
def test_results_and_causal_paths_reject_same_fixed_parameter_mismatch(
    family,
    wrong_count,
):
    """Lock the parameter-axis contract shared by results and causal paths."""

    eta = np.zeros((1, 1, wrong_count), dtype=np.float64)
    options = {}
    if family == "gaussian_known_sigma":
        options["sigma"] = 1.5
    elif family == "laplace_known_mu":
        options["mu"] = 0.25

    with pytest.raises(ValueError, match="require"):
        distribution_parameters_from_eta(eta, family=family, **options)
    with pytest.raises(ValueError, match="require"):
        plugin_ecd_from_eta(family, eta, eta)


@pytest.mark.parametrize(
    ("family", "eta", "options", "message"),
    [
        ("gaussian", [[[0.2, 0.0]]], {}, "negative"),
        ("exponential", [[[0.0]]], {}, "negative"),
        ("laplace_known_mu", [[[0.0]]], {"mu": 0.25}, "negative"),
        ("chi_square", [[[-1.0]]], {}, "greater than -1"),
    ],
)
def test_results_and_causal_paths_reject_same_domain_boundary(
    family,
    eta,
    options,
    message,
):
    """Lock strict natural-parameter domains at both active consumers."""

    eta = np.asarray(eta, dtype=np.float64)
    with pytest.raises(ValueError, match=message):
        distribution_parameters_from_eta(eta, family=family, **options)
    with pytest.raises(ValueError, match=message):
        plugin_ecd_from_eta(family, eta, eta)


@pytest.mark.parametrize(
    ("alias", "canonical", "eta", "options"),
    [
        (
            "gauss_univariate",
            "gaussian",
            [[[0.4, -0.7]]],
            {},
        ),
        (
            "gauss_univariate_known_sigma",
            "gaussian_known_sigma",
            [[[0.4]]],
            {"sigma": 1.5},
        ),
    ],
)
def test_inverse_link_and_causal_paths_preserve_family_aliases(
    alias,
    canonical,
    eta,
    options,
):
    """Lock legacy family spellings while internal names are consolidated."""

    eta = np.asarray(eta, dtype=np.float64)
    alias_result = distribution_parameters_from_eta(
        eta,
        family=alias,
        **options,
    )
    canonical_result = distribution_parameters_from_eta(
        eta,
        family=canonical,
        **options,
    )

    assert alias_result.family == canonical_result.family == canonical
    for name in alias_result.parameters:
        torch.testing.assert_close(
            alias_result[name],
            canonical_result[name],
        )
    np.testing.assert_allclose(
        plugin_ecd_from_eta(alias, eta, eta),
        plugin_ecd_from_eta(canonical, eta, eta),
    )
