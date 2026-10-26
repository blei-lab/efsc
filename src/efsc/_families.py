"""Shared structural definitions for EFSC exponential families.

This private module centralizes canonical family names, predictor dimensions,
fixed maps from unconstrained predictors to natural parameters, and shared
one-parameter exponential-family identities. It does not contain complete
likelihoods, sufficient statistics, or sampling behavior.

Author:
    Hector Rodriguez-Deniz

Affiliation:
    Data Science Institute, Columbia University

Date:
    August 2026

License:
    MIT
"""

from numbers import Real
from typing import Any, Dict, Optional

import numpy as np
import torch


FAMILY_ALIASES = {
    "gauss_univariate": "gaussian",
    "gauss_univariate_known_sigma": "gaussian_known_sigma",
}

FIXED_PARAMETER_COUNTS = {
    "gaussian": 2,
    "gaussian_known_sigma": 1,
    "poisson": 1,
    "bernoulli": 1,
    "exponential": 1,
    "laplace_known_mu": 1,
    "chi_square": 1,
}

SUPPORTED_FAMILIES = frozenset({*FIXED_PARAMETER_COUNTS, "multinomial"})
ONE_PARAMETER_FAMILIES = frozenset(
    family for family in FIXED_PARAMETER_COUNTS if family != "gaussian"
)


def normalize_family(family: str) -> str:
    """Return the canonical name of a supported EFSC family."""

    if not isinstance(family, str):
        raise ValueError("family must be supplied as a string.")
    normalized_input = family.strip().lower()
    normalized = FAMILY_ALIASES.get(normalized_input, normalized_input)
    if normalized not in SUPPORTED_FAMILIES:
        supported = ", ".join(sorted(SUPPORTED_FAMILIES))
        raise ValueError(
            f"Unsupported family '{family}'. Choose one of: {supported}."
        )
    return normalized


def predictor_block_count(
    family: str,
    n_categories: Optional[int] = None,
) -> int:
    """Return the number of unconstrained predictor blocks for a family."""

    family = normalize_family(family)
    if family == "multinomial":
        if n_categories is None or n_categories < 2:
            raise ValueError("Multinomial reconstruction requires n_categories >= 2.")
        return n_categories - 1
    return FIXED_PARAMETER_COUNTS[family]


def _is_torch_tensor(value: Any) -> bool:
    """Return whether a value is a PyTorch tensor."""

    return torch.is_tensor(value)


def _exp(value: Any) -> Any:
    """Evaluate a backend-preserving exponential."""

    if _is_torch_tensor(value):
        return torch.exp(value)
    return np.exp(value)


def _log(value: Any) -> Any:
    """Evaluate a backend-preserving natural logarithm."""

    if _is_torch_tensor(value):
        return torch.log(value)
    return np.log(value)


def _logaddexp_zero(value: Any) -> Any:
    """Evaluate ``log(1 + exp(value))`` stably for either backend."""

    if _is_torch_tensor(value):
        return torch.logaddexp(torch.zeros_like(value), value)
    return np.logaddexp(0.0, value)


def _log_two_like(value: Any) -> Any:
    """Return ``log(2)`` on the input tensor's device and dtype when needed."""

    if _is_torch_tensor(value):
        return torch.log(value.new_tensor(2.0))
    return np.log(2.0)


def _lgamma(value: Any) -> Any:
    """Evaluate the log-gamma function using the input backend."""

    if _is_torch_tensor(value):
        return torch.lgamma(value)
    array = np.asarray(value)
    dtype = None if np.issubdtype(array.dtype, np.floating) else torch.float64
    result = torch.lgamma(torch.as_tensor(array, dtype=dtype)).numpy()
    return result.item() if np.isscalar(value) else result


def _digamma(value: Any) -> Any:
    """Evaluate the digamma function using the input backend."""

    if _is_torch_tensor(value):
        return torch.digamma(value)
    array = np.asarray(value)
    dtype = None if np.issubdtype(array.dtype, np.floating) else torch.float64
    result = torch.digamma(torch.as_tensor(array, dtype=dtype)).numpy()
    return result.item() if np.isscalar(value) else result


def _normalize_one_parameter_family(family: str) -> str:
    """Return a canonical family supported by one-parameter EF identities."""

    normalized = normalize_family(family)
    if normalized not in ONE_PARAMETER_FAMILIES:
        raise ValueError(f"Unsupported one-parameter family: {family}.")
    return normalized


def one_parameter_log_partition(family: str, eta: Any) -> Any:
    """Evaluate the one-parameter log-partition function ``A(eta)``."""

    family = _normalize_one_parameter_family(family)
    if family == "poisson":
        return _exp(eta)
    if family == "bernoulli":
        return _logaddexp_zero(eta)
    if family == "exponential":
        return -_log(-eta)
    if family == "laplace_known_mu":
        return _log(-2.0 / eta)
    if family == "chi_square":
        return _lgamma(1.0 + eta) + (eta + 1.0) * _log_two_like(eta)
    if family == "gaussian_known_sigma":
        return 0.5 * eta**2
    raise AssertionError("Unreachable family dispatch.")


def one_parameter_mean_from_eta(family: str, eta: Any) -> Any:
    """Evaluate ``A'(eta)``, the mean of the sufficient statistic."""

    family = _normalize_one_parameter_family(family)
    if family == "poisson":
        return _exp(eta)
    if family == "bernoulli":
        return 1.0 / (1.0 + _exp(-eta))
    if family in {"exponential", "laplace_known_mu"}:
        return -1.0 / eta
    if family == "chi_square":
        return _digamma(1.0 + eta) + _log_two_like(eta)
    if family == "gaussian_known_sigma":
        return eta
    raise AssertionError("Unreachable family dispatch.")


def _all_true(condition: Any) -> bool:
    """Reduce a NumPy or PyTorch Boolean condition to one truth value."""

    if _is_torch_tensor(condition):
        return bool(torch.all(condition))
    return bool(np.all(condition))


def validate_natural_parameters(family: str, eta: Any) -> str:
    """Validate a natural-parameter axis and its family-specific domain."""

    family = normalize_family(family)
    if eta.ndim < 1:
        raise ValueError("Natural parameters must have a final parameter axis.")
    if family == "multinomial":
        if eta.shape[-1] < 1:
            raise ValueError(
                "multinomial requires at least one free natural parameter."
            )
    else:
        expected = FIXED_PARAMETER_COUNTS[family]
        if eta.shape[-1] != expected:
            word = "two" if expected == 2 else "one"
            noun = "parameter" if expected == 1 else "parameters"
            raise ValueError(
                f"{family} requires exactly {word} natural {noun}; "
                f"{expected} natural {noun} expected on the final axis."
            )

    if family == "gaussian" and not _all_true(eta[..., 1] < 0):
        raise ValueError("The Gaussian second natural parameter must be negative.")
    if family in {"exponential", "laplace_known_mu"} and not _all_true(eta < 0):
        raise ValueError(f"{family} natural parameters must be negative.")
    if family == "chi_square" and not _all_true(eta > -1):
        raise ValueError("chi_square natural parameters must be greater than -1.")
    return family


def _finite_scalar(value: Any, name: str) -> float:
    """Validate and return a finite scalar value."""

    if (
        not isinstance(value, Real)
        or isinstance(value, (bool, np.bool_))
        or not np.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number.")
    return float(value)


def _full_like(value: Any, fill_value: float) -> Any:
    """Construct a constant array using the input backend, shape, and dtype."""

    if _is_torch_tensor(value):
        return torch.full_like(value, fill_value)
    return np.full_like(value, fill_value)


def _sigmoid(value: Any) -> Any:
    """Evaluate the logistic function using the input backend."""

    if _is_torch_tensor(value):
        return torch.sigmoid(value)
    return 1.0 / (1.0 + np.exp(-value))


def multinomial_probabilities_from_eta(eta: Any) -> Any:
    """Append a zero reference logit and return category probabilities."""

    validate_natural_parameters("multinomial", eta)
    if _is_torch_tensor(eta):
        reference = torch.zeros(
            (*eta.shape[:-1], 1),
            dtype=eta.dtype,
            device=eta.device,
        )
        return torch.softmax(torch.cat([eta, reference], dim=-1), dim=-1)

    reference = np.zeros((*eta.shape[:-1], 1), dtype=eta.dtype)
    eta_with_reference = np.concatenate([eta, reference], axis=-1)
    shifted = eta_with_reference - np.max(eta_with_reference, axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / np.sum(weights, axis=-1, keepdims=True)


def conventional_parameters_from_eta(
    family: str,
    eta: Any,
    *,
    sigma: Optional[float] = None,
    mu: Optional[float] = None,
) -> Dict[str, Any]:
    """Convert natural parameters into familiar distribution parameters."""

    family = validate_natural_parameters(family, eta)
    if family == "gaussian":
        eta0 = eta[..., 0]
        eta1 = eta[..., 1]
        return {
            "mean": -0.5 * eta0 / eta1,
            "variance": -0.5 / eta1,
        }
    if family == "gaussian_known_sigma":
        resolved_sigma = _finite_scalar(sigma, "sigma")
        if resolved_sigma <= 0:
            raise ValueError("sigma must be greater than zero.")
        return {
            "mean": eta[..., 0] * resolved_sigma,
            "variance": _full_like(eta[..., 0], resolved_sigma**2),
        }
    if family == "poisson":
        return {"rate": _exp(eta[..., 0])}
    if family == "bernoulli":
        return {"probability": _sigmoid(eta[..., 0])}
    if family == "exponential":
        rate = -eta[..., 0]
        return {"rate": rate, "mean": 1.0 / rate}
    if family == "laplace_known_mu":
        resolved_mu = _finite_scalar(mu, "mu")
        natural = eta[..., 0]
        return {
            "location": _full_like(natural, resolved_mu),
            "scale": -1.0 / natural,
        }
    if family == "chi_square":
        return {"degrees_of_freedom": 2.0 * (eta[..., 0] + 1.0)}
    if family == "multinomial":
        return {"probabilities": multinomial_probabilities_from_eta(eta)}
    raise AssertionError("Unreachable family dispatch.")


def constrain_raw_component(
    raw: torch.Tensor,
    *,
    family: str,
    component: int = 0,
) -> torch.Tensor:
    """Map one unconstrained predictor component to its natural parameter."""

    if (family == "gaussian" and component == 1) or family in {
        "exponential",
        "laplace_known_mu",
    }:
        return -torch.exp(raw)
    if family == "chi_square":
        return torch.exp(raw) - 1
    return raw


def constrain_raw_parameters(raw: Any, *, family: str) -> torch.Tensor:
    """Apply the family map from predictor to natural-parameter space."""

    family = normalize_family(family)
    raw_tensor = torch.as_tensor(raw)
    expected = FIXED_PARAMETER_COUNTS.get(family)
    if expected is not None and (
        raw_tensor.ndim < 1 or raw_tensor.shape[-1] != expected
    ):
        raise ValueError(
            f"{family} requires {expected} unconstrained predictor component(s)."
        )
    return torch.stack(
        [
            constrain_raw_component(
                raw_tensor[..., component],
                family=family,
                component=component,
            )
            for component in range(raw_tensor.shape[-1])
        ],
        dim=-1,
    )
