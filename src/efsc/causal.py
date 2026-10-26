"""Plug-in causal estimands for Exponential Family Synthetic Controls.

This module evaluates the manuscript's expected causal effect (ECE) and
expected causal divergence (ECD) from treated and counterfactual natural
parameters reconstructed at variational posterior means.

Author:
    Hector Rodriguez-Deniz

Affiliation:
    Data Science Institute, Columbia University

Date:
    August 2026

License:
    MIT
"""

from typing import Any, Optional, Tuple

import numpy as np
import torch

from ._families import (
    conventional_parameters_from_eta,
    normalize_family,
    validate_natural_parameters,
)
from .divergences import (
    kl_categorical_probabilities,
    kl_gaussian_univariate,
    kl_one_parameter_from_eta,
)


_ONE_PARAMETER_DIVERGENCE_NAMES = {
    "poisson": "poisson",
    "bernoulli": "bernoulli",
    "exponential": "exponential",
    "laplace_known_mu": "laplace_known_mu",
    "chi_square": "chi_square",
    "gaussian_known_sigma": "gauss_univariate_known_sigma",
}
def _validated_eta_pair(
    treated_eta: Any,
    counterfactual_eta: Any,
) -> Tuple[Any, Any]:
    """Validate matching finite natural-parameter arrays with one backend."""

    treated_is_tensor = torch.is_tensor(treated_eta)
    counterfactual_is_tensor = torch.is_tensor(counterfactual_eta)
    if treated_is_tensor != counterfactual_is_tensor:
        raise TypeError(
            "treated_eta and counterfactual_eta must both be NumPy arrays or "
            "both be torch tensors."
        )

    if treated_is_tensor:
        treated = treated_eta
        counterfactual = counterfactual_eta
        if treated.shape != counterfactual.shape:
            raise ValueError(
                "treated_eta and counterfactual_eta must have identical shapes."
            )
        if treated.ndim < 1:
            raise ValueError(
                "Natural-parameter inputs must have a final parameter axis."
            )
        if not torch.is_floating_point(treated) or not torch.is_floating_point(
            counterfactual
        ):
            raise ValueError("Natural-parameter inputs must be floating point.")
        if not torch.isfinite(treated).all() or not torch.isfinite(
            counterfactual
        ).all():
            raise ValueError("Natural-parameter inputs must be finite.")
        return treated, counterfactual

    treated = np.asarray(treated_eta)
    counterfactual = np.asarray(counterfactual_eta)
    if treated.shape != counterfactual.shape:
        raise ValueError(
            "treated_eta and counterfactual_eta must have identical shapes."
        )
    if treated.ndim < 1:
        raise ValueError(
            "Natural-parameter inputs must have a final parameter axis."
        )
    if not np.issubdtype(treated.dtype, np.floating) or not np.issubdtype(
        counterfactual.dtype, np.floating
    ):
        raise ValueError("Natural-parameter inputs must be floating point.")
    if not np.isfinite(treated).all() or not np.isfinite(counterfactual).all():
        raise ValueError("Natural-parameter inputs must be finite.")
    return treated, counterfactual


def plugin_ece_from_eta(treated_eta: Any, counterfactual_eta: Any) -> Any:
    """Return the plug-in ECE, ``treated_eta - counterfactual_eta``."""

    treated, counterfactual = _validated_eta_pair(
        treated_eta,
        counterfactual_eta,
    )
    return treated - counterfactual


def plugin_ecd_from_eta(
    family: str,
    treated_eta: Any,
    counterfactual_eta: Any,
    *,
    reduction: Optional[str] = None,
) -> Any:
    """Return plug-in ECD as KL(treated || counterfactual).

    Multinomial inputs contain the free logits, with an implicit final
    reference-category logit of zero. Its ECD is reported on the unscaled
    probability-vector convention used in the manuscript.
    """

    family = normalize_family(family)
    treated, counterfactual = _validated_eta_pair(
        treated_eta,
        counterfactual_eta,
    )
    validate_natural_parameters(family, treated)
    validate_natural_parameters(family, counterfactual)

    if family in _ONE_PARAMETER_DIVERGENCE_NAMES:
        return kl_one_parameter_from_eta(
            _ONE_PARAMETER_DIVERGENCE_NAMES[family],
            treated[..., 0],
            counterfactual[..., 0],
            reduction=reduction,
        )

    if family == "gaussian":
        treated_parameters = conventional_parameters_from_eta(family, treated)
        counterfactual_parameters = conventional_parameters_from_eta(
            family,
            counterfactual,
        )
        return kl_gaussian_univariate(
            mu_p=treated_parameters["mean"],
            var_p=treated_parameters["variance"],
            mu_q=counterfactual_parameters["mean"],
            var_q=counterfactual_parameters["variance"],
            reduction=reduction,
        )

    if family == "multinomial":
        treated_probabilities = conventional_parameters_from_eta(
            family,
            treated,
        )["probabilities"]
        counterfactual_probabilities = conventional_parameters_from_eta(
            family,
            counterfactual,
        )["probabilities"]
        return kl_categorical_probabilities(
            treated_probabilities,
            counterfactual_probabilities,
            reduction=reduction,
        )

    raise AssertionError("Unreachable family dispatch.")
