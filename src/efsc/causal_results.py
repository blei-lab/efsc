"""Tabular results for Exponential Family Synthetic Controls causal fits.

This module combines treated and counterfactual factorization outputs into
cell-level counterfactual predictions, formal natural-parameter effects,
distributional divergences, and interpretable family-specific contrasts.

Author:
    Hector Rodriguez-Deniz

Affiliation:
    Data Science Institute, Columbia University

Date:
    August 2026

License:
    MIT
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .causal import plugin_ecd_from_eta, plugin_ece_from_eta
from .fitting import FactorizationFit
from .panel import CausalMasks, PanelData
from .results import DistributionParameters, distribution_parameters_from_eta


@dataclass(frozen=True)
class CausalResult:
    """Counterfactual predictions and effects from a two-fit EFSC analysis.

    The object contains plug-in counterfactuals and effects for treated-post
    cells, both variational fits, their reconstructed distributions, and the
    masks that define the absorbing-treatment analysis.
    """

    counterfactuals: pd.DataFrame
    effects: pd.DataFrame
    category_effects: Optional[pd.DataFrame]
    counterfactual_fit: FactorizationFit
    treated_fit: FactorizationFit
    counterfactual_distributions: DistributionParameters
    treated_distributions: DistributionParameters
    masks: CausalMasks
    panel: PanelData
    posterior_effects_seed: int

    def sample_effects(
        self,
        n_draws: int,
        *,
        random_state: Optional[int] = None,
    ) -> pd.DataFrame:
        """Draw ECE and ECD values from the two variational fits.

        Draws from the treated and counterfactual variational distributions
        are independent. The returned table has one row per draw and observed
        treated-post cell.
        """

        from .posterior import sample_causal_effects

        return sample_causal_effects(
            self,
            n_draws=n_draws,
            random_state=random_state,
        )

    def summarize_effects(
        self,
        n_draws: int,
        *,
        level: float = 0.95,
        random_state: Optional[int] = None,
    ) -> pd.DataFrame:
        """Summarize posterior ECE and ECD draws by treated-post cell.

        ``level`` is the central interval probability. The result includes
        posterior means and equal-tailed lower and upper bounds.
        """

        from .posterior import summarize_causal_effects

        return summarize_causal_effects(
            self,
            n_draws=n_draws,
            level=level,
            random_state=random_state,
        )


def _validate_inputs(
    panel: PanelData,
    masks: CausalMasks,
    counterfactual_fit: FactorizationFit,
    treated_fit: FactorizationFit,
) -> None:
    """Validate alignment among a panel, masks, and both causal fits."""

    if not isinstance(panel, PanelData):
        raise TypeError("panel must be a PanelData object.")
    if not isinstance(masks, CausalMasks):
        raise TypeError("masks must be a CausalMasks object.")
    if not isinstance(counterfactual_fit, FactorizationFit) or not isinstance(
        treated_fit,
        FactorizationFit,
    ):
        raise TypeError(
            "counterfactual_fit and treated_fit must be FactorizationFit objects."
        )

    expected_shape = (panel.n, panel.T)
    for name, mask in {
        "treated_post_mask": masks.treated_post_mask,
        "counterfactual_mask": masks.counterfactual_mask,
        "treated_mask": masks.treated_mask,
    }.items():
        if np.asarray(mask).shape != expected_shape:
            raise ValueError(f"{name} must have shape (n, T).")

    if counterfactual_fit.family != panel.family or treated_fit.family != panel.family:
        raise ValueError("The panel and both factorization fits must share a family.")
    if counterfactual_fit.rank != treated_fit.rank:
        raise ValueError("Counterfactual and treated fits must use the same rank.")
    if counterfactual_fit.natural_parameters.shape[:2] != expected_shape:
        raise ValueError("Counterfactual natural parameters do not match the panel.")
    if treated_fit.natural_parameters.shape[:2] != expected_shape:
        raise ValueError("Treated natural parameters do not match the panel.")
    if not np.array_equal(counterfactual_fit.mask, masks.counterfactual_mask):
        raise ValueError("counterfactual_fit does not use the counterfactual mask.")
    if not np.array_equal(treated_fit.mask, masks.treated_mask):
        raise ValueError("treated_fit does not use the treated mask.")


def _base_records(panel: PanelData, selected: np.ndarray) -> pd.DataFrame:
    """Build identifying records for selected panel cells."""

    records = []
    for i, unit_label in enumerate(panel.unit_labels):
        for j, time_label in enumerate(panel.time_labels):
            if not selected[i, j]:
                continue
            records.append(
                {
                    "unit": unit_label,
                    "time": time_label,
                    "unit_index": i,
                    "time_index": j,
                    "observation_count": int(panel.cell_counts[i, j]),
                    "observed": bool(panel.observed_mask[i, j]),
                }
            )
    return pd.DataFrame.from_records(records)


def _selected_values(values, selected: np.ndarray) -> np.ndarray:
    """Return selected tensor values as a NumPy array."""

    return values.detach().cpu().numpy()[selected]


def _counterfactual_table(
    panel: PanelData,
    masks: CausalMasks,
    fit: FactorizationFit,
    distributions: DistributionParameters,
) -> pd.DataFrame:
    """Build counterfactual predictions for all treated-post targets."""

    selected = masks.treated_post_mask
    table = _base_records(panel, selected)
    table["effect_available"] = masks.treated_mask[selected]
    table["role"] = np.where(
        table["effect_available"],
        "treated_post_observed",
        "treated_post_missing",
    )

    eta = _selected_values(fit.natural_parameters, selected)
    for parameter_index in range(eta.shape[1]):
        table[f"counterfactual_eta_{parameter_index}"] = eta[:, parameter_index]

    for name, values in distributions.parameters.items():
        selected_values = _selected_values(values, selected)
        if name == "probabilities":
            for category_index in range(selected_values.shape[1]):
                table[f"counterfactual_probability_{category_index}"] = (
                    selected_values[:, category_index]
                )
        else:
            table[f"counterfactual_{name}"] = selected_values
    return table


def _effect_table(
    panel: PanelData,
    masks: CausalMasks,
    counterfactual_fit: FactorizationFit,
    treated_fit: FactorizationFit,
    counterfactual_distributions: DistributionParameters,
    treated_distributions: DistributionParameters,
) -> pd.DataFrame:
    """Build plug-in effects for observed treated-post cells."""

    selected = masks.treated_mask
    table = _base_records(panel, selected)
    counterfactual_eta = _selected_values(
        counterfactual_fit.natural_parameters,
        selected,
    )
    treated_eta = _selected_values(treated_fit.natural_parameters, selected)
    ece = plugin_ece_from_eta(treated_eta, counterfactual_eta)
    ecd = plugin_ecd_from_eta(panel.family, treated_eta, counterfactual_eta)

    for parameter_index in range(counterfactual_eta.shape[1]):
        table[f"counterfactual_eta_{parameter_index}"] = counterfactual_eta[
            :, parameter_index
        ]
        table[f"treated_eta_{parameter_index}"] = treated_eta[:, parameter_index]
        table[f"ece_eta_{parameter_index}"] = ece[:, parameter_index]
    table["ecd"] = ecd

    if panel.family != "multinomial":
        for name in counterfactual_distributions.parameters:
            counterfactual_values = _selected_values(
                counterfactual_distributions[name],
                selected,
            )
            treated_values = _selected_values(
                treated_distributions[name],
                selected,
            )
            table[f"counterfactual_{name}"] = counterfactual_values
            table[f"treated_{name}"] = treated_values
            table[f"difference_{name}"] = treated_values - counterfactual_values
    return table


def _category_effect_table(
    panel: PanelData,
    masks: CausalMasks,
    counterfactual_distributions: DistributionParameters,
    treated_distributions: DistributionParameters,
) -> pd.DataFrame:
    """Build category-probability effects for multinomial causal fits."""

    categories = panel.categories
    if categories is None:
        raise ValueError("Multinomial panel is missing its category order.")
    counterfactual = counterfactual_distributions["probabilities"]
    treated = treated_distributions["probabilities"]
    counterfactual_array = counterfactual.detach().cpu().numpy()
    treated_array = treated.detach().cpu().numpy()
    reference_category = categories[-1]

    records = []
    for i, unit_label in enumerate(panel.unit_labels):
        for j, time_label in enumerate(panel.time_labels):
            if not masks.treated_mask[i, j]:
                continue
            for k, category in enumerate(categories):
                records.append(
                    {
                        "unit": unit_label,
                        "time": time_label,
                        "unit_index": i,
                        "time_index": j,
                        "observation_count": int(panel.cell_counts[i, j]),
                        "category": category,
                        "category_index": k,
                        "reference_category": reference_category,
                        "is_reference_category": k == len(categories) - 1,
                        "counterfactual_probability": counterfactual_array[i, j, k],
                        "treated_probability": treated_array[i, j, k],
                        "difference_probability": (
                            treated_array[i, j, k]
                            - counterfactual_array[i, j, k]
                        ),
                    }
                )
    return pd.DataFrame.from_records(records)


def build_causal_result(
    panel: PanelData,
    masks: CausalMasks,
    counterfactual_fit: FactorizationFit,
    treated_fit: FactorizationFit,
    *,
    posterior_effects_seed: Optional[int] = None,
) -> CausalResult:
    """Build causal tables from full-coordinate masked factorization fits."""

    _validate_inputs(panel, masks, counterfactual_fit, treated_fit)
    if posterior_effects_seed is None:
        posterior_effects_seed = (counterfactual_fit.seed + 2) % (2**32)
    if (
        not isinstance(posterior_effects_seed, (int, np.integer))
        or isinstance(posterior_effects_seed, (bool, np.bool_))
        or posterior_effects_seed < 0
        or posterior_effects_seed > np.iinfo(np.uint32).max
    ):
        raise ValueError(
            "posterior_effects_seed must be an integer between 0 and 2**32 - 1."
        )
    conversion_kwargs = {
        "family": panel.family,
        "sigma": panel.sigma,
        "mu": panel.mu,
        "categories": panel.categories,
    }
    counterfactual_distributions = distribution_parameters_from_eta(
        counterfactual_fit.natural_parameters,
        **conversion_kwargs,
    )
    treated_distributions = distribution_parameters_from_eta(
        treated_fit.natural_parameters,
        **conversion_kwargs,
    )
    counterfactuals = _counterfactual_table(
        panel,
        masks,
        counterfactual_fit,
        counterfactual_distributions,
    )
    effects = _effect_table(
        panel,
        masks,
        counterfactual_fit,
        treated_fit,
        counterfactual_distributions,
        treated_distributions,
    )
    category_effects = None
    if panel.family == "multinomial":
        category_effects = _category_effect_table(
            panel,
            masks,
            counterfactual_distributions,
            treated_distributions,
        )
    return CausalResult(
        counterfactuals=counterfactuals,
        effects=effects,
        category_effects=category_effects,
        counterfactual_fit=counterfactual_fit,
        treated_fit=treated_fit,
        counterfactual_distributions=counterfactual_distributions,
        treated_distributions=treated_distributions,
        masks=masks,
        panel=panel,
        posterior_effects_seed=int(posterior_effects_seed),
    )
