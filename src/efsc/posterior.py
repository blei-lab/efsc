"""Posterior Monte Carlo summaries for EFSC causal estimands.

This module independently samples the treated and counterfactual variational
factorizations and reports draws or summaries of the manuscript's formal ECE
and ECD on observed treated-post cells.

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
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from .causal import plugin_ecd_from_eta, plugin_ece_from_eta
from .fitting import FactorizationFit, reconstruct_natural_parameters
from ._validation import resolve_seed, validate_positive_integer


def _sample_eta(
    fit: FactorizationFit,
    *,
    n_draws: int,
    n: int,
    T: int,
    n_categories: Optional[int],
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw factorization parameters and reconstruct natural-parameter surfaces."""

    mean = fit.variational_mean
    standard_deviation = torch.exp(0.5 * fit.variational_log_variance)
    noise = torch.randn(
        (n_draws, fit.latent_dimension),
        dtype=mean.dtype,
        device=mean.device,
        generator=generator,
    )
    parameter_draws = mean[None, :] + standard_deviation[None, :] * noise
    eta_draws = [
        reconstruct_natural_parameters(
            parameter_draw,
            family=fit.family,
            n=n,
            T=T,
            rank=fit.rank,
            n_categories=n_categories,
        )
        for parameter_draw in parameter_draws
    ]
    return torch.stack(eta_draws, dim=0)


def _effect_draw_arrays(result, n_draws: int, seed: int):
    """Draw independently coupled posterior ECE and ECD arrays."""

    counterfactual_fit = result.counterfactual_fit
    treated_fit = result.treated_fit
    if counterfactual_fit.variational_mean.device != treated_fit.variational_mean.device:
        raise ValueError("Counterfactual and treated fits must use the same device.")
    device = counterfactual_fit.variational_mean.device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    n_categories = (
        None if result.panel.categories is None else len(result.panel.categories)
    )

    with torch.no_grad():
        counterfactual_eta = _sample_eta(
            counterfactual_fit,
            n_draws=n_draws,
            n=result.panel.n,
            T=result.panel.T,
            n_categories=n_categories,
            generator=generator,
        )
        treated_eta = _sample_eta(
            treated_fit,
            n_draws=n_draws,
            n=result.panel.n,
            T=result.panel.T,
            n_categories=n_categories,
            generator=generator,
        )
        selected = result.masks.treated_mask
        counterfactual_selected = counterfactual_eta[:, selected, :]
        treated_selected = treated_eta[:, selected, :]
        ece = plugin_ece_from_eta(treated_selected, counterfactual_selected)
        ecd = plugin_ecd_from_eta(
            result.panel.family,
            treated_selected,
            counterfactual_selected,
        )

    if not torch.isfinite(ece).all() or not torch.isfinite(ecd).all():
        raise FloatingPointError(
            "Posterior effect sampling produced nonfinite ECE or ECD draws."
        )
    return ece.detach().cpu().numpy(), ecd.detach().cpu().numpy()


def _selected_cells(result):
    """Return metadata for observed treated-post effect cells."""

    cells = []
    for i, unit_label in enumerate(result.panel.unit_labels):
        for j, time_label in enumerate(result.panel.time_labels):
            if result.masks.treated_mask[i, j]:
                cells.append(
                    (
                        unit_label,
                        time_label,
                        i,
                        j,
                        int(result.panel.cell_counts[i, j]),
                    )
                )
    return cells


def sample_causal_effects(
    result,
    *,
    n_draws: int,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Return long-form posterior draws of formal ECE and ECD."""

    n_draws = validate_positive_integer(n_draws, "n_draws")
    seed = resolve_seed(random_state, default=result.posterior_effects_seed)
    ece, ecd = _effect_draw_arrays(result, n_draws, seed)
    cells = _selected_cells(result)
    records = []
    for cell_index, (unit, time, i, j, observation_count) in enumerate(cells):
        for draw_index in range(n_draws):
            record = {
                "unit": unit,
                "time": time,
                "unit_index": i,
                "time_index": j,
                "observation_count": observation_count,
                "draw": draw_index + 1,
            }
            for parameter_index in range(ece.shape[2]):
                record[f"ece_eta_{parameter_index}"] = ece[
                    draw_index,
                    cell_index,
                    parameter_index,
                ]
            record["ecd"] = ecd[draw_index, cell_index]
            records.append(record)
    return pd.DataFrame.from_records(records)


def summarize_causal_effects(
    result,
    *,
    n_draws: int,
    level: float = 0.95,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Return posterior means, SDs, and central intervals by effect cell."""

    n_draws = validate_positive_integer(n_draws, "n_draws", minimum=2)
    if (
        not isinstance(level, Real)
        or isinstance(level, (bool, np.bool_))
        or not np.isfinite(level)
        or level <= 0
        or level >= 1
    ):
        raise ValueError("level must be a finite number strictly between 0 and 1.")
    seed = resolve_seed(random_state, default=result.posterior_effects_seed)
    ece, ecd = _effect_draw_arrays(result, n_draws, seed)
    lower_probability = (1.0 - float(level)) / 2.0
    upper_probability = 1.0 - lower_probability
    cells = _selected_cells(result)
    records = []
    for cell_index, (unit, time, i, j, observation_count) in enumerate(cells):
        record = {
            "unit": unit,
            "time": time,
            "unit_index": i,
            "time_index": j,
            "observation_count": observation_count,
            "n_draws": n_draws,
            "level": float(level),
        }
        for parameter_index in range(ece.shape[2]):
            values = ece[:, cell_index, parameter_index]
            prefix = f"ece_eta_{parameter_index}"
            record[f"{prefix}_mean"] = np.mean(values)
            record[f"{prefix}_sd"] = np.std(values, ddof=1)
            record[f"{prefix}_lower"] = np.quantile(values, lower_probability)
            record[f"{prefix}_upper"] = np.quantile(values, upper_probability)
        ecd_values = ecd[:, cell_index]
        record["ecd_mean"] = np.mean(ecd_values)
        record["ecd_sd"] = np.std(ecd_values, ddof=1)
        record["ecd_lower"] = np.quantile(ecd_values, lower_probability)
        record["ecd_upper"] = np.quantile(ecd_values, upper_probability)
        records.append(record)
    return pd.DataFrame.from_records(records)
