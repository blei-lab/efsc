"""Posterior-mean predictive sampling and panel completion for EFSC.

This module draws replicated observations from fitted cell distributions and
constructs one long-format completed panel while preserving all observed
outcomes.

Author:
    Hector Rodriguez-Deniz

Affiliation:
    Data Science Institute, Columbia University

Date:
    August 2026

License:
    MIT
"""

from numbers import Integral
from typing import Any, Optional

import numpy as np
import pandas as pd

from ._validation import resolve_seed, validate_positive_integer

_CELL_SELECTORS = {"all", "observed", "missing"}


def _numpy_parameters(result):
    """Copy fitted distribution parameters from tensors to NumPy arrays."""

    return {
        name: value.detach().cpu().numpy()
        for name, value in result.distributions.parameters.items()
    }


def _draw_observations(
    result,
    parameters,
    i: int,
    j: int,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw observations from one fitted cell distribution."""

    family = result.panel.family
    if family in {"gaussian", "gaussian_known_sigma"}:
        return rng.normal(
            loc=parameters["mean"][i, j],
            scale=np.sqrt(parameters["variance"][i, j]),
            size=size,
        )
    if family == "poisson":
        return rng.poisson(lam=parameters["rate"][i, j], size=size)
    if family == "bernoulli":
        return rng.binomial(n=1, p=parameters["probability"][i, j], size=size)
    if family == "exponential":
        return rng.exponential(scale=1.0 / parameters["rate"][i, j], size=size)
    if family == "laplace_known_mu":
        return rng.laplace(
            loc=parameters["location"][i, j],
            scale=parameters["scale"][i, j],
            size=size,
        )
    if family == "chi_square":
        return rng.chisquare(
            df=parameters["degrees_of_freedom"][i, j],
            size=size,
        )
    if family == "multinomial":
        return rng.choice(
            np.asarray(result.panel.categories, dtype=object),
            size=size,
            p=parameters["probabilities"][i, j],
        )
    raise ValueError(f"Unsupported family '{family}'.")


def _role_array(result) -> np.ndarray:
    """Reshape tabular cell-role labels to the panel grid."""

    ordered = result.cells.sort_values(["unit_index", "time_index"])
    return ordered["role"].to_numpy(dtype=object).reshape(
        result.panel.n,
        result.panel.T,
    )


def sample_predictive(
    result,
    *,
    n_draws: int,
    cells: str = "all",
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Draw outcomes conditional on the top-level posterior-mean parameters.

    The variational parameters are not resampled. In causal mode, the
    top-level fit is the counterfactual factorization.
    """

    n_draws = validate_positive_integer(n_draws, "n_draws")
    if cells not in _CELL_SELECTORS:
        raise ValueError("cells must be 'all', 'observed', or 'missing'.")
    seed = resolve_seed(
        random_state,
        default=result.derived_seeds["predictive"],
    )
    rng = np.random.default_rng(seed)
    parameters = _numpy_parameters(result)
    roles = _role_array(result)

    if cells == "all":
        selected = np.ones((result.panel.n, result.panel.T), dtype=bool)
    elif cells == "observed":
        selected = result.panel.observed_mask
    else:
        selected = ~result.panel.observed_mask

    records = []
    for i, unit_label in enumerate(result.panel.unit_labels):
        for j, time_label in enumerate(result.panel.time_labels):
            if not selected[i, j]:
                continue
            outcomes = _draw_observations(
                result,
                parameters,
                i,
                j,
                n_draws,
                rng,
            )
            for draw, outcome in enumerate(outcomes, start=1):
                records.append(
                    {
                        "unit": unit_label,
                        "time": time_label,
                        "unit_index": i,
                        "time_index": j,
                        "draw": draw,
                        "outcome": outcome,
                        "observed": bool(result.panel.observed_mask[i, j]),
                        "role": roles[i, j],
                    }
                )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "unit",
            "time",
            "unit_index",
            "time_index",
            "draw",
            "outcome",
            "observed",
            "role",
        ],
    )


def _missing_cell_size(panel, i: int, j: int, cell_size: Any) -> int:
    """Resolve an explicit or temporal-neighbor missing-cell sample size."""

    if isinstance(cell_size, Integral) and not isinstance(
        cell_size,
        (bool, np.bool_),
    ):
        return validate_positive_integer(cell_size, "cell_size")
    if cell_size != "temporal_neighbor_mean":
        raise ValueError(
            "cell_size must be a positive integer or 'temporal_neighbor_mean'."
        )

    observed_times = np.flatnonzero(panel.observed_mask[i])
    earlier = observed_times[observed_times < j]
    later = observed_times[observed_times > j]
    neighbor_counts = []
    if earlier.size:
        neighbor_counts.append(panel.cell_counts[i, earlier[-1]])
    if later.size:
        neighbor_counts.append(panel.cell_counts[i, later[0]])
    if not neighbor_counts:
        raise ValueError(
            "cell_size cannot be inferred because this unit has no observed "
            "temporal neighbor."
        )
    mean_count = float(np.mean(neighbor_counts))
    return max(1, int(np.floor(mean_count + 0.5)))


def complete_panel(
    result,
    *,
    cell_size: Any = "temporal_neighbor_mean",
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Fill only missing cells from posterior-mean fitted distributions.

    In causal mode, the top-level fitted distribution is counterfactual.
    """

    if not (
        isinstance(cell_size, Integral)
        and not isinstance(cell_size, (bool, np.bool_))
    ) and cell_size != "temporal_neighbor_mean":
        raise ValueError(
            "cell_size must be a positive integer or 'temporal_neighbor_mean'."
        )
    if isinstance(cell_size, Integral):
        validate_positive_integer(cell_size, "cell_size")

    seed = resolve_seed(
        random_state,
        default=result.derived_seeds["predictive"],
    )
    rng = np.random.default_rng(seed)
    parameters = _numpy_parameters(result)
    panel = result.panel
    observed = panel.data[
        [panel.unit_name, panel.time_name, panel.outcome_name]
    ].rename(
        columns={
            panel.unit_name: "unit",
            panel.time_name: "time",
            panel.outcome_name: "outcome",
        }
    )
    observed = observed.copy()
    observed["imputed"] = False

    imputed_records = []
    for i, unit_label in enumerate(panel.unit_labels):
        for j, time_label in enumerate(panel.time_labels):
            if panel.observed_mask[i, j]:
                continue
            size = _missing_cell_size(panel, i, j, cell_size)
            outcomes = _draw_observations(
                result,
                parameters,
                i,
                j,
                size,
                rng,
            )
            for outcome in outcomes:
                imputed_records.append(
                    {
                        "unit": unit_label,
                        "time": time_label,
                        "outcome": outcome,
                        "imputed": True,
                    }
                )

    if not imputed_records:
        return observed.reset_index(drop=True)
    imputed = pd.DataFrame.from_records(
        imputed_records,
        columns=["unit", "time", "outcome", "imputed"],
    )
    return pd.concat([observed, imputed], ignore_index=True)
