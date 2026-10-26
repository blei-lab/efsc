"""Distribution conversion and tabular results for EFSC fits.

This module transforms fitted natural-parameter surfaces into familiar
family-specific distribution parameters and formats complete panel tables for
downstream imputation and causal workflows.

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
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from ._families import conventional_parameters_from_eta, normalize_family
from .fitting import FactorizationFit
from .panel import PanelData


@dataclass(frozen=True)
class DistributionParameters:
    """Named distribution-parameter tensors reconstructed from one fit."""

    family: str
    parameters: Dict[str, torch.Tensor]
    categories: Optional[tuple] = None

    def __getitem__(self, name: str) -> torch.Tensor:
        """Return a named reconstructed distribution-parameter tensor."""

        return self.parameters[name]

    def keys(self):
        """Return the available distribution-parameter names."""

        return self.parameters.keys()


@dataclass(frozen=True)
class FactorizationTables:
    """User-facing tables for a fitted panel factorization."""

    cells: pd.DataFrame
    imputations: pd.DataFrame
    category_probabilities: Optional[pd.DataFrame]
    distributions: DistributionParameters


@dataclass(frozen=True)
class EFSCResult:
    """Complete result returned by the high-level EFSC estimator.

    ``cells`` reports every panel cell and ``imputations`` selects missing
    cells. ``causal`` is a :class:`~efsc.CausalResult` in causal mode and
    ``None`` in imputation mode. The remaining fields retain fitted tensors,
    configuration, seeds, and software versions for reproducibility.
    """

    mode: str
    cells: pd.DataFrame
    imputations: pd.DataFrame
    category_probabilities: Optional[pd.DataFrame]
    causal: Optional[Any]
    panel: PanelData
    factorization: FactorizationFit
    distributions: DistributionParameters
    configuration: Dict[str, Any]
    derived_seeds: Dict[str, int]
    software_versions: Dict[str, str]

    def sample_predictive(
        self,
        n_draws: int,
        *,
        cells: str = "all",
        random_state: Optional[int] = None,
    ) -> pd.DataFrame:
        """Draw outcomes conditional on posterior-mean fitted parameters.

        These are plug-in predictive replications, not draws from the
        variational parameter distribution. In causal mode, the fitted
        parameters describe the counterfactual distribution.

        ``cells`` must be ``"all"``, ``"observed"``, or ``"missing"``. The
        returned long-format table contains one row per simulated outcome.
        """

        from .predictive import sample_predictive

        return sample_predictive(
            self,
            n_draws=n_draws,
            cells=cells,
            random_state=random_state,
        )

    def complete_panel(
        self,
        *,
        cell_size: Any = "temporal_neighbor_mean",
        random_state: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fill missing cells from posterior-mean fitted distributions.

        Observed outcomes are preserved. In causal mode, missing outcomes are
        drawn from the counterfactual distribution.

        ``cell_size`` may be a positive common size or
        ``"temporal_neighbor_mean"``, which uses adjacent observed cells from
        the same unit. The returned table is in raw long format.
        """

        from .predictive import complete_panel

        return complete_panel(
            self,
            cell_size=cell_size,
            random_state=random_state,
        )


def _check_parameter_outputs(parameters: Dict[str, torch.Tensor]) -> None:
    """Ensure reconstructed distribution parameters are finite."""

    for name, value in parameters.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(
                f"Reconstructed distribution parameter '{name}' is nonfinite."
            )


def distribution_parameters_from_eta(
    natural_parameters: Any,
    *,
    family: str,
    sigma: Optional[float] = None,
    mu: Optional[float] = None,
    categories: Optional[Sequence[Any]] = None,
) -> DistributionParameters:
    """Convert natural parameters to familiar family-specific parameters."""

    family = normalize_family(family)
    eta = torch.as_tensor(natural_parameters)
    if eta.ndim != 3:
        raise ValueError("natural_parameters must have shape (n, T, p).")
    if not torch.is_floating_point(eta):
        eta = eta.to(dtype=torch.float32)
    if not torch.isfinite(eta).all():
        raise ValueError("natural_parameters must be finite.")

    resolved_categories = None
    if family == "multinomial":
        if categories is None:
            raise ValueError("Multinomial conversion requires ordered categories.")
        resolved_categories = tuple(categories)
        if len(resolved_categories) < 2 or eta.shape[2] != len(resolved_categories) - 1:
            raise ValueError(
                "Multinomial categories must contain exactly one more label than "
                "the number of natural-parameter logits."
            )
        if any(
            pd.isna(value) or (isinstance(value, str) and not value.strip())
            for value in resolved_categories
        ):
            raise ValueError("Multinomial categories cannot contain blank labels.")
        try:
            unique_categories = len(set(resolved_categories))
        except TypeError as exc:
            raise ValueError("Multinomial categories must be scalar labels.") from exc
        if unique_categories != len(resolved_categories):
            raise ValueError("Multinomial categories must be unique.")
    parameters = conventional_parameters_from_eta(
        family,
        eta,
        sigma=sigma,
        mu=mu,
    )

    _check_parameter_outputs(parameters)
    return DistributionParameters(
        family=family,
        parameters=parameters,
        categories=resolved_categories,
    )


def _validate_roles(panel: PanelData, roles: Optional[Any]) -> np.ndarray:
    """Validate or construct cell-role labels for a fitted panel."""

    if roles is None:
        return np.where(panel.observed_mask, "observed", "missing")
    role_array = np.asarray(roles, dtype=object)
    if role_array.shape != (panel.n, panel.T):
        raise ValueError("roles must have shape (n, T).")
    for value in role_array.reshape(-1):
        if pd.isna(value) or (isinstance(value, str) and not value.strip()):
            raise ValueError("roles cannot contain blank values.")
    return role_array.copy()


def _base_cell_table(panel: PanelData, roles: np.ndarray) -> pd.DataFrame:
    """Build the common unit-time records for a fitted panel."""

    records = []
    for i, unit_label in enumerate(panel.unit_labels):
        for j, time_label in enumerate(panel.time_labels):
            records.append(
                {
                    "unit": unit_label,
                    "time": time_label,
                    "unit_index": i,
                    "time_index": j,
                    "observation_count": int(panel.cell_counts[i, j]),
                    "observed": bool(panel.observed_mask[i, j]),
                    "role": roles[i, j],
                }
            )
    return pd.DataFrame.from_records(records)


def _category_probability_table(
    panel: PanelData,
    probabilities: np.ndarray,
    roles: np.ndarray,
) -> pd.DataFrame:
    """Build a long-form multinomial category-probability table."""

    records = []
    categories = panel.categories
    if categories is None:
        raise ValueError("Multinomial panel is missing its category order.")
    for i, unit_label in enumerate(panel.unit_labels):
        for j, time_label in enumerate(panel.time_labels):
            for k, category in enumerate(categories):
                records.append(
                    {
                        "unit": unit_label,
                        "time": time_label,
                        "unit_index": i,
                        "time_index": j,
                        "category": category,
                        "category_index": k,
                        "reference_category": k == len(categories) - 1,
                        "probability": probabilities[i, j, k],
                        "observed": bool(panel.observed_mask[i, j]),
                        "role": roles[i, j],
                    }
                )
    return pd.DataFrame.from_records(records)


def build_factorization_tables(
    panel: PanelData,
    fit: FactorizationFit,
    *,
    roles: Optional[Any] = None,
) -> FactorizationTables:
    """Format a completed factorization fit as panel-level pandas tables."""

    if not isinstance(panel, PanelData):
        raise TypeError("panel must be a PanelData object.")
    if not isinstance(fit, FactorizationFit):
        raise TypeError("fit must be a FactorizationFit object.")
    if fit.family != panel.family:
        raise ValueError("The panel and fit family must match.")
    eta = fit.natural_parameters
    if eta.shape[:2] != (panel.n, panel.T):
        raise ValueError("The fitted natural parameters do not match the panel shape.")

    role_array = _validate_roles(panel, roles)
    distributions = distribution_parameters_from_eta(
        eta,
        family=panel.family,
        sigma=panel.sigma,
        mu=panel.mu,
        categories=panel.categories,
    )
    cells = _base_cell_table(panel, role_array)
    eta_array = eta.detach().cpu().numpy()
    for index in range(eta_array.shape[2]):
        cells[f"eta_{index}"] = eta_array[..., index].reshape(-1)

    category_probabilities = None
    for name, values in distributions.parameters.items():
        values_array = values.detach().cpu().numpy()
        if name == "probabilities":
            for index in range(values_array.shape[2]):
                cells[f"probability_{index}"] = values_array[..., index].reshape(-1)
            category_probabilities = _category_probability_table(
                panel,
                values_array,
                role_array,
            )
        else:
            cells[name] = values_array.reshape(-1)

    imputations = cells.loc[~cells["observed"]].copy().reset_index(drop=True)
    return FactorizationTables(
        cells=cells,
        imputations=imputations,
        category_probabilities=category_probabilities,
        distributions=distributions,
    )
