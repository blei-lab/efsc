"""High-level estimator interface for Exponential Family Synthetic Controls.

This module exposes the user-facing EFSC class. It fits validated long-format
panels in either imputation or causal mode and returns complete tabular results
while delegating all mathematics to the shared fitting and result layers.

Author:
    Hector Rodriguez-Deniz

Affiliation:
    Data Science Institute, Columbia University

Date:
    August 2026

License:
    MIT
"""

from dataclasses import asdict
import platform
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .causal_results import build_causal_result
from .fitting import InferenceConfig, fit_factorization
from .panel import (
    CausalMasks,
    PanelData,
    build_causal_masks,
    prepare_panel_data,
    read_panel_data_csv,
)
from .results import EFSCResult, build_factorization_tables
from ._validation import derive_seed, validate_positive_integer, validate_seed


class EFSC:
    """Fit exponential-family panel factorizations and synthetic controls.

    Parameters
    ----------
    family:
        Exponential-family name. See the README for the supported values.
    rank:
        Positive rank of each latent factorization.
    random_state:
        Seed used to derive reproducible fitting and sampling seeds.
    sigma:
        Known standard deviation for ``"gaussian_known_sigma"``.
    mu:
        Known location for ``"laplace_known_mu"``.
    categories:
        Ordered categories for ``"multinomial"``. The last is the reference.
    inference_config:
        Optimization settings. Defaults to :class:`InferenceConfig`.
    """

    def __init__(
        self,
        family: str,
        rank: int,
        random_state: int = 0,
        *,
        sigma: Optional[float] = None,
        mu: Optional[float] = None,
        categories: Optional[Sequence[Any]] = None,
        inference_config: Optional[InferenceConfig] = None,
    ) -> None:
        """Initialize an EFSC estimator and its inference configuration."""

        rank = validate_positive_integer(rank, "rank")
        random_state = validate_seed(random_state)
        if random_state is None:
            raise AssertionError("Unreachable seed validation state.")
        if inference_config is None:
            inference_config = InferenceConfig()
        if not isinstance(inference_config, InferenceConfig):
            raise TypeError("inference_config must be an InferenceConfig object.")

        self.family = family
        self.rank = rank
        self.random_state = random_state
        self.sigma = sigma
        self.mu = mu
        self.categories = None if categories is None else tuple(categories)
        self.inference_config = inference_config
        self.result_: Optional[EFSCResult] = None

    def _configuration(
        self,
        panel: PanelData,
        *,
        mode: str,
        masks: Optional[CausalMasks] = None,
    ) -> Dict[str, Any]:
        """Record resolved data, model, inference, and causal settings."""

        inference = asdict(self.inference_config)
        inference["device"] = str(inference["device"])
        configuration = {
            "mode": mode,
            "family": panel.family,
            "rank": self.rank,
            "random_state": self.random_state,
            "sigma": panel.sigma,
            "mu": panel.mu,
            "categories": panel.categories,
            "outcome": panel.outcome_name,
            "unit": panel.unit_name,
            "time": panel.time_name,
            "time_kind": panel.time_kind,
            "time_step": panel.time_step,
            "inference": inference,
        }
        if masks is not None:
            configuration["causal"] = {
                "adoption_times": dict(masks.adoption_times),
                "treated_fit_scope": "full",
                "counterfactual_training_cells": int(
                    masks.counterfactual_mask.sum()
                ),
                "treated_training_cells": int(masks.treated_mask.sum()),
                "treated_post_targets": int(masks.treated_post_mask.sum()),
                "missing_treated_post_targets": int(
                    (masks.treated_post_mask & ~panel.observed_mask).sum()
                ),
            }
        return configuration

    @staticmethod
    def _software_versions() -> Dict[str, str]:
        """Return versions of the principal runtime dependencies."""

        return {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
        }

    def _derived_seed(self, offset: int) -> int:
        """Derive a deterministic unsigned seed from the estimator seed."""

        return derive_seed(self.random_state, offset)

    def _fit_imputation_panel(self, panel: PanelData) -> EFSCResult:
        """Fit one factorization and format an imputation-mode result."""

        factorization = fit_factorization(
            panel,
            panel.observed_mask,
            rank=self.rank,
            config=self.inference_config,
            seed=self.random_state,
        )
        tables = build_factorization_tables(panel, factorization)
        result = EFSCResult(
            mode="imputation",
            cells=tables.cells,
            imputations=tables.imputations,
            category_probabilities=tables.category_probabilities,
            causal=None,
            panel=panel,
            factorization=factorization,
            distributions=tables.distributions,
            configuration=self._configuration(panel, mode="imputation"),
            derived_seeds={
                "factorization": self.random_state,
                "predictive": self._derived_seed(1),
            },
            software_versions=self._software_versions(),
        )
        self.result_ = result
        return result

    @staticmethod
    def _causal_roles(panel: PanelData, masks: CausalMasks) -> np.ndarray:
        """Label panel cells by their role in a causal fit."""

        roles = np.full(
            (panel.n, panel.T),
            "ordinary_missing",
            dtype=object,
        )
        roles[masks.counterfactual_mask] = "counterfactual_training"
        roles[masks.treated_mask] = "treated_post_observed"
        roles[masks.treated_post_mask & ~panel.observed_mask] = (
            "treated_post_missing"
        )
        return roles

    def _fit_causal_panel(
        self,
        panel: PanelData,
        causal_metadata: Any,
    ) -> EFSCResult:
        """Fit treated and counterfactual factorizations for causal mode."""

        masks = build_causal_masks(panel, causal_metadata)
        counterfactual_fit = fit_factorization(
            panel,
            masks.counterfactual_mask,
            rank=self.rank,
            config=self.inference_config,
            seed=self.random_state,
        )
        treated_fit = fit_factorization(
            panel,
            masks.treated_mask,
            rank=self.rank,
            config=self.inference_config,
            seed=self._derived_seed(1),
        )
        roles = self._causal_roles(panel, masks)
        counterfactual_tables = build_factorization_tables(
            panel,
            counterfactual_fit,
            roles=roles,
        )
        causal = build_causal_result(
            panel,
            masks,
            counterfactual_fit,
            treated_fit,
            posterior_effects_seed=self._derived_seed(2),
        )
        result = EFSCResult(
            mode="causal",
            cells=counterfactual_tables.cells,
            imputations=counterfactual_tables.imputations,
            category_probabilities=counterfactual_tables.category_probabilities,
            causal=causal,
            panel=panel,
            factorization=counterfactual_fit,
            distributions=causal.counterfactual_distributions,
            configuration=self._configuration(
                panel,
                mode="causal",
                masks=masks,
            ),
            derived_seeds={
                "counterfactual": self.random_state,
                "treated": self._derived_seed(1),
                "posterior_effects": self._derived_seed(2),
                "predictive": self._derived_seed(3),
            },
            software_versions=self._software_versions(),
        )
        self.result_ = result
        return result

    def fit(
        self,
        data: pd.DataFrame,
        *,
        outcome: str,
        unit: str,
        time: str,
        time_order: Optional[Sequence[Any]] = None,
        causal_metadata: Optional[Any] = None,
    ) -> EFSCResult:
        """Fit EFSC to a raw long-format pandas DataFrame.

        Each row is one outcome inside a unit-time cell. Omitting
        ``causal_metadata`` selects imputation mode; supplying first-treatment
        times selects absorbing-treatment causal mode.

        Parameters
        ----------
        data:
            Raw long-format observations.
        outcome, unit, time:
            Column names identifying the outcome and panel coordinates.
        time_order:
            Explicit order for nonnumeric, non-date-like time labels.
        causal_metadata:
            First treated periods as a mapping, DataFrame, or CSV path.

        Returns
        -------
        EFSCResult
            Fitted panel results and, in causal mode, a ``causal`` result.
        """

        panel = prepare_panel_data(
            data,
            outcome=outcome,
            unit=unit,
            time=time,
            family=self.family,
            time_order=time_order,
            categories=self.categories,
            sigma=self.sigma,
            mu=self.mu,
        )
        if causal_metadata is None:
            return self._fit_imputation_panel(panel)
        return self._fit_causal_panel(panel, causal_metadata)

    def fit_csv(
        self,
        path: Any,
        *,
        outcome: str,
        unit: str,
        time: str,
        time_order: Optional[Sequence[Any]] = None,
        causal_metadata: Optional[Any] = None,
        read_csv_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> EFSCResult:
        """Read a raw long-format CSV and fit EFSC.

        This is the file-based counterpart of :meth:`fit`. Extra CSV-reader
        options, such as delimiter or missing-value settings, may be supplied
        through ``read_csv_kwargs``.
        """
        if read_csv_kwargs is None:
            read_csv_kwargs = {}
        if not isinstance(read_csv_kwargs, Mapping):
            raise TypeError("read_csv_kwargs must be a mapping.")
        panel = read_panel_data_csv(
            path,
            outcome=outcome,
            unit=unit,
            time=time,
            family=self.family,
            time_order=time_order,
            categories=self.categories,
            sigma=self.sigma,
            mu=self.mu,
            **dict(read_csv_kwargs),
        )
        if causal_metadata is None:
            return self._fit_imputation_panel(panel)
        return self._fit_causal_panel(panel, causal_metadata)
