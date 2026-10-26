"""Validated panel ingestion and causal-mask construction for EFSC.

This module converts raw long-format observations into the canonical panel
representation used by the high-level EFSC interface. It also validates
absorbing-treatment metadata and constructs the counterfactual and treated
fitting masks.

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
from numbers import Real
from os import PathLike
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ._families import normalize_family


PathInput = Union[str, PathLike]


@dataclass
class PanelData:
    """Validated raw observations and their resolved panel structure."""

    data: pd.DataFrame
    outcome_name: str
    unit_name: str
    time_name: str
    family: str
    unit_labels: Tuple[Any, ...]
    time_labels: Tuple[Any, ...]
    time_kind: str
    time_step: Optional[Real]
    observations: np.ndarray
    observed_mask: np.ndarray
    cell_counts: np.ndarray
    categories: Optional[Tuple[Any, ...]] = None
    sigma: Optional[float] = None
    mu: Optional[float] = None

    @property
    def n(self) -> int:
        """Return the number of panel units."""

        return len(self.unit_labels)

    @property
    def T(self) -> int:
        """Return the number of panel time points."""

        return len(self.time_labels)

    @property
    def reference_category(self) -> Optional[Any]:
        """Return the multinomial reference category, when applicable."""

        if self.categories is None:
            return None
        return self.categories[-1]


@dataclass
class CausalMasks:
    """Absorbing-treatment assignments and observed fitting masks."""

    treated_post_mask: np.ndarray
    counterfactual_mask: np.ndarray
    treated_mask: np.ndarray
    adoption_times: Dict[Any, Optional[Any]]


def _is_blank(value: Any) -> bool:
    """Return whether a scalar label or observation is blank or missing."""

    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _validate_labels(values: pd.Series, name: str) -> None:
    """Validate that a label series contains nonblank hashable values."""

    invalid = values.map(_is_blank)
    if invalid.any():
        rows = values.index[invalid].tolist()
        raise ValueError(
            f"Column '{name}' contains blank or missing values at rows {rows}."
        )
    for value in values:
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError(f"Column '{name}' must contain scalar labels.") from exc


def _ordered_unique(values: pd.Series) -> Tuple[Any, ...]:
    """Return deterministically ordered unique values from a series."""

    unique = list(pd.unique(values))
    try:
        unique.sort()
    except TypeError:
        unique.sort(key=lambda value: (type(value).__name__, repr(value)))
    return tuple(unique)


def _same_labels(left: Sequence[Any], right: Sequence[Any]) -> bool:
    """Return whether two sequences contain the same labels exactly once."""

    if len(left) != len(right):
        return False
    unmatched = list(right)
    for value in left:
        for index, candidate in enumerate(unmatched):
            if value == candidate:
                unmatched.pop(index)
                break
        else:
            return False
    return not unmatched


def _validate_explicit_time_order(
    labels: Sequence[Any], time_order: Sequence[Any]
) -> Tuple[Any, ...]:
    """Validate and return an explicit ordering of all observed time labels."""

    ordered = tuple(time_order)
    if not _same_labels(ordered, labels) or len(set(ordered)) != len(ordered):
        raise ValueError(
            "time_order must contain every observed time label exactly once."
        )
    return ordered


def _all_numeric(labels: Sequence[Any]) -> bool:
    """Return whether every label is a non-Boolean real number."""

    return all(
        isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
        for value in labels
    )


def _resolve_time_axis(
    values: pd.Series,
    time_order: Optional[Sequence[Any]],
) -> Tuple[Tuple[Any, ...], str, Optional[Real]]:
    """Resolve ordered time labels, their kind, and any numeric step size."""

    labels = tuple(pd.unique(values))

    if _all_numeric(labels):
        ordered = tuple(sorted(labels))
        if time_order is not None:
            supplied = _validate_explicit_time_order(labels, time_order)
            if supplied != ordered:
                raise ValueError("Numeric time_order must be increasing.")
        if len(ordered) < 2:
            return ordered, "numeric", None
        numeric = np.asarray(ordered, dtype=float)
        differences = np.diff(numeric)
        step = differences[0]
        tolerance = max(1e-12, abs(step) * 1e-9)
        if step <= 0 or not np.allclose(
            differences,
            step,
            rtol=1e-9,
            atol=tolerance,
        ):
            raise ValueError("Numeric time labels must be equally spaced.")
        return ordered, "numeric", ordered[1] - ordered[0]

    parsed_dates = [pd.to_datetime(value, errors="coerce") for value in labels]
    if all(not pd.isna(value) for value in parsed_dates):
        positions = np.argsort(np.asarray(parsed_dates, dtype="datetime64[ns]"))
        ordered = tuple(labels[index] for index in positions)
        if time_order is not None:
            supplied = _validate_explicit_time_order(labels, time_order)
            if supplied != ordered:
                raise ValueError("Parseable date time_order must be chronological.")
        return ordered, "datetime", None

    if time_order is None:
        raise ValueError(
            "Arbitrary time labels require an explicit time_order containing "
            "every observed label."
        )
    return _validate_explicit_time_order(labels, time_order), "ordered", None


def _validate_categories(categories: Optional[Sequence[Any]]) -> Tuple[Any, ...]:
    """Validate and return an ordered multinomial category sequence."""

    if categories is None:
        raise ValueError("Multinomial panels require explicit ordered categories.")
    categories = tuple(categories)
    if len(categories) < 2:
        raise ValueError("categories must contain at least two labels.")
    if any(_is_blank(value) for value in categories):
        raise ValueError("categories cannot contain blank or missing labels.")
    try:
        unique_count = len(set(categories))
    except TypeError as exc:
        raise ValueError("categories must contain scalar labels.") from exc
    if unique_count != len(categories):
        raise ValueError("categories must contain unique labels.")
    return categories


def _validate_outcome(
    frame: pd.DataFrame,
    outcome: str,
    family: str,
    categories: Optional[Sequence[Any]],
    sigma: Optional[float],
    mu: Optional[float],
) -> Tuple[pd.Series, Optional[Tuple[Any, ...]], Optional[float], Optional[float]]:
    """Validate outcomes and resolve family-specific auxiliary parameters."""

    values = frame[outcome]
    invalid = values.map(_is_blank)
    if invalid.any():
        rows = values.index[invalid].tolist()
        raise ValueError(
            f"Column '{outcome}' contains blank or missing values at rows {rows}."
        )

    if family == "multinomial":
        resolved_categories = _validate_categories(categories)
        unlisted = [value for value in pd.unique(values) if value not in resolved_categories]
        if unlisted:
            raise ValueError(
                f"Outcome column '{outcome}' contains categories not listed in "
                f"categories: {unlisted}."
            )
        return values.copy(), resolved_categories, None, None

    numeric = pd.to_numeric(values, errors="coerce")
    numeric_array = numeric.to_numpy(dtype=float)
    if numeric.isna().any() or not np.isfinite(numeric_array).all():
        raise ValueError(f"Outcome column '{outcome}' must contain finite numbers.")

    resolved_sigma = None
    resolved_mu = None
    if family == "poisson":
        if np.any(numeric_array < 0) or not np.equal(
            numeric_array, np.floor(numeric_array)
        ).all():
            raise ValueError("Poisson outcomes must be nonnegative integers.")
    elif family == "bernoulli":
        if not np.isin(numeric_array, [0.0, 1.0]).all():
            raise ValueError("Bernoulli outcomes must be exactly 0 or 1.")
    elif family == "exponential":
        if np.any(numeric_array < 0):
            raise ValueError("Exponential outcomes must be nonnegative.")
    elif family == "chi_square":
        if np.any(numeric_array <= 0):
            raise ValueError("Chi-square outcomes must be strictly positive.")
    elif family == "gaussian_known_sigma":
        if sigma is None or not isinstance(sigma, Real) or not np.isfinite(sigma):
            raise ValueError("gaussian_known_sigma requires a finite sigma > 0.")
        if sigma <= 0:
            raise ValueError("gaussian_known_sigma requires sigma > 0.")
        resolved_sigma = float(sigma)
    elif family == "laplace_known_mu":
        if mu is None or not isinstance(mu, Real) or not np.isfinite(mu):
            raise ValueError("laplace_known_mu requires a finite mu.")
        resolved_mu = float(mu)

    return pd.Series(numeric_array, index=values.index), None, resolved_sigma, resolved_mu


def prepare_panel_data(
    data: pd.DataFrame,
    *,
    outcome: str,
    unit: str,
    time: str,
    family: str,
    time_order: Optional[Sequence[Any]] = None,
    categories: Optional[Sequence[Any]] = None,
    sigma: Optional[float] = None,
    mu: Optional[float] = None,
) -> PanelData:
    """Validate a raw long-format DataFrame and construct its panel cells."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("The panel data cannot be empty.")
    if len({outcome, unit, time}) != 3:
        raise ValueError("outcome, unit, and time must name three distinct columns.")

    required = {outcome, unit, time}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}.")

    frame = data.copy().reset_index(drop=True)
    _validate_labels(frame[unit], unit)
    _validate_labels(frame[time], time)
    normalized_family = normalize_family(family)
    validated_outcome, resolved_categories, resolved_sigma, resolved_mu = (
        _validate_outcome(
            frame,
            outcome,
            normalized_family,
            categories,
            sigma,
            mu,
        )
    )
    frame[outcome] = validated_outcome

    unit_labels = _ordered_unique(frame[unit])
    time_labels, time_kind, time_step = _resolve_time_axis(frame[time], time_order)
    unit_to_index = {label: index for index, label in enumerate(unit_labels)}
    time_to_index = {label: index for index, label in enumerate(time_labels)}

    numeric_outcome = normalized_family != "multinomial"
    empty_dtype = float if numeric_outcome else object
    observations = np.empty((len(unit_labels), len(time_labels)), dtype=object)
    for i in range(len(unit_labels)):
        for j in range(len(time_labels)):
            observations[i, j] = np.asarray([], dtype=empty_dtype)

    grouped = frame.groupby([unit, time], sort=False, dropna=False)[outcome]
    for (unit_label, time_label), cell_values in grouped:
        i = unit_to_index[unit_label]
        j = time_to_index[time_label]
        observations[i, j] = cell_values.to_numpy(
            dtype=float if numeric_outcome else object,
            copy=True,
        )

    cell_counts = np.zeros((len(unit_labels), len(time_labels)), dtype=int)
    for i in range(len(unit_labels)):
        for j in range(len(time_labels)):
            cell_counts[i, j] = observations[i, j].size
    observed_mask = cell_counts > 0

    if not observed_mask.any(axis=1).all():
        raise ValueError("Every panel unit must contain at least one observed cell.")
    if not observed_mask.any(axis=0).all():
        raise ValueError("Every panel time period must contain at least one observed cell.")

    return PanelData(
        data=frame,
        outcome_name=outcome,
        unit_name=unit,
        time_name=time,
        family=normalized_family,
        unit_labels=unit_labels,
        time_labels=time_labels,
        time_kind=time_kind,
        time_step=time_step,
        observations=observations,
        observed_mask=observed_mask,
        cell_counts=cell_counts,
        categories=resolved_categories,
        sigma=resolved_sigma,
        mu=resolved_mu,
    )


def read_panel_data_csv(
    path: PathInput,
    *,
    outcome: str,
    unit: str,
    time: str,
    family: str,
    time_order: Optional[Sequence[Any]] = None,
    categories: Optional[Sequence[Any]] = None,
    sigma: Optional[float] = None,
    mu: Optional[float] = None,
    **read_csv_kwargs: Any,
) -> PanelData:
    """Read a long-format CSV and return a validated :class:`PanelData`."""

    frame = pd.read_csv(path, **read_csv_kwargs)
    return prepare_panel_data(
        frame,
        outcome=outcome,
        unit=unit,
        time=time,
        family=family,
        time_order=time_order,
        categories=categories,
        sigma=sigma,
        mu=mu,
    )


def _metadata_frame(causal_metadata: Any) -> pd.DataFrame:
    """Normalize causal metadata supplied as a table, mapping, or CSV path."""

    if isinstance(causal_metadata, pd.DataFrame):
        return causal_metadata.copy().reset_index(drop=True)
    if isinstance(causal_metadata, Mapping):
        return pd.DataFrame(
            {
                "unit": list(causal_metadata.keys()),
                "adoption_time": list(causal_metadata.values()),
            }
        )
    if isinstance(causal_metadata, (str, PathLike)):
        return pd.read_csv(causal_metadata)
    raise TypeError(
        "causal_metadata must be a CSV path, pandas DataFrame, or mapping."
    )


def _match_time_label(panel: PanelData, value: Any) -> Optional[Any]:
    """Match a metadata adoption time to a canonical panel time label."""

    for label in panel.time_labels:
        if value == label:
            return label
    if panel.time_kind == "numeric" and isinstance(value, str):
        try:
            numeric_value = float(value.strip())
        except ValueError:
            return None
        for label in panel.time_labels:
            if numeric_value == label:
                return label
    return None


def build_causal_masks(
    panel: PanelData,
    causal_metadata: Any,
    *,
    metadata_unit: str = "unit",
    adoption_time: str = "adoption_time",
) -> CausalMasks:
    """Validate absorbing-treatment metadata and construct causal masks."""

    if not isinstance(panel, PanelData):
        raise TypeError("panel must be a PanelData object.")
    metadata = _metadata_frame(causal_metadata)
    required = {metadata_unit, adoption_time}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Missing causal metadata columns: {sorted(missing)}.")
    _validate_labels(metadata[metadata_unit], metadata_unit)
    if metadata[metadata_unit].duplicated().any():
        duplicates = metadata.loc[
            metadata[metadata_unit].duplicated(keep=False), metadata_unit
        ].tolist()
        raise ValueError(f"Causal metadata contains duplicate units: {duplicates}.")

    panel_units = set(panel.unit_labels)
    unknown_units = [
        value for value in metadata[metadata_unit] if value not in panel_units
    ]
    if unknown_units:
        raise ValueError(
            f"Causal metadata units are not present in the panel: {unknown_units}."
        )

    adoption_times: Dict[Any, Optional[Any]] = {
        unit_label: None for unit_label in panel.unit_labels
    }
    for row in metadata.itertuples(index=False, name=None):
        row_values = dict(zip(metadata.columns, row))
        unit_label = row_values[metadata_unit]
        value = row_values[adoption_time]
        if _is_blank(value) or (
            isinstance(value, str) and value.strip().lower() == "never"
        ):
            adoption_times[unit_label] = None
            continue
        matched = _match_time_label(panel, value)
        if matched is None:
            raise ValueError(
                f"Adoption time {value!r} for unit {unit_label!r} is not a panel "
                "time label."
            )
        adoption_times[unit_label] = matched

    treated_post = np.zeros((panel.n, panel.T), dtype=bool)
    time_to_index = {label: index for index, label in enumerate(panel.time_labels)}
    for i, unit_label in enumerate(panel.unit_labels):
        adoption = adoption_times[unit_label]
        if adoption is None:
            continue
        first_treated = time_to_index[adoption]
        if not panel.observed_mask[i, :first_treated].any():
            raise ValueError(
                f"Treated unit {unit_label!r} must have at least one observed "
                "pretreatment cell."
            )
        treated_post[i, first_treated:] = True

    counterfactual = panel.observed_mask & ~treated_post
    treated = panel.observed_mask & treated_post
    if not counterfactual.any(axis=1).all():
        raise ValueError(
            "The counterfactual fit must retain observed cells for every unit."
        )
    if not counterfactual.any(axis=0).all():
        raise ValueError(
            "The counterfactual fit must retain observed cells for every time period."
        )
    if not treated.any():
        raise ValueError("Causal mode requires at least one observed treated-post cell.")

    return CausalMasks(
        treated_post_mask=treated_post,
        counterfactual_mask=counterfactual,
        treated_mask=treated,
        adoption_times=adoption_times,
    )
