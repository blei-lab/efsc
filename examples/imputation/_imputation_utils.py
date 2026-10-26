"""Shared simulation and plotting utilities for EFSC imputation examples."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from efsc import EFSCResult


N_UNITS = 16
N_PERIODS = 48
OBSERVATIONS_PER_CELL = 32
CATEGORIES = ("low", "medium", "high")
OFF_DIAGONAL_MISSING_RATE = 0.075
MISSINGNESS_SEED = 2026


@dataclass(frozen=True)
class SurfaceProfile:
    """Family-specific seed and additive-plus-rank-one surface settings."""

    seed: int
    offset: float
    unit_scale: float
    time_scale: float
    interaction_scale: float
    phase: float


SURFACE_PROFILES = {
    "gaussian": SurfaceProfile(101, 0.45, 0.72, 0.50, 0.62, 0.15),
    "gaussian_known_sigma": SurfaceProfile(
        211, 0.20, -0.65, 0.52, 0.48, 0.65
    ),
    "poisson": SurfaceProfile(307, 0.55, 0.62, 0.45, -0.50, 1.10),
    "bernoulli": SurfaceProfile(401, -0.15, -0.75, 0.60, 0.65, 1.80),
    "exponential": SurfaceProfile(503, 0.25, 0.50, -0.48, 0.58, 2.40),
    "laplace_known_mu": SurfaceProfile(601, 0.15, -0.55, 0.50, -0.60, 0.90),
    "chi_square": SurfaceProfile(701, 0.50, 0.45, 0.58, 0.52, 2.80),
    "multinomial": SurfaceProfile(809, 0.20, 0.70, 0.55, 0.60, 0.35),
}


@dataclass(frozen=True)
class ImputationSpec:
    """Family-specific settings for one imputation example."""

    family: str
    parameter: str
    parameter_label: str
    observed_label: str
    sigma: Optional[float] = None
    mu: Optional[float] = None
    category_index: Optional[int] = None


@dataclass(frozen=True)
class SimulatedImputationPanel:
    """A simulated long panel and the surface highlighted in its figure."""

    data: pd.DataFrame
    truth: np.ndarray
    observed_summary: np.ndarray
    missing_mask: np.ndarray
    observed_standard_deviation: Optional[np.ndarray] = None
    true_standard_deviation: Optional[np.ndarray] = None
    observed_category_proportions: Optional[np.ndarray] = None
    true_category_probabilities: Optional[np.ndarray] = None


def _surface(
    *,
    offset: float,
    unit_scale: float,
    time_scale: float,
    interaction_scale: float,
    phase: float = 0.0,
) -> np.ndarray:
    """Construct a smooth additive-plus-rank-one predictor surface."""

    unit_axis = np.linspace(-1.0, 1.0, N_UNITS)
    time = np.arange(N_PERIODS)
    time_wave = np.sin(2.0 * np.pi * time / 12.0 + phase)
    unit_loading = np.cos(np.pi * (unit_axis + 1.0) / 2.0)
    time_loading = np.cos(2.0 * np.pi * time / 12.0 + phase)
    return (
        offset
        + unit_scale * unit_axis[:, None]
        + time_scale * time_wave[None, :]
        + interaction_scale * np.outer(unit_loading, time_loading)
    )


def _missing_mask() -> np.ndarray:
    """Return a diagonal pattern plus reproducible off-diagonal missing cells."""

    mask = np.zeros((N_UNITS, N_PERIODS), dtype=bool)
    for unit_index in range(N_UNITS):
        start = (3 * unit_index + 1) % N_PERIODS
        mask[unit_index, start : min(start + 3, N_PERIODS)] = True
        if start + 3 > N_PERIODS:
            mask[unit_index, : start + 3 - N_PERIODS] = True

    off_diagonal = np.flatnonzero(~mask.ravel())
    n_additional = round(OFF_DIAGONAL_MISSING_RATE * off_diagonal.size)
    rng = np.random.default_rng(MISSINGNESS_SEED)
    selected = rng.choice(off_diagonal, size=n_additional, replace=False)
    mask.ravel()[selected] = True
    return mask


def _family_parameters(spec: ImputationSpec) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Construct data-generating parameters and the plotted true surface."""

    profile = SURFACE_PROFILES[spec.family]
    predictor = _surface(
        offset=profile.offset,
        unit_scale=profile.unit_scale,
        time_scale=profile.time_scale,
        interaction_scale=profile.interaction_scale,
        phase=profile.phase,
    )

    if spec.family == "gaussian":
        eta_mean = _surface(
            offset=2.2,
            unit_scale=profile.unit_scale,
            time_scale=profile.time_scale,
            interaction_scale=profile.interaction_scale,
            phase=profile.phase,
        )
        log_precision = _surface(
            offset=-0.15,
            unit_scale=-0.48,
            time_scale=0.40,
            interaction_scale=-0.36,
            phase=profile.phase + np.pi / 2.0,
        )
        variance = 0.5 * np.exp(-log_precision)
        mean = 0.5 * eta_mean * np.exp(-log_precision)
        return {"mean": mean, "variance": variance}, mean

    if spec.family == "gaussian_known_sigma":
        assert spec.sigma is not None
        mean = spec.sigma * predictor
        return {"mean": mean}, mean

    if spec.family == "poisson":
        rate = np.exp(predictor)
        return {"rate": rate}, rate

    if spec.family == "bernoulli":
        probability = 1.0 / (1.0 + np.exp(-predictor))
        return {"probability": probability}, probability

    if spec.family == "exponential":
        rate = np.exp(predictor)
        mean = 1.0 / rate
        return {"rate": rate, "mean": mean}, mean

    if spec.family == "laplace_known_mu":
        scale = np.exp(-predictor)
        return {"scale": scale}, scale

    if spec.family == "chi_square":
        degrees_of_freedom = 2.0 * np.exp(predictor)
        return {"degrees_of_freedom": degrees_of_freedom}, degrees_of_freedom

    if spec.family == "multinomial":
        logits = np.stack(
            [
                _surface(
                    offset=0.45,
                    unit_scale=profile.unit_scale,
                    time_scale=profile.time_scale,
                    interaction_scale=profile.interaction_scale,
                    phase=profile.phase,
                ),
                _surface(
                    offset=-0.25,
                    unit_scale=-0.55,
                    time_scale=0.70,
                    interaction_scale=-0.50,
                    phase=profile.phase + np.pi / 2.0,
                ),
                np.zeros((N_UNITS, N_PERIODS)),
            ],
            axis=-1,
        )
        logits -= logits.max(axis=-1, keepdims=True)
        weights = np.exp(logits)
        probabilities = weights / weights.sum(axis=-1, keepdims=True)
        assert spec.category_index is not None
        return {"probabilities": probabilities}, probabilities[..., spec.category_index]

    raise ValueError(f"Unsupported imputation family: {spec.family}")


def simulate_panel(
    spec: ImputationSpec,
    *,
    random_state: Optional[int] = None,
) -> SimulatedImputationPanel:
    """Simulate a panel with a structured missing-cell pattern."""

    resolved_seed = (
        SURFACE_PROFILES[spec.family].seed
        if random_state is None
        else random_state
    )
    rng = np.random.default_rng(resolved_seed)
    parameters, truth = _family_parameters(spec)
    missing = _missing_mask()
    observed_summary = np.full((N_UNITS, N_PERIODS), np.nan)
    observed_standard_deviation = (
        np.full((N_UNITS, N_PERIODS), np.nan)
        if spec.family == "gaussian"
        else None
    )
    observed_category_proportions = (
        np.full((N_UNITS, N_PERIODS, len(CATEGORIES)), np.nan)
        if spec.family == "multinomial"
        else None
    )
    records: list[dict[str, object]] = []
    unit_label_width = len(str(N_UNITS))

    for i in range(N_UNITS):
        unit = f"U{i + 1:0{unit_label_width}d}"
        for j in range(N_PERIODS):
            if spec.family == "gaussian":
                outcomes = rng.normal(
                    parameters["mean"][i, j],
                    np.sqrt(parameters["variance"][i, j]),
                    size=OBSERVATIONS_PER_CELL,
                )
            elif spec.family == "gaussian_known_sigma":
                assert spec.sigma is not None
                outcomes = rng.normal(
                    parameters["mean"][i, j],
                    spec.sigma,
                    size=OBSERVATIONS_PER_CELL,
                )
            elif spec.family == "poisson":
                outcomes = rng.poisson(
                    parameters["rate"][i, j],
                    size=OBSERVATIONS_PER_CELL,
                )
            elif spec.family == "bernoulli":
                outcomes = rng.binomial(
                    1,
                    parameters["probability"][i, j],
                    size=OBSERVATIONS_PER_CELL,
                )
            elif spec.family == "exponential":
                outcomes = rng.exponential(
                    1.0 / parameters["rate"][i, j],
                    size=OBSERVATIONS_PER_CELL,
                )
            elif spec.family == "laplace_known_mu":
                assert spec.mu is not None
                outcomes = rng.laplace(
                    spec.mu,
                    parameters["scale"][i, j],
                    size=OBSERVATIONS_PER_CELL,
                )
            elif spec.family == "chi_square":
                outcomes = rng.chisquare(
                    parameters["degrees_of_freedom"][i, j],
                    size=OBSERVATIONS_PER_CELL,
                )
            elif spec.family == "multinomial":
                outcomes = rng.choice(
                    CATEGORIES,
                    size=OBSERVATIONS_PER_CELL,
                    p=parameters["probabilities"][i, j],
                )
            else:
                raise AssertionError("Unreachable family dispatch.")

            if spec.family == "multinomial":
                assert spec.category_index is not None
                assert observed_category_proportions is not None
                for category_index, category in enumerate(CATEGORIES):
                    observed_category_proportions[i, j, category_index] = np.mean(
                        outcomes == category
                    )
                observed_summary[i, j] = observed_category_proportions[
                    i,
                    j,
                    spec.category_index,
                ]
            elif spec.family == "laplace_known_mu":
                assert spec.mu is not None
                observed_summary[i, j] = np.mean(
                    np.abs(outcomes.astype(float) - spec.mu)
                )
            else:
                observed_summary[i, j] = np.mean(outcomes.astype(float))

            if observed_standard_deviation is not None:
                observed_standard_deviation[i, j] = np.std(
                    outcomes.astype(float),
                    ddof=1,
                )

            if missing[i, j]:
                continue
            records.extend(
                {
                    "outcome": outcome.item() if hasattr(outcome, "item") else outcome,
                    "unit": unit,
                    "time": j + 1,
                }
                for outcome in outcomes
            )

    observed_summary[missing] = np.nan
    if observed_standard_deviation is not None:
        observed_standard_deviation[missing] = np.nan
    if observed_category_proportions is not None:
        observed_category_proportions[missing] = np.nan
    true_standard_deviation = (
        np.sqrt(parameters["variance"])
        if spec.family == "gaussian"
        else None
    )
    true_category_probabilities = (
        parameters["probabilities"]
        if spec.family == "multinomial"
        else None
    )
    return SimulatedImputationPanel(
        data=pd.DataFrame.from_records(records),
        truth=truth,
        observed_summary=observed_summary,
        missing_mask=missing,
        observed_standard_deviation=observed_standard_deviation,
        true_standard_deviation=true_standard_deviation,
        observed_category_proportions=observed_category_proportions,
        true_category_probabilities=true_category_probabilities,
    )


def parse_args(spec: ImputationSpec) -> argparse.Namespace:
    """Parse the common command-line options for one family example."""

    parser = argparse.ArgumentParser(
        description=(
            f"Fit the {spec.family} EFSC model to a simulated "
            f"{N_UNITS}-by-{N_PERIODS} imputation panel."
        )
    )
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--figure", type=Path)
    return parser.parse_args()


def _plot_limits(
    observed: np.ndarray,
    truth: np.ndarray,
    fitted: np.ndarray,
    *,
    unit_interval: bool = False,
    nonnegative: bool = False,
) -> tuple[float, float]:
    """Return shared color limits for one observed/true/fitted row."""

    finite_observed = observed[np.isfinite(observed)]
    lower = float(min(finite_observed.min(), truth.min(), fitted.min()))
    upper = float(max(finite_observed.max(), truth.max(), fitted.max()))
    if unit_interval:
        return 0.0, 1.0
    if nonnegative:
        lower = 0.0
    return lower, upper


def _format_axis(axis, title: str) -> None:
    """Apply common labels and ticks to one imputation panel."""

    axis.set_title(title)
    axis.set_xlabel("Time")
    time_ticks = np.linspace(0, N_PERIODS - 1, 5, dtype=int)
    axis.set_xticks(time_ticks, time_ticks + 1)
    unit_label_width = len(str(N_UNITS))
    axis.set_yticks(
        range(N_UNITS),
        [f"U{i + 1:0{unit_label_width}d}" for i in range(N_UNITS)],
    )


def plot_result(
    simulated: SimulatedImputationPanel,
    fitted: np.ndarray,
    fitted_standard_deviation: Optional[np.ndarray],
    fitted_category_probabilities: Optional[np.ndarray],
    spec: ImputationSpec,
    output: Path,
) -> None:
    """Save observed, true, and fitted parameter surfaces."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting imputation examples requires the optional 'examples' dependencies."
        ) from exc

    if spec.family == "gaussian":
        assert simulated.observed_standard_deviation is not None
        assert simulated.true_standard_deviation is not None
        assert fitted_standard_deviation is not None
        figure, axes = plt.subplots(
            2,
            3,
            figsize=(12, 6.6),
            constrained_layout=True,
        )
        rows = (
            (
                simulated.observed_summary,
                simulated.truth,
                fitted,
                "means",
                "mean",
                False,
            ),
            (
                simulated.observed_standard_deviation,
                simulated.true_standard_deviation,
                fitted_standard_deviation,
                "standard deviations",
                "standard deviation",
                False,
            ),
        )
        for row_index, (
            observed,
            truth,
            fitted_surface,
            observed_label,
            parameter_label,
            nonnegative,
        ) in enumerate(rows):
            lower, upper = _plot_limits(
                observed,
                truth,
                fitted_surface,
                nonnegative=nonnegative,
            )
            row_axes = axes[row_index]
            images = (
                row_axes[0].imshow(
                    observed,
                    aspect="auto",
                    vmin=lower,
                    vmax=upper,
                ),
                row_axes[1].imshow(
                    truth,
                    aspect="auto",
                    vmin=lower,
                    vmax=upper,
                ),
                row_axes[2].imshow(
                    fitted_surface,
                    aspect="auto",
                    vmin=lower,
                    vmax=upper,
                ),
            )
            titles = (
                f"Observed cell {observed_label}",
                f"True {parameter_label}",
                f"Fitted {parameter_label}",
            )
            for axis, title in zip(row_axes, titles):
                _format_axis(axis, title)
            row_axes[0].set_ylabel("Unit")
            figure.colorbar(
                images[2],
                ax=row_axes,
                label=parameter_label,
                shrink=0.85,
            )
    elif spec.family == "multinomial":
        assert simulated.observed_category_proportions is not None
        assert simulated.true_category_probabilities is not None
        assert fitted_category_probabilities is not None
        figure, axes = plt.subplots(
            len(CATEGORIES),
            3,
            figsize=(12, 9.6),
            constrained_layout=True,
        )
        for category_index, category in enumerate(CATEGORIES):
            row_axes = axes[category_index]
            images = (
                row_axes[0].imshow(
                    simulated.observed_category_proportions[
                        ...,
                        category_index,
                    ],
                    aspect="auto",
                    vmin=0.0,
                    vmax=1.0,
                ),
                row_axes[1].imshow(
                    simulated.true_category_probabilities[
                        ...,
                        category_index,
                    ],
                    aspect="auto",
                    vmin=0.0,
                    vmax=1.0,
                ),
                row_axes[2].imshow(
                    fitted_category_probabilities[
                        ...,
                        category_index,
                    ],
                    aspect="auto",
                    vmin=0.0,
                    vmax=1.0,
                ),
            )
            titles = (
                f"Observed cell proportions: {category}",
                f"True P({category})",
                f"Fitted P({category})",
            )
            for axis, title in zip(row_axes, titles):
                _format_axis(axis, title)
            row_axes[0].set_ylabel("Unit")
            figure.colorbar(
                images[2],
                ax=row_axes,
                label=f"P({category})",
                shrink=0.85,
            )
    else:
        lower, upper = _plot_limits(
            simulated.observed_summary,
            simulated.truth,
            fitted,
            unit_interval=spec.family == "bernoulli",
        )
        figure, axes = plt.subplots(
            1,
            3,
            figsize=(12, 3.4),
            constrained_layout=True,
        )
        images = (
            axes[0].imshow(
                simulated.observed_summary,
                aspect="auto",
                vmin=lower,
                vmax=upper,
            ),
            axes[1].imshow(
                simulated.truth,
                aspect="auto",
                vmin=lower,
                vmax=upper,
            ),
            axes[2].imshow(
                fitted,
                aspect="auto",
                vmin=lower,
                vmax=upper,
            ),
        )
        titles = (
            f"Observed cell {spec.observed_label}",
            f"True {spec.parameter_label}",
            f"Fitted {spec.parameter_label}",
        )
        for axis, title in zip(axes, titles):
            _format_axis(axis, title)
        axes[0].set_ylabel("Unit")
        figure.colorbar(
            images[2],
            ax=axes,
            label=spec.parameter_label,
            shrink=0.85,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)

def report_result(
    simulated: SimulatedImputationPanel,
    result: EFSCResult,
    fitted: np.ndarray,
    fitted_standard_deviation: Optional[np.ndarray],
    fitted_category_probabilities: Optional[np.ndarray],
    spec: ImputationSpec,
    elapsed: float,
) -> None:
    """Print dimensions, missing-cell accuracy, and example imputations."""

    print(
        f"{spec.family}: {N_UNITS} units, {N_PERIODS} periods, "
        f"{OBSERVATIONS_PER_CELL} observations per observed cell"
    )
    if fitted_category_probabilities is None:
        errors = (
            fitted[simulated.missing_mask]
            - simulated.truth[simulated.missing_mask]
        )
        rmse = float(np.sqrt(np.mean(errors**2)))
        print(
            f"Fit completed in {elapsed:.2f} seconds; "
            f"missing-cell {spec.parameter_label} RMSE = {rmse:.4f}"
        )
    else:
        assert simulated.true_category_probabilities is not None
        print(f"Fit completed in {elapsed:.2f} seconds")
        for category_index, category in enumerate(CATEGORIES):
            category_errors = (
                fitted_category_probabilities[
                    simulated.missing_mask,
                    category_index,
                ]
                - simulated.true_category_probabilities[
                    simulated.missing_mask,
                    category_index,
                ]
            )
            category_rmse = float(np.sqrt(np.mean(category_errors**2)))
            print(f"Missing-cell P({category}) RMSE = {category_rmse:.4f}")
    if fitted_standard_deviation is not None:
        assert simulated.true_standard_deviation is not None
        standard_deviation_errors = (
            fitted_standard_deviation[simulated.missing_mask]
            - simulated.true_standard_deviation[simulated.missing_mask]
        )
        standard_deviation_rmse = float(
            np.sqrt(np.mean(standard_deviation_errors**2))
        )
        print(
            f"Missing-cell standard deviation RMSE = {standard_deviation_rmse:.4f}"
        )
    columns = [
        column
        for column in result.imputations.columns
        if column
        not in {
            "unit_index",
            "time_index",
            "observation_count",
            "observed",
            "role",
        }
    ]
    print(result.imputations[columns].head(8).to_string(index=False))
