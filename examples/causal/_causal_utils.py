"""Shared simulation, reporting, and plotting for EFSC causal examples."""

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
FIRST_TREATED_PERIOD = 41
N_TREATED_UNITS = 4
CATEGORIES = ("low", "medium", "high")


@dataclass(frozen=True)
class SurfaceProfile:
    """Family-specific seed and counterfactual-surface settings."""

    seed: int
    offset: float
    unit_scale: float
    time_scale: float
    interaction_scale: float
    phase: float


SURFACE_PROFILES = {
    "gaussian": SurfaceProfile(1101, 0.45, 0.72, 0.50, 0.62, 0.15),
    "gaussian_known_sigma": SurfaceProfile(
        1211, 0.20, -0.65, 0.52, 0.48, 0.65
    ),
    "poisson": SurfaceProfile(1307, 0.55, 0.62, 0.45, -0.50, 1.10),
    "bernoulli": SurfaceProfile(1401, -0.15, -0.75, 0.60, 0.65, 1.80),
    "exponential": SurfaceProfile(1503, 0.25, 0.50, -0.48, 0.58, 2.40),
    "laplace_known_mu": SurfaceProfile(1601, 0.15, -0.55, 0.50, -0.60, 0.90),
    "chi_square": SurfaceProfile(1701, 0.50, 0.45, 0.58, 0.52, 2.80),
    "multinomial": SurfaceProfile(1809, 0.20, 0.70, 0.55, 0.60, 0.35),
}


@dataclass(frozen=True)
class CausalSpec:
    """Family-specific settings for one causal-gallery example."""

    family: str
    parameter: str
    parameter_label: str
    observed_label: str
    sigma: Optional[float] = None
    mu: Optional[float] = None
    category_index: Optional[int] = None


@dataclass(frozen=True)
class SimulatedCausalPanel:
    """A simulated treated panel and its true counterfactual surfaces."""

    data: pd.DataFrame
    adoption: dict[str, int]
    treated_post_mask: np.ndarray
    counterfactual_parameters: dict[str, np.ndarray]
    treated_parameters: dict[str, np.ndarray]
    observed_summary: np.ndarray
    observed_standard_deviation: Optional[np.ndarray] = None
    observed_category_proportions: Optional[np.ndarray] = None


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


def _counterfactual_parameters(spec: CausalSpec) -> dict[str, np.ndarray]:
    """Construct the untreated distribution-parameter surfaces."""

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
        return {"mean": mean, "variance": variance}
    if spec.family == "gaussian_known_sigma":
        assert spec.sigma is not None
        return {"mean": spec.sigma * predictor}
    if spec.family == "poisson":
        return {"rate": np.exp(predictor)}
    if spec.family == "bernoulli":
        return {"probability": 1.0 / (1.0 + np.exp(-predictor))}
    if spec.family == "exponential":
        rate = np.exp(predictor)
        return {"rate": rate, "mean": 1.0 / rate}
    if spec.family == "laplace_known_mu":
        return {"scale": np.exp(-predictor)}
    if spec.family == "chi_square":
        return {"degrees_of_freedom": 2.0 * np.exp(predictor)}
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
        return {"probabilities": weights / weights.sum(axis=-1, keepdims=True)}
    raise ValueError(f"Unsupported causal example family: {spec.family}")


def _treated_parameters(
    spec: CausalSpec,
    counterfactual: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Apply a family-appropriate intervention to distribution parameters."""

    treated = {name: values.copy() for name, values in counterfactual.items()}
    selected = np.zeros((N_UNITS, N_PERIODS), dtype=bool)
    selected[-N_TREATED_UNITS:, FIRST_TREATED_PERIOD - 1 :] = True
    if spec.family == "gaussian":
        treated["mean"][selected] += 3.0
        treated["variance"][selected] *= 2.50**2
    elif spec.family == "gaussian_known_sigma":
        treated["mean"][selected] += 3.0
    elif spec.family == "poisson":
        treated["rate"][selected] *= np.exp(1.20)
    elif spec.family == "bernoulli":
        probability = treated["probability"]
        logits = np.log(probability / (1.0 - probability))
        logits[selected] += 2.50
        treated["probability"] = 1.0 / (1.0 + np.exp(-logits))
    elif spec.family == "exponential":
        treated["rate"][selected] *= 0.30
        treated["mean"] = 1.0 / treated["rate"]
    elif spec.family == "laplace_known_mu":
        treated["scale"][selected] *= 3.0
    elif spec.family == "chi_square":
        treated["degrees_of_freedom"][selected] *= 3.0
    elif spec.family == "multinomial":
        probabilities = treated["probabilities"]
        logits = np.log(probabilities[..., :2] / probabilities[..., 2:3])
        logits[..., 0][selected] += 2.20
        logits[..., 1][selected] -= 1.60
        full_logits = np.concatenate(
            [logits, np.zeros((*logits.shape[:2], 1))], axis=-1
        )
        full_logits -= full_logits.max(axis=-1, keepdims=True)
        weights = np.exp(full_logits)
        treated["probabilities"] = weights / weights.sum(axis=-1, keepdims=True)
    else:
        raise ValueError(f"Unsupported causal example family: {spec.family}")
    return treated


def _draw_outcomes(
    rng: np.random.Generator,
    spec: CausalSpec,
    parameters: dict[str, np.ndarray],
    i: int,
    j: int,
) -> np.ndarray:
    """Draw one repeated-observation cell from its distribution."""

    if spec.family == "gaussian":
        return rng.normal(
            parameters["mean"][i, j],
            np.sqrt(parameters["variance"][i, j]),
            size=OBSERVATIONS_PER_CELL,
        )
    if spec.family == "gaussian_known_sigma":
        assert spec.sigma is not None
        return rng.normal(parameters["mean"][i, j], spec.sigma, OBSERVATIONS_PER_CELL)
    if spec.family == "poisson":
        return rng.poisson(parameters["rate"][i, j], OBSERVATIONS_PER_CELL)
    if spec.family == "bernoulli":
        return rng.binomial(1, parameters["probability"][i, j], OBSERVATIONS_PER_CELL)
    if spec.family == "exponential":
        return rng.exponential(parameters["mean"][i, j], OBSERVATIONS_PER_CELL)
    if spec.family == "laplace_known_mu":
        assert spec.mu is not None
        return rng.laplace(spec.mu, parameters["scale"][i, j], OBSERVATIONS_PER_CELL)
    if spec.family == "chi_square":
        return rng.chisquare(
            parameters["degrees_of_freedom"][i, j], OBSERVATIONS_PER_CELL
        )
    if spec.family == "multinomial":
        return rng.choice(
            CATEGORIES,
            size=OBSERVATIONS_PER_CELL,
            p=parameters["probabilities"][i, j],
        )
    raise AssertionError("Unreachable family dispatch.")


def simulate_panel(
    spec: CausalSpec,
    *,
    random_state: Optional[int] = None,
) -> SimulatedCausalPanel:
    """Simulate a simultaneous-adoption causal panel."""

    seed = SURFACE_PROFILES[spec.family].seed if random_state is None else random_state
    rng = np.random.default_rng(seed)
    counterfactual = _counterfactual_parameters(spec)
    treated = _treated_parameters(spec, counterfactual)
    treated_post = np.zeros((N_UNITS, N_PERIODS), dtype=bool)
    treated_post[-N_TREATED_UNITS:, FIRST_TREATED_PERIOD - 1 :] = True
    observed_summary = np.empty((N_UNITS, N_PERIODS))
    observed_standard_deviation = (
        np.empty((N_UNITS, N_PERIODS)) if spec.family == "gaussian" else None
    )
    observed_category_proportions = (
        np.empty((N_UNITS, N_PERIODS, len(CATEGORIES)))
        if spec.family == "multinomial"
        else None
    )
    records: list[dict[str, object]] = []
    unit_label_width = len(str(N_UNITS))
    for i in range(N_UNITS):
        unit = f"U{i + 1:0{unit_label_width}d}"
        for j in range(N_PERIODS):
            parameters = treated if treated_post[i, j] else counterfactual
            outcomes = _draw_outcomes(rng, spec, parameters, i, j)
            if spec.family == "multinomial":
                assert observed_category_proportions is not None
                for category_index, category in enumerate(CATEGORIES):
                    observed_category_proportions[i, j, category_index] = np.mean(
                        outcomes == category
                    )
                assert spec.category_index is not None
                observed_summary[i, j] = observed_category_proportions[
                    i, j, spec.category_index
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
                    outcomes.astype(float), ddof=1
                )
            records.extend(
                {
                    "outcome": outcome.item() if hasattr(outcome, "item") else outcome,
                    "unit": unit,
                    "time": j + 1,
                }
                for outcome in outcomes
            )
    treated_units = range(N_UNITS - N_TREATED_UNITS, N_UNITS)
    adoption = {
        f"U{i + 1:0{unit_label_width}d}": FIRST_TREATED_PERIOD
        for i in treated_units
    }
    return SimulatedCausalPanel(
        data=pd.DataFrame.from_records(records),
        adoption=adoption,
        treated_post_mask=treated_post,
        counterfactual_parameters=counterfactual,
        treated_parameters=treated,
        observed_summary=observed_summary,
        observed_standard_deviation=observed_standard_deviation,
        observed_category_proportions=observed_category_proportions,
    )


def parse_args(spec: CausalSpec) -> argparse.Namespace:
    """Parse common command-line options for a causal example."""

    parser = argparse.ArgumentParser(
        description=(
            f"Fit causal EFSC to a simulated {spec.family} "
            f"{N_UNITS}-by-{N_PERIODS} panel."
        )
    )
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--figure", type=Path)
    return parser.parse_args()


def _to_numpy(value) -> np.ndarray:
    """Move a fitted tensor to a NumPy array."""

    return value.detach().cpu().numpy()


def _parameter_effect(
    result: EFSCResult,
    parameter: str,
) -> np.ndarray:
    """Return a treated-minus-counterfactual parameter surface."""

    assert result.causal is not None
    counterfactual = _to_numpy(result.causal.counterfactual_distributions[parameter])
    treated = _to_numpy(result.causal.treated_distributions[parameter])
    effect = np.full(counterfactual.shape, np.nan)
    mask = result.causal.masks.treated_mask
    effect[mask] = treated[mask] - counterfactual[mask]
    return effect


def _standard_deviation_effect(
    result: EFSCResult,
) -> np.ndarray:
    """Return the Gaussian treated-minus-counterfactual scale surface."""

    assert result.causal is not None
    counterfactual = np.sqrt(
        _to_numpy(result.causal.counterfactual_distributions["variance"])
    )
    treated = np.sqrt(_to_numpy(result.causal.treated_distributions["variance"]))
    effect = np.full(counterfactual.shape, np.nan)
    mask = result.causal.masks.treated_mask
    effect[mask] = treated[mask] - counterfactual[mask]
    return effect


def _format_axis(axis, title: str) -> None:
    """Apply common labels, ticks, and treatment boundaries."""

    axis.set_title(title)
    axis.set_xlabel("Time")
    time_ticks = np.linspace(0, N_PERIODS - 1, 5, dtype=int)
    axis.set_xticks(time_ticks, time_ticks + 1)
    width = len(str(N_UNITS))
    axis.set_yticks(
        range(N_UNITS), [f"U{i + 1:0{width}d}" for i in range(N_UNITS)]
    )
    axis.axvline(FIRST_TREATED_PERIOD - 1.5, color="white", linestyle="--")
    axis.axhline(N_UNITS - N_TREATED_UNITS - 0.5, color="white", linestyle="--")


def _plot_row(
    axes,
    observed: np.ndarray,
    truth: np.ndarray,
    effect: np.ndarray,
    *,
    observed_label: str,
    parameter_label: str,
    unit_interval: bool = False,
    nonnegative: bool = False,
) -> tuple[object, object]:
    """Plot one observed/counterfactual/effect row."""

    lower = float(min(observed.min(), truth.min()))
    upper = float(max(observed.max(), truth.max()))
    if unit_interval:
        lower, upper = 0.0, 1.0
    elif nonnegative:
        lower = 0.0
    magnitude = max(float(np.nanmax(np.abs(effect))), 1e-8)
    images = (
        axes[0].imshow(observed, aspect="auto", vmin=lower, vmax=upper),
        axes[1].imshow(truth, aspect="auto", vmin=lower, vmax=upper),
        axes[2].imshow(
            effect,
            aspect="auto",
            cmap="coolwarm",
            vmin=-magnitude,
            vmax=magnitude,
        ),
    )
    titles = (
        f"Observed cell {observed_label}",
        f"True counterfactual {parameter_label}",
        f"Fitted {parameter_label} effect",
    )
    for axis, title in zip(axes, titles):
        _format_axis(axis, title)
    axes[0].set_ylabel("Unit")
    return images[1], images[2]


def plot_result(
    simulated: SimulatedCausalPanel,
    result: EFSCResult,
    spec: CausalSpec,
    output: Path,
) -> None:
    """Save observed, counterfactual, and causal-effect surfaces."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting causal examples requires the optional 'examples' dependencies."
        ) from exc
    if spec.family == "gaussian":
        assert simulated.observed_standard_deviation is not None
        mean_effect = _parameter_effect(result, "mean")
        scale_effect = _standard_deviation_effect(result)
        figure, axes = plt.subplots(2, 3, figsize=(12, 6.6), constrained_layout=True)
        rows = (
            (
                simulated.observed_summary,
                simulated.counterfactual_parameters["mean"],
                mean_effect,
                "means",
                "mean",
                False,
            ),
            (
                simulated.observed_standard_deviation,
                np.sqrt(simulated.counterfactual_parameters["variance"]),
                scale_effect,
                "standard deviations",
                "standard deviation",
                True,
            ),
        )
        for row_index, row in enumerate(rows):
            truth_image, effect_image = _plot_row(
                axes[row_index],
                row[0], row[1], row[2],
                observed_label=row[3],
                parameter_label=row[4],
                nonnegative=row[5],
            )
            figure.colorbar(truth_image, ax=axes[row_index, :2], label=row[4])
            figure.colorbar(
                effect_image,
                ax=axes[row_index, 2],
                label="Treated − counterfactual",
            )
    elif spec.family == "multinomial":
        assert simulated.observed_category_proportions is not None
        assert result.causal is not None
        fitted = _to_numpy(result.causal.counterfactual_distributions["probabilities"])
        treated = _to_numpy(result.causal.treated_distributions["probabilities"])
        effect = np.full(fitted.shape, np.nan)
        mask = result.causal.masks.treated_mask
        effect[mask] = treated[mask] - fitted[mask]
        figure, axes = plt.subplots(3, 3, figsize=(12, 9.6), constrained_layout=True)
        for category_index, category in enumerate(CATEGORIES):
            truth_image, effect_image = _plot_row(
                axes[category_index],
                simulated.observed_category_proportions[..., category_index],
                simulated.counterfactual_parameters["probabilities"]
                [..., category_index],
                effect[..., category_index],
                observed_label=f"proportions: {category}",
                parameter_label=f"P({category})",
                unit_interval=True,
            )
            figure.colorbar(
                truth_image, ax=axes[category_index, :2], label=f"P({category})"
            )
            figure.colorbar(
                effect_image,
                ax=axes[category_index, 2],
                label="Treated − counterfactual",
            )
    else:
        effect = _parameter_effect(result, spec.parameter)
        figure, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
        truth_image, effect_image = _plot_row(
            axes,
            simulated.observed_summary,
            simulated.counterfactual_parameters[spec.parameter],
            effect,
            observed_label=spec.observed_label,
            parameter_label=spec.parameter_label,
            unit_interval=spec.family == "bernoulli",
            nonnegative=spec.family != "bernoulli",
        )
        figure.colorbar(truth_image, ax=axes[:2], label=spec.parameter_label)
        figure.colorbar(
            effect_image,
            ax=axes[2],
            label="Treated − counterfactual",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def report_result(result: EFSCResult, spec: CausalSpec, elapsed: float) -> None:
    """Print dimensions, runtime, and average treated-post effects."""

    assert result.causal is not None
    print(
        f"{spec.family}: {N_UNITS} units, {N_PERIODS} periods, "
        f"{OBSERVATIONS_PER_CELL} observations per cell"
    )
    print(
        f"Fit completed in {elapsed:.2f} seconds; "
        f"{N_TREATED_UNITS} units treated from period {FIRST_TREATED_PERIOD}"
    )
    if spec.family == "multinomial":
        summary = result.causal.category_effects.groupby(
            ["unit", "category"], as_index=False
        )["difference_probability"].mean()
    else:
        difference = f"difference_{spec.parameter}"
        summary = result.causal.effects.groupby("unit", as_index=False)[
            [difference, "ecd"]
        ].mean()
    print(summary.to_string(index=False))
