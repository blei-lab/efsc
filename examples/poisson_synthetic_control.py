"""Fit EFSC to a small simulated Poisson synthetic-control panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from efsc import EFSC, EFSCResult, InferenceConfig


N_UNITS = 8
N_PERIODS = 24
OBSERVATIONS_PER_CELL = 32
FIRST_TREATED_PERIOD = 20
TREATED_UNITS = ("U7", "U8")


def simulate_panel(random_state: int = 123) -> tuple[pd.DataFrame, dict[str, object]]:
    """Simulate repeated Poisson observations with two treated units."""

    rng = np.random.default_rng(random_state)
    unit_effect = np.linspace(-0.35, 0.35, N_UNITS)
    unit_loading = np.linspace(-0.45, 0.45, N_UNITS)
    times = np.arange(1, N_PERIODS + 1)
    time_effect = 0.012 * times + 0.16 * np.sin(2 * np.pi * times / 12)
    time_loading = np.cos(2 * np.pi * times / 12)

    records = []
    for i in range(N_UNITS):
        unit = f"U{i + 1}"
        for j, time in enumerate(times):
            log_rate = (
                0.65
                + unit_effect[i]
                + time_effect[j]
                + unit_loading[i] * time_loading[j]
            )
            if unit in TREATED_UNITS and time >= FIRST_TREATED_PERIOD:
                log_rate += 0.65
            for outcome in rng.poisson(np.exp(log_rate), size=OBSERVATIONS_PER_CELL):
                records.append(
                    {"outcome": int(outcome), "unit": unit, "time": int(time)}
                )

    adoption = {unit: FIRST_TREATED_PERIOD for unit in TREATED_UNITS}
    return pd.DataFrame.from_records(records), adoption


def fit_example(epochs: int = 2000) -> tuple[pd.DataFrame, EFSCResult]:
    """Simulate and fit the public causal-analysis workflow."""

    panel, adoption = simulate_panel()
    result = EFSC(
        family="poisson",
        rank=1,
        random_state=123,
        inference_config=InferenceConfig(epochs=epochs, device="cpu"),
    ).fit(
        panel,
        outcome="outcome",
        unit="unit",
        time="time",
        causal_metadata=adoption,
    )
    return panel, result


def effect_summary(result: EFSCResult) -> pd.DataFrame:
    """Summarize fitted treated-post effects by treated unit."""

    assert result.causal is not None
    columns = ["counterfactual_rate", "treated_rate", "difference_rate", "ecd"]
    return result.causal.effects.groupby("unit", as_index=False)[columns].mean()


def plot_result(panel: pd.DataFrame, result: EFSCResult, output: Path) -> None:
    """Save observed, counterfactual, and treatment-effect panel matrices."""

    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError as exc:
        raise RuntimeError("Plotting this example requires matplotlib.") from exc

    unit_order = [f"U{i + 1}" for i in range(N_UNITS)]
    time_order = list(range(1, N_PERIODS + 1))
    observed = (
        panel.groupby(["unit", "time"])["outcome"]
        .mean()
        .unstack("time")
        .reindex(index=unit_order, columns=time_order)
        .to_numpy()
    )
    counterfactual = (
        result.distributions["rate"].detach().cpu().numpy()
    )
    effect = np.full((N_UNITS, N_PERIODS), np.nan)
    assert result.causal is not None
    for row in result.causal.effects.itertuples(index=False):
        effect[row.unit_index, row.time_index] = row.difference_rate

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    common_max = max(float(np.nanmax(observed)), float(np.nanmax(counterfactual)))
    images = [
        axes[0].imshow(observed, aspect="auto", vmin=0, vmax=common_max),
        axes[1].imshow(counterfactual, aspect="auto", vmin=0, vmax=common_max),
        axes[2].imshow(
            effect,
            aspect="auto",
            cmap="coolwarm",
            norm=TwoSlopeNorm(vcenter=0),
        ),
    ]
    titles = ("Observed cell means", "Estimated counterfactual rates", "Rate effects")
    for axis, title in zip(axes, titles):
        axis.set_title(title)
        axis.set_xlabel("Time")
        axis.set_xticks([0, 5, 11, 17, 23], [1, 6, 12, 18, 24])
        axis.set_yticks(range(N_UNITS), unit_order)
        axis.axvline(FIRST_TREATED_PERIOD - 1.5, color="white", linestyle="--")
        axis.axhline(N_UNITS - len(TREATED_UNITS) - 0.5, color="white", linestyle="--")
    axes[0].set_ylabel("Unit")
    figure.colorbar(images[0], ax=axes[:2], label="Poisson rate", shrink=0.85)
    figure.colorbar(images[2], ax=axes[2], label="Treated − counterfactual", shrink=0.85)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Run the example, print effects, and optionally save its figure."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()

    panel, result = fit_example(epochs=args.epochs)
    print("Average treated-post effects by unit:")
    print(effect_summary(result).to_string(index=False))
    if args.figure is not None:
        plot_result(panel, result, args.figure)
        print(f"Saved figure to {args.figure}")


if __name__ == "__main__":
    main()
