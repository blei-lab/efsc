"""Fit EFSC in causal mode to a small Gaussian panel."""

from pathlib import Path

from efsc import EFSC, InferenceConfig


DATA_PATH = Path(__file__).with_name("data") / "gaussian_causal_toy.csv"
METADATA_PATH = (
    Path(__file__).with_name("data") / "gaussian_causal_metadata_toy.csv"
)


def main() -> None:
    """Run the public CSV-based absorbing-treatment workflow."""

    result = EFSC(
        family="gaussian",
        rank=1,
        random_state=123,
        inference_config=InferenceConfig(
            epochs=20,
            learning_rate=0.03,
            device="cpu",
        ),
    ).fit_csv(
        DATA_PATH,
        outcome="outcome",
        unit="unit",
        time="time",
        causal_metadata=METADATA_PATH,
    )

    print("Plug-in effects for observed treated-post cells:")
    columns = ["unit", "time", "difference_mean", "ecd"]
    print(result.causal.effects[columns].to_string(index=False))

    print("\nPosterior effect summary (90% central intervals):")
    summary = result.causal.summarize_effects(
        n_draws=100,
        level=0.90,
        random_state=456,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
