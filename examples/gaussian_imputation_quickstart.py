"""Fit EFSC to a small Gaussian panel and report its missing cells."""

from pathlib import Path

from efsc import EFSC, InferenceConfig


DATA_PATH = Path(__file__).with_name("data") / "gaussian_panel_toy.csv"


def main() -> None:
    """Run the public CSV-based imputation workflow."""

    result = EFSC(
        family="gaussian",
        rank=1,
        random_state=123,
        inference_config=InferenceConfig(epochs=2000, device="cpu"),
    ).fit_csv(
        DATA_PATH,
        outcome="outcome",
        unit="unit",
        time="time",
    )

    print("Imputed parameters for missing cells:")
    print(
        result.imputations[["unit", "time", "mean", "variance"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
