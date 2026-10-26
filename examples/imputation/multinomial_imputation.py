"""Fit EFSC to a simulated multinomial panel and visualize probabilities."""

from time import perf_counter

from efsc import EFSC, InferenceConfig

from _imputation_utils import (
    ImputationSpec,
    parse_args,
    plot_result,
    report_result,
    simulate_panel,
)


SPEC = ImputationSpec(
    family="multinomial",
    parameter="probabilities",
    parameter_label="P(high)",
    observed_label="proportions: high",
    category_index=2,
)


def main() -> None:
    """Simulate a long-format panel, fit EFSC, and report imputations."""

    args = parse_args(SPEC)

    # In an application, replace this simulation with your own long-format data.
    simulated = simulate_panel(SPEC)
    data = simulated.data

    categories = ["low", "medium", "high"]
    model = EFSC(
        family="multinomial",
        categories=categories,
        rank=1,
        random_state=123,
        inference_config=InferenceConfig(epochs=args.epochs, device="cpu"),
    )
    started = perf_counter()
    result = model.fit(data, outcome="outcome", unit="unit", time="time")
    elapsed = perf_counter() - started

    fitted_probabilities = (
        result.distributions["probabilities"].detach().cpu().numpy()
    )
    high_index = categories.index("high")
    fitted_high_probability = fitted_probabilities[..., high_index]
    report_result(
        simulated,
        result,
        fitted_high_probability,
        None,
        fitted_probabilities,
        SPEC,
        elapsed,
    )

    if args.figure is not None:
        plot_result(
            simulated,
            fitted_high_probability,
            None,
            fitted_probabilities,
            SPEC,
            args.figure,
        )
        print(f"Saved figure to {args.figure}")


if __name__ == "__main__":
    main()
