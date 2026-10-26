"""Fit EFSC to a simulated Bernoulli panel and visualize probabilities."""

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
    family="bernoulli",
    parameter="probability",
    parameter_label="success probability",
    observed_label="proportions",
)


def main() -> None:
    """Simulate a long-format panel, fit EFSC, and report imputations."""

    args = parse_args(SPEC)

    # In an application, replace this simulation with your own long-format data.
    simulated = simulate_panel(SPEC)
    data = simulated.data

    model = EFSC(
        family="bernoulli",
        rank=1,
        random_state=123,
        inference_config=InferenceConfig(epochs=args.epochs, device="cpu"),
    )
    started = perf_counter()
    result = model.fit(data, outcome="outcome", unit="unit", time="time")
    elapsed = perf_counter() - started

    fitted_probability = (
        result.distributions["probability"].detach().cpu().numpy()
    )
    report_result(
        simulated,
        result,
        fitted_probability,
        None,
        None,
        SPEC,
        elapsed,
    )

    if args.figure is not None:
        plot_result(
            simulated,
            fitted_probability,
            None,
            None,
            SPEC,
            args.figure,
        )
        print(f"Saved figure to {args.figure}")


if __name__ == "__main__":
    main()
