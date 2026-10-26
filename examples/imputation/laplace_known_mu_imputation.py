"""Fit EFSC to a Laplace panel with known location."""

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
    family="laplace_known_mu",
    parameter="scale",
    parameter_label="scale",
    observed_label="mean absolute deviations",
    mu=0.25,
)


def main() -> None:
    """Simulate a long-format panel, fit EFSC, and report imputations."""

    args = parse_args(SPEC)

    # In an application, replace this simulation with your own long-format data.
    simulated = simulate_panel(SPEC)
    data = simulated.data

    model = EFSC(
        family="laplace_known_mu",
        mu=0.25,
        rank=1,
        random_state=123,
        inference_config=InferenceConfig(epochs=args.epochs, device="cpu"),
    )
    started = perf_counter()
    result = model.fit(data, outcome="outcome", unit="unit", time="time")
    elapsed = perf_counter() - started

    fitted_scale = result.distributions["scale"].detach().cpu().numpy()
    report_result(simulated, result, fitted_scale, None, None, SPEC, elapsed)

    if args.figure is not None:
        plot_result(simulated, fitted_scale, None, None, SPEC, args.figure)
        print(f"Saved figure to {args.figure}")


if __name__ == "__main__":
    main()
