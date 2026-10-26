"""Fit causal EFSC to a simulated Laplace panel with known location."""

from time import perf_counter

from efsc import EFSC, InferenceConfig

from _causal_utils import (
    CausalSpec,
    parse_args,
    plot_result,
    report_result,
    simulate_panel,
)


SPEC = CausalSpec(
    "laplace_known_mu", "scale", "scale", "mean absolute deviations", mu=0.25
)


def main() -> None:
    """Simulate treatment, fit both causal models, and report effects."""

    args = parse_args(SPEC)
    simulated = simulate_panel(SPEC)
    model = EFSC(
        family="laplace_known_mu",
        mu=0.25,
        rank=1,
        random_state=123,
        inference_config=InferenceConfig(epochs=args.epochs, device="cpu"),
    )
    started = perf_counter()
    result = model.fit(
        simulated.data,
        outcome="outcome",
        unit="unit",
        time="time",
        causal_metadata=simulated.adoption,
    )
    report_result(result, SPEC, perf_counter() - started)
    if args.figure is not None:
        plot_result(simulated, result, SPEC, args.figure)
        print(f"Saved figure to {args.figure}")


if __name__ == "__main__":
    main()
