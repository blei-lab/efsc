"""Fit EFSC to a Gaussian panel with unknown mean and variance."""

from time import perf_counter

import numpy as np

from efsc import EFSC, InferenceConfig

from _imputation_utils import (
    ImputationSpec,
    parse_args,
    plot_result,
    report_result,
    simulate_panel,
)


SPEC = ImputationSpec(
    family="gaussian",
    parameter="mean",
    parameter_label="mean",
    observed_label="means",
)


def main() -> None:
    """Simulate a long-format panel, fit EFSC, and report imputations."""

    args = parse_args(SPEC)

    # In an application, replace this simulation with your own long-format data.
    simulated = simulate_panel(SPEC)
    data = simulated.data

    model = EFSC(
        family="gaussian",
        rank=1,
        random_state=123,
        inference_config=InferenceConfig(epochs=args.epochs, device="cpu"),
    )
    started = perf_counter()
    result = model.fit(data, outcome="outcome", unit="unit", time="time")
    elapsed = perf_counter() - started

    fitted_mean = result.distributions["mean"].detach().cpu().numpy()
    fitted_standard_deviation = np.sqrt(
        result.distributions["variance"].detach().cpu().numpy()
    )
    report_result(
        simulated,
        result,
        fitted_mean,
        fitted_standard_deviation,
        None,
        SPEC,
        elapsed,
    )

    if args.figure is not None:
        plot_result(
            simulated,
            fitted_mean,
            fitted_standard_deviation,
            None,
            SPEC,
            args.figure,
        )
        print(f"Saved figure to {args.figure}")


if __name__ == "__main__":
    main()
