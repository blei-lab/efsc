# Imputation mode

[Documentation index](README.md) · [Main README](../README.md) ·
[Imputation examples](../examples/IMPUTATION.md)

If `causal_metadata` is omitted, one [model factorization](FACTORIZATION.md) is
fitted to all observed cells.

```python
from efsc import EFSC, InferenceConfig

model = EFSC(
    family="gaussian",
    rank=2,
    random_state=123,
    inference_config=InferenceConfig(
        epochs=2000,
        learning_rate=0.1,
        device="cpu",
    ),
)

result = model.fit_csv(
    "examples/data/gaussian_panel_toy.csv",
    outcome="outcome",
    unit="unit",
    time="time",
)

# One row per missing cell, containing fitted distribution parameters.
print(result.imputations)

# Predictive draws from each missing cell's fitted distribution.
draws = result.sample_predictive(n_draws=100, cells="missing")

# Preserve observed rows and fill every missing cell with simulated outcomes.
completed = result.complete_panel(cell_size="temporal_neighbor_mean")
```

These are plug-in predictive draws: the fitted distribution parameters are
fixed at the variational posterior means, and `n_draws` controls repeated
outcome sampling conditional on those parameters. It does not request draws
from the variational parameter distribution.

`complete_panel` determines the number of outcomes in a missing cell from the
nearest observed cells for that unit. Pass a positive integer as `cell_size`
to use one common size instead.

For complete runnable workflows, see the [imputation examples](../examples/IMPUTATION.md).
