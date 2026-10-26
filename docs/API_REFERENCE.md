# Public API reference

EFSC intentionally exposes five public names from the top-level package. The
lower-level modules are implementation details and may change between releases.

## `EFSC`

```python
EFSC(
    family,
    rank,
    random_state=0,
    *,
    sigma=None,
    mu=None,
    categories=None,
    inference_config=None,
)
```

The high-level estimator. `family` selects the observation model and `rank`
sets the latent-factor rank. `sigma` is required for
`"gaussian_known_sigma"`, `mu` for `"laplace_known_mu"`, and `categories` for
`"multinomial"`.

### `fit`

```python
result = model.fit(
    data,
    outcome="outcome",
    unit="unit",
    time="time",
    time_order=None,
    causal_metadata=None,
)
```

Fits raw long-format observations held in a pandas DataFrame. Without
`causal_metadata`, EFSC imputes missing panel cells. Supplying first-treatment
times activates absorbing-treatment causal mode.

### `fit_csv`

```python
result = model.fit_csv(
    path,
    outcome="outcome",
    unit="unit",
    time="time",
    time_order=None,
    causal_metadata=None,
    read_csv_kwargs=None,
)
```

Reads a long-format CSV and then follows the same workflow as `fit`.
`causal_metadata` may be a unit-to-adoption-time mapping, a DataFrame, or a CSV
path. `read_csv_kwargs` passes options to `pandas.read_csv` for the panel file.

After fitting, the latest result is also available as `model.result_`.

## `InferenceConfig`

```python
InferenceConfig(
    epochs=2000,
    learning_rate=None,
    elbo_samples=2,
    prior_scale=2.0,
    init_scale=0.01,
    init_log_variance=-6.0,
    device="cpu",
    print_every=None,
)
```

Configuration settings for one variational fit. An `InferenceConfig` instance
cannot be modified after creation. A `None` learning rate selects the
family-specific default. The configuration is recorded in the fitted result.

## `EFSCResult`

Returned by `EFSC.fit` and `EFSC.fit_csv`. The main tabular fields are:

- `cells`: fitted parameters and roles for every unit-time cell;
- `imputations`: fitted parameters for missing cells;
- `category_probabilities`: category-level probabilities for multinomial data;
- `causal`: a `CausalResult` in causal mode and `None` otherwise.

The `panel`, `factorization`, `distributions`, `configuration`, `derived_seeds`,
and `software_versions` fields preserve the underlying fit and reproducibility
metadata.

### `sample_predictive`

```python
draws = result.sample_predictive(
    n_draws=100,
    cells="missing",
    random_state=123,
)
```

Draws outcomes conditional on posterior-mean fitted distribution parameters.
`cells` is one of `"all"`, `"observed"`, or `"missing"`. In causal mode these
are counterfactual predictive draws.

### `complete_panel`

```python
completed = result.complete_panel(
    cell_size="temporal_neighbor_mean",
    random_state=123,
)
```

Preserves observed rows and simulates outcomes for missing cells. A positive
integer requests a common number of outcomes per missing cell. The default
uses the mean size of the nearest observed cells in time for the same unit.

## `CausalResult`

Available as `result.causal` after a causal fit. `counterfactuals` contains all
treated-post targets; `effects` contains cells whose treated outcomes were
observed. Effects include natural-parameter ECE, ECD, and family-specific
parameter contrasts. Multinomial category contrasts are in `category_effects`.

### `sample_effects`

```python
draws = result.causal.sample_effects(n_draws=2000, random_state=123)
```

Draws ECE and ECD from independently sampled treated and counterfactual
variational distributions. It returns one row per posterior draw and observed
treated-post cell. Monte Carlo ECD draws are optional summaries and can be
unstable; the default reported ECD is the plug-in estimate in `effects`.

### `summarize_effects`

```python
summary = result.causal.summarize_effects(
    n_draws=2000,
    level=0.95,
    random_state=123,
)
```

Returns posterior means and equal-tailed central intervals by observed
treated-post cell. ECE intervals are generally more stable than the optional
Monte Carlo ECD summaries.

## `__version__`

The installed EFSC package version as a string.
