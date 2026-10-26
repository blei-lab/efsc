# Results

[Documentation index](README.md) · [Main README](../README.md) ·
[API reference](API_REFERENCE.md)

`EFSC.fit` and `EFSC.fit_csv` return an `EFSCResult` containing:

- `cells`: one row per unit-time cell, including how the cell was used in the
  fit and its fitted parameters;
- `imputations`: fitted parameters for missing cells;
- `category_probabilities`: long category-probability table for multinomial
  panels;
- `distributions`: fitted distribution-parameter tensors;
- `factorization`: variational parameters, natural-parameter surfaces, loss
  history, mask, seed, and inference settings;
- `configuration`, `derived_seeds`, and `software_versions`: reproducibility
  metadata;
- `causal`: counterfactuals and effects in causal mode, otherwise `None`.

The `role` column in `cells` describes how each cell was used. In imputation
mode, its values are `observed` and `missing`. In causal mode, the values are
`counterfactual_training`, `treated_post_observed`, `treated_post_missing`,
and `ordinary_missing`.

In causal mode, `result.factorization`, `result.distributions`,
`result.sample_predictive`, and `result.complete_panel` use the fitted
untreated counterfactual model. Treated-model outputs, counterfactual tables,
and treatment effects are available through `result.causal`.

See the [public API reference](API_REFERENCE.md) for all five top-level public
names and their principal methods.
