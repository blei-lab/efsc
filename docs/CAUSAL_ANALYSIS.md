# Causal/Synthetic Controls mode

[Documentation index](README.md) · [Main README](../README.md) ·
[Causal examples](../examples/CAUSAL.md)

Causal metadata records the first treated period for each unit. Treatment is
assumed to remain active thereafter. EFSC does not infer treatment timing:
the user must supply the first period in which treatment is active for each
treated unit.

Metadata can be a mapping:

```python
adoption = {
    "A": 2014,
    "B": "never",
    "C": None,
}

result = model.fit_csv(
    "panel.csv",
    outcome="outcome",
    unit="unit",
    time="time",
    causal_metadata=adoption,
)
```

It can also be a DataFrame or CSV with columns `unit` and `adoption_time`:

```text
unit,adoption_time
A,2014
B,never
C,
```

Blank values, `None`, and `"never"` mean never treated. Supplied units must be
unique and must occur in the panel; omitted panel units are also treated as
never treated. In causal/SC mode EFSC fits two models on the full panel
coordinates using separate masks:

- the counterfactual fit excludes observed treated-post cells;
- the treated fit uses observed treated-post cells;
- missing treated-post cells receive counterfactual predictions but no
  observed treatment effect.

The principal causal outputs are:

```python
# Counterfactual parameters for all treated-post targets.
counterfactuals = result.causal.counterfactuals

# Plug-in ECE, ECD, and familiar parameter contrasts where outcomes exist.
effects = result.causal.effects

# Posterior uncertainty from both variational distributions.
summary = result.causal.summarize_effects(
    n_draws=2000,
    level=0.95,
)
```

For multinomial outcomes, category-level probability contrasts are available
as `result.causal.category_effects`.

See [causal estimands](CAUSAL_ESTIMANDS.md) for a concise description of ECE,
ECD, and their reported summaries.

For complete runnable workflows, see the [causal/SC examples](../examples/CAUSAL.md).
