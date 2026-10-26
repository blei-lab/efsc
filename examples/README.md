# EFSC examples

[Main README](../README.md) · [Documentation index](../docs/README.md) ·
[Supported distributions](../docs/SUPPORTED_DISTRIBUTIONS.md)

EFSC provides separate workflows for probabilistic panel completion and
causal synthetic-control analysis. Choose the guide corresponding to your
task:

| Mode | Example guide | Purpose |
|---|---|---|
| Imputation | [Imputation examples](IMPUTATION.md) | Fit a panel and infer missing unit-time datasets |
| Causal/SC | [Causal/SC examples](CAUSAL.md) | Estimate untreated counterfactual distributions and treatment effects |

Both guides contain copyable public-API examples, complete runnable scripts,
and links to sample long-format CSV files. Run commands from the repository
root after installing the optional example dependencies:

```bash
python -m pip install --editable ".[examples]"
```

For the underlying data contract and returned objects, see the
[input-data](../docs/INPUT_DATA.md) and [results](../docs/RESULTS.md) guides.
