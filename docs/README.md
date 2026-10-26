# EFSC documentation

[Main README](../README.md) · [Examples](../examples/README.md) ·
[API reference](API_REFERENCE.md)

The main README provides installation and a minimal quick start. The guides
below document the complete user-facing workflow:

- [Input data](INPUT_DATA.md): long-format observations, missing cells, and
  time ordering.
- [Imputation mode](IMPUTATION.md): fitted missing-cell distributions,
  predictive draws, and completed panels.
- [Model factorization](FACTORIZATION.md): the shared latent model underlying
  imputation and causal/SC analyses.
- [Causal analysis](CAUSAL_ANALYSIS.md): adoption metadata, counterfactuals,
  and treatment effects.
- [Causal estimands](CAUSAL_ESTIMANDS.md): ECE, ECD, and the distinction
  between plug-in and posterior summaries.
- [Supported distributions](SUPPORTED_DISTRIBUTIONS.md): family names, required
  arguments, and returned parameters.
- [Inference](INFERENCE.md): variational settings, defaults, and seeds.
- [Results](RESULTS.md): the contents of `EFSCResult` and `CausalResult`.
- [Public API reference](API_REFERENCE.md): constructor and method signatures.
- [Manuscript reproduction](REPRODUCIBILITY.md): separation between the
  lightweight package and historical experimental code.
- [Example index](../examples/README.md): choose the imputation or causal/SC
  workflow.
- [Imputation examples](../examples/IMPUTATION.md): CSV and pandas quick
  starts, all supported families, and a Gaussian visualization.
- [Causal/SC examples](../examples/CAUSAL.md): adoption metadata, a Gaussian
  CSV example, and the Poisson synthetic-control visualization.
