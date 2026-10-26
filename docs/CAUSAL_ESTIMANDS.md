# Causal estimands

[Documentation index](README.md) · [Causal/SC mode](CAUSAL_ANALYSIS.md) ·
[Main README](../README.md)

EFSC reports two principal causal estimands for each observed treated
post-treatment cell $(i,j)$. See the
[EFSC paper](https://arxiv.org/abs/2609.23970) for further details.

## Expected causal effect

The expected causal effect (ECE) is the treated-minus-counterfactual change in
the natural parameters:

$$\mathrm{ECE}_{ij}=\boldsymbol\eta^{\mathrm{treat}}_{ij}-\boldsymbol\eta^{\mathrm{ctrl}}_{ij}.$$

ECE has one component per natural parameter. Plug-in estimates are stored in
`result.causal.effects` as `ece_eta_0`, `ece_eta_1`, and so forth. Posterior
draws and equal-tailed intervals, such as the 2.5th and 97.5th percentiles for
a 95% interval, are available through `sample_effects()` and
`summarize_effects()`.

## Expected causal divergence

The expected causal divergence (ECD) measures the full distributional change:

$$\mathrm{ECD}_{ij}=\mathrm{KL}\left(p^{\mathrm{treat}}_{ij} \Vert p^{\mathrm{ctrl}}_{ij}\right).$$

ECD is a scalar for each cell. The default reported ECD is the plug-in estimate
in `result.causal.effects["ecd"]`, evaluated at the posterior-mean treated and
counterfactual parameters. The posterior-sampling methods also return optional
Monte Carlo ECD draws, but these can be unstable because KL divergence is
nonlinear and may become very large for outlying parameter draws.
