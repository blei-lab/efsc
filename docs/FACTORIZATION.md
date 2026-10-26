# Latent factorization

[Documentation index](README.md) · [Imputation mode](IMPUTATION.md) ·
[Causal/SC mode](CAUSAL_ANALYSIS.md) · [Main README](../README.md)

EFSC models an $N\times T$ panel in which each unit-time cell contains a
dataset. Conditional on the cell-specific natural parameter
$\boldsymbol\eta_{ij}$, its observations follow a known exponential-family
distribution:

$$y_{ijk}\mid\boldsymbol\eta_{ij}\sim p(\cdot;\boldsymbol\eta_{ij}),\qquad k=1,\ldots,m_{ij}.$$

When the natural-parameter space is unrestricted, EFSC factorizes each
component $p=1,\ldots,P$ directly:

$$\eta_{ij}[p]=\alpha_i[p]+\gamma_j[p]+\boldsymbol\theta_i[p]^\top\boldsymbol\beta_j[p],$$

where $\alpha_i[p]$ and $\gamma_j[p]$ are unit and time effects, while
$\boldsymbol\theta_i[p],\boldsymbol\beta_j[p]\in\mathbb R^r$ are latent
factors and $r$ is the factorization rank. For constrained natural-parameter
spaces, the same factorization is placed on an unconstrained predictor
$\mathbf z_{ij}$, with $\boldsymbol\eta_{ij}=h(\mathbf z_{ij})$. See the
[EFSC paper](https://arxiv.org/abs/2609.23970) for further details.

EFSC places Gaussian priors on the effects and latent factors and approximates
their posterior with a mean-field Gaussian variational distribution. The
evidence lower bound is optimized using black-box variational inference and
automatic differentiation; see the available options in the [inference
guide](INFERENCE.md).

Observed cells contribute to the likelihood. Missing cells do not, but their
distributions are inferred through the unit effects, time effects, and latent
factors shared across the panel. In imputation mode, one factorization is fitted
to all observed cells. In causal/SC mode, separate masks define the treated and
counterfactual fits.

See the [EFSC paper](https://arxiv.org/abs/2609.23970) for the complete model
specification and theoretical details.
