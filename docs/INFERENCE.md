# Inference

[Documentation index](README.md) · [Main README](../README.md)

`InferenceConfig` controls one variational fit:

```python
from efsc import InferenceConfig

config = InferenceConfig(
    epochs=2000,
    learning_rate=None,  # use the family-specific default
    elbo_samples=2,
    prior_scale=2.0,
    init_scale=0.01,
    init_log_variance=-6.0,
    device="cpu",
    print_every=None,
)
```

`rank` is required by `EFSC`; `random_state` defaults to zero. Fixed seeds are
propagated to the factorization, predictive, and causal-effect computations
and are recorded in the returned result.
