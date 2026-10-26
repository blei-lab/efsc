"""Shared variational fitting engine for EFSC factorization models.

This module connects validated panel data to the existing family-specific
sufficient statistics, log-joints, and mean-field BBVI optimizer. It provides
one deterministic implementation of the latent layout and natural-parameter
reconstruction used across EFSC experiments.

Author:
    Hector Rodriguez-Deniz

Affiliation:
    Data Science Institute, Columbia University

Date:
    August 2026

License:
    MIT
"""

from dataclasses import dataclass
from numbers import Real
from typing import Any, Optional, Union

import numpy as np
import torch

from ._families import constrain_raw_parameters, predictor_block_count
from .log_joints import (
    bernoulli_log_joint_flat,
    chi_square_log_joint_flat,
    exponential_log_joint_flat,
    gaussian_known_sigma_log_joint_flat,
    gaussian_univariate_log_joint_flat,
    laplace_known_mu_log_joint_flat,
    multinomial_log_joint_flat,
    poisson_log_joint_flat,
)
from .panel import PanelData
from .preprocessing import (
    bernoulli_sufficient_statistics_flat,
    chi_square_sufficient_statistics_flat,
    exponential_sufficient_statistics_flat,
    gaussian_known_sigma_sufficient_statistics_flat,
    gaussian_sufficient_statistics_flat,
    laplace_sufficient_statistics_flat,
    multinomial_sufficient_statistics_flat,
    poisson_sufficient_statistics_flat,
)
from .vi import fit_vi, make_negative_elbo
from ._validation import validate_positive_integer, validate_seed


DeviceInput = Union[str, torch.device]

DEFAULT_LEARNING_RATES = {
    "gaussian": 0.1,
    "poisson": 0.1,
    "gaussian_known_sigma": 0.05,
    "bernoulli": 0.05,
    "exponential": 0.005,
    "laplace_known_mu": 0.005,
    "chi_square": 0.002,
    "multinomial": 0.01,
}


@dataclass(frozen=True)
class InferenceConfig:
    """Optimization settings for one EFSC factorization fit.

    ``learning_rate=None`` selects the family-specific default. ``device``
    accepts ``"cpu"``, ``"cuda"``, or a compatible PyTorch device. Instances
    are immutable so a fitted result records a stable configuration.
    """

    epochs: int = 2000
    learning_rate: Optional[float] = None
    elbo_samples: int = 2
    prior_scale: float = 2.0
    init_scale: float = 0.01
    init_log_variance: float = -6.0
    device: DeviceInput = "cpu"
    print_every: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate all inference settings after dataclass initialization."""

        validate_positive_integer(self.epochs, "epochs")
        validate_positive_integer(self.elbo_samples, "elbo_samples")
        if self.learning_rate is not None and (
            not isinstance(self.learning_rate, Real)
            or isinstance(self.learning_rate, (bool, np.bool_))
            or not np.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate must be finite and positive.")
        if (
            not isinstance(self.prior_scale, Real)
            or isinstance(self.prior_scale, (bool, np.bool_))
            or not np.isfinite(self.prior_scale)
            or self.prior_scale <= 0
        ):
            raise ValueError("prior_scale must be finite and positive.")
        if (
            not isinstance(self.init_scale, Real)
            or isinstance(self.init_scale, (bool, np.bool_))
            or not np.isfinite(self.init_scale)
            or self.init_scale < 0
        ):
            raise ValueError("init_scale must be finite and nonnegative.")
        if not isinstance(self.init_log_variance, Real) or not np.isfinite(
            self.init_log_variance
        ):
            raise ValueError("init_log_variance must be finite.")
        if self.print_every is not None:
            validate_positive_integer(self.print_every, "print_every")


@dataclass(frozen=True)
class PreparedStatistics:
    """Flattened cell statistics consumed by EFSC log-joints."""

    unit_index: torch.Tensor
    time_index: torch.Tensor
    cell_size: torch.Tensor
    statistics: torch.Tensor
    log_carrier: torch.Tensor


@dataclass(frozen=True)
class FactorizationFit:
    """Output of one variational factorization fit."""

    family: str
    rank: int
    seed: Optional[int]
    mask: np.ndarray
    variational_mean: torch.Tensor
    variational_log_variance: torch.Tensor
    raw_natural_parameters: torch.Tensor
    natural_parameters: torch.Tensor
    loss_history: np.ndarray
    elapsed_time: float
    learning_rate: float
    config: InferenceConfig
    prior_mean: Optional[torch.Tensor] = None
    prior_form: str = "independent_normal"
    initial_variational_parameters: Optional[torch.Tensor] = None

    @property
    def variational_parameters(self) -> torch.Tensor:
        """Return the concatenated variational mean and log-variance vector."""

        return torch.cat([self.variational_mean, self.variational_log_variance])

    @property
    def latent_dimension(self) -> int:
        """Return the number of latent parameters in the factorization."""

        return self.variational_mean.numel()


def _validate_rank(rank: Any, n: int, T: int) -> int:
    """Validate and return a positive rank compatible with the panel dimensions."""

    rank = validate_positive_integer(rank, "rank")
    if rank > min(n, T):
        raise ValueError("rank cannot exceed the smaller panel dimension.")
    return int(rank)


def _block_dimension(n: int, T: int, rank: int) -> int:
    """Return the number of scalar parameters in one PMF block."""

    return (n + T) * (rank + 1)


def _reconstruct_raw_block(
    block: torch.Tensor,
    *,
    n: int,
    T: int,
    rank: int,
) -> torch.Tensor:
    """Reconstruct one unconstrained predictor matrix from a PMF block."""

    alpha, gamma, unit_flat, time_flat = torch.tensor_split(
        block,
        [n, n + T, n + T + n * rank],
    )
    unit_factors = unit_flat.reshape(n, rank)
    time_factors = time_flat.reshape(T, rank)
    return alpha[:, None] + gamma[None, :] + unit_factors @ time_factors.T


def reconstruct_raw_natural_parameters(
    posterior_mean: torch.Tensor,
    *,
    family: str,
    n: int,
    T: int,
    rank: int,
    n_categories: Optional[int] = None,
) -> torch.Tensor:
    """Reconstruct unconstrained latent surfaces from blockwise PMF parameters."""

    rank = _validate_rank(rank, n, T)
    blocks = predictor_block_count(family, n_categories)
    block_dimension = _block_dimension(n, T, rank)
    posterior_mean = torch.as_tensor(posterior_mean)
    if posterior_mean.ndim != 1 or posterior_mean.numel() != blocks * block_dimension:
        raise ValueError("posterior_mean has incompatible length for this model.")
    surfaces = [
        _reconstruct_raw_block(block, n=n, T=T, rank=rank)
        for block in torch.split(posterior_mean, block_dimension)
    ]
    return torch.stack(surfaces, dim=2)


def reconstruct_natural_parameters(
    posterior_mean: torch.Tensor,
    *,
    family: str,
    n: int,
    T: int,
    rank: int,
    n_categories: Optional[int] = None,
) -> torch.Tensor:
    """Reconstruct valid family-specific natural-parameter surfaces."""

    raw = reconstruct_raw_natural_parameters(
        posterior_mean,
        family=family,
        n=n,
        T=T,
        rank=rank,
        n_categories=n_categories,
    )
    return constrain_raw_parameters(raw, family=family)


def _validate_mask(panel: PanelData, mask: Any) -> np.ndarray:
    """Validate and return the Boolean mask of panel cells used for fitting."""

    mask_array = np.asarray(mask)
    if mask_array.shape != (panel.n, panel.T):
        raise ValueError("mask must have shape (n, T).")
    if not (
        np.issubdtype(mask_array.dtype, np.bool_)
        or np.isin(mask_array, [0, 1]).all()
    ):
        raise ValueError("mask must contain only boolean or 0/1 values.")
    mask_array = mask_array.astype(bool)
    if np.any(mask_array & ~panel.observed_mask):
        raise ValueError("mask cannot select missing cells.")
    if not mask_array.any():
        raise ValueError("mask must select at least one observed cell.")
    return mask_array


def _resolve_parameter_vector(
    value: Any,
    *,
    dimension: int,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    """Return a finite scalar-expanded or dimension-length parameter vector."""

    vector = torch.as_tensor(value, device=device, dtype=torch.float32)
    if vector.ndim == 0:
        vector = vector.expand(dimension).clone()
    elif vector.ndim != 1 or vector.numel() != dimension:
        raise ValueError(f"{name} must be a scalar or have length {dimension}.")
    else:
        vector = vector.clone()
    if not torch.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values.")
    return vector


def _resolve_initial_parameters(
    value: Any,
    *,
    dimension: int,
    device: torch.device,
) -> torch.Tensor:
    """Validate an explicitly supplied mean/log-variance initialization."""

    parameters = torch.as_tensor(value, device=device, dtype=torch.float32)
    if parameters.ndim != 1 or parameters.numel() != 2 * dimension:
        raise ValueError(
            "initial_parameters must be one-dimensional with length "
            f"{2 * dimension}."
        )
    if not torch.isfinite(parameters).all():
        raise ValueError("initial_parameters must contain only finite values.")
    return parameters.clone()


def _multinomial_counts(panel: PanelData) -> np.ndarray:
    """Count category occurrences within every multinomial panel cell."""

    categories = panel.categories
    if categories is None:
        raise ValueError("Multinomial panel is missing its category order.")
    category_index = {category: index for index, category in enumerate(categories)}
    counts = np.zeros((panel.n, panel.T, len(categories)), dtype=np.int64)
    for i in range(panel.n):
        for j in range(panel.T):
            for value in panel.observations[i, j]:
                counts[i, j, category_index[value]] += 1
    return counts


def prepare_sufficient_statistics(
    panel: PanelData,
    mask: Any,
    *,
    device: DeviceInput = "cpu",
) -> PreparedStatistics:
    """Aggregate a validated raw panel for the existing flattened log-joints."""

    if not isinstance(panel, PanelData):
        raise TypeError("panel must be a PanelData object.")
    mask_array = _validate_mask(panel, mask)
    if panel.family == "gaussian":
        unit, time, cell_size, tx0, tx1, carrier = (
            gaussian_sufficient_statistics_flat(
                panel.observations,
                mask_array,
                device=device,
            )
        )
        statistics = torch.stack([tx0, tx1], dim=1)
    elif panel.family == "multinomial":
        unit, time, cell_size, statistics, carrier = (
            multinomial_sufficient_statistics_flat(
                _multinomial_counts(panel),
                mask_array,
                device=device,
            )
        )
    else:
        preprocessors = {
            "gaussian_known_sigma": gaussian_known_sigma_sufficient_statistics_flat,
            "poisson": poisson_sufficient_statistics_flat,
            "bernoulli": bernoulli_sufficient_statistics_flat,
            "exponential": exponential_sufficient_statistics_flat,
            "laplace_known_mu": laplace_sufficient_statistics_flat,
            "chi_square": chi_square_sufficient_statistics_flat,
        }
        kwargs = {}
        if panel.family == "gaussian_known_sigma":
            kwargs["sigma"] = panel.sigma
        if panel.family == "laplace_known_mu":
            kwargs["mu"] = panel.mu
        unit, time, cell_size, statistic, carrier = preprocessors[panel.family](
            panel.observations,
            mask_array,
            device=device,
            **kwargs,
        )
        statistics = statistic[:, None]

    return PreparedStatistics(
        unit_index=unit,
        time_index=time,
        cell_size=cell_size,
        statistics=statistics,
        log_carrier=carrier,
    )


def _make_log_joint(
    family: str,
    n: int,
    T: int,
    statistics: PreparedStatistics,
    *,
    rank: int,
    prior: torch.distributions.Distribution,
):
    """Construct the family-specific log-joint over prepared cell statistics."""

    common = {
        "unit_idx": statistics.unit_index,
        "time_idx": statistics.time_index,
        "m_vec": statistics.cell_size,
        "n": n,
        "T": T,
        "r": rank,
        "prior": prior,
    }
    if family == "gaussian":
        def log_joint(theta):
            """Evaluate the univariate Gaussian log-joint at ``theta``."""

            return gaussian_univariate_log_joint_flat(
                theta=theta,
                Tx0_vec=statistics.statistics[:, 0],
                Tx1_vec=statistics.statistics[:, 1],
                lhx=statistics.log_carrier,
                **common,
            )

        return log_joint

    log_joints = {
        "gaussian_known_sigma": gaussian_known_sigma_log_joint_flat,
        "poisson": poisson_log_joint_flat,
        "bernoulli": bernoulli_log_joint_flat,
        "exponential": exponential_log_joint_flat,
        "laplace_known_mu": laplace_known_mu_log_joint_flat,
        "chi_square": chi_square_log_joint_flat,
        "multinomial": multinomial_log_joint_flat,
    }

    def log_joint(theta):
        """Evaluate the selected one-parameter or multinomial log-joint."""

        sufficient_statistics = statistics.statistics
        if family != "multinomial":
            sufficient_statistics = sufficient_statistics[:, 0]
        return log_joints[family](
            theta=theta,
            Tx_vec=sufficient_statistics,
            log_carrier_vec=statistics.log_carrier,
            **common,
        )

    return log_joint


def _fit_prepared_factorization(
    *,
    family: str,
    n: int,
    T: int,
    n_categories: Optional[int],
    mask_array: np.ndarray,
    statistics: PreparedStatistics,
    rank: int,
    config: InferenceConfig,
    seed: Optional[int],
    prior_mean: Any,
    prior_form: str,
    initial_parameters: Any,
) -> FactorizationFit:
    """Run the shared optimizer from already-validated panel statistics."""

    blocks = predictor_block_count(family, n_categories)
    dimension = blocks * _block_dimension(n, T, rank)
    learning_rate = (
        DEFAULT_LEARNING_RATES[family]
        if config.learning_rate is None
        else float(config.learning_rate)
    )
    device = torch.device(config.device)

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    resolved_prior_mean = _resolve_parameter_vector(
        0.0 if prior_mean is None else prior_mean,
        dimension=dimension,
        name="prior_mean",
        device=device,
    )

    if prior_form == "independent_normal":
        prior = torch.distributions.Normal(
            resolved_prior_mean,
            torch.full(
                (dimension,),
                float(config.prior_scale),
                device=device,
                dtype=torch.float32,
            ),
        )
    else:
        prior = torch.distributions.MultivariateNormal(
            resolved_prior_mean,
            torch.eye(dimension, device=device, dtype=torch.float32)
            * float(config.prior_scale) ** 2,
        )
    log_joint = _make_log_joint(
        family,
        n,
        T,
        statistics,
        rank=rank,
        prior=prior,
    )
    objective = make_negative_elbo(
        log_joint=log_joint,
        dim=dimension,
        device=device,
        n_samples=int(config.elbo_samples),
    )
    if initial_parameters is None:
        init_mean = float(config.init_scale) * torch.randn(
            dimension,
            device=device,
            dtype=torch.float32,
        )
        init_log_variance = torch.full(
            (dimension,),
            float(config.init_log_variance),
            device=device,
            dtype=torch.float32,
        )
        init_params = torch.cat([init_mean, init_log_variance])
    else:
        init_params = _resolve_initial_parameters(
            initial_parameters,
            dimension=dimension,
            device=device,
        )
    parameters, loss_history, elapsed_time = fit_vi(
        negative_elbo=objective,
        init_params=init_params,
        lr=learning_rate,
        num_epochs=int(config.epochs),
        print_every=config.print_every,
    )

    losses = np.asarray(loss_history, dtype=float)
    if not np.isfinite(losses).all():
        raise FloatingPointError("The variational objective became nonfinite.")
    if not torch.isfinite(parameters).all():
        raise FloatingPointError("The fitted variational parameters are nonfinite.")

    variational_mean = parameters[:dimension]
    variational_log_variance = parameters[dimension:]
    raw_natural_parameters = reconstruct_raw_natural_parameters(
        variational_mean,
        family=family,
        n=n,
        T=T,
        rank=rank,
        n_categories=n_categories,
    )
    natural_parameters = reconstruct_natural_parameters(
        variational_mean,
        family=family,
        n=n,
        T=T,
        rank=rank,
        n_categories=n_categories,
    )
    if not torch.isfinite(natural_parameters).all():
        raise FloatingPointError("The reconstructed natural parameters are nonfinite.")

    return FactorizationFit(
        family=family,
        rank=rank,
        seed=seed,
        mask=mask_array.copy(),
        prior_mean=resolved_prior_mean,
        prior_form=prior_form,
        initial_variational_parameters=init_params,
        variational_mean=variational_mean,
        variational_log_variance=variational_log_variance,
        raw_natural_parameters=raw_natural_parameters,
        natural_parameters=natural_parameters,
        loss_history=losses,
        elapsed_time=elapsed_time,
        learning_rate=learning_rate,
        config=config,
    )


def fit_factorization(
    panel: PanelData,
    mask: Any,
    *,
    rank: int,
    config: Optional[InferenceConfig] = None,
    seed: Optional[int] = 0,
    prior_mean: Any = None,
    prior_form: str = "independent_normal",
    initial_parameters: Any = None,
) -> FactorizationFit:
    """Fit one family-specific factorization to the selected observed cells.

    The default prior, initialization, and seeded behavior are the public EFSC
    defaults. ``prior_mean``, ``prior_form``, and ``initial_parameters`` are
    low-level controls for reproducing historical fits. Passing ``seed=None``
    leaves the caller's NumPy and PyTorch RNG streams untouched instead of
    resetting them.
    """

    if not isinstance(panel, PanelData):
        raise TypeError("panel must be a PanelData object.")
    rank = _validate_rank(rank, panel.n, panel.T)
    if config is None:
        config = InferenceConfig()
    if not isinstance(config, InferenceConfig):
        raise TypeError("config must be an InferenceConfig object.")
    if not isinstance(prior_form, str) or prior_form not in {
        "independent_normal",
        "multivariate_normal",
    }:
        raise ValueError(
            "prior_form must be 'independent_normal' or 'multivariate_normal'."
        )
    seed = validate_seed(seed, name="seed", allow_none=True)

    device = torch.device(config.device)
    mask_array = _validate_mask(panel, mask)
    statistics = prepare_sufficient_statistics(panel, mask_array, device=device)
    n_categories = len(panel.categories) if panel.categories is not None else None
    return _fit_prepared_factorization(
        family=panel.family,
        n=panel.n,
        T=panel.T,
        n_categories=n_categories,
        mask_array=mask_array,
        statistics=statistics,
        rank=rank,
        config=config,
        seed=seed,
        prior_mean=prior_mean,
        prior_form=prior_form,
        initial_parameters=initial_parameters,
    )
