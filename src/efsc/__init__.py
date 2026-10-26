"""Public interface for Exponential Family Synthetic Controls."""

from importlib.metadata import PackageNotFoundError, version as _distribution_version

from .causal_results import CausalResult
from .estimator import EFSC
from .fitting import InferenceConfig
from .results import EFSCResult

try:
    __version__ = _distribution_version("efsc")
except PackageNotFoundError:  # pragma: no cover - only for an unpackaged source tree
    __version__ = "0+unknown"

__all__ = [
    "CausalResult",
    "EFSC",
    "EFSCResult",
    "InferenceConfig",
    "__version__",
]
