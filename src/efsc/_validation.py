"""Shared scalar validation and random-seed helpers for EFSC.

This private module contains small validation contracts reused across fitting,
posterior sampling, and predictive sampling. It deliberately remains separate
from family-specific mathematics and data validation.

Author:
    Hector Rodriguez-Deniz

Affiliation:
    Data Science Institute, Columbia University

Date:
    August 2026

License:
    MIT
"""

from numbers import Integral
from typing import Any, Optional

import numpy as np


MAX_SEED = int(np.iinfo(np.uint32).max)


def validate_positive_integer(
    value: Any,
    name: str,
    *,
    minimum: int = 1,
) -> int:
    """Validate and return an integer meeting the requested minimum."""

    if (
        not isinstance(value, Integral)
        or isinstance(value, (bool, np.bool_))
        or value < minimum
    ):
        if minimum == 2:
            raise ValueError(f"{name} must be an integer of at least two.")
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def validate_seed(
    value: Any,
    *,
    name: str = "random_state",
    allow_none: bool = False,
) -> Optional[int]:
    """Validate and return an unsigned 32-bit seed, optionally allowing None."""

    if value is None and allow_none:
        return None
    if (
        not isinstance(value, Integral)
        or isinstance(value, (bool, np.bool_))
        or value < 0
        or value > MAX_SEED
    ):
        optional = "None or " if allow_none else ""
        raise ValueError(
            f"{name} must be {optional}an integer between 0 and 2**32 - 1."
        )
    return int(value)


def resolve_seed(value: Any, *, default: int) -> int:
    """Return a validated explicit seed or an already-derived default seed."""

    if value is None:
        return int(default)
    resolved = validate_seed(value)
    if resolved is None:
        raise AssertionError("Unreachable seed validation state.")
    return resolved


def derive_seed(seed: int, offset: int) -> int:
    """Add an internal offset while remaining in the unsigned 32-bit range."""

    return (seed + offset) % (MAX_SEED + 1)
