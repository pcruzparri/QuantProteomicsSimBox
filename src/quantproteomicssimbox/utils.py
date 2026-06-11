"""Shared constants and small math helpers used across modules."""

import numpy as np

amino_acids = set("ACDEFGHIKLMNPQRSTVWY")

# Small offset that keeps a logit finite when a fraction hits exactly 0 or 1.
STOICH_EPS = 1e-6


def logit2(fraction, eps: float = STOICH_EPS):
    """Base-2 logit ``log2(f / (1 - f))`` with ``f`` clamped to ``[eps, 1 - eps]``.

    The clamp keeps the transform finite at the boundaries (f = 0 or 1). NaN passes through (both
    ``np.clip`` and ``np.log2`` propagate it), so an unobserved/missing fraction stays missing.
    Accepts scalars or arrays.
    """
    f = np.clip(fraction, eps, 1.0 - eps)
    return np.log2(f / (1.0 - f))
