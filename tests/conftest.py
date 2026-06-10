"""Shared pytest fixtures for the QuantProteomicsSimBox test suite."""

import numpy as np
import pytest

# Fixed seed -> deterministic simulation so statistical assertions don't flake.
SEED = 12345


@pytest.fixture
def rng() -> np.random.Generator:
    """A freshly seeded NumPy Generator for reproducible tests."""
    return np.random.default_rng(SEED)
