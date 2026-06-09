"""Shared pytest fixtures for the QuantProteomicsSimBox test suite."""

import numpy as np
import pytest

# A fixed seed keeps the stochastic simulation deterministic across runs so that
# statistical assertions (occupancy ranges, realized miscleavage rates, weighting)
# are reproducible rather than flaky.
SEED = 12345


@pytest.fixture
def rng() -> np.random.Generator:
    """A freshly seeded NumPy Generator for reproducible tests."""
    return np.random.default_rng(SEED)
