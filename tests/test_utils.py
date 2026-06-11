"""Tests for quantproteomicssimbox.utils."""

import numpy as np
import pytest

from quantproteomicssimbox.utils import STOICH_EPS, amino_acids, logit2


def test_amino_acids_is_the_20_standard_residues():
    assert amino_acids == set("ACDEFGHIKLMNPQRSTVWY")
    assert len(amino_acids) == 20


def test_logit2_matches_log2_odds():
    assert logit2(0.5) == pytest.approx(0.0)
    assert logit2(0.25) == pytest.approx(np.log2(0.25 / 0.75))


def test_logit2_clamps_zero_and_one_to_finite():
    assert np.isfinite(logit2(0.0))
    assert np.isfinite(logit2(1.0))
    # 0 -> log2(eps/(1-eps)); 1 -> the symmetric positive value.
    assert logit2(0.0) == pytest.approx(np.log2(STOICH_EPS / (1 - STOICH_EPS)))
    assert logit2(1.0) == pytest.approx(-logit2(0.0))


def test_logit2_propagates_nan():
    out = logit2(np.array([0.25, np.nan]))
    assert out[0] == pytest.approx(np.log2(1 / 3))
    assert np.isnan(out[1])
