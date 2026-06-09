"""Tests for quantproteomicssimbox.utils."""

from quantproteomicssimbox.utils import amino_acids


def test_amino_acids_is_the_20_standard_residues():
    assert amino_acids == set("ACDEFGHIKLMNPQRSTVWY")
    assert len(amino_acids) == 20
