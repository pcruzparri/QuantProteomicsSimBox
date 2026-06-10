"""Tests for quantproteomicssimbox.rollups.

Covers the implemented scaffold: the aggregation functions, the no-scaling ``rollup``, the PTM
site-table builder, and the `roll_up` orchestrator — plus the stubbed ``rrollup``/``zrollup``.
"""

import numpy as np
import pytest

from quantproteomicssimbox.observation import Sample
from quantproteomicssimbox.protgen import Peptide
from quantproteomicssimbox.rollups import (
    AGGREGATIONS,
    SCALINGS,
    RollupResult,
    SiteTable,
    build_site_tables,
    roll_up,
    scale_rollup,
)


def _two_sample_dataset() -> list[Sample]:
    # One protein, two subjects in group 0. Site 0 is carried by two peptide species (a short
    # peptide and a miscleavage form); site 5 by one species observed in only the second subject.
    s0 = Sample(
        protein_sequence="SKARMS",
        group=0,
        subject=0,
        peptides=[
            Peptide("SK", abundance=10.0, start_index=0, end_index=1, mod_sites=[0]),
            Peptide("SKAR", abundance=4.0, start_index=0, end_index=3, mod_sites=[0]),
            Peptide("AR", abundance=20.0, start_index=2, end_index=3, mod_sites=[]),
        ],
    )
    s1 = Sample(
        protein_sequence="SKARMS",
        group=0,
        subject=1,
        peptides=[
            Peptide("SK", abundance=30.0, start_index=0, end_index=1, mod_sites=[0]),
            Peptide("SKAR", abundance=6.0, start_index=0, end_index=3, mod_sites=[0]),
            Peptide("MS", abundance=5.0, start_index=4, end_index=5, mod_sites=[5]),
        ],
    )
    return [s0, s1]


# --------------------------------------------------------------------------- #
# Registries & aggregation functions
# --------------------------------------------------------------------------- #
def test_registries_expose_paper_methods():
    assert set(SCALINGS) == {"rollup", "rrollup", "zrollup"}
    assert set(AGGREGATIONS) == {"mean", "median", "sum"}


def test_aggregation_functions_reduce_over_peptides():
    m = np.array([[10.0, 30.0], [4.0, 6.0]])  # 2 peptides x 2 samples
    assert AGGREGATIONS["sum"](m) == pytest.approx([14.0, 36.0])
    assert AGGREGATIONS["mean"](m) == pytest.approx([7.0, 18.0])
    assert AGGREGATIONS["median"](m) == pytest.approx([7.0, 18.0])


def test_aggregation_skips_nan():
    m = np.array([[10.0, np.nan], [np.nan, 6.0]])
    assert AGGREGATIONS["sum"](m) == pytest.approx([10.0, 6.0])
    assert AGGREGATIONS["mean"](m) == pytest.approx([10.0, 6.0])


def test_scale_rollup_is_identity():
    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.array_equal(scale_rollup(m), m)


# --------------------------------------------------------------------------- #
# build_site_tables
# --------------------------------------------------------------------------- #
def test_build_site_tables_groups_by_modification_site():
    tables = build_site_tables(_two_sample_dataset())
    assert [t.site for t in tables] == [0, 5]  # ordered by position; unmodified "AR" excluded

    site0 = tables[0]
    assert site0.sample_keys == [(0, 0), (0, 1)]
    assert site0.peptide_keys == [("SK", (0,)), ("SKAR", (0,))]  # both species carrying site 0
    np.testing.assert_array_equal(site0.matrix, np.array([[10.0, 30.0], [4.0, 6.0]]))

    site5 = tables[1]
    # "MS" observed only in subject 1 -> NaN in subject 0's column.
    np.testing.assert_array_equal(site5.matrix, np.array([[np.nan, 5.0]]))


def test_build_site_tables_rejects_multiple_proteins():
    samples = [
        Sample(protein_sequence="SK", group=0, subject=0),
        Sample(protein_sequence="AR", group=0, subject=1),
    ]
    with pytest.raises(ValueError):
        build_site_tables(samples)


def test_build_site_tables_empty():
    assert build_site_tables([]) == []


# --------------------------------------------------------------------------- #
# roll_up orchestrator
# --------------------------------------------------------------------------- #
def test_roll_up_rollup_sum_end_to_end():
    res = roll_up(_two_sample_dataset(), scaling="rollup", aggregation="sum")
    assert isinstance(res, RollupResult)
    assert res.sites == [0, 5]
    assert res.sample_keys == [(0, 0), (0, 1)]
    # site 0: 10+4=14, 30+6=36 ; site 5: nansum(nan)=0, 5
    np.testing.assert_allclose(res.values, np.array([[14.0, 36.0], [0.0, 5.0]]))


def test_roll_up_rollup_median():
    res = roll_up(_two_sample_dataset(), scaling="rollup", aggregation="median")
    # site 0 medians over the two peptides: [7, 18]
    np.testing.assert_allclose(res.values[0], np.array([7.0, 18.0]))
    # site 5 is unobserved in subject 0 -> NaN (missing), present (5.0) in subject 1.
    assert np.isnan(res.values[1, 0])
    assert res.values[1, 1] == pytest.approx(5.0)


def test_roll_up_rejects_unknown_methods():
    with pytest.raises(ValueError):
        roll_up(_two_sample_dataset(), scaling="bogus")
    with pytest.raises(ValueError):
        roll_up(_two_sample_dataset(), aggregation="bogus")


@pytest.mark.parametrize("scaling", ["rrollup", "zrollup"])
def test_roll_up_advanced_scalings_are_stubbed(scaling):
    with pytest.raises(NotImplementedError):
        roll_up(_two_sample_dataset(), scaling=scaling)
