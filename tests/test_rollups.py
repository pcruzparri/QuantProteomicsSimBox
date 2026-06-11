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
    STOICHIOMETRY_METHODS,
    RollupResult,
    SiteTable,
    build_peptide_fraction_tables,
    build_site_tables,
    build_stoichiometry_tables,
    group_site_change,
    roll_up,
    roll_up_stoichiometry,
    scale_rollup,
)
from quantproteomicssimbox.utils import logit2


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
    res = roll_up(_two_sample_dataset(), scaling="rollup", aggregation="sum", space="linear")
    assert isinstance(res, RollupResult)
    assert res.sites == [0, 5]
    assert res.sample_keys == [(0, 0), (0, 1)]
    assert res.space == "linear"
    # site 0: 10+4=14, 30+6=36 ; site 5: nansum(nan)=0, 5
    np.testing.assert_allclose(res.values, np.array([[14.0, 36.0], [0.0, 5.0]]))


def test_roll_up_rollup_median():
    res = roll_up(_two_sample_dataset(), scaling="rollup", aggregation="median", space="linear")
    # site 0 medians over the two peptides: [7, 18]
    np.testing.assert_allclose(res.values[0], np.array([7.0, 18.0]))
    # site 5 is unobserved in subject 0 -> NaN (missing), present (5.0) in subject 1.
    assert np.isnan(res.values[1, 0])
    assert res.values[1, 1] == pytest.approx(5.0)


def test_roll_up_log2_space_transforms_before_aggregating():
    # Default space is log2: site-0 sum is log2(10)+log2(4) for subject 0, not 14.
    res = roll_up(_two_sample_dataset(), scaling="rollup", aggregation="sum")
    assert res.space == "log2"
    assert res.values[0, 0] == pytest.approx(np.log2(10.0) + np.log2(4.0))


def test_roll_up_rejects_unknown_methods():
    with pytest.raises(ValueError):
        roll_up(_two_sample_dataset(), scaling="bogus")
    with pytest.raises(ValueError):
        roll_up(_two_sample_dataset(), aggregation="bogus")
    with pytest.raises(ValueError):
        roll_up(_two_sample_dataset(), space="bogus")


@pytest.mark.parametrize("scaling", ["rrollup", "zrollup"])
def test_roll_up_advanced_scalings_are_stubbed(scaling):
    with pytest.raises(NotImplementedError):
        roll_up(_two_sample_dataset(), scaling=scaling)


# --------------------------------------------------------------------------- #
# group_site_change
# --------------------------------------------------------------------------- #
def test_group_site_change_linear_space_uses_ratio():
    result = RollupResult(
        sites=[5, 9],
        sample_keys=[(0, 0), (0, 1), (1, 0), (1, 1)],
        values=np.array([[2.0, 2.0, 4.0, 4.0], [1.0, 1.0, 1.0, 1.0]]),
        space="linear",
    )
    fc = group_site_change(result, group_a=0, group_b=1)
    assert fc[5] == pytest.approx(1.0)  # log2(mean[4,4] / mean[2,2])
    assert fc[9] == pytest.approx(0.0)  # log2(1 / 1)


def test_presence_filter_drops_one_sided_species():
    # Site 0 carried by a shared species (both groups) plus one present only in group 1.
    samples = [
        Sample("X", group=0, subject=0,
               peptides=[Peptide("SK", abundance=10.0, start_index=0, end_index=1, mod_sites=[0])]),
        Sample("X", group=1, subject=0, peptides=[
            Peptide("SK", abundance=20.0, start_index=0, end_index=1, mod_sites=[0]),
            Peptide("SAK", abundance=5.0, start_index=0, end_index=2, mod_sites=[0]),  # one-sided
        ]),
    ]
    col = lambda res: res.sample_keys.index((1, 0))
    raw = roll_up(samples, aggregation="sum", space="linear", min_per_group=0)
    filt = roll_up(samples, aggregation="sum", space="linear", min_per_group=1)
    assert raw.values[0, col(raw)] == pytest.approx(25.0)  # 20 (SK) + 5 (SAK)
    assert filt.values[0, col(filt)] == pytest.approx(20.0)  # one-sided SAK dropped


def test_group_site_change_log2_space_uses_difference():
    # Values are already log2 abundances -> FC is the difference of group means, not a log-ratio.
    result = RollupResult(
        sites=[5],
        sample_keys=[(0, 0), (0, 1), (1, 0), (1, 1)],
        values=np.array([[1.0, 1.0, 3.5, 3.5]]),
        space="log2",
    )
    fc = group_site_change(result, group_a=0, group_b=1)
    assert fc[5] == pytest.approx(2.5)  # 3.5 - 1.0


def test_group_site_change_logit_space_uses_difference():
    # Logit values are already in a log space -> change is the difference of group means.
    result = RollupResult(
        sites=[5],
        sample_keys=[(0, 0), (1, 0)],
        values=np.array([[-2.0, 1.0]]),
        space="logit",
    )
    change = group_site_change(result, group_a=0, group_b=1)
    assert change[5] == pytest.approx(3.0)  # 1.0 - (-2.0)


# --------------------------------------------------------------------------- #
# Stoichiometry roll-up
# --------------------------------------------------------------------------- #
def _stoich_dataset() -> list[Sample]:
    # Each serine site is spanned by a modified AND an unmodified species, so the denominator (total
    # spanning abundance) exceeds the numerator -> fraction < 1. Site 0: 10/(10+30); site 5: 5/(5+15).
    s = Sample(
        protein_sequence="SKARMS",
        group=0,
        subject=0,
        peptides=[
            Peptide("SK", abundance=10.0, start_index=0, end_index=1, mod_sites=[0]),
            Peptide("SK", abundance=30.0, start_index=0, end_index=1, mod_sites=[]),
            Peptide("MS", abundance=5.0, start_index=4, end_index=5, mod_sites=[5]),
            Peptide("MS", abundance=15.0, start_index=4, end_index=5, mod_sites=[]),
        ],
    )
    return [s]


def test_stoichiometry_registry_exposes_methods():
    assert set(STOICHIOMETRY_METHODS) == {
        "fraction", "logit", "logit_pseudocount",
        "peptide_mean", "peptide_median", "peptide_mean_logit", "peptide_median_logit",
    }
    # space is derived from the transform: bare fraction -> "fraction"; logit -> "logit".
    assert STOICHIOMETRY_METHODS["fraction"].space == "fraction"
    assert STOICHIOMETRY_METHODS["logit"].space == "logit"
    assert STOICHIOMETRY_METHODS["logit_pseudocount"].space == "logit"
    assert STOICHIOMETRY_METHODS["peptide_mean"].space == "fraction"
    assert STOICHIOMETRY_METHODS["peptide_median_logit"].space == "logit"


def test_build_stoichiometry_tables_sums_mod_and_total_spanning():
    tables = build_stoichiometry_tables(_stoich_dataset())
    assert [t.site for t in tables] == [0, 5]
    np.testing.assert_array_equal(tables[0].mod, [10.0])
    np.testing.assert_array_equal(tables[0].total, [40.0])  # numerator + unmodified spanning
    np.testing.assert_array_equal(tables[1].mod, [5.0])
    np.testing.assert_array_equal(tables[1].total, [20.0])


def test_build_stoichiometry_tables_marks_unobserved_site_nan():
    # "AR" spans neither serine (indices 0, 5) -> both sites unobserved in this sample.
    s = Sample("SKARMS", group=0, subject=0,
               peptides=[Peptide("AR", abundance=8.0, start_index=2, end_index=3, mod_sites=[])])
    assert build_stoichiometry_tables([s]) == []  # no serine spanned -> no tables


def test_build_stoichiometry_tables_rejects_multiple_proteins():
    samples = [Sample("SK", group=0, subject=0), Sample("AR", group=0, subject=1)]
    with pytest.raises(ValueError):
        build_stoichiometry_tables(samples)


def test_roll_up_stoichiometry_fraction_values():
    res = roll_up_stoichiometry(_stoich_dataset(), method="fraction", min_per_group=0)
    assert res.space == "fraction"
    assert res.sites == [0, 5]
    np.testing.assert_allclose(res.values.ravel(), [0.25, 0.25])  # 10/40, 5/20


def test_roll_up_stoichiometry_logit_values():
    res = roll_up_stoichiometry(_stoich_dataset(), method="logit", min_per_group=0)
    assert res.space == "logit"
    np.testing.assert_allclose(res.values.ravel(), [logit2(0.25), logit2(0.25)])


def test_roll_up_stoichiometry_logit_clamps_full_modification():
    # A site carried only by a modified species -> fraction 1.0 -> logit must stay finite (clamped).
    s = Sample("SK", group=0, subject=0,
               peptides=[Peptide("SK", abundance=7.0, start_index=0, end_index=1, mod_sites=[0])])
    res = roll_up_stoichiometry([s], method="logit", min_per_group=0)
    assert np.isfinite(res.values).all()
    assert res.values.ravel()[0] == pytest.approx(logit2(1.0))


def test_roll_up_stoichiometry_pseudocount_softens_boundary():
    # Haldane: fraction 1.0 -> (7+0.5)/(7+1) = 0.9375 before logit (less extreme than the hard clamp).
    s = Sample("SK", group=0, subject=0,
               peptides=[Peptide("SK", abundance=7.0, start_index=0, end_index=1, mod_sites=[0])])
    res = roll_up_stoichiometry([s], method="logit_pseudocount", min_per_group=0)
    assert res.values.ravel()[0] == pytest.approx(logit2(7.5 / 8.0))


def test_roll_up_stoichiometry_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown stoichiometry method"):
        roll_up_stoichiometry(_stoich_dataset(), method="bogus")


# --------------------------------------------------------------------------- #
# Per-peptide-span stoichiometry
# --------------------------------------------------------------------------- #
def _multi_span_dataset() -> list[Sample]:
    # Site 0 covered by two spans with UNEQUAL totals: SK (mod 10 / unmod 30 -> f=0.25, total 40) and
    # SKAR (mod 4 / unmod 6 -> f=0.40, total 10). pooled = 14/50 = 0.28; per-peptide mean = 0.325.
    s = Sample(
        protein_sequence="SKARMS",
        group=0,
        subject=0,
        peptides=[
            Peptide("SK", abundance=10.0, start_index=0, end_index=1, mod_sites=[0]),
            Peptide("SK", abundance=30.0, start_index=0, end_index=1, mod_sites=[]),
            Peptide("SKAR", abundance=4.0, start_index=0, end_index=3, mod_sites=[0]),
            Peptide("SKAR", abundance=6.0, start_index=0, end_index=3, mod_sites=[]),
        ],
    )
    return [s]


def test_build_peptide_fraction_tables_per_span_fractions():
    tables = build_peptide_fraction_tables(_multi_span_dataset())
    assert [t.site for t in tables] == [0]
    t = tables[0]
    assert t.span_keys == [(0, 1), (0, 3)]
    np.testing.assert_allclose(t.fractions.ravel(), [0.25, 0.40])


def test_peptide_mean_differs_from_pooled_with_unequal_spans():
    pooled = roll_up_stoichiometry(_multi_span_dataset(), "fraction", min_per_group=0)
    pep_mean = roll_up_stoichiometry(_multi_span_dataset(), "peptide_mean", min_per_group=0)
    pep_median = roll_up_stoichiometry(_multi_span_dataset(), "peptide_median", min_per_group=0)
    assert pooled.values.ravel()[0] == pytest.approx(14 / 50)  # 0.28, abundance-weighted
    assert pep_mean.values.ravel()[0] == pytest.approx(0.325)  # (0.25 + 0.40) / 2, span-unweighted
    assert pep_median.values.ravel()[0] == pytest.approx(0.325)


def test_peptide_and_pooled_agree_with_a_single_span():
    # One span per site -> per-peptide aggregation reduces to the pooled ratio.
    single = _stoich_dataset()  # each site carried by exactly one span
    pooled = roll_up_stoichiometry(single, "fraction", min_per_group=0)
    pep_mean = roll_up_stoichiometry(single, "peptide_mean", min_per_group=0)
    np.testing.assert_allclose(pooled.values, pep_mean.values)
