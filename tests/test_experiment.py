"""Tests for quantproteomicssimbox.experiment.

Covers the multi-protein orchestrator: dataset build, the shared-model effect structure (beta shared
across proteins, alpha per protein), pooled per-protein roll-up, and the end-to-end RMSE scoring of
estimated vs known per-site log2 fold-change.
"""

import numpy as np
import pytest

from quantproteomicssimbox.experiment import Experiment


def _experiment(**kwargs) -> Experiment:
    defaults = dict(
        n_proteins=3,
        protein_length=80,
        n_groups=2,
        n_subjects=4,
        abundance=50,
        miscleavage_rate=0.25,
        rng=np.random.default_rng(0),
    )
    defaults.update(kwargs)
    return Experiment(**defaults)


def test_build_creates_one_protein_per_group():
    exp = _experiment().build()
    assert len(exp.protein_groups) == 3
    for groups in exp.protein_groups:
        assert len(groups) == 2
        # Groups within a protein share the sequence (same coordinate system / shared alpha).
        assert groups[0].sequence == groups[1].sequence


def test_subject_effect_shared_across_proteins():
    exp = _experiment(var_subject=1.0, var_site=0.0)
    exp.observe()
    # beta keyed on (group, subject) only -> one draw per (group, subject), reused across proteins.
    assert len(exp.model.subject_effects) == exp.n_groups * exp.n_subjects
    assert set(exp.model.subject_effects) == {
        (g, s) for g in range(exp.n_groups) for s in range(exp.n_subjects)
    }


def test_site_effects_are_per_protein():
    exp = _experiment(var_subject=0.0, var_site=1.0)
    exp.observe()
    # alpha keyed on (sequence, site) -> distinct draws per protein sequence.
    sequences = {seq for seq, _site in exp.model.site_effects}
    assert sequences == {groups[0].sequence for groups in exp.protein_groups}


def test_each_protein_pools_both_groups():
    exp = _experiment()
    exp.observe()
    assert len(exp.samples) == exp.n_proteins
    for pooled in exp.samples:
        # 2 groups x 4 subjects, all sharing one protein sequence (single-protein roll-up).
        assert len(pooled) == exp.n_groups * exp.n_subjects
        assert len({s.protein_sequence for s in pooled}) == 1
        assert {(s.group, s.subject) for s in pooled} == {
            (g, sub) for g in range(exp.n_groups) for sub in range(exp.n_subjects)
        }


def test_score_zero_variance_linear_sum_recovers_truth():
    # In LINEAR space with no presence filter, var=0 -> observed == true counts; sum recovers
    # per-group occupancy at each site, so estimated FC == true FC and RMSE is 0.
    exp = _experiment(var_subject=0.0, var_site=0.0)
    assert exp.score("rollup", "sum", space="linear", min_per_group=0) == pytest.approx(0.0, abs=1e-9)


def test_score_positive_with_subject_variance():
    exp = _experiment(var_subject=1.0, var_site=1.0)
    rmse = exp.score("rollup", "mean", space="log2")
    assert np.isfinite(rmse)
    assert rmse > 0.0


def test_aggregation_space_flips_matched_aggregator():
    # Without the presence filter, the matched (low-RMSE) aggregator flips with the space:
    # sum in linear, mean in log2.
    exp = _experiment(var_subject=1.0, var_site=1.0)
    lin_sum = exp.score("rollup", "sum", space="linear", min_per_group=0)
    lin_mean = exp.score("rollup", "mean", space="linear", min_per_group=0)
    log_sum = exp.score("rollup", "sum", space="log2", min_per_group=0)
    log_mean = exp.score("rollup", "mean", space="log2", min_per_group=0)
    assert lin_sum < lin_mean  # linear: sum unbiased, mean biased by peptide count
    assert log_mean < log_sum  # log2: mean unbiased, sum inflated by peptide count (paper's result)


def test_presence_filter_curbs_log2_sum_inflation():
    # Dropping one-sided species (present in only one group) reduces the log2-sum blow-up.
    exp = _experiment(var_subject=1.0, var_site=1.0)
    unfiltered = exp.score("rollup", "sum", space="log2", min_per_group=0)
    filtered = exp.score("rollup", "sum", space="log2", min_per_group=1)
    assert filtered < unfiltered


def test_missingness_drops_observations_and_keeps_score_finite():
    clean = _experiment(var_subject=1.0, var_site=1.0, rng=np.random.default_rng(5))
    clean.observe()
    n_clean = sum(len(s.peptides) for pooled in clean.samples for s in pooled)

    missing = _experiment(var_subject=1.0, var_site=1.0, missingness=0.3, rng=np.random.default_rng(5))
    missing.observe()
    n_missing = sum(len(s.peptides) for pooled in missing.samples for s in pooled)

    assert n_missing < n_clean
    assert np.isfinite(missing.score("rollup", "mean", space="log2"))


def test_run_is_reproducible_under_seed():
    a = _experiment(var_subject=1.0, var_site=1.0, rng=np.random.default_rng(42))
    b = _experiment(var_subject=1.0, var_site=1.0, rng=np.random.default_rng(42))
    assert a.score("rollup", "median") == pytest.approx(b.score("rollup", "median"))
