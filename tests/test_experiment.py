"""Tests for quantproteomicssimbox.experiment.

Covers the multi-protein orchestrator: dataset build, the shared-model effect structure (beta shared
across proteins, alpha per protein), pooled per-protein roll-up, and the end-to-end RMSE scoring of
estimated vs known per-site change via a `QuantMethod`.
"""

import numpy as np
import pytest

from quantproteomicssimbox.experiment import Experiment
from quantproteomicssimbox.methods import intensity_method, stoichiometry_method


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
    method = intensity_method("rollup", "sum", "linear")
    assert exp.score(method, min_per_group=0) == pytest.approx(0.0, abs=1e-9)


def test_score_positive_with_subject_variance():
    exp = _experiment(var_subject=1.0, var_site=1.0)
    rmse = exp.score(intensity_method("rollup", "mean", "log2"))
    assert np.isfinite(rmse)
    assert rmse > 0.0


def test_aggregation_space_flips_matched_aggregator():
    # Without the presence filter, the matched (low-RMSE) aggregator flips with the space:
    # sum in linear, mean in log2.
    exp = _experiment(var_subject=1.0, var_site=1.0)
    lin_sum = exp.score(intensity_method("rollup", "sum", "linear"), min_per_group=0)
    lin_mean = exp.score(intensity_method("rollup", "mean", "linear"), min_per_group=0)
    log_sum = exp.score(intensity_method("rollup", "sum", "log2"), min_per_group=0)
    log_mean = exp.score(intensity_method("rollup", "mean", "log2"), min_per_group=0)
    assert lin_sum < lin_mean  # linear: sum unbiased, mean biased by peptide count
    assert log_mean < log_sum  # log2: mean unbiased, sum inflated by peptide count (paper's result)


def test_presence_filter_curbs_log2_sum_inflation():
    # Dropping one-sided species (present in only one group) reduces the log2-sum blow-up.
    exp = _experiment(var_subject=1.0, var_site=1.0)
    log2_sum = intensity_method("rollup", "sum", "log2")
    unfiltered = exp.score(log2_sum, min_per_group=0)
    filtered = exp.score(log2_sum, min_per_group=1)
    assert filtered < unfiltered


def test_missingness_drops_observations_and_keeps_score_finite():
    clean = _experiment(var_subject=1.0, var_site=1.0, rng=np.random.default_rng(5))
    clean.observe()
    n_clean = sum(len(s.peptides) for pooled in clean.samples for s in pooled)

    missing = _experiment(var_subject=1.0, var_site=1.0, missingness=0.3, rng=np.random.default_rng(5))
    missing.observe()
    n_missing = sum(len(s.peptides) for pooled in missing.samples for s in pooled)

    assert n_missing < n_clean
    assert np.isfinite(missing.score(intensity_method("rollup", "mean", "log2")))


@pytest.mark.parametrize("scaling", ["rollup", "rrollup", "zrollup"])
def test_every_scaling_scores_finite(scaling):
    exp = _experiment(var_subject=1.0, var_site=1.0)
    assert np.isfinite(exp.score(intensity_method(scaling, "median", "log2")))


def test_detection_limit_curbs_log2_sum_inflation():
    # The log2-sum aggregator is inflated by the species count per site, which a long tail of rare
    # miscleavage singletons blows up. A detection limit prunes them, sharply cutting log2-sum RMSE
    # (toward the paper's magnitude) while barely moving the matched mean aggregator.
    sum_method = intensity_method("rollup", "sum", "log2")
    mean_method = intensity_method("rollup", "mean", "log2")

    def exp(dl):
        return _experiment(var_subject=1.0, var_site=1.0, miscleavage_rate=0.5,
                           protein_length=160, abundance=200, detection_limit=dl)

    assert exp(3).score(sum_method) < exp(1).score(sum_method)
    assert exp(3).score(mean_method) < exp(1).score(sum_method)  # mean stays well below inflated sum


def test_zrollup_is_worst_scaling():
    # The paper's finding: z-scoring each peptide destroys the log2 scale, so zrollup has the highest
    # RMSE. Average a few seeds for a stable comparison against the no-scaling rollup.
    def mean_rmse(scaling):
        return np.mean([
            _experiment(var_subject=1.0, var_site=1.0, rng=np.random.default_rng(s))
            .score(intensity_method(scaling, "median", "log2"))
            for s in range(4)
        ])

    assert mean_rmse("zrollup") > mean_rmse("rollup")


def test_run_is_reproducible_under_seed():
    method = intensity_method("rollup", "median", "log2")
    a = _experiment(var_subject=1.0, var_site=1.0, rng=np.random.default_rng(42))
    b = _experiment(var_subject=1.0, var_site=1.0, rng=np.random.default_rng(42))
    assert a.score(method) == pytest.approx(b.score(method))


# --------------------------------------------------------------------------- #
# Stoichiometry scoring (via stoichiometry_method)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["fraction", "logit"])
def test_score_stoichiometry_zero_variance_recovers_truth(method):
    # var=0 + position-aware: each copy contributes exactly one spanning peptide per site, so observed
    # mod/total == m_r/M == true stoichiometry -> estimated change == true change, RMSE 0.
    exp = _experiment(var_subject=0.0, var_site=0.0, position_aware=True)
    assert exp.score(stoichiometry_method(method), min_per_group=0) == pytest.approx(0.0, abs=1e-9)


def test_score_stoichiometry_fraction_cancels_subject_effect():
    # The subject effect scales all of a sample's peptides equally -> cancels in the mod/total ratio.
    # Only the per-site effect (numerator-only) biases the fraction, so large subject variance alone
    # still recovers the truth exactly.
    exp = _experiment(var_subject=9.0, var_site=0.0, position_aware=True)
    assert exp.score(stoichiometry_method("fraction"), min_per_group=0) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("method", ["fraction", "logit"])
def test_score_stoichiometry_positive_with_site_variance(method):
    # The per-site effect does not cancel (it multiplies only the modified peptides), so RMSE > 0.
    exp = _experiment(var_subject=1.0, var_site=1.0, position_aware=True)
    rmse = exp.score(stoichiometry_method(method))
    assert np.isfinite(rmse)
    assert rmse > 0.0


@pytest.mark.parametrize("method", ["peptide_mean", "peptide_median"])
def test_score_peptide_stoichiometry_recovers_truth_without_miscleavage(method):
    # No miscleavage -> exactly one span per site, so the per-peptide aggregation reduces to the
    # pooled ratio and recovers the truth exactly at var=0.
    exp = _experiment(miscleavage_rate=0.0, var_subject=0.0, var_site=0.0, position_aware=True)
    assert exp.score(stoichiometry_method(method), min_per_group=0) == pytest.approx(0.0, abs=1e-9)


def test_peptide_methods_score_finite_under_miscleavage_and_missingness():
    # With fragmentation + missingness the per-peptide methods are valid (finite) estimators; this is
    # the regime where they can differ from pooled (studied in the notebook).
    exp = _experiment(var_subject=1.0, var_site=1.0, missingness=0.3, position_aware=True)
    for method in ("peptide_mean", "peptide_median", "peptide_mean_logit"):
        assert np.isfinite(exp.score(stoichiometry_method(method)))


# --------------------------------------------------------------------------- #
# Per-species ionization efficiency (var_species)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["fraction", "peptide_mean"])
def test_var_species_preserves_exact_recovery_without_miscleavage(method):
    # gamma is shared by a span's modified & unmodified forms, so it cancels in the fraction. With one
    # span per site (no miscleavage), even large species variance recovers the truth exactly.
    exp = _experiment(miscleavage_rate=0.0, var_subject=0.0, var_site=0.0, var_species=9.0,
                      position_aware=True)
    assert exp.score(stoichiometry_method(method), min_per_group=0) == pytest.approx(0.0, abs=1e-9)


def test_var_site_is_not_cancelled_by_per_peptide():
    # The per-site effect alpha shifts only the *modified* peptides -> a within-span mod-vs-unmod
    # ionization difference that the per-peptide fraction does NOT cancel (unlike var_subject and
    # var_species, which scale a span's mod & unmod forms together). With one span per site (no
    # miscleavage) alpha is the only error source, so peptide_mean RMSE grows with var_site.
    def rmse(var_site):
        return _experiment(
            miscleavage_rate=0.0, var_subject=0.0, var_site=var_site, position_aware=True
        ).score(stoichiometry_method("peptide_mean"), min_per_group=0)

    assert rmse(0.0) == pytest.approx(0.0, abs=1e-9)
    assert rmse(4.0) > 0.1


def test_var_species_biases_pooled_but_not_per_peptide_under_fragmentation():
    # Multiple spans per site: the abundance-weighted pooled ratio is distorted by per-species
    # efficiency, while the per-peptide fraction cancels it. Same seed -> same ground truth & digestion,
    # so the only difference is the observation's species term.
    kw = dict(miscleavage_rate=0.5, var_subject=0.0, var_site=0.0, position_aware=True)
    frac, pep = stoichiometry_method("fraction"), stoichiometry_method("peptide_mean")
    pooled_plain = _experiment(var_species=0.0, **kw).score(frac, min_per_group=0)
    pooled_eff = _experiment(var_species=9.0, **kw).score(frac, min_per_group=0)
    pep_plain = _experiment(var_species=0.0, **kw).score(pep, min_per_group=0)
    pep_eff = _experiment(var_species=9.0, **kw).score(pep, min_per_group=0)
    assert pooled_plain == pytest.approx(0.0, abs=1e-9)  # no efficiency weighting -> exact
    assert pooled_eff > 0.1  # efficiency biases the pooled ratio
    assert pep_eff == pytest.approx(pep_plain)  # per-peptide fraction is invariant to efficiency
