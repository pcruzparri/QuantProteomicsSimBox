"""Tests for quantproteomicssimbox.observation.

Covers the implemented ``aggregate_peptides`` collapse (position-aware vs agnostic) and
the structural scaffolding (``Sample`` dataclass, stubbed ``ObservationModel`` methods).
"""

import numpy as np
import pytest

from quantproteomicssimbox.observation import ObservationModel, Sample, aggregate_peptides
from quantproteomicssimbox.protgen import Peptide, Protein, ProteinGenerator


# --------------------------------------------------------------------------- #
# aggregate_peptides
# --------------------------------------------------------------------------- #
def test_agnostic_merges_same_species_from_different_loci():
    # Same sequence, same relative mod pattern (the S), different protein positions.
    p1 = Peptide("SK", abundance=3, start_index=0, end_index=1, mod_sites=[0])
    p2 = Peptide("SK", abundance=5, start_index=10, end_index=11, mod_sites=[10])
    out = aggregate_peptides([p1, p2], position_aware=False)
    assert len(out) == 1
    assert out[0].sequence == "SK"
    assert out[0].abundance == 8
    # First occurrence is the representative.
    assert out[0].start_index == 0 and out[0].mod_sites == [0]


def test_position_aware_keeps_loci_separate():
    p1 = Peptide("SK", abundance=3, start_index=0, end_index=1, mod_sites=[0])
    p2 = Peptide("SK", abundance=5, start_index=10, end_index=11, mod_sites=[10])
    out = aggregate_peptides([p1, p2], position_aware=True)
    assert len(out) == 2
    assert {pep.abundance for pep in out} == {3, 5}


def test_agnostic_distinguishes_modification_state():
    # Same sequence and locus, different modification signatures -> distinct species.
    modified = Peptide("SK", abundance=3, start_index=0, end_index=1, mod_sites=[0])
    unmodified = Peptide("SK", abundance=5, start_index=0, end_index=1, mod_sites=[])
    out = aggregate_peptides([modified, unmodified], position_aware=False)
    assert len(out) == 2


@pytest.mark.parametrize("position_aware", [True, False])
def test_aggregate_conserves_total_abundance(position_aware):
    gen = ProteinGenerator(rng=np.random.default_rng(11))
    p = gen.generate_protein(60)
    p.set_quantification(30, miscleavage_rate=0.25)
    out = aggregate_peptides(p.peptides, position_aware=position_aware)
    assert sum(pep.abundance for pep in out) == sum(pep.abundance for pep in p.peptides)


def test_aware_aggregation_is_identity_on_ground_truth(rng):
    # Protein.peptides are already position-aware/distinct, so aware aggregation must not merge.
    gen = ProteinGenerator(rng=rng)
    p = gen.generate_protein(60)
    p.set_quantification(20, miscleavage_rate=0.2)
    out = aggregate_peptides(p.peptides, position_aware=True)
    assert len(out) == len(p.peptides)


def test_aggregate_does_not_mutate_inputs():
    pep = Peptide("SK", abundance=3, start_index=0, end_index=1, mod_sites=[0])
    aggregate_peptides([pep, Peptide("SK", abundance=5, start_index=0, end_index=1, mod_sites=[0])])
    assert pep.abundance == 3  # original untouched


# --------------------------------------------------------------------------- #
# Sample / ObservationModel scaffold
# --------------------------------------------------------------------------- #
def test_sample_dataclass_holds_metadata_and_peptides():
    pep = Peptide("AK", abundance=1.5, start_index=0, end_index=1)
    s = Sample(protein_sequence="AKAR", group=0, subject=2, peptides=[pep])
    assert s.group == 0 and s.subject == 2
    assert s.protein_sequence == "AKAR"
    assert s.peptides == [pep]


def test_sample_defaults_to_empty_peptides():
    s = Sample(protein_sequence="AKAR", group=1, subject=0)
    assert s.peptides == []


def _missingness_samples(rng):
    gen = ProteinGenerator(rng=rng)
    p = gen.generate_protein(120)
    p.set_quantification(50, miscleavage_rate=0.25)
    model = ObservationModel(var_subject=1.0, var_site=1.0, rng=rng)
    return model, model.sample_group(p, group=0, n_subjects=10)


def test_missingness_rate_zero_is_identity():
    model, samples = _missingness_samples(np.random.default_rng(0))
    assert model.apply_missingness(samples, 0.0) is samples


def test_missingness_drops_target_fraction():
    model, samples = _missingness_samples(np.random.default_rng(1))
    total = sum(len(s.peptides) for s in samples)
    out = model.apply_missingness(samples, 0.3)
    assert sum(len(s.peptides) for s in out) == total - round(0.3 * total)


def test_missingness_is_abundance_dependent():
    # Low-abundance observations drop more often, so survivors skew high-abundance.
    model, samples = _missingness_samples(np.random.default_rng(2))
    all_ab = sorted(pep.abundance for s in samples for pep in s.peptides)
    median_ab = all_ab[len(all_ab) // 2]
    out = model.apply_missingness(samples, 0.4)
    kept = [pep.abundance for s in out for pep in s.peptides]
    assert np.mean([a < median_ab for a in kept]) < 0.5


def test_missingness_does_not_mutate_inputs():
    model, samples = _missingness_samples(np.random.default_rng(3))
    before = [len(s.peptides) for s in samples]
    model.apply_missingness(samples, 0.3)
    assert [len(s.peptides) for s in samples] == before


def test_missingness_deterministic_under_seed():
    def run():
        model, samples = _missingness_samples(np.random.default_rng(4))
        return [len(s.peptides) for s in model.apply_missingness(samples, 0.3)]

    assert run() == run()


def test_sample_sets_metadata_and_float_abundances():
    protein = Protein("SAKAR", rng=np.random.default_rng(0))
    protein.set_quantification(5)
    model = ObservationModel(rng=np.random.default_rng(0))
    s = model.sample(protein, group=1, subject=2)
    assert s.group == 1 and s.subject == 2 and s.protein_sequence == "SAKAR"
    assert all(isinstance(pep.abundance, float) for pep in s.peptides)


def test_sample_zero_variance_reproduces_truth():
    # sigma=0 -> 2^0 factors -> observed abundances equal the (aggregated) ground truth.
    gen = ProteinGenerator(rng=np.random.default_rng(7))
    p = gen.generate_protein(80)
    p.set_quantification(20, miscleavage_rate=0.25)
    model = ObservationModel(var_subject=0.0, var_site=0.0, rng=np.random.default_rng(1))
    s = model.sample(p, group=0, subject=0)
    truth = aggregate_peptides(p.peptides, position_aware=False)
    assert sorted(pep.abundance for pep in s.peptides) == pytest.approx(
        sorted(float(pep.abundance) for pep in truth)
    )


def test_subject_effect_is_constant_factor_across_peptides():
    # With only a subject effect (var_site=0), every peptide is scaled by the same 2^beta.
    gen = ProteinGenerator(rng=np.random.default_rng(3))
    p = gen.generate_protein(80)
    p.set_quantification(20, miscleavage_rate=0.0)
    model = ObservationModel(var_subject=1.0, var_site=0.0, rng=np.random.default_rng(2))
    s = model.sample(p, group=0, subject=0)
    truth = aggregate_peptides(p.peptides, position_aware=False)
    assert len(s.peptides) == len(truth)
    ratios = [o.abundance / t.abundance for o, t in zip(s.peptides, truth)]
    assert ratios == pytest.approx([ratios[0]] * len(ratios))


def test_site_effects_keyed_by_protein_not_peptide_form():
    # Regression for the alpha-keying bug: alpha is a property of the absolute site, shared across
    # the peptide forms that carry it (e.g. miscleavage variants). The cache therefore keys on the
    # protein sequence + absolute site, with exactly one draw per distinct modified site.
    gen = ProteinGenerator(rng=np.random.default_rng(5))
    p = gen.generate_protein(120)
    p.set_quantification(30, miscleavage_rate=0.5)
    model = ObservationModel(var_subject=0.0, var_site=1.0, rng=np.random.default_rng(9))
    model.sample(p, group=0, subject=0)
    modified_sites = {site for pep in p.peptides for site in pep.mod_sites}
    assert modified_sites  # sanity: the protein actually has modified sites
    assert all(seq == p.sequence for seq, _site in model.site_effects)
    assert {site for _seq, site in model.site_effects} == modified_sites


def test_species_effect_keyed_by_sequence_shared_across_mod_forms():
    # gamma_p is a per-sequence (backbone) ionization factor: one draw per distinct peptide sequence,
    # shared by every mod-form of that backbone. With only var_species, the modified and unmodified
    # "SK" species are scaled by the same 2^gamma, so it cancels in their within-span fraction.
    p = Protein("SK", rng=np.random.default_rng(0))
    p.set_quantification(100, miscleavage_rate=0.0)  # yields modified + unmodified "SK" species
    model = ObservationModel(var_subject=0.0, var_site=0.0, var_species=1.0, rng=np.random.default_rng(1))
    s = model.sample(p, group=0, subject=0)
    assert set(model.species_effects) == {"SK"}  # one draw, keyed by sequence
    truth = {tuple(t.mod_sites): t.abundance for t in p.peptides}
    ratios = [o.abundance / truth[tuple(o.mod_sites)] for o in s.peptides]
    assert ratios == pytest.approx([ratios[0]] * len(ratios))  # same 2^gamma for both forms


def test_sample_does_not_mutate_protein():
    gen = ProteinGenerator(rng=np.random.default_rng(4))
    p = gen.generate_protein(60)
    p.set_quantification(15, miscleavage_rate=0.2)
    before = [(pep.sequence, pep.abundance, tuple(pep.mod_sites)) for pep in p.peptides]
    model = ObservationModel(var_subject=1.0, var_site=1.0, rng=np.random.default_rng(0))
    model.sample(p, group=0, subject=0)
    after = [(pep.sequence, pep.abundance, tuple(pep.mod_sites)) for pep in p.peptides]
    assert before == after


def test_sample_is_deterministic_under_seed():
    gen = ProteinGenerator(rng=np.random.default_rng(8))
    p = gen.generate_protein(80)
    p.set_quantification(20, miscleavage_rate=0.25)
    s1 = ObservationModel(var_subject=1.0, var_site=1.0, rng=np.random.default_rng(123)).sample(p, 0, 0)
    s2 = ObservationModel(var_subject=1.0, var_site=1.0, rng=np.random.default_rng(123)).sample(p, 0, 0)
    assert [pep.abundance for pep in s1.peptides] == pytest.approx([pep.abundance for pep in s2.peptides])
