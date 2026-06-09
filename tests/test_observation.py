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


def test_observation_model_stub_raises():
    model = ObservationModel(sigma_subject=1.0, sigma_site=1.0)
    protein = Protein("SAKAR", rng=np.random.default_rng(0))
    protein.set_quantification(5)
    with pytest.raises(NotImplementedError):
        model.sample(protein, group=0, subject=0)
    with pytest.raises(NotImplementedError):
        model.apply_missingness([], rate=0.25)
