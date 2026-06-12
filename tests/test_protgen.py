"""Tests for quantproteomicssimbox.protgen.

These exercise the PTM digestion pipeline: protein/sequence generation, the trypsin
cleavage rules, the per-site serine occupancy model (paper Eq. 1), the proportion-
controlled weighted missed-cleavage sampling, and the per-copy modified-peptide
quantification stored as ``Protein.peptides`` (a list of ``Peptide`` objects).
"""

import numpy as np
import pytest

from quantproteomicssimbox.protgen import (
    AMINO_ACIDS,
    Peptide,
    Protein,
    ProteinGenerator,
    make_group_proteins,
    true_site_log2_fold_change,
    true_site_stoichiometry_change,
)
from quantproteomicssimbox.utils import amino_acids, logit2


def peptide_summary(protein):
    """Canonical {(start, end, mod_sites): abundance} view for comparisons."""
    return {
        (p.start_index, p.end_index, tuple(p.mod_sites)): p.abundance
        for p in protein.peptides
    }


# --------------------------------------------------------------------------- #
# ProteinGenerator
# --------------------------------------------------------------------------- #
def test_amino_acids_constant_matches_utils():
    assert set(AMINO_ACIDS) == amino_acids
    # Sorted tuple -> stable order for reproducible seeded sampling.
    assert AMINO_ACIDS == tuple(sorted(amino_acids))


def test_generate_sequence_length_and_alphabet(rng):
    gen = ProteinGenerator(rng=rng)
    seq = gen.generate_sequence(200)
    assert isinstance(seq, str)
    assert len(seq) == 200
    assert set(seq) <= set(AMINO_ACIDS)


def test_generate_sequence_is_reproducible():
    a = ProteinGenerator(rng=np.random.default_rng(0)).generate_sequence(100)
    b = ProteinGenerator(rng=np.random.default_rng(0)).generate_sequence(100)
    assert a == b


def test_generate_protein_wraps_sequence(rng):
    gen = ProteinGenerator(rng=rng)
    protein = gen.generate_protein(50)
    assert isinstance(protein, Protein)
    assert len(protein.sequence) == 50


def test_generate_sequence_repeat_units_creates_repeated_peptide():
    gen = ProteinGenerator(rng=np.random.default_rng(0))
    seq = gen.generate_sequence(80, repeat_units=3, unit_length=7)
    assert len(seq) == 80
    p = Protein(seq, rng=np.random.default_rng(0))
    p.set_quantification(1, miscleavage_rate=0.0)  # perfect digest, single copy
    counts: dict[str, int] = {}
    for pep in p.peptides:
        counts[pep.sequence] = counts.get(pep.sequence, 0) + 1
    repeated = [s for s, c in counts.items() if c == 3]  # the unit digests at all 3 loci
    assert repeated, "the forced unit should appear as the same peptide at every repeat locus"
    assert all(u.endswith("K") and "S" in u for u in repeated)  # clean tryptic peptide w/ a serine


def test_generate_sequence_repeat_units_must_fit():
    gen = ProteinGenerator(rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        gen.generate_sequence(20, repeat_units=3, unit_length=8)  # 24 unit residues > 20


def test_generate_sequence_zero_repeats_is_plain_random():
    gen = ProteinGenerator(rng=np.random.default_rng(0))
    seq = gen.generate_sequence(50)
    assert len(seq) == 50
    assert set(seq) <= set(AMINO_ACIDS)


def test_end_to_end_reproducible_with_same_seed():
    def run():
        gen = ProteinGenerator(rng=np.random.default_rng(7))
        p = gen.generate_protein(80)
        p.set_quantification(40, miscleavage_rate=0.3)
        return peptide_summary(p), p.digestion_map

    q1, d1 = run()
    q2, d2 = run()
    assert q1 == q2
    assert d1 == d2


# --------------------------------------------------------------------------- #
# Protein basics / serine map
# --------------------------------------------------------------------------- #
def test_serine_map_locates_serines():
    p = Protein("SAGSAK")
    assert p.serine_map == [0, 3]


def test_digest_requires_abundance():
    p = Protein("AKAR")
    with pytest.raises(ValueError, match="Abundance and modification table must be set"):
        p.digest()


def test_set_quantification_builds_mod_table(rng):
    p = Protein("SKSAR", rng=rng)
    p.set_quantification(25)
    assert p.abundance == 25
    assert p.mod_table.shape == (25, 5)


# --------------------------------------------------------------------------- #
# Trypsin cleavage rules
# --------------------------------------------------------------------------- #
def test_cleaves_after_k_and_r(rng):
    p = Protein("AAKAAR", rng=rng)
    p.set_quantification(1)
    # K at index 2, R at index 5 (terminal) -> both are cut sites.
    assert p.digestion_sites == [2, 5]


def test_no_cleavage_when_proline_follows(rng):
    p = Protein("AKPAAR", rng=rng)
    p.set_quantification(1)
    # K at index 1 is followed by P -> not a cut site; only R at index 5 remains.
    assert p.digestion_sites == [5]


def test_terminal_kr_is_a_cut_site(rng):
    p = Protein("AAAK", rng=rng)
    p.set_quantification(1)
    assert p.digestion_sites == [3]


def test_protein_with_no_kr_has_no_sites(rng):
    p = Protein("AAGGAA", rng=rng)
    p.set_quantification(5)
    assert p.digestion_sites == []
    assert p.digestion_map == [[] for _ in range(5)]


# --------------------------------------------------------------------------- #
# Occupancy model (paper Eq. 1: m_r ~ U(1, M))
# --------------------------------------------------------------------------- #
def test_modifications_only_on_serines(rng):
    p = Protein("SKSAR", rng=rng)
    p.set_quantification(100)
    non_serine = [i for i in range(len(p.sequence)) if i not in p.serine_map]
    assert p.mod_table[:, non_serine].sum() == 0


def test_per_site_occupancy_within_1_to_M(rng):
    M = 100
    p = Protein("S" * 10, rng=rng)  # 10 serine sites
    p.set_quantification(M)
    occupancy = p.mod_table.sum(axis=0)
    assert occupancy.min() >= 1  # every site modified in at least one copy
    assert occupancy.max() <= M  # never more copies than exist


def test_occupancy_reproducible():
    def occ():
        p = Protein("SSSSS", rng=np.random.default_rng(3))
        p.set_quantification(50)
        return p.mod_table.sum(axis=0).tolist()

    assert occ() == occ()


# --------------------------------------------------------------------------- #
# Missed-cleavage sampling
# --------------------------------------------------------------------------- #
def test_rate_zero_is_perfect_digest(rng):
    p = Protein("AKAKAKAKAR", rng=rng)
    p.set_quantification(20, miscleavage_rate=0.0)
    assert all(copy == p.digestion_sites for copy in p.digestion_map)


def test_rate_one_misses_all_missable_sites(rng):
    p = Protein("AKAKAKAKAR", rng=rng)
    p.set_quantification(20, miscleavage_rate=1.0)
    # Terminal-residue cuts are excluded as no-ops, so at most one site may survive.
    assert all(len(copy) <= 1 for copy in p.digestion_map)


def test_realized_missed_fraction_tracks_rate():
    for rate in (0.0, 0.25, 0.5):
        gen = ProteinGenerator(rng=np.random.default_rng(123))
        fractions = []
        for _ in range(100):
            p = gen.generate_protein(200)
            p.set_quantification(60, miscleavage_rate=rate)
            n = len(p.digestion_sites)
            if n:
                fractions.append(np.mean([1 - len(c) / n for c in p.digestion_map]))
        assert abs(np.mean(fractions) - rate) < 0.03


def test_shorter_flanks_are_missed_more_often():
    gen = ProteinGenerator(rng=np.random.default_rng(5))
    missed, total = {}, {}
    for _ in range(150):
        p = gen.generate_protein(120)
        p.set_quantification(50, miscleavage_rate=0.4)
        sites = p.digestion_sites
        if len(sites) < 3:
            continue
        cuts = [-1] + sites + [len(p.sequence) - 1]
        for i, s in enumerate(sites):
            min_flank = min(s - cuts[i], cuts[i + 2] - s)
            if min_flank <= 0:
                continue
            n_missed = p.abundance - sum(s in c for c in p.digestion_map)
            missed[min_flank] = missed.get(min_flank, 0) + n_missed
            total[min_flank] = total.get(min_flank, 0) + p.abundance

    short = sum(missed.get(k, 0) for k in total if k <= 3) / sum(total[k] for k in total if k <= 3)
    long = sum(missed.get(k, 0) for k in total if k >= 8) / sum(total[k] for k in total if k >= 8)
    assert short > long


def test_terminal_residue_does_not_emit_runtime_warning(recwarn):
    # A cut on the final residue has a zero-length downstream flank; the weight code
    # must not trigger a divide-by-zero RuntimeWarning.
    p = Protein("AAAKAAAK", rng=np.random.default_rng(0))
    p.set_quantification(5, miscleavage_rate=0.5)
    assert [w for w in recwarn if issubclass(w.category, RuntimeWarning)] == []


# --------------------------------------------------------------------------- #
# Miscleavage model fork: "global" (fixed count) vs "bernoulli" (per-site probability)
# --------------------------------------------------------------------------- #
# A sequence with well-separated, all-missable cut sites (none terminal, all flanks >= 2): the K
# after each "AAK" block, ending in "AA". Lets realized fractions track the rate exactly.
ALL_MISSABLE_SEQ = "AAK" * 8 + "AA"  # 8 cut sites, all missable


def test_unknown_miscleavage_model_raises(rng):
    p = Protein("AKAR", rng=rng)
    with pytest.raises(ValueError, match="unknown miscleavage_model"):
        p.set_quantification(5, miscleavage_model="poisson")


def test_global_digestion_map_is_fixed_length(rng):
    # round(rate * n_sites) cuts missed per copy -> every copy keeps the same number of cuts.
    p = Protein(ALL_MISSABLE_SEQ, rng=rng)
    p.set_quantification(200, miscleavage_rate=0.5, miscleavage_model="global")
    assert len({len(copy) for copy in p.digestion_map}) == 1


def test_bernoulli_digestion_map_is_variable_length(rng):
    # Binomial missed count per copy -> copies generally keep different numbers of cuts.
    p = Protein(ALL_MISSABLE_SEQ, rng=rng)
    p.set_quantification(200, miscleavage_rate=0.5, miscleavage_model="bernoulli")
    assert len({len(copy) for copy in p.digestion_map}) > 1


def test_bernoulli_rate_zero_is_perfect_digest(rng):
    p = Protein("AKAKAKAKAR", rng=rng)
    p.set_quantification(20, miscleavage_rate=0.0, miscleavage_model="bernoulli")
    assert all(copy == p.digestion_sites for copy in p.digestion_map)


def test_bernoulli_rate_one_misses_all_missable_sites(rng):
    p = Protein("AKAKAKAKAR", rng=rng)
    p.set_quantification(20, miscleavage_rate=1.0, miscleavage_model="bernoulli")
    # Terminal-residue cuts are never missable, so at most one site (the C-terminal cut) survives.
    assert all(len(copy) <= 1 for copy in p.digestion_map)


def test_bernoulli_realized_fraction_tracks_rate():
    for rate in (0.0, 0.25, 0.5):
        p = Protein(ALL_MISSABLE_SEQ, rng=np.random.default_rng(123))
        p.set_quantification(2000, miscleavage_rate=rate, miscleavage_model="bernoulli")
        n = len(p.digestion_sites)
        frac = np.mean([1 - len(c) / n for c in p.digestion_map])
        assert abs(frac - rate) < 0.03


def test_bernoulli_is_reproducible_with_same_seed():
    def run():
        p = Protein(ALL_MISSABLE_SEQ, rng=np.random.default_rng(9))
        p.set_quantification(100, miscleavage_rate=0.4, miscleavage_model="bernoulli")
        return p.digestion_map

    assert run() == run()


# --------------------------------------------------------------------------- #
# Digestion granularity: per_copy (default) vs per_subject (reference model)
# --------------------------------------------------------------------------- #
def test_unknown_digestion_raises(rng):
    p = Protein("AKAR", rng=rng)
    with pytest.raises(ValueError, match="unknown digestion"):
        p.set_quantification(5, digestion="per_molecule")


def test_per_subject_ground_truth_is_a_perfect_digest():
    # per_subject keeps a *perfect* tryptic digest as the ground truth; missed cleavages are realized
    # later, per subject. So every copy is fully cleaved in the Protein itself.
    p = Protein("AAKAAKAAKAAR", rng=np.random.default_rng(0))
    p.set_quantification(20, miscleavage_rate=0.5, digestion="per_subject")
    assert all(copy == p.digestion_sites for copy in p.digestion_map)


def test_per_subject_subjects_get_different_digestions():
    p = Protein("AAKAAKAAKAAR", rng=np.random.default_rng(0))
    p.set_quantification(30, miscleavage_rate=0.5, digestion="per_subject")
    perfect = frozenset((pep.start_index, pep.end_index) for pep in p.peptides)
    subject_spans = [
        frozenset((pep.start_index, pep.end_index) for pep in p.peptides_for_subject(np.random.default_rng(i)))
        for i in range(8)
    ]
    assert len(set(subject_spans)) > 1  # subjects merge different boundaries
    assert any(spans != perfect for spans in subject_spans)  # missed cleavages do merge peptides


def test_per_subject_rate_zero_returns_the_perfect_peptides():
    p = Protein("AAKAAKAAR", rng=np.random.default_rng(0))
    p.set_quantification(20, miscleavage_rate=0.0, digestion="per_subject")
    perfect = {(pep.start_index, pep.end_index) for pep in p.peptides}
    subject = {(pep.start_index, pep.end_index) for pep in p.peptides_for_subject(np.random.default_rng(1))}
    assert subject == perfect


def test_digestion_mode_leaves_occupancy_truth_unchanged():
    # Occupancy (m_r ~ U(1,M)) is drawn before digestion, so the per-site truth is identical across
    # digestion modes for the same seed — digestion granularity changes estimation, not ground truth.
    seq = ProteinGenerator(rng=np.random.default_rng(0)).generate_sequence(120)

    def occupancy(digestion):
        p = Protein(seq, rng=np.random.default_rng(1))
        p.set_quantification(200, miscleavage_rate=0.5, digestion=digestion)
        return p.true_site_abundances()

    assert occupancy("per_copy") == occupancy("per_subject")


# --------------------------------------------------------------------------- #
# Peptide quantification (Protein.peptides)
# --------------------------------------------------------------------------- #
def test_peptides_exact_for_unmodified_sequence(rng):
    # No serines, abundance 1, perfect digest -> one copy, deterministic span split.
    p = Protein("AAKAARAA", rng=rng)
    p.set_quantification(1, miscleavage_rate=0.0)
    # (start, end, mod_sites): abundance
    assert peptide_summary(p) == {
        (0, 2, ()): 1,  # AAK
        (3, 5, ()): 1,  # AAR
        (6, 7, ()): 1,  # AA
    }
    assert all(isinstance(pep, Peptide) for pep in p.peptides)
    assert [pep.sequence for pep in p.peptides] == ["AAK", "AAR", "AA"]


def test_peptide_indices_and_sequence_are_consistent(rng):
    # sequence must equal the protein slice [start_index : end_index + 1] for every peptide.
    gen = ProteinGenerator(rng=rng)
    p = gen.generate_protein(60)
    p.set_quantification(20, miscleavage_rate=0.3)
    for pep in p.peptides:
        assert pep.sequence == p.sequence[pep.start_index : pep.end_index + 1]
        # mod_sites are absolute serine positions inside the peptide span, all modified.
        assert all(pep.start_index <= s <= pep.end_index for s in pep.mod_sites)
        assert all(p.sequence[s] == "S" for s in pep.mod_sites)


def test_peptides_conserve_total_residues(rng):
    # Mass balance: total residues across all (peptide length x abundance) equals
    # abundance * sequence length, since every copy is partitioned without overlap or gap.
    gen = ProteinGenerator(rng=rng)
    p = gen.generate_protein(60)
    p.set_quantification(30, miscleavage_rate=0.25)
    total_residues = sum(len(pep.sequence) * pep.abundance for pep in p.peptides)
    assert total_residues == 30 * len(p.sequence)


def test_no_empty_peptides_when_sequence_ends_in_cut_site(rng):
    # "AAKAAR" ends in R (a cut site); the C-terminal split must not yield an empty peptide.
    p = Protein("AAKAAR", rng=rng)
    p.set_quantification(20, miscleavage_rate=0.0)
    assert all(pep.sequence != "" for pep in p.peptides)
    assert peptide_summary(p) == {(0, 2, ()): 20, (3, 5, ()): 20}  # AAK, AAR


def test_no_site_protein_quantifies_full_sequences(rng):
    p = Protein("SAAGAS", rng=rng)
    p.set_quantification(10, miscleavage_rate=0.0)
    # Every copy is a single full-length peptide spanning the whole protein.
    assert sum(pep.abundance for pep in p.peptides) == 10
    assert all(pep.start_index == 0 and pep.end_index == len(p.sequence) - 1 for pep in p.peptides)


def test_modifications_recorded_in_mod_sites(rng):
    p = Protein("SK", rng=rng)
    p.set_quantification(100, miscleavage_rate=0.0)
    # Serine at index 0 -> some copies modified (mod_sites == [0]), others not ([]).
    assert all(pep.sequence == "SK" for pep in p.peptides)
    mod_states = {tuple(pep.mod_sites) for pep in p.peptides}
    assert (0,) in mod_states  # at least one modified species exists
    assert mod_states <= {(), (0,)}
    assert sum(pep.abundance for pep in p.peptides) == 100


def test_repeat_digest_resets_peptides(rng):
    p = Protein("AAKAAR", rng=rng)
    p.set_quantification(20, miscleavage_rate=0.2)
    first = sum(pep.abundance for pep in p.peptides)
    p.digest(miscleavage_rate=0.2)
    second = sum(pep.abundance for pep in p.peptides)
    assert first == second == 40  # 2 peptides x 20 copies; must not accumulate across digests


# --------------------------------------------------------------------------- #
# Group occupancy / ground-truth fold change
# --------------------------------------------------------------------------- #
def test_true_site_abundances_match_mod_table(rng):
    p = Protein("SKSAS", rng=rng)
    p.set_quantification(100)
    occ = p.true_site_abundances()
    assert set(occ) == set(p.serine_map)
    for r in p.serine_map:
        assert occ[r] == int(p.mod_table[:, r].sum())
        assert 1 <= occ[r] <= 100  # m_r ~ U(1, M)


def test_make_group_proteins_share_sequence_independent_occupancy():
    sequence = ProteinGenerator(rng=np.random.default_rng(0)).generate_sequence(120)
    groups = make_group_proteins(sequence, n_groups=2, abundance=200, rng=np.random.default_rng(1))
    assert [g.sequence for g in groups] == [sequence, sequence]
    # Independent occupancy draws -> at least one site differs between groups.
    occ0, occ1 = groups[0].true_site_abundances(), groups[1].true_site_abundances()
    assert any(occ0[r] != occ1[r] for r in occ0)


def test_true_site_log2_fold_change_matches_occupancy_ratio():
    sequence = ProteinGenerator(rng=np.random.default_rng(2)).generate_sequence(120)
    a, b = make_group_proteins(sequence, 2, 200, rng=np.random.default_rng(3))
    fc = true_site_log2_fold_change(a, b)
    occ_a, occ_b = a.true_site_abundances(), b.true_site_abundances()
    for r in occ_a:
        assert fc[r] == pytest.approx(np.log2(occ_b[r] / occ_a[r]))


# --------------------------------------------------------------------------- #
# Stoichiometry truth (s_r = m_r / M)
# --------------------------------------------------------------------------- #
def test_true_site_stoichiometry_is_m_over_M(rng):
    p = Protein("SKSAS", rng=rng)
    p.set_quantification(100)
    s = p.true_site_stoichiometry()
    occ = p.true_site_abundances()
    assert set(s) == set(p.serine_map)
    for r in p.serine_map:
        assert s[r] == pytest.approx(occ[r] / 100)
        assert 0 < s[r] <= 1


def test_true_site_stoichiometry_change_fraction_coincides_with_count_fc():
    # Shared total abundance M -> log2(s_b/s_a) collapses to the raw-count FC log2(m_b/m_a).
    sequence = ProteinGenerator(rng=np.random.default_rng(2)).generate_sequence(120)
    a, b = make_group_proteins(sequence, 2, 200, rng=np.random.default_rng(3))
    frac_change = true_site_stoichiometry_change(a, b, "fraction")
    count_fc = true_site_log2_fold_change(a, b)
    for r in frac_change:
        assert frac_change[r] == pytest.approx(count_fc[r])


def test_true_site_stoichiometry_change_logit_matches_logit_difference():
    sequence = ProteinGenerator(rng=np.random.default_rng(2)).generate_sequence(120)
    a, b = make_group_proteins(sequence, 2, 200, rng=np.random.default_rng(3))
    logit_change = true_site_stoichiometry_change(a, b, "logit")
    s_a, s_b = a.true_site_stoichiometry(), b.true_site_stoichiometry()
    for r in logit_change:
        assert logit_change[r] == pytest.approx(logit2(s_b[r]) - logit2(s_a[r]))


def test_true_site_stoichiometry_change_rejects_unknown_space():
    sequence = ProteinGenerator(rng=np.random.default_rng(2)).generate_sequence(60)
    a, b = make_group_proteins(sequence, 2, 100, rng=np.random.default_rng(3))
    with pytest.raises(ValueError, match="unknown stoichiometry space"):
        true_site_stoichiometry_change(a, b, "odds")
