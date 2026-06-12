"""Observation layer: derive observed Samples from a ground-truth Protein.

Turns a Protein's true peptide copy numbers into the data the pipeline sees — observation
model (Eqs. 2-5), optional position-agnostic collapse, missingness — one Sample per subject.
Never mutates the Protein.

Implements the observation model and abundance-dependent missingness (apply_missingness, label-free
variant); the TMT-plex (block) missingness variant is a future refinement.
"""

from dataclasses import dataclass, field

import numpy as np

from .protgen import Peptide, Protein


@dataclass
class Sample:
    """One observed subject's sample: a Protein's peptides with simulated abundances, plus
    group/subject metadata. Lightweight (no digestion machinery); `protein_sequence` links
    back to the parent for scoring.
    """

    protein_sequence: str
    group: int
    subject: int
    peptides: list[Peptide] = field(default_factory=list)


class ObservationModel:
    """Generates observed Samples from a ground-truth Protein.

    Implements the paper's observation model (Eqs. 2-5): a per-subject effect
    beta_ik ~ N(0, var_subject) and a per-site effect alpha_r ~ N(0, var_site), combined as a
    2 ** (beta_ik + sum_{r in S} alpha_r) factor on each peptide's abundance. An optional extension
    adds a per-peptide-species (backbone) ionization efficiency gamma_p ~ N(0, var_species), a fixed
    factor shared by all mod-forms of a sequence — it cancels inside a span's modified fraction but
    drives between-span abundance differences (so it is what the per-peptide stoichiometry aggregations
    target, and what abundance-dependent missingness keys on). var_* are the Normal *variances* (the
    paper's swept "variance levels"), so the std-dev passed to the draw is their square root. Draws are
    cached so each subject, site and species is sampled once and reused.
    """

    def __init__(
        self,
        var_subject: float = 0.0,
        var_site: float = 0.0,
        var_species: float = 0.0,
        detection_limit: int = 1,
        position_aware: bool = False,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.var_subject = var_subject  # beta_ik ~ N(0, var_subject): per-subject variance (Eq. 4)
        self.var_site = var_site  # alpha_r ~ N(0, var_site): per-site variance (Eq. 5)
        # gamma_p ~ N(0, var_species): per-peptide-species (backbone) log2 ionization efficiency — a
        # fixed physicochemical factor shared by every mod-form of a sequence, so it cancels inside a
        # span's modified fraction but not between spans (the effect per-peptide aggregation targets).
        self.var_species = var_species
        # Limit of detection (proxy): a peptide species must arise from >= detection_limit proteoform
        # copies to be observed. Default 1 keeps every species; raising it prunes the long tail of rare
        # miscleavage forms (singletons dominate at high miscleavage) that real bottom-up MS would not
        # identify — which is what inflates the log2-`sum` aggregator beyond the paper's magnitude.
        self.detection_limit = detection_limit
        self.position_aware = position_aware  # collapse indistinguishable peptides in a Sample?
        self.rng = rng if rng is not None else np.random.default_rng()
        # beta_ik (Eq. 4): one draw per (group, subject), reused across all that subject's peptides and proteins.
        self.subject_effects: dict[tuple[int, int], float] = {}
        # alpha_r (Eq. 5): one draw per (protein sequence, absolute site), reused across all subjects and groups.
        self.site_effects: dict[tuple[str, int], float] = {}
        # gamma_p: one draw per peptide *sequence* (backbone), reused across subjects, groups and loci.
        self.species_effects: dict[str, float] = {}

    def sample(self, protein: Protein, group: int, subject: int) -> Sample:
        """One observed Sample for (group, subject).

        Applies the observation model (Eqs. 2-5) to each *true* peptide species — scaling its
        abundance by ``2 ** (beta_ik + sum_{r in S} alpha_r)`` — then collapses indistinguishable
        species via ``aggregate_peptides`` (summing observed intensities). Effects are applied
        before aggregation so each species uses its own absolute sites. Never mutates the Protein;
        missingness is applied separately (not yet implemented).
        """
        subject_key = (group, subject)
        if subject_key not in self.subject_effects:
            self.subject_effects[subject_key] = self.rng.normal(0, np.sqrt(self.var_subject))  # beta_ik (Eq. 4)
        beta = self.subject_effects[subject_key]

        observed: list[Peptide] = []
        for pep in protein.peptides_for_subject(self.rng):  # per-copy: shared set; per-subject: re-digest
            if pep.abundance < self.detection_limit:
                continue  # below the limit of detection: too few copies produce this peptide species
            exponent = beta  # log2-space shift = beta_ik + sum_{r in S} alpha_r (+ gamma_p) (Eq. 3)
            for mod_site in pep.mod_sites:
                site_key = (protein.sequence, mod_site)  # alpha is a property of the site, not the peptide form
                if site_key not in self.site_effects:
                    self.site_effects[site_key] = self.rng.normal(0, np.sqrt(self.var_site))  # alpha_r (Eq. 5)
                exponent += self.site_effects[site_key]
            if self.var_species:
                # gamma_p: per-sequence ionization efficiency, shared by all mod-forms of this backbone.
                if pep.sequence not in self.species_effects:
                    self.species_effects[pep.sequence] = self.rng.normal(0, np.sqrt(self.var_species))
                exponent += self.species_effects[pep.sequence]
            observed.append(
                Peptide(
                    pep.sequence,
                    abundance=pep.abundance * 2**exponent,
                    start_index=pep.start_index,
                    end_index=pep.end_index,
                    mod_sites=list(pep.mod_sites),
                )
            )

        return Sample(
            protein_sequence=protein.sequence,
            group=group,
            subject=subject,
            peptides=aggregate_peptides(observed, self.position_aware),
        )

    def sample_group(self, protein: Protein, group: int, n_subjects: int) -> list[Sample]:
        """Produce observed Samples for every subject in a group."""
        return [self.sample(protein, group, subject) for subject in range(n_subjects)]

    def apply_missingness(self, samples: list[Sample], rate: float) -> list[Sample]:
        """Drop observations to a global missingness target (Bramer et al., label-free variant).

        Missingness is abundance-dependent (MNAR): each (sample, peptide) observation is dropped with
        probability ``proportional to 1 / abundance``, so low-abundance peptides go missing more
        often. Exactly ``round(rate * total_observations)`` observations are removed, sampled without
        replacement by that weight. Returns new `Sample`s (inputs untouched); a dropped peptide is
        simply absent from its sample (→ NaN in the roll-up matrix). The TMT-plex (labeled, block)
        variant is a future refinement.
        """
        if rate <= 0:
            return samples
        observations = [
            (si, pi, pep.abundance)
            for si, s in enumerate(samples)
            for pi, pep in enumerate(s.peptides)
        ]
        if not observations:
            return samples
        n_drop = min(int(round(rate * len(observations))), len(observations))
        if n_drop == 0:
            return samples

        weights = np.array([1.0 / abundance for *_, abundance in observations])
        weights /= weights.sum()
        dropped = self.rng.choice(len(observations), size=n_drop, replace=False, p=weights)
        drop_set = {(observations[i][0], observations[i][1]) for i in dropped}

        return [
            Sample(
                protein_sequence=s.protein_sequence,
                group=s.group,
                subject=s.subject,
                peptides=[pep for pi, pep in enumerate(s.peptides) if (si, pi) not in drop_set],
            )
            for si, s in enumerate(samples)
        ]


def aggregate_peptides(peptides: list[Peptide], position_aware: bool = False) -> list[Peptide]:
    """Aggregate peptides into distinct species, summing abundances. Returns new Peptides.

    Key = (sequence, relative-mod signature), so the same sequence + mod pattern aggregates
    regardless of locus. position_aware=True also keys on start (ground-truth view); the
    default merges identical species across loci (bottom-up MS can't tell them apart), keeping
    the first occurrence's position fields. Used by ``ObservationModel.sample`` to collapse the
    indistinguishable peptides in a Sample.
    """
    aggregated: dict[tuple, Peptide] = {}
    for pep in peptides:
        if pep.start_index is None:
            rel_mods = tuple(pep.mod_sites)
        else:
            rel_mods = tuple(s - pep.start_index for s in pep.mod_sites)
        key: tuple = (pep.sequence, rel_mods)
        if position_aware:
            key = key + (pep.start_index,)

        existing = aggregated.get(key)
        if existing is None:
            aggregated[key] = Peptide(
                pep.sequence,
                abundance=pep.abundance,
                start_index=pep.start_index,
                end_index=pep.end_index,
                mod_sites=list(pep.mod_sites),
            )
        else:
            existing.abundance += pep.abundance
    return list(aggregated.values())
