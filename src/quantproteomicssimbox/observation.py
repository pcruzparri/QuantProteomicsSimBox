"""Observation layer: derive observed Samples from a ground-truth Protein.

Turns a Protein's true peptide copy numbers into the data the pipeline sees — observation
model (Eqs. 2-5), optional position-agnostic collapse, missingness — one Sample per subject.
Never mutates the Protein.

Scaffold: noise model and missingness are stubbed; only aggregate_peptides is implemented.
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


def aggregate_peptides(peptides: list[Peptide], position_aware: bool = False) -> list[Peptide]:
    """Aggregate peptides into distinct species, summing abundances. Returns new Peptides.

    Key = (sequence, relative-mod signature), so the same sequence + mod pattern aggregates
    regardless of locus. position_aware=True also keys on start (ground-truth view); the
    default merges identical species across loci (bottom-up MS can't tell them apart), keeping
    the first occurrence's position fields.
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


class ObservationModel:
    """Generates observed Samples from a ground-truth Protein.

    Parameters mirror the paper's observation model. The random draws are not implemented
    yet — this is the structural scaffold for the future simulation layer.
    """

    def __init__(
        self,
        sigma_subject: float = 0.0,
        sigma_site: float = 0.0,
        position_aware: bool = False,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.sigma_subject = sigma_subject  # beta_k ~ N(0, .): per-subject effect (Eq. 4)
        self.sigma_site = sigma_site  # alpha_r ~ N(0, .): per-site effect (Eq. 5)
        self.position_aware = position_aware  # collapse indistinguishable peptides in a Sample?
        self.rng = rng if rng is not None else np.random.default_rng()

    def sample(self, protein: Protein, group: int, subject: int) -> Sample:
        """One observed Sample for (group, subject): apply Eqs. 2-5, aggregate per
        position_aware, apply missingness. Not implemented yet.
        """
        raise NotImplementedError("Observation model not yet implemented")

    def sample_group(self, protein: Protein, group: int, n_subjects: int) -> list[Sample]:
        """Produce observed Samples for every subject in a group."""
        return [self.sample(protein, group, subject) for subject in range(n_subjects)]

    def apply_missingness(self, samples: list[Sample], rate: float) -> list[Sample]:
        """Remove observations to reach a global missingness target (Bramer et al.). Not implemented yet."""
        raise NotImplementedError("Missingness not yet implemented")
