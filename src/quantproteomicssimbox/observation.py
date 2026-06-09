"""Observation layer: derive observed Samples from a ground-truth Protein.

`protgen.Protein` produces *ground truth*: exact, position-aware peptide species with
integer copy-number abundances. This layer derives the *observed* data the analysis
pipeline actually sees. For each experimental group and subject it (1) applies the
paper's observation model — subject and site random effects, Eqs. 2-5 — to turn true
copy numbers into measured intensities, (2) optionally collapses peptides that mass
spectrometry cannot distinguish (position-agnostic aggregation), and (3) applies
missingness. Each subject yields one `Sample`.

Nothing here mutates the input Protein; the ground truth stays pristine so it can be
observed under many conditions and compared against at scoring time.

NOTE: the noise model and missingness are scaffolding only (signatures + docstrings);
`aggregate_peptides` is implemented because it is deterministic and shared.
"""

from dataclasses import dataclass, field

import numpy as np

from .protgen import Peptide, Protein


@dataclass
class Sample:
    """One observed proteoform sample — a single subject within a single group.

    A lightweight view of a Protein's peptides carrying observed/simulated abundances
    rather than the integer ground-truth counts. It deliberately does not carry the
    Protein's digestion machinery (mod_table, digestion_map, RNG); `protein_sequence`
    references the parent protein's identity for mapping back to ground truth.
    """

    protein_sequence: str
    group: int
    subject: int
    peptides: list[Peptide] = field(default_factory=list)


def aggregate_peptides(peptides: list[Peptide], position_aware: bool = False) -> list[Peptide]:
    """Aggregate peptides into distinct observed species, summing their abundances.

    The modification signature in the key is peptide-*relative* (offsets from the peptide
    start), so the same sequence carrying the same modification pattern aggregates the same
    way regardless of where it sits in the protein.

    - ``position_aware=False`` (default): key on ``(sequence, relative mod signature)`` only.
      Identical species from different protein loci merge, mirroring the fact that bottom-up
      MS cannot distinguish them. A merged species keeps the *first occurrence's* position
      fields as a representative (lossy only for short, repeated peptides) and sums abundance.
    - ``position_aware=True``: also key on the start position, so peptides from different loci
      stay separate — the exact ground-truth view.

    Returns new ``Peptide`` objects; the inputs are not modified.
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
        # beta_k ~ N(0, sigma_subject): per-subject random effect (Eq. 4).
        self.sigma_subject = sigma_subject
        # alpha_r ~ N(0, sigma_site): per-site random effect (Eq. 5).
        self.sigma_site = sigma_site
        # How to aggregate indistinguishable peptides when building a Sample.
        self.position_aware = position_aware
        self.rng = rng if rng is not None else np.random.default_rng()

    def sample(self, protein: Protein, group: int, subject: int) -> Sample:
        """Produce one observed Sample for ``(group, subject)`` from the ground-truth protein.

        Will apply the observation model (Eqs. 2-5) to each peptide's true abundance, aggregate
        per ``self.position_aware``, and apply missingness. Not implemented yet.
        """
        raise NotImplementedError("Observation model not yet implemented")

    def sample_group(self, protein: Protein, group: int, n_subjects: int) -> list[Sample]:
        """Produce observed Samples for every subject in a group."""
        return [self.sample(protein, group, subject) for subject in range(n_subjects)]

    def apply_missingness(self, samples: list[Sample], rate: float) -> list[Sample]:
        """Remove observations to reach a global missingness target (Bramer et al.). Not implemented yet."""
        raise NotImplementedError("Missingness not yet implemented")
