"""Ground-truth helpers: build the per-group proteins and read the known per-site between-group change."""

import numpy as np

from ..utils import logit2
from .protein import Protein


def make_group_proteins(
    sequence: str,
    n_groups: int,
    abundance: int,
    miscleavage_rate: float = 0.0,
    miscleavage_model: str = "global",
    rng: np.random.Generator | None = None,
) -> list[Protein]:
    """One ground-truth Protein per group: same sequence, independent per-group occupancy.

    Group differences come from independent m_r ~ U(1, M) draws, so each group gets its own true
    site abundances (a known per-site log2FC). The shared sequence keeps the observation layer's
    site effects (keyed on sequence) shared across groups — a site effect is measurement bias, not a
    group difference. `miscleavage_model` selects the digest fork (see ``MISCLEAVAGE_MODELS``).
    """
    rng = rng if rng is not None else np.random.default_rng()
    proteins = []
    for _ in range(n_groups):
        protein = Protein(sequence, rng=rng)
        protein.set_quantification(abundance, miscleavage_rate, miscleavage_model)
        proteins.append(protein)
    return proteins


def true_site_log2_fold_change(protein_a: Protein, protein_b: Protein) -> dict[int, float]:
    """Known per-site log2 fold-change (group B vs A) from ground-truth occupancy."""
    if protein_a.sequence != protein_b.sequence:
        raise ValueError("Proteins must have the same sequence to compare true site log2FC")

    occ_a = protein_a.true_site_abundances()
    occ_b = protein_b.true_site_abundances()
    return {r: float(np.log2(occ_b[r] / occ_a[r])) for r in occ_a}


def true_site_stoichiometry_change(
    protein_a: Protein, protein_b: Protein, space: str = "fraction"
) -> dict[int, float]:
    """Known per-site between-group change in stoichiometry (group B vs A).

    Derived from each group's true stoichiometry ``s = m_r / M``:
      - ``space="fraction"`` -> ``log2(s_b / s_a)`` (log2 fold-change of the modified fraction)
      - ``space="logit"``    -> ``logit2(s_b) - logit2(s_a)`` (change in log-odds)

    These are the truths the corresponding stoichiometry roll-up methods estimate.
    """
    if protein_a.sequence != protein_b.sequence:
        raise ValueError("Proteins must have the same sequence to compare true site stoichiometry change")
    if space not in ("fraction", "logit"):
        raise ValueError(f"unknown stoichiometry space {space!r}; choose from ['fraction', 'logit']")

    s_a = protein_a.true_site_stoichiometry()
    s_b = protein_b.true_site_stoichiometry()
    if space == "fraction":
        # log2(s_b/s_a) collapses to the raw-count FC log2(m_b/m_a) ONLY because both groups share the
        # same total abundance M (s = m_r / M, so the M's cancel). With M_a != M_b they would diverge.
        return {r: float(np.log2(s_b[r] / s_a[r])) for r in s_a}
    return {r: float(logit2(s_b[r]) - logit2(s_a[r])) for r in s_a}
