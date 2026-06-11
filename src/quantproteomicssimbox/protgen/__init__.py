"""Ground-truth simulation: random proteins, per-site serine occupancy, trypsin digestion, and the
known per-site between-group truths.

Split into one module per concern — `peptide` (the species), `protein` (occupancy + digestion),
`generator` (sequence generation), `truth` (group construction + known change). Everything public is
re-exported here, so ``from quantproteomicssimbox.protgen import X`` keeps working.
"""

from .generator import AMINO_ACIDS, ProteinGenerator
from .peptide import Peptide
from .protein import MISCLEAVAGE_MODELS, Protein
from .truth import (
    make_group_proteins,
    true_site_log2_fold_change,
    true_site_stoichiometry_change,
)

__all__ = [
    "Peptide",
    "Protein",
    "MISCLEAVAGE_MODELS",
    "ProteinGenerator",
    "AMINO_ACIDS",
    "make_group_proteins",
    "true_site_log2_fold_change",
    "true_site_stoichiometry_change",
]
