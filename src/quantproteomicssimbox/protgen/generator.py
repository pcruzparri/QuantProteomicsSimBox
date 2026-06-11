"""ProteinGenerator: build random (optionally repeat-seeded) sequences and wrap them as Proteins."""

import numpy as np

from ..utils import amino_acids
from .protein import Protein

# Sorted for a stable order -> reproducible seeded sampling (set iteration order isn't stable).
AMINO_ACIDS = tuple(sorted(amino_acids))


class ProteinGenerator:
    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()

    def generate_sequence(self, length: int, repeat_units: int = 0, unit_length: int = 8) -> str:
        """Random amino-acid sequence of the given length.

        With ``repeat_units > 0``, embed that many identical copies of one shared peptide ``unit`` — a
        clean tryptic peptide (no internal K/R, contains a serine, ends in K), each flanked by cut sites
        so it digests as the **same** peptide species at distinct loci. This forces a repeated peptide,
        used to probe robustness when bottom-up MS cannot resolve which locus a shared peptide came from
        (the position-agnostic ambiguity).
        """
        if repeat_units <= 0:
            return "".join(self.rng.choice(AMINO_ACIDS, size=length))
        if unit_length < 2:
            raise ValueError("unit_length must be >= 2 (body + terminal K)")
        if repeat_units * unit_length >= length:
            raise ValueError("repeated units do not fit in the requested length")

        # Body residues avoid K/R (no internal cut) and P (no K-before-P next to a boundary).
        non_cut = tuple(a for a in AMINO_ACIDS if a not in ("K", "R", "P"))
        body = list(self.rng.choice(non_cut, size=unit_length - 1))
        if "S" not in body:
            body[int(self.rng.integers(unit_length - 1))] = "S"  # guarantee a modifiable site
        unit = "".join(body) + "K"  # one clean tryptic peptide

        # Split the remaining residues into (repeat_units + 1) fillers; each non-final filler ends in K
        # so the next unit starts a fresh peptide (an empty filler relies on the prior unit's own K).
        remaining = length - unit_length * repeat_units
        n_fill = repeat_units + 1
        sizes = [remaining // n_fill + (1 if i < remaining % n_fill else 0) for i in range(n_fill)]
        parts: list[str] = []
        for i in range(repeat_units):
            parts.append(self._filler(sizes[i], non_cut, terminal=False))
            parts.append(unit)
        parts.append(self._filler(sizes[-1], non_cut, terminal=True))
        return "".join(parts)

    def _filler(self, n: int, alphabet: tuple[str, ...], terminal: bool) -> str:
        """Random filler of length `n`; a non-terminal filler ends in K to cut before the next unit."""
        if n <= 0:
            return ""
        chars = list(self.rng.choice(alphabet, size=n))
        if not terminal:
            chars[-1] = "K"
        return "".join(chars)

    def generate_protein(self, length: int, repeat_units: int = 0, unit_length: int = 8) -> Protein:
        return Protein(self.generate_sequence(length, repeat_units, unit_length), rng=self.rng)
