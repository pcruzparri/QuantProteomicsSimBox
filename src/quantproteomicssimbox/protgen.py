import numpy as np

from .utils import amino_acids, logit2

# Sorted for a stable order -> reproducible seeded sampling (set iteration order isn't stable).
AMINO_ACIDS = tuple(sorted(amino_acids))

# Miscleavage sampling models for digest() — how missed cleavages are drawn per proteoform copy:
#   "global"    - miss a fixed round(rate * n_sites) cuts per copy (the paper's realized-proportion
#                 model), sampled without replacement weighted toward shorter flanks. The missed
#                 *count* is identical across copies -> fixed-length digestion map.
#   "bernoulli" - miss each cut independently with probability `rate`; the count per copy is
#                 Binomial(n_missable, rate) -> variable-length digestion map.
MISCLEAVAGE_MODELS = ("global", "bernoulli")

class Peptide:
    """A distinct peptide species from digesting a Protein.

    `abundance` is the ground-truth copy count, or the simulated intensity in a Sample
    (observation.py). Indices are absolute/inclusive; `mod_sites` are absolute modified serines.
    """

    def __init__(
        self,
        sequence: str,
        abundance: float = 0,
        start_index: int | None = None,
        end_index: int | None = None,
        mod_sites: list[int] | None = None,
    ) -> None:
        self.sequence = sequence
        self.abundance = abundance
        self.start_index = start_index
        self.end_index = end_index
        self.mod_sites: list[int] = mod_sites if mod_sites is not None else []

class Protein:
    def __init__(self, sequence: str, rng: np.random.Generator | None = None) -> None:
        self.sequence = sequence
        self.abundance: int | None = None
        self.mod_table: np.ndarray | None = None
        self.serine_map: list[int] = [i for i, aa in enumerate(sequence) if aa == "S"]
        self.rng = rng if rng is not None else np.random.default_rng()
        self.digestion_sites: list[int] = []
        self.digestion_map: list[list[int]] = []
        self.peptides: list[Peptide] = []

    def digest(self, miscleavage_rate: float = 0.0, miscleavage_model: str = "global") -> None:
        if self.abundance is None or self.mod_table is None:
            raise ValueError("Abundance and modification table must be set before digestion by using the set_quantification() method.")
        if miscleavage_model not in MISCLEAVAGE_MODELS:
            raise ValueError(f"unknown miscleavage_model {miscleavage_model!r}; choose from {list(MISCLEAVAGE_MODELS)}")

        # Trypsin cleaves after K or R, except when the following residue is P.
        self.digestion_sites = [
            i
            for i, aa in enumerate(self.sequence)
            if aa in ("K", "R") and (i + 1 >= len(self.sequence) or self.sequence[i + 1] != "P")
        ]

        sites = np.asarray(self.digestion_sites, dtype=int)
        n_sites = sites.size
        if n_sites == 0:
            self.digestion_map = [[] for _ in range(self.abundance)]

        else:
            # Flanking-peptide geometry, shared by both models: each site's two adjacent peptides are
            # bounded by the previous/next cut (termini stand in at -1 / len-1). A zero-length terminal
            # flank can never merge, so such cuts are never missable.
            prev_cut = np.empty(n_sites, dtype=int)
            prev_cut[0] = -1
            prev_cut[1:] = sites[:-1]
            next_cut = np.empty(n_sites, dtype=int)
            next_cut[-1] = len(self.sequence) - 1
            next_cut[:-1] = sites[1:]
            min_len = np.minimum(sites - prev_cut, next_cut - sites)
            missable = min_len > 0  # exclude zero-length (terminal) flanks: no merge, avoids /0

            # Fork: how many / which cuts each copy misses. digestion_map[m] is the kept (cleaved)
            # complement either way; the models differ in whether the missed count is fixed or random.
            if miscleavage_model == "global":
                self.digestion_map = self._digest_global(sites, min_len, missable, miscleavage_rate)
            else:  # "bernoulli"
                self.digestion_map = self._digest_bernoulli(sites, missable, miscleavage_rate)

        # Split each copy at its kept cuts; aggregate identical species (same span + modified sites)
        # into Peptides whose abundance counts the copies that produced them.
        peptides_by_id: dict[tuple[int, int, tuple[int, ...]], Peptide] = {}
        for form in range(self.abundance):
            mod_row = self.mod_table[form]
            # Peptide spans [start, end] inclusive between kept cuts.
            bounds = []
            start = 0
            for site in self.digestion_map[form]:
                bounds.append((start, site))
                start = site + 1
            if start < len(self.sequence):  # trailing peptide, unless the last cut is the C-terminus
                bounds.append((start, len(self.sequence) - 1))

            for start_index, end_index in bounds:
                mod_sites = [i for i in self.serine_map if start_index <= i <= end_index and mod_row[i]]
                key = (start_index, end_index, tuple(mod_sites))
                peptide = peptides_by_id.get(key)
                if peptide is None:
                    peptide = Peptide(
                        self.sequence[start_index : end_index + 1],
                        abundance=0,
                        start_index=start_index,
                        end_index=end_index,
                        mod_sites=mod_sites,
                    )
                    peptides_by_id[key] = peptide
                peptide.abundance += 1
        self.peptides = list(peptides_by_id.values())

    def _digest_global(
        self, sites: np.ndarray, min_len: np.ndarray, missable: np.ndarray, rate: float
    ) -> list[list[int]]:
        """Global realized-proportion model (paper): every copy misses exactly round(rate * n_sites)
        cuts, sampled without replacement weighted by 1 / shorter-flank length (shorter flanks merge
        more). The missed *count* is identical across copies -> fixed-length digestion map; copies
        differ only in *which* cuts are missed. Returns each copy's kept (cleaved) sites.
        """
        n_sites = sites.size
        weights = np.zeros(n_sites, dtype=float)
        weights[missable] = 1.0 / min_len[missable]
        total_weight = weights.sum()
        if total_weight > 0:  # 0 only when no site is missable (lone terminal cut)
            weights /= total_weight

        n_missed = int(round(rate * n_sites))
        n_missed = min(max(n_missed, 0), int(missable.sum()))
        digestion_map: list[list[int]] = []
        for _ in range(self.abundance):
            if n_missed == 0:
                digestion_map.append(sites.tolist())
                continue
            missed = self.rng.choice(n_sites, size=n_missed, replace=False, p=weights)
            keep_mask = np.ones(n_sites, dtype=bool)
            keep_mask[missed] = False
            digestion_map.append(sites[keep_mask].tolist())
        return digestion_map

    def _digest_bernoulli(
        self, sites: np.ndarray, missable: np.ndarray, rate: float
    ) -> list[list[int]]:
        """Bernoulli model: each missable cut is missed independently with probability `rate`, so the
        missed count per copy is Binomial(n_missable, rate) and the digestion map is variable-length
        across copies (the contrast to the fixed-length ``global`` model). Terminal zero-length flanks
        are never missed (no merge). Returns each copy's kept (cleaved) sites.
        """
        n_sites = sites.size
        digestion_map: list[list[int]] = []
        for _ in range(self.abundance):
            missed = missable & (self.rng.random(n_sites) < rate)
            digestion_map.append(sites[~missed].tolist())
        return digestion_map

    def set_quantification(
        self, abundance: int = 1, miscleavage_rate: float = 0.0, miscleavage_model: str = "global"
    ) -> None:
        self.abundance = abundance

        # Occupancy model (paper Eq. 1): for each serine, modify m_r ~ U(1, M) randomly chosen
        # copies. Uniform over [1, M], unlike a per-copy 50/50 coin (which pins occupancy at M/2).
        table = np.zeros((abundance, len(self.sequence)))
        for i in self.serine_map:
            m_r = int(self.rng.integers(1, abundance + 1))  # U(1, M), inclusive
            modified = self.rng.choice(abundance, size=m_r, replace=False)
            table[modified, i] = 1
        self.mod_table = table
        self.digest(miscleavage_rate, miscleavage_model)

    def true_site_abundances(self) -> dict[int, int]:
        """Ground-truth modified-copy count (occupancy m_r) per serine site."""
        if self.mod_table is None:
            raise ValueError("set_quantification() must be called before true_site_abundances()")
        return {r: int(self.mod_table[:, r].sum()) for r in self.serine_map}

    def true_site_stoichiometry(self) -> dict[int, float]:
        """Ground-truth modification stoichiometry s_r = m_r / M per serine site.

        The fraction of copies modified at the site. Exact because every ground-truth copy is
        full-length, so each copy spans (and is counted at) every serine.
        """
        occupancy = self.true_site_abundances()
        return {r: occupancy[r] / self.abundance for r in occupancy}


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
