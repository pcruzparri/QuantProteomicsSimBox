"""The Protein ground-truth generator: per-site serine occupancy + trypsin digestion into Peptides."""

import numpy as np

from .peptide import Peptide

# Miscleavage sampling models for digest() — how missed cleavages are drawn per proteoform copy:
#   "global"    - miss a fixed round(rate * n_sites) cuts per copy (the paper's realized-proportion
#                 model), sampled without replacement weighted toward shorter flanks. The missed
#                 *count* is identical across copies -> fixed-length digestion map.
#   "bernoulli" - miss each cut independently with probability `rate`; the count per copy is
#                 Binomial(n_missable, rate) -> variable-length digestion map.
MISCLEAVAGE_MODELS = ("global", "bernoulli")


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
