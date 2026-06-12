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

# Digestion granularity — at what level missed cleavages are realized:
#   "per_copy"    - each proteoform copy is digested independently (closer to physical ground truth);
#                   `Protein.peptides` is the shared per-copy digest, observed identically by every subject.
#   "per_subject" - the paper/reference model: the ground truth is a *perfect* tryptic digest, and each
#                   subject gets its own missed-cleavage realization at observation time (one digestion
#                   per sample, merging adjacent tryptic peptides) via `peptides_for_subject`.
DIGESTION_MODES = ("per_copy", "per_subject")


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
        self.miscleavage_rate: float = 0.0  # remembered for per-subject re-digestion
        self.digestion: str = "per_copy"

    def set_quantification(
        self,
        abundance: int = 1,
        miscleavage_rate: float = 0.0,
        miscleavage_model: str = "global",
        digestion: str = "per_copy",
    ) -> None:
        if digestion not in DIGESTION_MODES:
            raise ValueError(f"unknown digestion {digestion!r}; choose from {list(DIGESTION_MODES)}")
        self.abundance = abundance
        self.miscleavage_rate = miscleavage_rate
        self.digestion = digestion

        # Occupancy model (paper Eq. 1): for each serine, modify m_r ~ U(1, M) randomly chosen
        # copies. Uniform over [1, M], unlike a per-copy 50/50 coin (which pins occupancy at M/2).
        table = np.zeros((abundance, len(self.sequence)))
        for i in self.serine_map:
            m_r = int(self.rng.integers(1, abundance + 1))  # U(1, M), inclusive
            modified = self.rng.choice(abundance, size=m_r, replace=False)
            table[modified, i] = 1
        self.mod_table = table
        # per_subject: the ground truth is a *perfect* tryptic digest; each subject re-digests later via
        # peptides_for_subject. per_copy: realize the missed cleavages now, shared by all subjects.
        self.digest(0.0 if digestion == "per_subject" else miscleavage_rate, miscleavage_model)

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

        self.peptides = self._peptides_from_digestion_map(self.digestion_map)

    def _peptides_from_digestion_map(self, digestion_map: list[list[int]]) -> list[Peptide]:
        """Split each copy at its kept cuts and aggregate identical species (same span + modified
        sites) into Peptides whose abundance counts the copies that produced them.

        `digestion_map[m]` is copy m's kept (cleaved) sites. Passing each copy its own cut list gives
        the per-copy digest; passing the *same* cut list for every copy realizes one shared digestion
        (the per-subject case) while still counting modification patterns over the copies — this is the
        reference's ``get_peptide_counts``.
        """
        peptides_by_id: dict[tuple[int, int, tuple[int, ...]], Peptide] = {}
        for form in range(self.abundance):
            mod_row = self.mod_table[form]
            # Peptide spans [start, end] inclusive between kept cuts.
            bounds = []
            start = 0
            for site in digestion_map[form]:
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
        return list(peptides_by_id.values())

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

    def peptides_for_subject(self, rng: np.random.Generator) -> list[Peptide]:
        """The peptide species observed for one subject.

        ``per_copy``: the shared per-copy digest (`self.peptides`), identical for every subject.
        ``per_subject``: a fresh missed-cleavage realization for this subject — the perfect tryptic
        peptides minus a set of merged boundaries — counted over the copies. Drawn from the
        observation `rng`, so each subject gets its own digestion (the reference's per-sample model).
        """
        if self.digestion != "per_subject":
            return self.peptides
        subject_cuts = self._merge_boundaries(rng)
        return self._peptides_from_digestion_map([subject_cuts] * self.abundance)

    def _merge_boundaries(self, rng: np.random.Generator) -> list[int]:
        """Per-subject missed cleavages (reference ``imperfect_digest``): from the perfect tryptic cut
        sites, drop ``round(rate · n_boundaries)`` of them — merging those adjacent peptides — sampled
        without replacement weighted by ``1 / (preceding tryptic peptide length)`` (shorter peptides
        merge more). Returns the kept cut sites for this subject.
        """
        sites = self.digestion_sites
        if not sites or self.miscleavage_rate <= 0:
            return list(sites)
        # A cut is a mergeable boundary if a peptide follows it (exclude a C-terminal-residue cut). Its
        # weight is 1 / length of the peptide ending at that cut (prev cut .. this cut).
        boundaries: list[int] = []
        weights: list[float] = []
        prev = -1
        for k, s in enumerate(sites):
            nxt = sites[k + 1] if k + 1 < len(sites) else len(self.sequence) - 1
            if nxt - s > 0:  # a peptide follows -> this boundary can be merged
                boundaries.append(k)
                weights.append(1.0 / (s - prev))
            prev = s
        n_merge = min(max(int(round(self.miscleavage_rate * len(boundaries))), 0), len(boundaries))
        if n_merge == 0:
            return list(sites)
        w = np.asarray(weights)
        w /= w.sum()
        merged = set(rng.choice(boundaries, size=n_merge, replace=False, p=w).tolist())
        return [s for k, s in enumerate(sites) if k not in merged]

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
