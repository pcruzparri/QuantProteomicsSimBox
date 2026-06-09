import numpy as np

from .utils import amino_acids

# Stable, ordered alphabet so that sequence sampling is reproducible under a seeded RNG
# (iteration order of the `amino_acids` set is not guaranteed stable across processes).
AMINO_ACIDS = tuple(sorted(amino_acids))

class Peptide:
    """A distinct peptide species produced by digesting a Protein.

    `abundance` is the species' copy count in a ground-truth Protein, or its simulated
    intensity in an observed Sample (see observation.py) — the object's role determines which.
    `start_index`/`end_index` are absolute, inclusive protein coordinates; `mod_sites` are the
    absolute serine positions modified in this species.
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

    def digest(self, miscleavage_rate: float = 0.0) -> None:
        if self.abundance is None or self.mod_table is None:
            raise ValueError("Abundance and modification table must be set before digestion by using the set_quantification() method.")

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
            # Per-site miss weight, inversely proportional to the shorter of the two flanking peptides
            # (shorter peptides are merged with higher probability, per the paper). prev_cut/next_cut
            # are the cut positions bounding each site's two adjacent peptides, with the sequence
            # start (-1) and end (len-1) standing in at the termini.
            prev_cut = np.empty(n_sites, dtype=int)
            prev_cut[0] = -1
            prev_cut[1:] = sites[:-1]
            next_cut = np.empty(n_sites, dtype=int)
            next_cut[-1] = len(self.sequence) - 1
            next_cut[:-1] = sites[1:]
            left_len = sites - prev_cut
            right_len = next_cut - sites
            min_len = np.minimum(left_len, right_len)

            missable = min_len > 0 # zero-length flanking peptide can lead to division by zero, so exclude them.
            weights = np.zeros(n_sites, dtype=float)
            weights[missable] =  1.0 / min_len[missable]
            total_weight = weights.sum()
            if total_weight > 0:  # all sites non-missable (e.g. a lone terminal cut) -> no normalisation
                weights /= total_weight

            # The miscleavage_rate sets the proportion of sites missed; sample exactly that many per
            # proteoform copy, without replacement, weighted toward shorter flanking peptides. The
            # kept (cleaved) sites are the complement, so digestion_map[m] lists copy m's cleaved sites.
            n_missed = int(round(miscleavage_rate * n_sites))
            n_missed = min(max(n_missed, 0), int(missable.sum()))
            self.digestion_map = []
            for _ in range(self.abundance):
                if n_missed == 0:
                    self.digestion_map.append(sites.tolist())
                    continue
                missed = self.rng.choice(n_sites, size=n_missed, replace=False, p=weights)
                keep_mask = np.ones(n_sites, dtype=bool)
                keep_mask[missed] = False
                self.digestion_map.append(sites[keep_mask].tolist())
 
        # Peptide quantification: split each proteoform copy at its kept cut sites and aggregate
        # identical peptide species (same span and same modified sites, in protein coordinates) into
        # Peptide objects. abundance counts how many copies produced each species.
        peptides_by_id: dict[tuple[int, int, tuple[int, ...]], Peptide] = {}
        for form in range(self.abundance):
            mod_row = self.mod_table[form]
            # Peptide spans [start_index, end_index], inclusive, between consecutive kept cut sites.
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


    def set_quantification(self, abundance: int = 1, miscleavage_rate: float = 0.0) -> None:
        self.abundance = abundance

        # Per-site occupancy follows the paper's model (Eq. 1): for each modifiable serine r,
        # draw a count m_r ~ U(1, M) of copies to modify, then mark that many randomly chosen
        # copies as modified. This is uniform over [1, M], unlike an independent 50/50 coin per
        # copy (which would instead pin expected occupancy at M/2).
        table = np.zeros((abundance, len(self.sequence)))
        for i in self.serine_map:
            m_r = int(self.rng.integers(1, abundance + 1))  # inclusive upper bound -> U(1, M)
            modified = self.rng.choice(abundance, size=m_r, replace=False)
            table[modified, i] = 1
        self.mod_table = table
        self.digest(miscleavage_rate)


class ProteinGenerator:
    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()

    def generate_sequence(self, length: int) -> str:
        return "".join(self.rng.choice(AMINO_ACIDS, size=length))

    def generate_protein(self, length: int) -> Protein:
        return Protein(self.generate_sequence(length), rng=self.rng)
