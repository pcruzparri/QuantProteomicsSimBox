"""The Peptide species — the unit a digested Protein is composed of (and the carrier of observed
intensity in the observation layer)."""


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
