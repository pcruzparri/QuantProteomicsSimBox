"""Experiment layer: a multi-protein simulated study, end-to-end.

Owns a dataset of `n_proteins`, each observed across `n_groups` x `n_subjects`, and ties the pipeline
together: ground-truth generation (per-group occupancy), observation, per-protein roll-up, and RMSE
of estimated vs known per-site log2 fold-change. A single shared `ObservationModel` makes the subject
effect beta shared across proteins for a (group, subject), while the site effect alpha stays
per-protein (keyed on sequence). One `rng` makes a run reproducible.
"""

import numpy as np

from .methods import QuantMethod
from .observation import ObservationModel, Sample
from .protgen import Protein, ProteinGenerator, make_group_proteins
from .rollups import RollupResult, group_site_change


class Experiment:
    def __init__(
        self,
        n_proteins: int = 5,
        protein_length: int = 200,
        n_groups: int = 2,
        n_subjects: int = 25,
        abundance: int = 250,
        repeat_units: int = 0,
        repeat_unit_length: int = 8,
        miscleavage_rate: float = 0.0,
        miscleavage_model: str = "global",
        var_subject: float = 0.0,
        var_site: float = 0.0,
        var_species: float = 0.0,
        detection_limit: int = 1,
        missingness: float = 0.0,
        position_aware: bool = False,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.n_proteins = n_proteins
        self.protein_length = protein_length
        self.n_groups = n_groups
        self.n_subjects = n_subjects
        self.abundance = abundance
        self.repeat_units = repeat_units  # identical repeated peptides per protein (0 = none)
        self.repeat_unit_length = repeat_unit_length
        self.miscleavage_rate = miscleavage_rate
        self.miscleavage_model = miscleavage_model  # digest fork: "global" | "bernoulli"
        self.missingness = missingness  # global induced-missingness rate (abundance-dependent)
        self.rng = rng if rng is not None else np.random.default_rng()
        # One model across all proteins: beta per (group, subject) shared; alpha per (sequence, site).
        self.model = ObservationModel(
            var_subject=var_subject,
            var_site=var_site,
            var_species=var_species,
            detection_limit=detection_limit,
            position_aware=position_aware,
            rng=self.rng,
        )
        # Per protein, the list of its per-group ground-truth realizations (same sequence,
        # group-specific occupancy): protein_groups[protein][group].
        self.protein_groups: list[list[Protein]] = []
        self.samples: list[list[Sample]] = []  # [protein] -> all groups' Samples, pooled

    def build(self) -> "Experiment":
        """Generate n_proteins, each with one ground-truth Protein per group (per-group occupancy)."""
        generator = ProteinGenerator(rng=self.rng)
        self.protein_groups = [
            make_group_proteins(
                generator.generate_sequence(
                    self.protein_length, self.repeat_units, self.repeat_unit_length
                ),
                self.n_groups,
                self.abundance,
                self.miscleavage_rate,
                self.miscleavage_model,
                self.rng,
            )
            for _ in range(self.n_proteins)
        ]
        return self

    def observe(self) -> list[list[Sample]]:
        """Observe every protein across all groups/subjects with the shared model."""
        if not self.protein_groups:
            self.build()
        self.samples = []
        for groups in self.protein_groups:
            pooled: list[Sample] = []
            for group, group_protein in enumerate(groups):
                pooled += self.model.sample_group(group_protein, group=group, n_subjects=self.n_subjects)
            pooled = self.model.apply_missingness(pooled, self.missingness)
            self.samples.append(pooled)
        return self.samples

    def roll_up(self, method: QuantMethod, min_per_group: int = 1) -> list[RollupResult]:
        """Roll up each protein's pooled samples to site level with `method` (one result per protein).

        `method` is a `QuantMethod` (intensity or stoichiometry) — see `quantproteomicssimbox.methods`.
        `min_per_group` is the presence filter. Best run with ``position_aware=True`` for stoichiometry
        methods (the spanning denominator is exact only then; a study axis otherwise).
        """
        if not self.samples:
            self.observe()
        return [method.roll_up(pooled, min_per_group) for pooled in self.samples]

    def score(
        self,
        method: QuantMethod,
        min_per_group: int = 1,
        baseline_group: int = 0,
        treatment_group: int = 1,
    ) -> float:
        """RMSE of estimated vs true per-site change across all sites of all proteins, for `method`.

        One scoring loop for every quantification method: roll up each protein with ``method.roll_up``,
        estimate the per-site between-group change (``group_site_change``), and compare to
        ``method.true_change`` (the truth matched to that method). Build methods with
        `methods.intensity_method` / `methods.stoichiometry_method` or pull one from `methods.QUANT_METHODS`.
        """
        if not self.samples:
            self.observe()
        squared_errors: list[float] = []
        for groups, pooled in zip(self.protein_groups, self.samples):
            result = method.roll_up(pooled, min_per_group)
            truth = method.true_change(groups[baseline_group], groups[treatment_group])
            estimated = group_site_change(result, baseline_group, treatment_group)
            for site, true_change in truth.items():
                est = estimated.get(site)
                if est is not None and np.isfinite(est):
                    squared_errors.append((est - true_change) ** 2)
        if not squared_errors:
            return float("nan")
        return float(np.sqrt(np.mean(squared_errors)))
