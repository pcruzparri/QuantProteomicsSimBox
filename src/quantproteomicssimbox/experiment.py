"""Experiment layer: a multi-protein simulated study, end-to-end.

Owns a dataset of `n_proteins`, each observed across `n_groups` x `n_subjects`, and ties the pipeline
together: ground-truth generation (per-group occupancy), observation, per-protein roll-up, and RMSE
of estimated vs known per-site log2 fold-change. A single shared `ObservationModel` makes the subject
effect beta shared across proteins for a (group, subject), while the site effect alpha stays
per-protein (keyed on sequence). One `rng` makes a run reproducible.
"""

import numpy as np

from .observation import ObservationModel, Sample
from .protgen import (
    Protein,
    ProteinGenerator,
    make_group_proteins,
    true_site_log2_fold_change,
    true_site_stoichiometry_change,
)
from .rollups import (
    STOICHIOMETRY_METHODS,
    RollupResult,
    group_log2_fold_change,
    roll_up,
    roll_up_stoichiometry,
)


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

    def roll_up(
        self,
        scaling: str = "rollup",
        aggregation: str = "median",
        space: str = "log2",
        min_per_group: int = 1,
    ) -> list[RollupResult]:
        """Roll up each protein's pooled samples to site level (one RollupResult per protein).

        `space` ("log2" | "linear") selects the aggregation space; `min_per_group` is the presence
        filter — see ``rollups.roll_up``.
        """
        if not self.samples:
            self.observe()
        return [roll_up(pooled, scaling, aggregation, space, min_per_group) for pooled in self.samples]

    def score(
        self,
        scaling: str = "rollup",
        aggregation: str = "median",
        space: str = "log2",
        min_per_group: int = 1,
        baseline_group: int = 0,
        treatment_group: int = 1,
    ) -> float:
        """RMSE of estimated vs true per-site log2FC across all sites of all proteins.

        Defaults to log2 aggregation space (the paper's convention) with the `min_per_group`=1
        presence filter. The matched (unbiased) aggregator is mean/median in log2 space and sum in
        linear space.
        """
        results = self.roll_up(scaling, aggregation, space, min_per_group)
        squared_errors: list[float] = []
        for groups, result in zip(self.protein_groups, results):
            truth = true_site_log2_fold_change(groups[baseline_group], groups[treatment_group])
            estimated = group_log2_fold_change(result, baseline_group, treatment_group)
            for site, true_fc in truth.items():
                est = estimated.get(site)
                if est is not None and np.isfinite(est):
                    squared_errors.append((est - true_fc) ** 2)
        if not squared_errors:
            return float("nan")
        return float(np.sqrt(np.mean(squared_errors)))

    def roll_up_stoichiometry(self, method: str = "fraction", min_per_group: int = 1) -> list[RollupResult]:
        """Stoichiometry roll-up per protein — per-site modified fraction (see
        ``rollups.roll_up_stoichiometry``). `method` selects a ``STOICHIOMETRY_METHODS`` entry.

        Best run with ``position_aware=True``: the spanning denominator is exact only under
        position-aware observation; under the agnostic default it is biased (a study axis, not enforced).
        """
        if not self.samples:
            self.observe()
        return [roll_up_stoichiometry(pooled, method, min_per_group) for pooled in self.samples]

    def score_stoichiometry(
        self,
        method: str = "fraction",
        min_per_group: int = 1,
        baseline_group: int = 0,
        treatment_group: int = 1,
    ) -> float:
        """RMSE of estimated vs true per-site stoichiometry change across all sites of all proteins.

        The ``fraction`` method estimates the log2 fold-change of the modified fraction; ``logit`` the
        change in log-odds. Each is scored against its matching truth
        (``true_site_stoichiometry_change``, keyed off the method's space). Best run with
        ``position_aware=True`` (exact spanning denominator).
        """
        results = self.roll_up_stoichiometry(method, min_per_group)
        space = STOICHIOMETRY_METHODS[method].space
        squared_errors: list[float] = []
        for groups, result in zip(self.protein_groups, results):
            truth = true_site_stoichiometry_change(groups[baseline_group], groups[treatment_group], space)
            estimated = group_log2_fold_change(result, baseline_group, treatment_group)
            for site, true_change in truth.items():
                est = estimated.get(site)
                if est is not None and np.isfinite(est):
                    squared_errors.append((est - true_change) ** 2)
        if not squared_errors:
            return float("nan")
        return float(np.sqrt(np.mean(squared_errors)))
