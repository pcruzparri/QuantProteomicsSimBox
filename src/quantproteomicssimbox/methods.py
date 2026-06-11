"""Quantification methods: bundle a roll-up with the ground-truth change it should be scored against.

A `QuantMethod` pairs a ``roll_up(samples, min_per_group) -> RollupResult`` with the matching
``true_change(protein_a, protein_b) -> {site: change}``, so ``Experiment.score(method)`` is a single
loop over any method — intensity, pooled/per-peptide stoichiometry, and future LiP roll-ups alike.
Build them with the factories, or iterate the ``QUANT_METHODS`` registry.
"""

from collections.abc import Callable
from dataclasses import dataclass

from .observation import Sample
from .protgen import Protein, true_site_log2_fold_change, true_site_stoichiometry_change
from .rollups import STOICHIOMETRY_METHODS, RollupResult, roll_up, roll_up_stoichiometry

RollUpFn = Callable[[list[Sample], int], RollupResult]
TrueChangeFn = Callable[[Protein, Protein], dict[int, float]]


@dataclass(frozen=True)
class QuantMethod:
    """A scoring-ready quantification strategy.

    `roll_up(samples, min_per_group)` -> per-site values; `true_change(a, b)` -> the matching known
    per-site between-group change. `name` is for display and registry keys.
    """

    name: str
    roll_up: RollUpFn
    true_change: TrueChangeFn


def intensity_method(
    scaling: str = "rollup", aggregation: str = "median", space: str = "log2", name: str | None = None
) -> QuantMethod:
    """Intensity roll-up (scaling x aggregation in `space`), scored vs the occupancy log2 fold-change."""
    return QuantMethod(
        name=name or f"intensity:{scaling}/{aggregation}/{space}",
        roll_up=lambda samples, min_per_group=1: roll_up(samples, scaling, aggregation, space, min_per_group),
        true_change=true_site_log2_fold_change,
    )


def stoichiometry_method(method: str = "fraction", name: str | None = None) -> QuantMethod:
    """Stoichiometry roll-up (a ``STOICHIOMETRY_METHODS`` entry), scored vs the matching stoichiometry change."""
    space = STOICHIOMETRY_METHODS[method].space
    return QuantMethod(
        name=name or f"stoich:{method}",
        roll_up=lambda samples, min_per_group=1: roll_up_stoichiometry(samples, method, min_per_group),
        true_change=lambda a, b: true_site_stoichiometry_change(a, b, space),
    )


# Convenience registry of common named methods — iterate this in notebooks / sweeps.
QUANT_METHODS: dict[str, QuantMethod] = {
    "int_mean": intensity_method(aggregation="mean", name="int_mean"),
    "int_median": intensity_method(aggregation="median", name="int_median"),
    "int_sum": intensity_method(aggregation="sum", name="int_sum"),
    "stoich_fraction": stoichiometry_method("fraction", name="stoich_fraction"),
    "stoich_logit": stoichiometry_method("logit", name="stoich_logit"),
    "stoich_pep_mean": stoichiometry_method("peptide_mean", name="stoich_pep_mean"),
    "stoich_pep_median": stoichiometry_method("peptide_median", name="stoich_pep_median"),
}


def paper_ptm_methods() -> dict[str, QuantMethod]:
    """The paper's 9 PTM intensity methods: every (scaling × aggregation) pair, in log2 space."""
    return {
        f"{scaling}/{aggregation}": intensity_method(scaling, aggregation, "log2", name=f"{scaling}/{aggregation}")
        for scaling in ("rollup", "rrollup", "zrollup")
        for aggregation in ("mean", "median", "sum")
    }
