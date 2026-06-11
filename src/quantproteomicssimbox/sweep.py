"""Parameter-sweep harness: run replicate Experiments over a parameter grid and tabulate mean RMSE
± standard error per quantification method.

The paper's PTM benchmark is a large grid (protein length, abundance, missingness, miscleavage rate,
subject/site variance, number of proteins) scored for every (scaling × aggregation) method.
`run_sweep` walks the grid: per combo it builds + observes each replicate Experiment **once**, then
scores every method on that shared data, and returns tidy `SweepRecord`s (mean ± SE over replicates).
"""

import itertools
from dataclasses import dataclass

import numpy as np

from .experiment import Experiment
from .methods import QuantMethod

# The paper's PTM benchmark grid (Fig. S1–S3). Crossed with the 9 (scaling × aggregation) intensity
# methods this is the >600-combo sweep; subject and site variance are swept independently (3×3).
PAPER_PTM_GRID: dict[str, list] = {
    "protein_length": [100, 200],
    "abundance": [100, 250],
    "missingness": [0.0, 0.25, 0.5],
    "miscleavage_rate": [0.0, 0.25, 0.5],
    "var_subject": [0.0, 1.0, 9.0],
    "var_site": [0.0, 1.0, 9.0],
    "n_proteins": [5, 10],
}


@dataclass
class SweepRecord:
    """One method's score at one parameter combo: mean ± SE of RMSE over the finite replicates."""

    params: dict
    method: str
    rmse_mean: float
    rmse_se: float  # standard error of the mean across replicates
    n: int  # replicates contributing a finite RMSE


def run_sweep(
    grid: dict[str, list],
    methods: dict[str, QuantMethod],
    *,
    n_replicates: int = 5,
    base: dict | None = None,
    min_per_group: int = 1,
    seed: int = 0,
) -> list[SweepRecord]:
    """Score every `methods` value at every combo of `grid`, averaged over `n_replicates` Experiments.

    `grid` maps `Experiment` kwargs to value lists (their Cartesian product is the swept combos);
    `base` holds fixed kwargs shared by every Experiment. Each (combo, replicate) is built + observed
    once and scored by all methods, so adding methods is nearly free. Returns one `SweepRecord` per
    (combo, method). Reproducible: the rng seed is derived from `seed`, the combo index and replicate.
    """
    base = dict(base or {})
    keys = list(grid)
    records: list[SweepRecord] = []
    for combo_i, values in enumerate(itertools.product(*(grid[k] for k in keys))):
        params = dict(zip(keys, values))
        scores: dict[str, list[float]] = {name: [] for name in methods}
        for rep in range(n_replicates):
            rng = np.random.default_rng(seed + combo_i * 10_000 + rep)
            exp = Experiment(rng=rng, **{**base, **params})
            exp.build()
            exp.observe()
            for name, method in methods.items():
                scores[name].append(exp.score(method, min_per_group=min_per_group))
        for name in methods:
            finite = np.array(scores[name], dtype=float)
            finite = finite[np.isfinite(finite)]
            n = int(finite.size)
            mean = float(finite.mean()) if n else float("nan")
            se = float(finite.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            records.append(SweepRecord(params=params, method=name, rmse_mean=mean, rmse_se=se, n=n))
    return records


def records_to_rows(records: list[SweepRecord]) -> list[dict]:
    """Flatten `SweepRecord`s to plain dict rows (params + method + stats) for a DataFrame / CSV."""
    return [
        {**r.params, "method": r.method, "rmse_mean": r.rmse_mean, "rmse_se": r.rmse_se, "n": r.n}
        for r in records
    ]
