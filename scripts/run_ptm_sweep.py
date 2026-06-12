"""Run the paper's full PTM intensity sweep and write a per-combo results table + headline summary.

PAPER_PTM_GRID (648 combos) x the 9 (scaling x aggregation) methods, averaged over N_REPLICATES
Experiments each. Long-running (~30-50 min). Subjects are fixed (not a PTM sweep axis); 10 keeps the
run tractable while preserving the method ranking (which is invariant to subject count). Uses the
faithful per-subject digestion (the reference realizes missed cleavages once per sample, not per copy),
so the log2-sum magnitude tracks the paper rather than the inflated per-copy value.

    uv run python scripts/run_ptm_sweep.py
"""

import pathlib

import pandas as pd

from quantproteomicssimbox.methods import paper_ptm_methods
from quantproteomicssimbox.sweep import PAPER_PTM_GRID, records_to_rows, run_sweep

N_REPLICATES = 5
BASE = dict(n_groups=2, n_subjects=10, digestion="per_subject")
OUT = pathlib.Path("results/ptm_sweep.csv")


def main() -> None:
    records = run_sweep(PAPER_PTM_GRID, paper_ptm_methods(), n_replicates=N_REPLICATES, base=BASE)
    df = pd.DataFrame(records_to_rows(records))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    n_methods = df["method"].nunique()
    print(f"wrote {OUT}  ({len(df)} rows = {n_methods} methods x {len(df) // n_methods} combos, "
          f"{N_REPLICATES} replicates each)")

    df[["scaling", "aggregation"]] = df["method"].str.split("/", expand=True)
    print("\nMean RMSE per method over all combos (rows=scaling, cols=aggregation):")
    print(df.groupby(["scaling", "aggregation"])["rmse_mean"].mean().round(3).unstack())

    print("\nRobustness to missingness (rollup scaling) — mean RMSE by missingness x aggregation:")
    rollup = df[df["scaling"] == "rollup"]
    print(rollup.groupby(["missingness", "aggregation"])["rmse_mean"].mean().round(3).unstack())


main()
