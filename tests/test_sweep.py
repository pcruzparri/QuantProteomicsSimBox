"""Tests for quantproteomicssimbox.sweep — the parameter-sweep harness."""

import numpy as np

from quantproteomicssimbox.methods import paper_ptm_methods
from quantproteomicssimbox.sweep import PAPER_PTM_GRID, SweepRecord, records_to_rows, run_sweep

_SMALL_BASE = dict(n_proteins=2, protein_length=60, n_subjects=4, abundance=40, var_subject=1.0, var_site=1.0)
_GRID = {"missingness": [0.0, 0.3], "miscleavage_rate": [0.0, 0.5]}  # 4 combos


def _methods():
    m = paper_ptm_methods()
    return {"rollup/median": m["rollup/median"], "zrollup/median": m["zrollup/median"]}


def test_paper_ptm_methods_has_nine():
    methods = paper_ptm_methods()
    assert len(methods) == 9
    assert set(methods) == {
        f"{s}/{a}" for s in ("rollup", "rrollup", "zrollup") for a in ("mean", "median", "sum")
    }


def test_run_sweep_record_count_and_shape():
    records = run_sweep(_GRID, _methods(), n_replicates=2, base=_SMALL_BASE)
    assert len(records) == 4 * 2  # 4 combos × 2 methods
    assert all(isinstance(r, SweepRecord) for r in records)
    for r in records:
        assert set(r.params) == {"missingness", "miscleavage_rate"}
        assert np.isfinite(r.rmse_mean)
        assert r.rmse_se >= 0.0
        assert r.n == 2


def test_run_sweep_is_reproducible():
    a = run_sweep(_GRID, _methods(), n_replicates=2, base=_SMALL_BASE, seed=7)
    b = run_sweep(_GRID, _methods(), n_replicates=2, base=_SMALL_BASE, seed=7)
    assert [r.rmse_mean for r in a] == [r.rmse_mean for r in b]


def test_records_to_rows_flattens():
    records = run_sweep(_GRID, _methods(), n_replicates=2, base=_SMALL_BASE)
    rows = records_to_rows(records)
    assert rows[0].keys() == {"missingness", "miscleavage_rate", "method", "rmse_mean", "rmse_se", "n"}


def test_paper_grid_is_the_documented_axes():
    assert set(PAPER_PTM_GRID) == {
        "protein_length", "abundance", "missingness", "miscleavage_rate",
        "var_subject", "var_site", "n_proteins",
    }
