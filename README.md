# QuantProteomicsSimBox

A simulation sandbox for **benchmarking peptide → site roll-up (aggregation) methods** in bottom-up
proteomics. Real experiments have no ground truth, so this package generates data with a *known answer*,
runs each candidate method on it, and scores how well the method recovers the truth. It replicates and
extends VonKaenel et al., *J. Proteome Res.* 2026 (DOI `10.1021/acs.jproteome.5c00782`).

```python
import numpy as np
from quantproteomicssimbox.experiment import Experiment
from quantproteomicssimbox.methods import intensity_method

exp = Experiment(n_proteins=5, n_subjects=25, miscleavage_rate=0.25,
                 var_subject=1.0, var_site=1.0, rng=np.random.default_rng(0))

# RMSE of the estimated vs true per-site log2 fold-change (lower is better)
print(exp.score(intensity_method("rollup", "median", "log2")))
```

The pipeline has four layers — **ground truth → observation → roll-up → scoring** — and you study a
question by turning the simulation's knobs (noise, missing data, digestion granularity, …) and comparing
methods.

## Get started
```bash
uv sync          # install (uses uv, Python 3.12)
uv run pytest    # run the test suite
```

## Documentation
- **[docs/GUIDE.md](docs/GUIDE.md)** — the instructional guide: architecture, usage, the full knob tree,
  and experiment recipes (what to set to study each question), in plain English.
- **[AGENTS.md](AGENTS.md)** — dense code-to-paper reference and the project backlog.
- **[notebooks/playground.ipynb](notebooks/playground.ipynb)** — worked
  figures for every topic.
- **[scripts/run_ptm_sweep.py](scripts/run_ptm_sweep.py)** — the full paper sweep; output in
  `results/ptm_sweep.csv`.
