# QuantProteomicsSimBox — Guide

A simulation sandbox for **benchmarking peptide → site roll-up (aggregation) methods** in bottom-up
proteomics. Because real experiments have no ground truth, this package *generates* data with a
**known answer**, runs each candidate method on it, and scores how well the method recovers the truth.
It replicates (and extends) VonKaenel et al., *J. Proteome Res.* 2026 (DOI `10.1021/acs.jproteome.5c00782`).

This guide is the friendly entry point. For a dense code-to-paper reference, see
[`AGENTS.md`](../AGENTS.md). For runnable examples, see the notebook at
[`src/quantproteomicssimbox/playground.ipynb`](../src/quantproteomicssimbox/playground.ipynb).

## Contents
1. [The big idea (plain English)](#1-the-big-idea-plain-english)
2. [The pipeline: four layers](#2-the-pipeline-four-layers)
3. [Architecture & code map](#3-architecture--code-map)
4. [Install & quickstart](#4-install--quickstart)
5. [Usage patterns](#5-usage-patterns)
6. [The experimental knobs (the tree)](#6-the-experimental-knobs-the-tree)
7. [Experiment recipes (question → settings)](#7-experiment-recipes-question--settings)
8. [What we've found so far](#8-what-weve-found-so-far)
9. [Extending the framework](#9-extending-the-framework)
10. [Reference tables](#10-reference-tables)

---

## 1. The big idea (plain English)

In bottom-up proteomics you don't measure proteins directly. You chop each protein into **peptides**
with an enzyme (trypsin), measure the peptides in a mass spectrometer, and then try to add the peptide
measurements back up to say something about the **protein or a specific site on it** — for example,
"how much is this serine phosphorylated, and did it change between two groups?"

That "add the peptides back up" step is called **roll-up** (or aggregation), and there are many ways to
do it: take the mean of the peptides, the median, the sum; rescale the peptides first; work with raw
intensities or with a *fraction modified*; and so on. Each choice has different blind spots and biases.
The catch: in a real experiment you can't tell which roll-up is best, because **you never know the true
answer** the method is supposed to recover.

This package gets around that by **simulating the whole process with a known truth**:

> We invent proteins, decide exactly which copies are modified at which sites (so we *know* the true
> per-site amounts), simulate a realistic, noisy measurement of the resulting peptides, run each roll-up
> method on that simulated data, and score it by how close its estimate is to the truth we planted.

The score is an **RMSE** (root-mean-square error) of the estimated per-site *change between groups*
versus the true change. Lower is better. By sweeping the simulation's knobs (noise, missing data,
digestion quality, …) we can ask *which method wins, and under what conditions*.

---

## 2. The pipeline: four layers

Data flows through four stages. Each later stage only depends on the one before it, and the **ground
truth is kept separate from the noisy observation** so the truth is never contaminated.

```
   ┌────────────────────┐   ┌──────────────────────┐   ┌──────────────────┐   ┌───────────────┐
   │  1. GROUND TRUTH   │ → │   2. OBSERVATION     │ → │   3. ROLL-UP     │ → │  4. SCORING   │
   │   (protgen)        │   │  (observation.py)    │   │  (rollups/)      │   │ (methods +    │
   │                    │   │                      │   │                  │   │  experiment)  │
   │ proteins, which    │   │ what the instrument  │   │ aggregate peptides│   │ compare to    │
   │ copies are modified│   │ actually sees: noise,│   │ back up to a      │   │ the planted   │
   │ where, digestion   │   │ missing values,      │   │ per-site number   │   │ truth → RMSE  │
   │ into peptides      │   │ detection limits     │   │ (the methods)     │   │               │
   └────────────────────┘   └──────────────────────┘   └──────────────────┘   └───────────────┘
```

1. **Ground truth** — Build proteins; for each modifiable site decide how many of the protein's copies
   carry the modification (the *occupancy*, the thing we want to recover); digest the copies into
   peptides. This layer is exact and known.
2. **Observation** — Turn the true peptide amounts into what a mass spectrometer would *report*: add
   per-sample, per-site, and per-peptide variation, optionally drop undetectable peptides, and remove a
   fraction of measurements to mimic missing data.
3. **Roll-up** — The methods under test. Aggregate the observed peptides into one number per site, per
   sample. Two families: **intensity** roll-up and **stoichiometry** (fraction-modified) roll-up.
4. **Scoring** — Turn site values into a between-group change, compare to the known true change, report
   RMSE. A sweep harness repeats this over a grid of settings.

---

## 3. Architecture & code map

A small `src/`-layout library. Each module maps to one pipeline stage.

```
src/quantproteomicssimbox/
├── utils.py              shared constants + the logit2 helper (amino acids, STOICH_EPS)
│
├── protgen/              ── LAYER 1: ground truth ──────────────────────────────────
│   ├── peptide.py        Peptide: one distinct peptide species (sequence, span, mod sites, abundance)
│   ├── protein.py        Protein: occupancy model + trypsin digestion; per_copy / per_subject modes
│   ├── generator.py      ProteinGenerator: random sequences (optionally with forced repeated peptides)
│   └── truth.py          make_group_proteins; true_site_log2_fold_change / _stoichiometry_change
│
├── observation.py        ── LAYER 2 ── Sample (a subject's data); ObservationModel (noise, detection,
│                          missingness); aggregate_peptides (position-aware vs agnostic collapse)
│
├── rollups/              ── LAYER 3: the methods under test ────────────────────────
│   ├── core.py           RollupResult, the Space enum, group_site_change, shared span helpers
│   ├── intensity.py      SCALINGS × AGGREGATIONS, build_site_tables, roll_up
│   └── stoichiometry.py  STOICHIOMETRY_METHODS, the two builders, roll_up_stoichiometry
│
├── methods.py            ── LAYER 4 ── QuantMethod (roll-up + matching truth), intensity_method /
│                          stoichiometry_method factories, the QUANT_METHODS registry
├── experiment.py         Experiment: ties all four layers together; build → observe → roll_up → score
└── sweep.py              run_sweep: replicate Experiments over a parameter grid → mean RMSE ± SE
```

Two architectural ideas worth knowing:

- **Ground truth vs observation are separate layers.** A `Protein` is exact and immutable after setup;
  the `ObservationModel` derives noisy `Sample`s from it without ever mutating it. This is what lets the
  score compare against an uncontaminated truth.
- **Methods are a registry, scoring is one loop.** A `QuantMethod` bundles *how to roll up* with *the
  matching ground-truth change to score against*. So `Experiment.score(method)` is a single code path
  that works for intensity, stoichiometry, per-peptide, and (future) LiP methods alike. New methods are
  registry entries, not new code paths.

---

## 4. Install & quickstart

The project uses **`uv`** (not pip) and Python 3.12.

```bash
uv sync                 # install dependencies into .venv
uv run pytest           # run the test suite (should be all green)
```

Smallest end-to-end example — build a study, score one method:

```python
import numpy as np
from quantproteomicssimbox.experiment import Experiment
from quantproteomicssimbox.methods import intensity_method

exp = Experiment(n_proteins=5, n_subjects=25, miscleavage_rate=0.25,
                 var_subject=1.0, var_site=1.0, rng=np.random.default_rng(0))

rmse = exp.score(intensity_method("rollup", "median", "log2"))
print(rmse)   # RMSE of estimated vs true per-site log2 fold-change (lower is better)
```

---

## 5. Usage patterns

**Score any method.** Methods are built by two factories or pulled from a registry:

```python
from quantproteomicssimbox.methods import (
    intensity_method, stoichiometry_method, QUANT_METHODS,
)

exp.score(intensity_method("rollup", "sum", "log2"))     # intensity, log2-sum
exp.score(stoichiometry_method("fraction"))              # stoichiometry, bare fraction
exp.score(stoichiometry_method("peptide_mean"))          # per-peptide mean fraction
exp.score(QUANT_METHODS["int_median"])                   # a pre-named registry method
```

**Lifecycle.** `score()` lazily runs `build()` (make proteins) then `observe()` (simulate data) if you
haven't. To reuse one simulated dataset across several methods, call them explicitly:

```python
exp.build(); exp.observe()
for name, method in QUANT_METHODS.items():
    print(name, exp.score(method))   # all scored on the same simulated data
```

**Sweep a grid.** The harness builds + observes each replicate once and scores every method on it:

```python
from quantproteomicssimbox.sweep import run_sweep, PAPER_PTM_GRID, records_to_rows
from quantproteomicssimbox.methods import paper_ptm_methods

records = run_sweep(
    grid={"missingness": [0.0, 0.25, 0.5], "miscleavage_rate": [0.0, 0.5]},
    methods=paper_ptm_methods(),          # the 9 scaling × aggregation intensity methods
    n_replicates=5,
    base=dict(n_proteins=5, n_subjects=10, digestion="per_subject"),
)
rows = records_to_rows(records)           # tidy dicts → pandas.DataFrame(rows)
```

The full paper grid (`PAPER_PTM_GRID`, 648 combos) and a ready-made runner live in
[`scripts/run_ptm_sweep.py`](../scripts/run_ptm_sweep.py); its output is committed at
`results/ptm_sweep.csv`.

**Explore interactively.** The notebook
[`playground.ipynb`](../src/quantproteomicssimbox/playground.ipynb) has worked figures for every topic
in §7–8.

---

## 6. The experimental knobs (the tree)

Everything you can control is a keyword on `Experiment(...)` (plus the method you score). They split
cleanly by pipeline stage. Read this as a tree of *what you can branch on*:

```
GROUND TRUTH ─ what is really there
├── study size
│     ├── n_proteins ........ how many proteins in the study (more = steadier averages)
│     ├── protein_length .... longer proteins → more sites & peptides
│     ├── abundance (M) ...... copies of each protein (the denominator of "fraction modified")
│     ├── n_groups ........... experimental groups to compare (default 2)
│     └── n_subjects ......... replicate samples per group
├── repeats
│     └── repeat_units ....... force N identical peptides at different places (a "can't tell apart" stress test)
└── digestion ─ how the enzyme cuts
      ├── digestion = per_copy ...... each molecule cut independently (finer; closer to physical reality)
      │     └── miscleavage_model = global | bernoulli   (how missed cuts are drawn per copy)
      └── digestion = per_subject ... one cut pattern per sample (the paper's model; matches its magnitudes)
      └── miscleavage_rate = 0.0 | 0.25 | 0.5            (how often the enzyme misses a cut)

OBSERVATION ─ what the instrument reports
├── noise (variances; the paper's "variance levels" 0 / 1 / 9)
│     ├── var_subject ....... whole-sample up/down shift (cancels in fraction methods)
│     ├── var_site .......... per-modification-site shift (the within-peptide "a PTM changes signal" effect)
│     └── var_species ....... per-peptide ionization efficiency (cancels in per-peptide fractions)
├── position_aware = False | True   can you tell identical peptide sequences from different loci apart?
├── detection_limit = 1 | k .......  optional: ignore peptides made by fewer than k copies (a sensitivity floor)
└── missingness = 0.0 | 0.25 | 0.5   fraction of measurements lost (low-abundance lost more often)

ROLL-UP ─ the method you score (passed to score(), not to Experiment)
├── INTENSITY ............ aggregate peptide *intensities*
│     ├── scaling     = rollup (none) | rrollup (re-reference) | zrollup (z-score)
│     ├── aggregation = mean | median | sum
│     └── space       = log2 (the paper) | linear
└── STOICHIOMETRY ....... aggregate the *fraction modified* at a site
      ├── combine   = pooled (one ratio) | peptide_mean | peptide_median (per-peptide then average)
      └── transform = fraction (raw) | logit (log-odds)
```

A few plain-English notes on the less obvious knobs:

- **Digestion granularity (`per_copy` vs `per_subject`).** Physically, every molecule is digested on its
  own (`per_copy`). But the paper — and most bulk experiments — effectively see *one* digestion pattern
  per sample (`per_subject`). This matters a lot for the `sum` aggregator: `per_copy` produces far more
  distinct peptide variants per site, which inflates `sum`. Use `per_subject` to match the paper's
  numbers; use `per_copy` for the more granular, physically-faithful view.
- **The three variances.** Think of them as three independent "dials of messiness." `var_subject` shifts
  a whole sample; `var_site` shifts only the *modified* form of a site; `var_species` shifts a whole
  peptide backbone (its ionization efficiency). Which dial you turn decides which method looks good
  (see §7).
- **`position_aware`.** Bottom-up MS often can't tell two identical peptide sequences apart even if they
  came from different places on the protein. `position_aware=False` (the realistic default) merges them;
  `True` keeps them separate (an idealized instrument). Combine with `repeat_units` to stress-test it.
- **`detection_limit`.** An *optional* knob (off by default). It is **not** how we match the paper's
  magnitudes (that's `per_subject` digestion) — it's a separate "what if faint peptides vanish" study.

---

## 7. Experiment recipes (question → settings)

Each research question lights up a different branch of the tree above. Set the knobs that matter and
leave the rest at defaults.

### A. "Which intensity roll-up best recovers PTM fold-changes?" (the paper's headline)
- **Method:** intensity, `space="log2"`, sweep `scaling ∈ {rollup, rrollup, zrollup}` × `aggregation ∈
  {mean, median, sum}` (that's `paper_ptm_methods()`).
- **Knobs:** `digestion="per_subject"` (to match magnitudes); sweep `missingness`, `miscleavage_rate`,
  `var_subject`, `var_site`, `protein_length`, `abundance`, `n_proteins` over `PAPER_PTM_GRID`.
- **Read-out:** mean RMSE per method. Expect `mean ≈ median ≪ sum`, and `zrollup` worst for mean/median.

### B. "Is stoichiometry (fraction modified) better than intensity?"
- **Method:** compare `intensity_method(...)` against `stoichiometry_method("fraction" | "logit")`.
- **Knobs:** `position_aware=True` (stoichiometry needs honest spans to count the denominator).
- **Read-out:** stoichiometry is on a different, often lower-error footing, and `logit` is steadier than
  raw `fraction` near 0 % / 100 % modification.

### C. "When does per-peptide aggregation beat the pooled ratio?"
- **Method:** `stoichiometry_method("fraction")` (pooled) vs `("peptide_mean")` (per-peptide).
- **Knobs:** turn up `var_species` (per-peptide ionization differences) and `miscleavage_rate` (so each
  site is covered by several peptides); keep `position_aware=True`.
- **Read-out:** per-peptide *cancels* the ionization differences and wins once `var_species` is large;
  with no ionization effect it just pays extra variance and loses. (It does **not** escape `var_site`.)

### D. "How robust is each method to missing data?"
- **Knobs:** sweep `missingness ∈ {0, 0.25, 0.5}`; pick a couple of methods to compare.
- **Read-out:** mean/median intensity and stoichiometry hold up; watch how per-peptide methods behave
  when whole peptides drop out.

### E. "How bad is the can't-tell-peptides-apart problem?"
- **Knobs:** sweep `repeat_units ∈ {0, 2, 4, 8}` × `position_aware ∈ {True, False}`.
- **Read-out:** the position-agnostic penalty grows with repeated peptides and hits the **pooled**
  stoichiometry ratio hardest.

### F. "Does abundance noise alone fool a method?"
- **Knobs:** `var_subject` large, `var_site=var_species=0`.
- **Read-out:** stoichiometry methods are *immune* (the per-sample shift cancels in any ratio); intensity
  RMSE climbs with the noise.

### G. "How granular is our digestion vs the paper's?"
- **Knobs:** the same config under `digestion="per_copy"` vs `"per_subject"`.
- **Read-out:** `per_copy` fragments each site into many more peptides → inflated `sum`; `per_subject`
  reproduces the paper's `sum ≈ 3`.

---

## 8. What we've found so far

- **Intensity ranking reproduces the paper.** Over the full 648-combo grid (`results/ptm_sweep.csv`,
  `per_subject` digestion): `rollup` mean 1.13 / median 1.16 / **sum 2.98**; `zrollup` is worst for
  mean/median (1.34 / 1.35). `mean ≈ median ≪ sum`, robust to 0/25/50 % missingness — matching the paper
  in both ranking *and* magnitude.
- **The `sum` "magnitude gap" was a digestion-granularity artifact, not a detection effect.** Our
  original per-copy digestion over-fragmented sites; switching to the reference's `per_subject` model
  pulls `sum` from ~10 back to ~3 honestly. (`detection_limit` reaches the same place but is *not* how
  the paper does it, so it's kept as an optional, separate knob.)
- **Stoichiometry cancels abundance noise.** A per-sample (`var_subject`) shift cancels inside any
  fraction, so stoichiometry RMSE is flat against it while intensity RMSE grows.
- **Per-peptide aggregation cancels ionization (`var_species`) but not site effects (`var_site`).** So it
  wins where between-peptide abundance differences dominate (high ionization variance + fragmentation),
  and the per-site effect remains the one bias no fraction method escapes.
- **Position-agnostic grouping biases the denominator.** The penalty grows with repeated-peptide density
  and is worst for the pooled stoichiometry ratio.

---

## 9. Extending the framework

The design is registries + one scoring loop, so most extensions are small, local additions:

- **A new intensity scaling** (e.g. finish `rrollup` variants): add a `scale_*` function and register it
  in `SCALINGS` (`rollups/intensity.py`).
- **A new stoichiometry method:** add an entry to `STOICHIOMETRY_METHODS` (`rollups/stoichiometry.py`) —
  an `(aggregation, transform)` pair; the builders and scoring already handle it.
- **A new quantification method end-to-end:** write a `QuantMethod` (a `roll_up` + its matching
  `true_change`) in `methods.py`; `Experiment.score` and `run_sweep` pick it up for free.
- **A new model effect or digestion option:** add a knob to `ObservationModel` / `Protein`, thread it
  through `Experiment`, default it to a no-op so existing behavior and tests are unchanged.

Known backlog (see [`AGENTS.md`](../AGENTS.md) for detail): the TMT-plex (block) missingness variant,
and the entire **LiP-MS** pipeline (region masking, proteinase-K digestion, two-stage digestion) — a
second application the framework is structured to host alongside PTM.

---

## 10. Reference tables

**`Experiment(...)` knobs** (defaults in parentheses):

| Knob | Default | Meaning |
|---|---|---|
| `n_proteins` | 5 | proteins in the study |
| `protein_length` | 200 | residues per protein |
| `n_groups` / `n_subjects` | 2 / 25 | groups compared / replicates per group |
| `abundance` | 250 | copies per protein (M) |
| `miscleavage_rate` | 0.0 | fraction of cleavage sites missed |
| `miscleavage_model` | `"global"` | per-copy missed-cut model: `global` \| `bernoulli` |
| `digestion` | `"per_copy"` | granularity: `per_copy` \| `per_subject` (paper-faithful) |
| `var_subject` / `var_site` / `var_species` | 0.0 | per-sample / per-site / per-peptide noise variances |
| `detection_limit` | 1 | optional min copies for a peptide to be observed |
| `missingness` | 0.0 | fraction of measurements dropped (abundance-dependent) |
| `position_aware` | `False` | keep identical peptides from different loci separate? |
| `repeat_units` | 0 | forced identical peptides at distinct loci |
| `rng` | new | NumPy `Generator` for reproducibility |

**Method registries** (built via `intensity_method(scaling, aggregation, space)` /
`stoichiometry_method(name)`):

| Family | Choices |
|---|---|
| intensity `scaling` | `rollup`, `rrollup`, `zrollup` |
| intensity `aggregation` | `mean`, `median`, `sum` |
| intensity `space` | `log2`, `linear` |
| `STOICHIOMETRY_METHODS` | `fraction`, `logit`, `logit_pseudocount`, `peptide_mean`, `peptide_median`, `peptide_mean_logit`, `peptide_median_logit` |

**Scoring** — `score(method)` returns RMSE of the estimated vs true per-site between-group change.
`group_site_change` forms that change in the right space automatically: a **difference** for `log2`/
`logit` values, a **log2 ratio** for `linear`/`fraction` values.
