# QuantProteomicsSimBox — Agent Notes

## Source Paper & Goal

This project replicates the simulation framework from:

> **Aggregation Methods for Quantifying PTM and Structural Changes in Bottom-Up Proteomics**
> VonKaenel, Rozum, Zhang, Stratton, Bramer, Wiley, Qian, Sims, Melchior, Feng.
> *J. Proteome Res.* (2026), DOI: `10.1021/acs.jproteome.5c00782`. CC-BY 4.0.
> Reference implementation: https://github.com/PNNL-Predictive-Phenomics/OptRollingup

**Central question**: which peptide→site **roll-up** (aggregation) strategy most accurately recovers
ground-truth changes in two under-explored bottom-up applications — (1) **PTM** site-level
quantification and (2) **LiP-MS** structural-change detection. Because real experiments lack ground
truth, the paper builds **simulations with known answers** and scores methods by RMSE against the
injected truth.

**Headline findings** (the replication target):
- Roll-up is a **two-stage** process: (a) intensity **scaling** then (b) feature **aggregation**.
- Scaling methods: `rollup` (no scaling), `rrollup` (scale each peptide by the most frequently
  observed peptide), `zrollup` (scale by estimated standard error across peptides).
- Aggregation functions: `mean`, `median`, `sum`.
- **PTM**: `mean`/`median` aggregation with `rollup`/`rrollup` scaling wins; `sum` is worst;
  `zrollup` is the worst scaling (RMSE > 0.70 vs ~0.35–0.45). Robust to 0/25/50% missingness.
- **LiP**: aggregating intensities at **proteinase-K (ProK) cleavage sites** beats the traditional
  fully-tryptic-peptide analysis for detecting differentially masked (structurally changed) regions
  (median RMSE ~0.55 vs ~1.0 — a 50% error reduction).

## PTM Simulation Pipeline

1. **Protein**: amino-acid sequence of length `Q` with `Q*` potential modification sites. Paper
   restricts potential PTM sites to **serine**, denoting a modified residue with `#`.
2. **Per-site occupancy**: `M` = absolute abundance (copy number) of the sequence. For each
   modifiable site `r ∈ {1..R}`, sample `m_r ~ U(1, M)` and modify that many of the `M` copies.
   This yields a binary `M × R` matrix `A` where `A[m,r]=1` iff copy `m` is modified at site `r`.
3. **Digestion**: inject `#` into modified copies, then **trypsin** digest (cleave after K/R unless
   followed by P). Simulate **imperfect digestion** via a second pass that merges adjacent peptides;
   missed-cleavage sites are sampled with probability **proportional to adjacent peptide length**
   (shorter peptides merge more readily). Reference uses `OrgMassSpecR` for the perfect digest. The
   simulator offers two `miscleavage_model`s for this (see the `digest()` mapping below): the paper's
   fixed-proportion `"global"` model and an independent-probability `"bernoulli"` model.
4. **Observation model**: the modified-peptide abundance carries a **site effect** (how a PTM at a
   position shifts the peptide's m/z-derived intensity) and a **subject effect**:
   `abundance_{i,k,S} = p_S · 2^{β_k + Σ_{r∈S} α_r}`, i.e.
   `log2 abundance = log2 p_S + β_k + Σ α_r`, with random effects
   `β_k ~ N(0, σ_subj)` (per-subject) and `α_r ~ N(0, σ_site)` (per-site).
5. **Design**: `K` experimental groups × `N_k` subjects; group differences come from differing
   per-site occupancy → known true **log₂ fold-change** per site.
6. **Missingness** (Bramer et al. approach): default is the **labeled (TMT plex)** case — assign
   subjects to plexes, drop measurements with probability **inversely proportional to abundance**,
   continue until a global missingness target (0 / 25 / 50%) is reached.
7. **Sweep** (>600 combos): protein length (100/200), abundance `M` (100/250), global missingness
   (0/25/50%), missed-cleavage rate (0/25/50%), site/subject variance (0/1/9), 5 or 10 proteins.
8. **Scoring**: per method, compute RMSE of estimated vs true per-site log₂FC across all sites;
   report mean RMSE ± std error over replicates.

## LiP-MS Simulation Pipeline

1. **Two-stage digestion** modeling limited proteolysis under native conditions:
   - **Masking**: select `q` non-overlapping masked (folded/protected) regions `M_i` separated by
     gaps `G_j`, alternating along the sequence. Lengths drawn from Poisson:
     `|G_j| ~ Pois(λ_G)`, `|M_i| ~ Pois(λ_M)` (min masked length ≥ 5).
   - **Stage 1 — Proteinase K**: cleaves immediately after aliphatic/aromatic/hydrophobic residues
     (A, V, L, I, F, Y, W, M, P) but **cannot cleave inside a masked region**. Perfect digest then
     adjacent-merge for imperfect digestion (same scheme as PTM).
   - **Denature**: remove all masks. **Stage 2 — trypsin** digest (ideal, then merged).
2. **Group masking**: each group gets a masking **prevalence** (proportion 0–1) per region; each
   replicate randomly masks/unmasks regions until the group proportion is met → known masked vs
   unmasked distribution and thus known per-region log₂FC. Subject-level noise added to both digest
   stages.
3. **Missingness**: **label-free** variant of Bramer et al. — remove observations from individual
   subjects until a global threshold is hit.
4. **Two inference strategies compared**:
   - *Traditional tryptic*: log₂FC of fully-tryptic peptides, masked vs unmasked (median aggregation).
   - *Proposed site-level roll-up*: aggregate peptide intensities immediately **upstream and
     downstream of each ProK cleavage site** (sum/mean/median), then infer per-site.
5. **Sweep**: samples/group (5/10/25/50), protein copy number (100/500/1000), missingness
   (0/25/50%), ProK missed-cleavage (0/25/50%), mask length `λ_M`, gap `λ_G`. Fixed: protein
   length 1000, `λ_G=25`, `λ_M=50`.
6. **Scoring**: RMSE of estimated vs known per-region log₂FC, split into differential-masking
   (real change) and non-differential (1:1, no change → measures false positives).

## Code ↔ Paper Mapping (current state)

The **`protgen/` package** implements the **PTM digestion stage** (protein → distinct modified peptide
species), one module per concern — `peptide.py` (`Peptide`), `protein.py` (`MISCLEAVAGE_MODELS`,
`Protein`), `generator.py` (`AMINO_ACIDS`, `ProteinGenerator`), `truth.py` (`make_group_proteins`,
`true_site_*`); `__init__` re-exports all, so `from quantproteomicssimbox.protgen import X` is unchanged:
- `Peptide` — one distinct peptide species. Fields: `sequence` (plain residues, no mod markers),
  `abundance` (a single field: the copy count in a ground-truth `Protein`, or the simulated
  intensity in an observed `Sample` — the holding object's role decides which),
  `start_index`/`end_index` (inclusive, **absolute protein coordinates**), and `mod_sites`
  (absolute serine positions modified in this species).
- `Protein` — holds `sequence`, `abundance` (copy number `M`), `mod_table` (M×len binary serine
  modifications), `serine_map`, `digestion_sites`, `digestion_map` (per-copy kept cut positions),
  and `peptides: list[Peptide]` (the digestion output). Takes an optional `rng: np.random.Generator`
  for reproducible simulations (defaults to `np.random.default_rng()`).
- `Protein.set_quantification(abundance, miscleavage_rate=0.0, miscleavage_model="global")` — assigns
  serine mods using the paper's per-site occupancy model (Eq. 1): for each serine draw `m_r ~ U(1, M)`
  and modify that many randomly chosen copies; then digests at the given rate and model.
- `Protein.digest(miscleavage_rate=0.0, miscleavage_model="global")` — trypsin sites (K/R not before
  P). The **miscleavage-model fork** (`protgen.MISCLEAVAGE_MODELS`) chooses how missed cleavages are
  drawn per proteoform copy:
  - `"global"` (default, the paper's model) — `miscleavage_rate` is the **realized proportion**: each
    copy misses exactly `round(rate · n_sites)` sites, sampled **without replacement, weighted by
    `1/min(flanking peptide lengths)`** so shorter flanks merge more. The missed *count* is identical
    across copies → **fixed-length `digestion_map`**; copies differ only in *which* cuts are missed
    (`_digest_global`).
  - `"bernoulli"` — each cut is missed **independently with probability `rate`**, so the per-copy
    missed count is `Binomial(n_missable, rate)` → **variable-length `digestion_map`** (`_digest_bernoulli`).
  Terminal-residue cuts (zero-length flank) are never missable in either model. Then it splits each
  copy into peptide spans and aggregates identical species (same `start`/`end`/`mod_sites`) into
  `Peptide` objects whose `abundance` counts the copies that produced them. **Requires `mod_table`**
  (the guard raises `ValueError` otherwise), so call via `set_quantification`, not `digest()` directly.
  There is no `get_peptides()` method.
- `ProteinGenerator(rng=...)` — `generate_sequence(length, repeat_units=0, unit_length=8) -> str` and
  `generate_protein(...) -> Protein`; samples over the shared `AMINO_ACIDS` alphabet (a sorted tuple
  derived from `utils.amino_acids`). With `repeat_units > 0` it embeds that many copies of one shared
  clean tryptic peptide (a `unit`: no internal K/R, contains a serine, ends in K) so it digests as the
  **same peptide species at distinct loci** — forces the bottom-up position ambiguity (identical
  sequences that position-agnostic grouping merges). Threaded through `Experiment(repeat_units, …)`.

**Architecture — ground truth vs. observation are separate layers.** `Protein` is the *ground-truth*
generator: position-aware, exact, treated as immutable after `set_quantification`. `observation.py`
is the *observation* layer that derives noisy observed data from that truth without mutating it:
- `Sample` (dataclass) — one observed subject's data: `protein_sequence`, `group`, `subject`, and a
  `peptides: list[Peptide]` carrying simulated abundances. A lightweight type (no digestion
  machinery), since a run holds many samples.
- `aggregate_peptides(peptides, position_aware=False)` — **implemented**. Collapses peptides into
  observed species, summing abundance, keyed on `(sequence, relative-mod signature)`; with
  `position_aware=True` it also keys on start position (no cross-locus merge). Agnostic (default)
  mirrors that bottom-up MS cannot distinguish identical sequences at different loci; merged species
  keep the first occurrence's position fields. Returns new `Peptide`s (inputs untouched).
- `ObservationModel(var_subject, var_site, var_species, position_aware, rng)` — applies the observation
  model (Eqs. 2–5): `sample()`/`sample_group()` apply per-subject `beta_ik` and per-site `alpha_r`
  Normal effects (`var_*` are variances). **`var_species` (extension)** adds a per-peptide-species
  (backbone) log2 ionization efficiency `gamma_p` keyed on the peptide *sequence* — shared by a span's
  mod & unmod forms, so it **cancels inside a span's modified fraction** (per-peptide stoichiometry is
  invariant to it) but drives between-span abundance differences that bias the pooled ratio and that
  abundance-dependent missingness keys on. Threaded through `Experiment(var_species, …)`. `apply_missingness(samples, rate)` drops
  `round(rate · n_obs)` observations, **abundance-dependent** (prob ∝ 1/abundance, MNAR; Bramer
  et al. label-free variant) — the TMT-plex (block) variant is the remaining refinement.

**Groups, fold change, and the Experiment layer.** Group effects live in the *ground truth* (per-group
occupancy), not the noise model:
- `make_group_proteins(sequence, n_groups, abundance, …)` (protgen/truth.py) — one `Protein` per group,
  **same sequence**, **independent occupancy**. Same sequence ⇒ `alpha` shared across groups
  (site_effects keyed on sequence) and group samples pool in one `build_site_tables` call; `beta`
  stays per `(group, subject)`.
- `Protein.true_site_abundances()` (protgen/protein.py) + `true_site_log2_fold_change(a, b)`
  (protgen/truth.py) — known per-site truth: occupancy `mod_table[:, r].sum()`, and `log2(occ_b/occ_a)`.
- `group_site_change(result, group_a, group_b)` (rollups.core) — estimated per-site change from a
  `RollupResult`, branching on `result.space` (a `Space` enum): log-space values (`log2`/`logit`) →
  `mean_b - mean_a`; ratio-space (`linear`/`fraction`) → `log2(mean_b/mean_a)`. Covers both the paper's
  *modified-peptide-intensity* FC and the stoichiometry roll-up below.

**Quantification methods & scoring (`methods.py` + `Experiment.score`).** A `QuantMethod(name, roll_up,
true_change)` bundles a roll-up with the ground-truth change it should be scored against, so **one**
`Experiment.score(method, min_per_group=…)` covers every approach (intensity, pooled / per-peptide
stoichiometry, future LiP). Build them with `intensity_method(scaling, aggregation, space)` /
`stoichiometry_method(method)`, or iterate the `QUANT_METHODS` registry. `score` is a single RMSE loop:
roll up each protein, `group_site_change`, compare to `method.true_change`.

**Stoichiometry roll-up (rollups/stoichiometry.py / protgen/truth.py / methods.py).** A second quantification approach:
per-site **modified fraction** `s = (abundance modified at the site) / (abundance of all peptides
spanning the site, mod + unmod)`, a ratio of two sums (not a scale-then-aggregate over a peptide
matrix), then a selectable transform.
- Two builders (per serine, read off `Sample.protein_sequence`; a peptide contributes to a site for
  every serine in its span `[start,end]`, and to the numerator when `r ∈ mod_sites`):
  `build_stoichiometry_tables` sums **pooled** `mod`/`total` spanning abundance over samples;
  `build_peptide_fraction_tables` keeps a fraction **per peptide span** `(start,end)` covering the site
  (a `PeptideFractionSite.fractions` matrix `[spans × samples]`). Unobserved = NaN; spanned-but-unmodified = 0.
- `STOICHIOMETRY_METHODS` registry (extensible, like `SCALINGS`) — each entry is an `aggregation` ×
  `transform`. **aggregation**: `pooled` (sum mod / sum total), `pooled_pseudocount` (Haldane
  `(mod+0.5)/(total+1)`), or per-span `peptide_mean` / `peptide_median` (mean/median of per-span
  fractions — abundance-cancelling). **transform** (`FRACTION_TRANSFORMS`): `fraction` (bare, space
  `"fraction"`) or `logit` (`logit2`, space `"logit"`). Shipped names: `fraction`, `logit`,
  `logit_pseudocount`, `peptide_mean`, `peptide_median`, `peptide_mean_logit`, `peptide_median_logit`.
  `roll_up_stoichiometry(samples, method, min_per_group)` applies one → a `RollupResult` whose `space`
  drives `group_site_change`. Per-span fractions are unbiased for `s`, so `peptide_*` are scored
  against the same truth as `pooled`; with miscleavage they fragment a site into several spans, so
  `peptide_*` recover truth exactly only at miscleavage 0 (single span).
- **Truth** `Protein.true_site_stoichiometry()` = `m_r/M` and `true_site_stoichiometry_change(a, b,
  space)` (protgen/truth.py): `fraction` → `log2(s_b/s_a)`, `logit` → `logit2(s_b)−logit2(s_a)`. The
  fraction change equals the count FC `log2(m_b/m_a)` **only because both groups share `M`** (noted in
  code). Scored via `Experiment.score(stoichiometry_method(method))` (each method carries its matching
  truth — see the methods section above).
- **Position-aware requirement**: the spanning denominator is exact only under `position_aware=True`
  observation; the agnostic merge biases it (a deliberate study axis — see backlog). With `var=0` +
  position-aware, `mod/total == m_r/M` exactly (RMSE 0). The subject effect `beta` **cancels** in the
  fraction (it scales numerator and denominator equally); only the per-site `alpha` (numerator-only)
  biases it. `logit2`/`STOICH_EPS` live in `utils.py`.
- **Aggregation `space` (`"linear"` | `"log2"`, default `"log2"`).** `roll_up`/`Experiment` take a
  `space`. The paper/pmartR aggregate **log2** abundances (`edata_transform(., "log2")` first), which
  flips which aggregator is unbiased vs the occupancy truth:
  `log2` → mean/median matched, **sum inflated by peptide count** (the paper's "sum is worst");
  `linear` → **sum matched** (a total), mean/median biased by peptide count. All four combos are
  intentionally allowed (no constraint) — the "biased" ones (log2+sum, linear+mean) are exactly the
  demonstrations of the finding.
- **Presence filter `min_per_group`** (`rollups.roll_up`, default 1): pmartR-style — keep a peptide
  row only if observed in ≥ that many samples of **every** group; a site with no survivors is dropped.
  Default 1 drops "one-sided" species (present in one group only) that otherwise blow up log2-`sum`.
  Set 0 to disable (needed for the exact linear+sum recovery check).
- **Replication status (defaults: log2 + filter):** reproduces Fig S1–S3's *ranking* across 0/25/50%
  missingness — mean (~0.7) < median (~0.9) ≪ sum (~5–6) — and the "robust to missingness" behaviour
  (mean/median barely move). mean/median magnitudes ≈ the paper's ~1; **sum is still ~2× the paper's
  ~3**, a residual from our peptide-species granularity (more span/mod variants per site than the
  reference) — a finer modeling detail, see backlog.
- `experiment.py` `Experiment(n_proteins, …, repeat_units, miscleavage_rate, miscleavage_model, var_subject, var_site, var_species, missingness, position_aware, rng)` — multi-protein
  study facade: `build()` (per-group proteins) → `observe()` (one **shared** `ObservationModel`, so
  `beta` is shared across proteins, `alpha` per protein; applies `missingness` per protein) →
  `roll_up(method)` (one `RollupResult` per protein) → `score(method, min_per_group)` = RMSE of
  estimated vs true per-site change over all sites of all proteins, for any `QuantMethod`.
  Zero-variance + `intensity_method("rollup","sum","linear")` + `min_per_group=0` recovers truth
  exactly (RMSE 0).

**Not yet implemented** (replication backlog): the residual **log2-`sum` magnitude gap** (~2× the
paper's ~3 — from over-fragmented peptide species per site; would need coarser feature granularity to
fully match); the **TMT-plex (block) missingness** variant (label-free MNAR is done); the LiP ProK-site
table builder in `rollups/`; and the entire **LiP-MS** pipeline (masking, ProK digest, two-stage
digestion). The full PTM intensity + stoichiometry roll-up is implemented (all three scalings
`rollup`/`rrollup`/`zrollup`, mean/median/sum aggregations, linear/log2 `space`, the `min_per_group`
presence filter, the PTM site-table builder, `roll_up`, `group_site_change`, `apply_missingness`, and
the unified `Experiment.score(method)`); the **sweep harness** (`sweep.run_sweep`, mean RMSE ± std error
across replicate `Experiment`s over a parameter grid) is implemented — running the full >600-combo grid
to a results table is the remaining replication step. The **stoichiometry / logit-FC** analysis is also
implemented (see the Stoichiometry roll-up section above); its remaining exploration directions are in
the backlog below.

**Stoichiometry exploration** (per-site fraction = mod abundance / total spanning abundance, with
`fraction` / `logit` transforms; sweeps live in `playground.ipynb` §C1–C2 via `sweep.run_sweep`):
- **Position-aware vs position-agnostic denominator** *(C1, characterized)* — the spanning-abundance
  denominator is exact only under **position-aware** grouping; the agnostic cross-loci merge
  (`aggregate_peptides`, `position_aware=False`) collapses same-sequence peptides and mis-attributes
  their intensity across loci. Finding: the agnostic RMSE penalty **grows with repeated-peptide density**
  (`repeat_units`) and hits the **pooled** ratio harder than per-peptide. Both observation modes stay
  runnable; agnostic is a deliberate study axis, not a guard.
- **Per-peptide-span fraction aggregation** *(C2, done)* — `peptide_mean`/`peptide_median` (+ `_logit`)
  via `build_peptide_fraction_tables`. Per-peptide fractions are **provably invariant** to `var_species`
  (per-backbone ionization; it cancels in each span's ratio) and to `var_subject`, while the pooled ratio
  degrades — so per-peptide **wins where the between-span abundance effect dominates** (moderate
  miscleavage + large `var_species`, low–moderate missingness; heavy missingness starves the per-span
  average). **Within-span efficiency decision — resolved: no new term.** The within-span mod-vs-unmod
  efficiency is already the per-site effect `var_site` (alpha_r), which shifts only the modified peptides
  and so biases the fraction numerator — the one effect per-peptide does *not* cancel
  (`test_var_site_is_not_cancelled_by_per_peptide`). A *systematic* (non-zero-mean) mod efficiency would
  largely cancel in the between-group fold-change, so the mean-0 alpha already models what matters for RMSE.

**Structural refactor backlog** — *all done* (kept here as a record of the current architecture):
- ✅ **Unified `QuantMethod` registry** — `methods.py` bundles each method's `roll_up` + `true_change`;
  one `Experiment.score(method)` covers intensity / pooled-stoich / per-peptide / future LiP, and
  notebooks iterate `QUANT_METHODS` (or the factories). The old `score`/`score_stoichiometry` are gone.
- ✅ **Split `rollups.py` into the `rollups/` package** — `core` (`RollupResult`, `Space`,
  `group_site_change`, shared serine-span helpers), `intensity`, `stoichiometry`; `__init__` re-exports.
- ✅ **Renamed `group_log2_fold_change` → `group_site_change`** with a `Space` str-enum
  (`linear`/`log2`/`fraction`/`logit`).
- ✅ **DRY'd the builders** — the two stoichiometry builders share `core.iter_peptide_spans` /
  `core.serine_sites_and_keys`.

## Package Overview

- **Package name**: `quantproteomicssimbox`
- **Entry point**: `src/quantproteomicssimbox/__init__.py`
- **Core modules**: `protgen/` package (ground-truth simulation + group occupancy: `peptide`,
  `protein`, `generator`, `truth`), `observation.py`
  (observed-sample layer), `rollups/` package (`core` shared types + `group_site_change` + span
  helpers; `intensity`; `stoichiometry`), `methods.py`
  (`QuantMethod` strategy + `intensity_method`/`stoichiometry_method` factories + `QUANT_METHODS`),
  `experiment.py` (multi-protein study + unified `score(method)`), `utils.py` (shared constants +
  the `logit2` helper)
- **README.md is empty** — do not rely on it for context or requirements.
- **Type hints**: `py.typed` is present, so type-checking tools should respect it.

## Environment & Tooling

- **Dependency manager**: `uv` (not `pip`). Use `uv` for all package operations.
- **Python version**: `3.12` (pinned in `.python-version`).
- **Virtual environment**: `.venv` (managed by `uv`, ignored by git).
- **Lock file**: `uv.lock` is present.

## Developer Commands

- **Install / sync dependencies**: `uv sync`
- **Run a Python script**: `uv run python <script.py>`
- **Run a module**: `uv run python -m quantproteomicssimbox`
- **Add a dependency**: `uv add <package>`
- **Run tests**: `uv run pytest` (suite lives in `tests/`)
- **Type check** (if `mypy`/`pyright` are added): `uv run mypy src/quantproteomicssimbox` or `uv run pyright`

## Architecture & Conventions

- **Comment concisely.** Prefer short, single-line comments that explain the *why* / non-obvious
  intent — not line-by-line narration. Lean on clear names, type hints, and this file's paper-mapping
  for deeper context rather than long in-code prose.
- **Library / package** (not a CLI app). No main entrypoint script besides `__init__.py`.
- **`src/` layout** — imports should reference the package name, e.g. `from quantproteomicssimbox.protgen import ProteinGenerator`.
- **`rollups/` package** holds the roll-up families: `intensity.py` implements the paper's two-stage
  roll-up — `SCALINGS` (`rollup`/`rrollup`/`zrollup`, all implemented) × `AGGREGATIONS`
  (`mean`/`median`/`sum`), `build_site_tables`, and the `roll_up` orchestrator; `stoichiometry.py`
  holds the fraction roll-up; `core.py` the shared `RollupResult`/`Space`/`group_site_change` + span
  helpers. Extend here: add a LiP ProK-site builder, register new stoichiometry methods. New scoring
  methods go in `methods.py`.
- **`utils.py` is minimal** (`amino_acids` set) — shared constants only.
- **`protgen/` holds the core simulation logic** — `protein.py` (`Protein`: trypsin digestion +
  serine-modification occupancy), `generator.py` (`ProteinGenerator`), `peptide.py` (`Peptide`),
  `truth.py` (group construction + known per-site change).
- **No `__main__` blocks** anywhere — use `uv run python -c "from quantproteomicssimbox import ..."` for quick testing.
- **Tests live in `tests/`** at the repo root (`pytest`, a dev dependency). `tests/conftest.py` exposes
  a seeded `rng` fixture (`SEED = 12345`) so the stochastic simulation is deterministic in tests.
  `test_protgen.py` covers the trypsin rules, the `U(1, M)` occupancy model, proportion-controlled
  weighted miscleavage, and the `Peptide` quantification (`Protein.peptides`);
  `test_observation.py` covers `aggregate_peptides` (agnostic vs aware) and the `ObservationModel`;
  `test_rollups.py` covers the roll-up + `group_site_change` (intensity & stoichiometry, incl.
  per-peptide); `test_methods.py` covers the `QuantMethod` factories/registry; `test_experiment.py`
  covers the multi-protein `Experiment` (shared β / per-protein α, unified `score(method)`,
  zero-variance RMSE = 0); `test_utils.py` covers the amino-acid set + `logit2`. Run with `uv run
  pytest`. Config is in
  `[tool.pytest.ini_options]` of `pyproject.toml`.
- **No CI/CD workflows** (no `.github/workflows/` directory). No pre-commit hooks or linting/formatting configured.
- **No build or deploy scripts** — the project is currently a local library.

## Notebook

- `playground.ipynb` exists in the package directory — may contain exploratory code or examples. Treat it as scratch space, not a definitive source of truth.
