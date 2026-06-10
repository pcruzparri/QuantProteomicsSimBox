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
   (shorter peptides merge more readily). Reference uses `OrgMassSpecR` for the perfect digest.
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

`protgen.py` implements the **PTM digestion stage** (protein → distinct modified peptide species):
- `Peptide` — one distinct peptide species. Fields: `sequence` (plain residues, no mod markers),
  `abundance` (a single field: the copy count in a ground-truth `Protein`, or the simulated
  intensity in an observed `Sample` — the holding object's role decides which),
  `start_index`/`end_index` (inclusive, **absolute protein coordinates**), and `mod_sites`
  (absolute serine positions modified in this species).
- `Protein` — holds `sequence`, `abundance` (copy number `M`), `mod_table` (M×len binary serine
  modifications), `serine_map`, `digestion_sites`, `digestion_map` (per-copy kept cut positions),
  and `peptides: list[Peptide]` (the digestion output). Takes an optional `rng: np.random.Generator`
  for reproducible simulations (defaults to `np.random.default_rng()`).
- `Protein.set_quantification(abundance, miscleavage_rate=0.0)` — assigns serine mods using the
  paper's per-site occupancy model (Eq. 1): for each serine draw `m_r ~ U(1, M)` and modify that
  many randomly chosen copies; then digests at the given rate.
- `Protein.digest(miscleavage_rate=0.0)` — trypsin sites (K/R not before P). `miscleavage_rate` is
  the **realized proportion of missed cleavages**: per proteoform copy it samples
  `round(rate · n_sites)` sites to miss, **without replacement, weighted by `1/min(flanking peptide
  lengths)`** so shorter flanking peptides are merged more often (paper's model). Terminal-residue
  cuts (zero-length flank) are excluded as no-ops. Then it splits each copy into peptide spans and
  aggregates identical species (same `start`/`end`/`mod_sites`) into `Peptide` objects whose
  `abundance` counts the copies that produced them. **Requires `mod_table`** (the guard raises `ValueError` otherwise), so
  call via `set_quantification`, not `digest()` directly. There is no `get_peptides()` method.
- `ProteinGenerator(rng=...)` — `generate_sequence(length) -> str` and
  `generate_protein(length) -> Protein`; samples over the shared `AMINO_ACIDS` alphabet (a sorted
  tuple derived from `utils.amino_acids`).

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
- `ObservationModel(var_subject, var_site, position_aware, rng)` — applies the observation model
  (Eqs. 2–5): `sample()`/`sample_group()` are **implemented** (per-subject `beta_ik` and per-site
  `alpha_r` Normal effects, with `var_*` interpreted as variances). Only `apply_missingness()` is
  still stubbed (`NotImplementedError`).

**Not yet implemented** (replication backlog): TMT & label-free missingness
(`ObservationModel.apply_missingness`); experimental groups & true log₂FC; the `rrollup`/`zrollup`
scaling methods and the LiP ProK-site table builder in `rollups.py` (the `rollup` scaling, the
mean/median/sum aggregations, the PTM site-table builder, and the `roll_up` orchestrator are
implemented); the entire **LiP-MS** pipeline (masking, ProK digest, two-stage digestion); and RMSE
evaluation/sweep harness.

## Package Overview

- **Package name**: `quantproteomicssimbox`
- **Entry point**: `src/quantproteomicssimbox/__init__.py`
- **Core modules**: `protgen.py` (ground-truth simulation), `observation.py` (observed-sample layer),
  `rollups.py` (peptide→site roll-up: two-stage scaling × aggregation; `rrollup`/`zrollup` stubbed),
  `utils.py` (shared constants)
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
- **`rollups.py`** scaffolds the paper's two-stage roll-up: `SCALINGS` (`rollup` implemented;
  `rrollup`/`zrollup` raise `NotImplementedError`) × `AGGREGATIONS` (`mean`/`median`/`sum`),
  `build_site_tables` (PTM peptide→site matrices), and the `roll_up` orchestrator. Extend new
  roll-up logic here: implement `scale_rrollup`/`scale_zrollup`, and add a LiP ProK-site builder.
- **`utils.py` is minimal** (`amino_acids` set) — shared constants only.
- **`protgen.py` contains the core simulation logic** (`Protein` class, `ProteinGenerator` class, trypsin digestion simulation, serine modification assignment).
- **No `__main__` blocks** in `protgen.py` or `__init__.py` — use `uv run python -c "from quantproteomicssimbox import ..."` for quick testing.
- **Tests live in `tests/`** at the repo root (`pytest`, a dev dependency). `tests/conftest.py` exposes
  a seeded `rng` fixture (`SEED = 12345`) so the stochastic simulation is deterministic in tests.
  `test_protgen.py` covers the trypsin rules, the `U(1, M)` occupancy model, proportion-controlled
  weighted miscleavage, and the `Peptide` quantification (`Protein.peptides`);
  `test_observation.py` covers `aggregate_peptides` (agnostic vs aware) and the `Sample`/
  `ObservationModel` scaffold; `test_utils.py` covers the amino-acid set. Run
  with `uv run pytest`. Config is in `[tool.pytest.ini_options]` of `pyproject.toml`.
- **No CI/CD workflows** (no `.github/workflows/` directory). No pre-commit hooks or linting/formatting configured.
- **No build or deploy scripts** — the project is currently a local library.

## Notebook

- `playground.ipynb` exists in the package directory — may contain exploratory code or examples. Treat it as scratch space, not a definitive source of truth.
