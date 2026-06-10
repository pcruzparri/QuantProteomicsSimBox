"""Roll-up methods: aggregate peptide abundances to site-level quantification.

The paper operationalizes roll-up as a two-stage process (Fig. 1):

  1. intensity SCALING   — ``rollup`` (none) | ``rrollup`` (scale to the most-observed peptide)
                           | ``zrollup`` (z-score by estimated standard error)
  2. feature AGGREGATION — ``mean`` | ``median`` | ``sum``  (over the peptides mapping to a site)

This module builds per-site peptide x sample tables from observed `Sample`s (`build_site_tables`),
then `roll_up` applies a (scaling, aggregation) pair to produce site-level quantification. Group
log2 fold-changes and RMSE scoring are downstream of this module.

Scaffold status: the three aggregation functions, the no-scaling ``rollup``, the PTM site-table
builder, and the `roll_up` orchestrator are implemented; ``rrollup``/``zrollup`` scaling are stubbed
(`NotImplementedError`). The site-table builder is PTM-specific (peptide -> modification site); the
LiP variant (peptide -> proteinase-K cut site) will be a separate builder.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .observation import Sample

# --------------------------------------------------------------------------- #
# Stage 2 — aggregation functions: peptides (rows) -> one value per sample.
# NaN-aware so missing observations are skipped; reduce over the peptide axis (axis=0).
# --------------------------------------------------------------------------- #
AggregationFunc = Callable[[np.ndarray], np.ndarray]

AGGREGATIONS: dict[str, AggregationFunc] = {
    "mean": lambda m: np.nanmean(m, axis=0),
    "median": lambda m: np.nanmedian(m, axis=0),
    "sum": lambda m: np.nansum(m, axis=0),
}


# --------------------------------------------------------------------------- #
# Stage 1 — scaling methods: peptide x sample matrix -> scaled matrix (same shape).
# --------------------------------------------------------------------------- #
ScalingFunc = Callable[[np.ndarray], np.ndarray]


def scale_rollup(matrix: np.ndarray) -> np.ndarray:
    """``rollup``: no scaling — aggregate raw abundances directly (paper)."""
    return matrix


def scale_rrollup(matrix: np.ndarray) -> np.ndarray:
    """``rrollup``: scale each peptide to the most-frequently-observed peptide (the reference) by
    the median log-ratio across samples before aggregation. Not implemented yet.
    """
    raise NotImplementedError("rrollup scaling not yet implemented")


def scale_zrollup(matrix: np.ndarray) -> np.ndarray:
    """``zrollup``: standardize each peptide by its estimated standard error across samples
    (z-score) before aggregation. Not implemented yet.
    """
    raise NotImplementedError("zrollup scaling not yet implemented")


SCALINGS: dict[str, ScalingFunc] = {
    "rollup": scale_rollup,
    "rrollup": scale_rrollup,
    "zrollup": scale_zrollup,
}


# --------------------------------------------------------------------------- #
# Aggregation structures
# --------------------------------------------------------------------------- #
@dataclass
class SiteTable:
    """Peptide abundances mapping to one site, arranged peptides x samples for roll-up.

    `matrix` is shape ``(n_peptides, n_samples)``; rows are distinct peptide species carrying the
    site (one per `peptide_keys` entry), columns follow `sample_keys`. NaN marks a peptide that is
    unobserved in that sample.
    """

    site: int
    peptide_keys: list[tuple]  # species identity per row: (sequence, sorted absolute mod sites)
    sample_keys: list[tuple[int, int]]  # (group, subject) per column
    matrix: np.ndarray


@dataclass
class RollupResult:
    """Site-level quantification from a (scaling, aggregation) roll-up.

    `values` is shape ``(n_sites, n_samples)``; row ``i`` is `sites[i]`, column ``j`` is
    `sample_keys[j]`. `space` records whether the aggregation was over linear or log2 abundances,
    so fold-change is computed correctly downstream.
    """

    sites: list[int]
    sample_keys: list[tuple[int, int]]
    values: np.ndarray
    space: str = "log2"


def build_site_tables(samples: list[Sample]) -> list[SiteTable]:
    """Group observed peptides into per-PTM-site peptide x sample matrices.

    A peptide contributes to every modification site in its ``mod_sites``. Rows are the distinct
    peptide species (keyed by sequence + sorted absolute mod sites) carrying a given site; columns
    are the samples, one per ``(group, subject)``. Assumes all `samples` come from a single protein
    (sites are absolute positions in that protein); raises otherwise. Returns one `SiteTable` per
    modified site, ordered by site position.
    """
    if not samples:
        return []
    if len({s.protein_sequence for s in samples}) > 1:
        raise ValueError("build_site_tables expects samples from a single protein")

    sample_keys = [(s.group, s.subject) for s in samples]
    n = len(sample_keys)

    # site -> {species_key -> row vector over samples (NaN where unobserved)}
    by_site: dict[int, dict[tuple, np.ndarray]] = {}
    for col, s in enumerate(samples):
        for pep in s.peptides:
            species = (pep.sequence, tuple(sorted(pep.mod_sites)))
            for site in pep.mod_sites:
                rows = by_site.setdefault(site, {})
                vec = rows.get(species)
                if vec is None:
                    vec = np.full(n, np.nan)
                    rows[species] = vec
                vec[col] = pep.abundance

    tables: list[SiteTable] = []
    for site in sorted(by_site):
        rows = by_site[site]
        keys = list(rows)
        matrix = np.vstack([rows[k] for k in keys])
        tables.append(SiteTable(site=site, peptide_keys=keys, sample_keys=sample_keys, matrix=matrix))
    return tables


def roll_up(
    samples: list[Sample],
    scaling: str = "rollup",
    aggregation: str = "median",
    space: str = "log2",
    min_per_group: int = 1,
) -> RollupResult:
    """Two-stage roll-up: scale each site's peptide matrix, then aggregate over peptides.

    Produces per-site, per-sample quantification. `scaling` in {rollup, rrollup, zrollup};
    `aggregation` in {mean, median, sum}. `space` selects whether peptides are aggregated in
    ``log2`` (the paper's / pmartR convention — abundances are log2-transformed first) or ``linear``
    space. The two spaces flip which aggregator is unbiased against the occupancy truth, so the
    matched pairs are:

        space=log2   -> mean / median are unbiased; sum is inflated by the peptide count
        space=linear -> sum is unbiased (a total); mean / median are biased by the peptide count

    All four combinations are intentionally allowed (the "biased" ones — log2+sum, linear+mean —
    are exactly what demonstrate the paper's finding).

    `min_per_group` is a pmartR-style presence filter: a peptide row is kept only if observed in at
    least that many samples of **every** group; a site with no surviving peptides is dropped (not
    quantifiable). Default 1 drops "one-sided" species (present in one group only), which otherwise
    blow up log2-`sum`; set 0 to disable filtering. Group log2 fold-changes / RMSE are computed
    downstream via ``group_log2_fold_change``, which reads ``space`` off the result.
    """
    if scaling not in SCALINGS:
        raise ValueError(f"unknown scaling {scaling!r}; choose from {sorted(SCALINGS)}")
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"unknown aggregation {aggregation!r}; choose from {sorted(AGGREGATIONS)}")
    if space not in ("linear", "log2"):
        raise ValueError(f"unknown space {space!r}; choose from ['linear', 'log2']")
    scale = SCALINGS[scaling]
    aggregate = AGGREGATIONS[aggregation]

    tables = build_site_tables(samples)
    sample_keys = tables[0].sample_keys if tables else []
    sites: list[int] = []
    rows: list[np.ndarray] = []
    with warnings.catch_warnings():
        # A site unobserved in a sample yields an all-NaN column -> NaN (intended "missing"); the
        # nan-reducers warn on that expected case, so silence just those messages.
        warnings.filterwarnings("ignore", "(All-NaN slice|Mean of empty slice)", RuntimeWarning)
        for t in tables:
            matrix = t.matrix[_presence_mask(t.matrix, sample_keys, min_per_group)]
            if matrix.shape[0] == 0:
                continue  # no peptide passes the presence filter -> site not quantifiable
            sites.append(t.site)
            # log2-transform abundances before scaling/aggregation in log space; NaN (missing) and
            # the strictly-positive counts both pass through cleanly.
            rows.append(aggregate(scale(_to_space(matrix, space))))
    values = np.vstack(rows) if rows else np.empty((0, len(sample_keys)))
    return RollupResult(sites=sites, sample_keys=sample_keys, values=values, space=space)


def _to_space(matrix: np.ndarray, space: str) -> np.ndarray:
    return np.log2(matrix) if space == "log2" else matrix


def _presence_mask(matrix: np.ndarray, sample_keys: list[tuple[int, int]], min_per_group: int) -> np.ndarray:
    """Boolean row mask: keep peptides observed in >= min_per_group samples of every group."""
    if min_per_group <= 0:
        return np.ones(matrix.shape[0], dtype=bool)
    observed = ~np.isnan(matrix)
    mask = np.ones(matrix.shape[0], dtype=bool)
    for group in {g for g, _ in sample_keys}:
        cols = [j for j, (g, _s) in enumerate(sample_keys) if g == group]
        mask &= observed[:, cols].sum(axis=1) >= min_per_group
    return mask


def group_log2_fold_change(result: RollupResult, group_a: int, group_b: int) -> dict[int, float]:
    """Estimated per-site log2 fold-change (group B vs A) from rolled-up site values.

    Takes the nan-aware mean of each site's quantification over each group's samples. If the roll-up
    was in ``log2`` space the values are already log2 abundances, so FC = ``mean_b - mean_a``; in
    ``linear`` space FC = ``log2(mean_b / mean_a)``. Sites missing from a group yield nan/inf.

    This is the paper's modified-peptide-intensity FC; a stoichiometry variant (modified fraction
    over all peptides spanning the site, logit-transformed) would be a sibling helper.
    """
    cols_a = [j for j, (g, _s) in enumerate(result.sample_keys) if g == group_a]
    cols_b = [j for j, (g, _s) in enumerate(result.sample_keys) if g == group_b]
    fold_changes: dict[int, float] = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        for i, site in enumerate(result.sites):
            mean_a = np.nanmean(result.values[i, cols_a]) if cols_a else np.nan
            mean_b = np.nanmean(result.values[i, cols_b]) if cols_b else np.nan
            if result.space == "log2":
                fold_changes[site] = float(mean_b - mean_a)
            else:
                fold_changes[site] = float(np.log2(mean_b / mean_a))
    return fold_changes
