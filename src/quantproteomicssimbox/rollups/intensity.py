"""Intensity roll-up: the paper's two-stage modified-peptide-intensity aggregation (Fig. 1).

  1. intensity SCALING   — ``rollup`` (none) | ``rrollup`` (scale to the most-observed peptide)
                           | ``zrollup`` (z-score by estimated standard error)
  2. feature AGGREGATION — ``mean`` | ``median`` | ``sum``  (over the peptides mapping to a site)

`build_site_tables` builds per-site peptide x sample matrices from observed `Sample`s; `roll_up`
applies a (scaling, aggregation) pair in linear or log2 space. All three scalings are implemented.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..observation import Sample
from .core import RollupResult, Space

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
    """``rrollup`` (pmartR): re-reference every peptide to the most-frequently-observed peptide.

    The reference is the row (peptide) with the fewest missing samples. Each peptide is shifted by the
    **median, across samples, of `(reference - peptide)`** — its median log-ratio to the reference in
    log2 space — bringing all peptides onto the reference's level before aggregation. NaN (missing)
    entries are ignored in the medians and pass through; a peptide sharing no observed sample with the
    reference is left unshifted. Intended for the log2-space roll-up.
    """
    if matrix.shape[0] <= 1:
        return matrix  # 0/1 peptides -> nothing to re-reference
    observed = ~np.isnan(matrix)
    reference = matrix[int(np.argmax(observed.sum(axis=1)))]  # most-observed peptide
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
        offsets = np.nanmedian(reference - matrix, axis=1)  # per-peptide median log-ratio to reference
    offsets = np.nan_to_num(offsets, nan=0.0)  # no shared sample with reference -> no shift
    return matrix + offsets[:, None]


def scale_zrollup(matrix: np.ndarray) -> np.ndarray:
    """``zrollup`` (pmartR): standardize each peptide to a z-score across samples before aggregation.

    For each peptide (row) ``z = (x - mean) / sd`` with mean/sd over its observed samples (ddof=1).
    This strips each peptide's level *and* scale, so the aggregated value is in pooled-sd units rather
    than log2 abundance — which is why `zrollup` is the paper's worst scaling (the magnitude no longer
    matches the occupancy log2FC). Peptides with <2 observations (sd undefined) become NaN and are
    skipped by the nan-aware aggregation.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "(Degrees of freedom|invalid value|Mean of empty slice)", RuntimeWarning
        )
        mean = np.nanmean(matrix, axis=1, keepdims=True)
        sd = np.nanstd(matrix, axis=1, ddof=1, keepdims=True)
        return (matrix - mean) / sd


SCALINGS: dict[str, ScalingFunc] = {
    "rollup": scale_rollup,
    "rrollup": scale_rrollup,
    "zrollup": scale_zrollup,
}


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
    space: str = Space.LOG2,
    min_per_group: int = 1,
) -> RollupResult:
    """Two-stage roll-up: scale each site's peptide matrix, then aggregate over peptides.

    Produces per-site, per-sample quantification. `scaling` in {rollup, rrollup, zrollup};
    `aggregation` in {mean, median, sum}. `space` selects whether peptides are aggregated in
    ``log2`` (the paper's / pmartR convention — abundances are log2-transformed first) or ``linear``
    space. The two spaces flip which aggregator is unbiased against the occupancy truth:

        space=log2   -> mean / median are unbiased; sum is inflated by the peptide count
        space=linear -> sum is unbiased (a total); mean / median are biased by the peptide count

    All four combinations are intentionally allowed (the "biased" ones — log2+sum, linear+mean —
    are exactly what demonstrate the paper's finding).

    `min_per_group` is a pmartR-style presence filter: a peptide row is kept only if observed in at
    least that many samples of **every** group; a site with no surviving peptides is dropped (not
    quantifiable). Default 1 drops "one-sided" species (present in one group only), which otherwise
    blow up log2-`sum`; set 0 to disable filtering. The change / RMSE are computed downstream via
    ``group_site_change``, which reads ``space`` off the result.
    """
    if scaling not in SCALINGS:
        raise ValueError(f"unknown scaling {scaling!r}; choose from {sorted(SCALINGS)}")
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"unknown aggregation {aggregation!r}; choose from {sorted(AGGREGATIONS)}")
    if space not in (Space.LINEAR, Space.LOG2):
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
    return np.log2(matrix) if space == Space.LOG2 else matrix


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
