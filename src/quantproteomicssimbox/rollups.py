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
    `sample_keys[j]`.
    """

    sites: list[int]
    sample_keys: list[tuple[int, int]]
    values: np.ndarray


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
) -> RollupResult:
    """Two-stage roll-up: scale each site's peptide matrix, then aggregate over peptides.

    Produces per-site, per-sample quantification. `scaling` in {rollup, rrollup, zrollup};
    `aggregation` in {mean, median, sum}. Group-level log2 fold-changes / RMSE scoring are computed
    downstream from the returned `RollupResult`.
    """
    if scaling not in SCALINGS:
        raise ValueError(f"unknown scaling {scaling!r}; choose from {sorted(SCALINGS)}")
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"unknown aggregation {aggregation!r}; choose from {sorted(AGGREGATIONS)}")
    scale = SCALINGS[scaling]
    aggregate = AGGREGATIONS[aggregation]

    tables = build_site_tables(samples)
    sample_keys = tables[0].sample_keys if tables else []
    sites = [t.site for t in tables]
    with warnings.catch_warnings():
        # A site unobserved in a sample yields an all-NaN column -> NaN (intended "missing"); the
        # nan-reducers warn on that expected case, so silence just those messages.
        warnings.filterwarnings("ignore", "(All-NaN slice|Mean of empty slice)", RuntimeWarning)
        values = (
            np.vstack([aggregate(scale(t.matrix)) for t in tables])
            if tables
            else np.empty((0, len(sample_keys)))
        )
    return RollupResult(sites=sites, sample_keys=sample_keys, values=values)
