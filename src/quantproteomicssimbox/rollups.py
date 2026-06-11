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

import bisect
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .observation import Sample
from .utils import logit2

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
    """Estimated per-site between-group change (group B vs A) from rolled-up site values.

    Takes the nan-aware mean of each site's quantification over each group's samples, then forms the
    change according to the result's ``space``:
      - ``log2`` / ``logit`` (already a log space) -> ``mean_b - mean_a`` (a difference)
      - ``linear`` / ``fraction``                  -> ``log2(mean_b / mean_a)`` (a log2 ratio)
    Sites missing from a group yield nan/inf.

    Covers both the paper's modified-peptide-intensity FC (``linear``/``log2``) and the stoichiometry
    roll-up (``fraction``/``logit``); the value's meaning (log2 fold-change vs log-odds change) follows
    the space.
    """
    cols_a = [j for j, (g, _s) in enumerate(result.sample_keys) if g == group_a]
    cols_b = [j for j, (g, _s) in enumerate(result.sample_keys) if g == group_b]
    fold_changes: dict[int, float] = {}
    with warnings.catch_warnings():
        # Degenerate sites (a group mean of 0 in fraction space, or all-NaN) yield inf/nan by design;
        # the scorer filters those, so silence the expected reducer/division warnings.
        warnings.filterwarnings(
            "ignore", "(Mean of empty slice|divide by zero|invalid value)", RuntimeWarning
        )
        for i, site in enumerate(result.sites):
            mean_a = np.nanmean(result.values[i, cols_a]) if cols_a else np.nan
            mean_b = np.nanmean(result.values[i, cols_b]) if cols_b else np.nan
            if result.space in ("log2", "logit"):
                fold_changes[site] = float(mean_b - mean_a)
            else:
                fold_changes[site] = float(np.log2(mean_b / mean_a))
    return fold_changes


# --------------------------------------------------------------------------- #
# Stoichiometry roll-up: per-site modified fraction = (abundance modified at the site) / (abundance of
# all peptides spanning the site, mod + unmod). A separate path from the intensity roll-up above. The
# peptides covering a site can be combined two ways — pooled (one ratio of summed abundances) or
# per-peptide-span (a fraction per span, then mean/median) — then a fraction/logit transform.
# NOTE: the spanning denominator is exact only under position-aware observation; the agnostic merge
# biases it (a deliberate study axis — see AGENTS.md backlog).
# --------------------------------------------------------------------------- #
# Two ways to combine the peptides covering a site into one fraction per sample:
#   - "pooled" (/ "pooled_pseudocount"): sum all modified / sum all spanning abundance, then divide.
#   - "peptide_mean" / "peptide_median": a fraction *per peptide span*, then mean/median over spans.
#     The per-span fraction cancels that span's own abundance/ionization, so between-span abundance
#     differences (and missingness that drops whole spans) bias it less than the pooled ratio.
# A transform then maps the per-sample fraction to the site value ("fraction" -> bare; "logit" -> log-odds).
FractionTransform = Callable[[np.ndarray], np.ndarray]

# name -> (transform on a per-sample fraction array, the FC space it implies)
FRACTION_TRANSFORMS: dict[str, tuple[FractionTransform, str]] = {
    "fraction": (lambda f: f, "fraction"),
    "logit": (lambda f: logit2(f), "logit"),
}


@dataclass(frozen=True)
class StoichiometryMethod:
    """A selectable stoichiometry roll-up method (registered in ``STOICHIOMETRY_METHODS``).

    `aggregation` is how the peptides covering a site reduce to one fraction per sample
    (``pooled`` | ``pooled_pseudocount`` | ``peptide_mean`` | ``peptide_median``); `transform` names
    the ``FRACTION_TRANSFORMS`` entry applied to that fraction. `space` (derived from the transform)
    tells ``group_log2_fold_change`` how to form the between-group change ("fraction" -> log2 ratio;
    "logit" -> difference of log-odds).
    """

    aggregation: str
    transform: str

    @property
    def space(self) -> str:
        return FRACTION_TRANSFORMS[self.transform][1]


STOICHIOMETRY_METHODS: dict[str, StoichiometryMethod] = {
    # pooled ratio (sum modified / sum spanning) — the canonical site stoichiometry
    "fraction": StoichiometryMethod("pooled", "fraction"),
    "logit": StoichiometryMethod("pooled", "logit"),
    "logit_pseudocount": StoichiometryMethod("pooled_pseudocount", "logit"),
    # per-peptide-span fractions, aggregated over spans (abundance-cancelling)
    "peptide_mean": StoichiometryMethod("peptide_mean", "fraction"),
    "peptide_median": StoichiometryMethod("peptide_median", "fraction"),
    "peptide_mean_logit": StoichiometryMethod("peptide_mean", "logit"),
    "peptide_median_logit": StoichiometryMethod("peptide_median", "logit"),
}


@dataclass
class StoichiometrySite:
    """Per-site pooled modified vs total spanning abundance over samples (the "pooled" aggregation).

    `mod` and `total` are shape ``(n_samples,)``: `mod[j]` is the abundance modified at `site` in
    sample `j`, `total[j]` the abundance of all peptides spanning `site`. Both are NaN where no peptide
    spans the site in that sample (unobserved); a site spanned but never modified is a real 0.
    """

    site: int
    sample_keys: list[tuple[int, int]]
    mod: np.ndarray
    total: np.ndarray


@dataclass
class PeptideFractionSite:
    """Per-site modified fraction of each peptide *span* covering the site, over samples.

    `fractions` is shape ``(n_spans, n_samples)``: row `i` is span `span_keys[i]` = ``(start, end)``,
    entry ``[i, j]`` is (abundance modified at `site`) / (total abundance) of that span in sample `j`,
    or NaN if the span is unobserved there. The per-peptide aggregations reduce over the span axis.
    """

    site: int
    sample_keys: list[tuple[int, int]]
    span_keys: list[tuple[int, int]]
    fractions: np.ndarray


def build_stoichiometry_tables(samples: list[Sample]) -> list[StoichiometrySite]:
    """Per serine site, sum modified and total spanning abundance across each sample.

    A peptide adds its abundance to `total[site]` for every serine `site` in its span
    ``[start_index, end_index]``, and additionally to `mod[site]` when ``site in pep.mod_sites``.
    Serine sites are read off the (single) protein sequence. A ``(site, sample)`` with no spanning
    peptide is NaN (unobserved); spanned-but-unmodified is a real 0. Assumes all `samples` come from a
    single protein; raises otherwise. Returns one `StoichiometrySite` per serine observed in >= 1
    sample, ordered by position.
    """
    if not samples:
        return []
    if len({s.protein_sequence for s in samples}) > 1:
        raise ValueError("build_stoichiometry_tables expects samples from a single protein")

    serines = [i for i, aa in enumerate(samples[0].protein_sequence) if aa == "S"]
    if not serines:
        return []
    sample_keys = [(s.group, s.subject) for s in samples]
    n = len(sample_keys)

    row_of = {r: k for k, r in enumerate(serines)}
    mod = np.zeros((len(serines), n))
    total = np.zeros((len(serines), n))
    for col, s in enumerate(samples):
        for pep in s.peptides:
            if pep.start_index is None or pep.end_index is None:
                continue  # no span info -> cannot attribute to sites
            lo = bisect.bisect_left(serines, pep.start_index)
            hi = bisect.bisect_right(serines, pep.end_index)
            mod_set = set(pep.mod_sites)
            for r in serines[lo:hi]:
                total[row_of[r], col] += pep.abundance
                if r in mod_set:
                    mod[row_of[r], col] += pep.abundance

    unobserved = total == 0  # exactly 0 -> no peptide spanned this site in this sample
    mod[unobserved] = np.nan
    total[unobserved] = np.nan

    tables: list[StoichiometrySite] = []
    for row, r in enumerate(serines):
        if np.isnan(total[row]).all():
            continue  # site never spanned in any sample
        tables.append(
            StoichiometrySite(site=r, sample_keys=sample_keys, mod=mod[row], total=total[row])
        )
    return tables


def build_peptide_fraction_tables(samples: list[Sample]) -> list[PeptideFractionSite]:
    """Per serine site, the modified fraction of each distinct peptide span covering it, over samples.

    Peptides are grouped by span ``(start_index, end_index)``; within a span, a site's fraction is
    (abundance of species modified at the site) / (abundance of all species of that span). A
    ``(span, sample)`` with no observation is NaN. Same single-protein assumption as
    ``build_stoichiometry_tables``; returns one `PeptideFractionSite` per serine spanned in >= 1
    sample, ordered by position (spans ordered by ``(start, end)``).
    """
    if not samples:
        return []
    if len({s.protein_sequence for s in samples}) > 1:
        raise ValueError("build_peptide_fraction_tables expects samples from a single protein")

    serines = [i for i, aa in enumerate(samples[0].protein_sequence) if aa == "S"]
    if not serines:
        return []
    sample_keys = [(s.group, s.subject) for s in samples]
    n = len(sample_keys)

    # site -> {span -> (num, denom) arrays over samples}
    by_site: dict[int, dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]] = {}
    for col, s in enumerate(samples):
        for pep in s.peptides:
            if pep.start_index is None or pep.end_index is None:
                continue  # no span info -> cannot attribute to sites
            span = (pep.start_index, pep.end_index)
            lo = bisect.bisect_left(serines, pep.start_index)
            hi = bisect.bisect_right(serines, pep.end_index)
            mod_set = set(pep.mod_sites)
            for r in serines[lo:hi]:
                spans = by_site.setdefault(r, {})
                cell = spans.get(span)
                if cell is None:
                    cell = (np.zeros(n), np.zeros(n))
                    spans[span] = cell
                num, denom = cell
                denom[col] += pep.abundance
                if r in mod_set:
                    num[col] += pep.abundance

    tables: list[PeptideFractionSite] = []
    for r in sorted(by_site):
        span_keys = sorted(by_site[r])
        fractions = np.empty((len(span_keys), n))
        for row, span in enumerate(span_keys):
            num, denom = by_site[r][span]
            fractions[row] = np.divide(num, denom, out=np.full(n, np.nan), where=denom > 0)
        tables.append(
            PeptideFractionSite(site=r, sample_keys=sample_keys, span_keys=span_keys, fractions=fractions)
        )
    return tables


def roll_up_stoichiometry(
    samples: list[Sample], method: str = "fraction", min_per_group: int = 1
) -> RollupResult:
    """Stoichiometry roll-up: per-site modified fraction, combined and transformed per the `method`.

    `method` selects a ``STOICHIOMETRY_METHODS`` entry. Its `aggregation` decides how peptides combine
    into one fraction per sample (``pooled`` ratio, Haldane ``pooled_pseudocount``, or per-span
    ``peptide_mean`` / ``peptide_median``); its `transform` (`fraction` / `logit`) then maps that
    fraction to the site value, and `space` drives ``group_log2_fold_change``. `min_per_group` keeps a
    site only if observed (some peptide spans it) in >= that many samples of **every** group; 0 disables.

    The spanning denominator is exact only under position-aware observation; see the module note.
    """
    if method not in STOICHIOMETRY_METHODS:
        raise ValueError(
            f"unknown stoichiometry method {method!r}; choose from {sorted(STOICHIOMETRY_METHODS)}"
        )
    spec = STOICHIOMETRY_METHODS[method]
    transform, space = FRACTION_TRANSFORMS[spec.transform]
    sample_keys = [(s.group, s.subject) for s in samples]
    sites: list[int] = []
    rows: list[np.ndarray] = []
    with warnings.catch_warnings():
        # Unobserved (NaN) entries flow through reducers/divisions; their warnings are the intended case.
        warnings.filterwarnings(
            "ignore", "(invalid value|divide by zero|Mean of empty slice|All-NaN slice)", RuntimeWarning
        )
        if spec.aggregation in ("pooled", "pooled_pseudocount"):
            for t in build_stoichiometry_tables(samples):
                if not _site_present(~np.isnan(t.total), sample_keys, min_per_group):
                    continue
                if spec.aggregation == "pooled_pseudocount":
                    fraction = (t.mod + 0.5) / (t.total + 1.0)  # Haldane continuity correction
                else:
                    fraction = t.mod / t.total
                sites.append(t.site)
                rows.append(transform(fraction))
        else:  # per-peptide-span aggregation
            reduce = np.nanmean if spec.aggregation == "peptide_mean" else np.nanmedian
            for t in build_peptide_fraction_tables(samples):
                observed = (~np.isnan(t.fractions)).any(axis=0)  # any span seen -> site observed
                if not _site_present(observed, sample_keys, min_per_group):
                    continue
                sites.append(t.site)
                rows.append(transform(reduce(t.fractions, axis=0)))
    values = np.vstack(rows) if rows else np.empty((0, len(sample_keys)))
    return RollupResult(sites=sites, sample_keys=sample_keys, values=values, space=space)


def _site_present(observed: np.ndarray, sample_keys: list[tuple[int, int]], min_per_group: int) -> bool:
    """True if the site is observed (``observed[j]`` True) in >= min_per_group samples of every group."""
    if min_per_group <= 0:
        return True
    for group in {g for g, _ in sample_keys}:
        cols = [j for j, (g2, _s) in enumerate(sample_keys) if g2 == group]
        if int(observed[cols].sum()) < min_per_group:
            return False
    return True
