"""Stoichiometry roll-up: per-site modified fraction = (abundance modified at the site) / (abundance
of all peptides spanning the site, mod + unmod).

The peptides covering a site can be combined two ways — **pooled** (one ratio of summed abundances)
or **per-peptide-span** (a fraction per span, then mean/median) — then a ``fraction``/``logit``
transform maps the fraction to the site value. NOTE: the spanning denominator is exact only under
position-aware observation; the agnostic merge biases it (a deliberate study axis — see AGENTS.md).
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..observation import Sample
from ..utils import logit2
from .core import RollupResult, Space, iter_peptide_spans, serine_sites_and_keys, site_present

# Two ways to combine the peptides covering a site into one fraction per sample:
#   - "pooled" (/ "pooled_pseudocount"): sum all modified / sum all spanning abundance, then divide.
#   - "peptide_mean" / "peptide_median": a fraction *per peptide span*, then mean/median over spans.
#     The per-span fraction cancels that span's own abundance/ionization, so between-span abundance
#     differences (and missingness that drops whole spans) bias it less than the pooled ratio.
# A transform then maps the per-sample fraction to the site value ("fraction" -> bare; "logit" -> log-odds).
FractionTransform = Callable[[np.ndarray], np.ndarray]

# name -> (transform on a per-sample fraction array, the FC space it implies)
FRACTION_TRANSFORMS: dict[str, tuple[FractionTransform, Space]] = {
    "fraction": (lambda f: f, Space.FRACTION),
    "logit": (lambda f: logit2(f), Space.LOGIT),
}


@dataclass(frozen=True)
class StoichiometryMethod:
    """A selectable stoichiometry roll-up method (registered in ``STOICHIOMETRY_METHODS``).

    `aggregation` is how the peptides covering a site reduce to one fraction per sample
    (``pooled`` | ``pooled_pseudocount`` | ``peptide_mean`` | ``peptide_median``); `transform` names
    the ``FRACTION_TRANSFORMS`` entry applied to that fraction. `space` (derived from the transform)
    tells ``group_site_change`` how to form the between-group change.
    """

    aggregation: str
    transform: str

    @property
    def space(self) -> Space:
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
    A ``(site, sample)`` with no spanning peptide is NaN (unobserved); spanned-but-unmodified is a
    real 0. Single-protein only (raises otherwise). One `StoichiometrySite` per serine observed in
    >= 1 sample, ordered by position.
    """
    if not samples:
        return []
    serines, sample_keys = serine_sites_and_keys(samples)
    if not serines:
        return []
    n = len(sample_keys)

    row_of = {r: k for k, r in enumerate(serines)}
    mod = np.zeros((len(serines), n))
    total = np.zeros((len(serines), n))
    for col, pep, span_serines, mod_set in iter_peptide_spans(samples, serines):
        for r in span_serines:
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
    ``(span, sample)`` with no observation is NaN. Single-protein only; one `PeptideFractionSite` per
    serine spanned in >= 1 sample, ordered by position (spans ordered by ``(start, end)``).
    """
    if not samples:
        return []
    serines, sample_keys = serine_sites_and_keys(samples)
    if not serines:
        return []
    n = len(sample_keys)

    # site -> {span -> (num, denom) arrays over samples}
    by_site: dict[int, dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]] = {}
    for col, pep, span_serines, mod_set in iter_peptide_spans(samples, serines):
        span = (pep.start_index, pep.end_index)
        for r in span_serines:
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
    fraction to the site value, and `space` drives ``group_site_change``. `min_per_group` keeps a site
    only if observed (some peptide spans it) in >= that many samples of **every** group; 0 disables.

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
                if not site_present(~np.isnan(t.total), sample_keys, min_per_group):
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
                if not site_present(observed, sample_keys, min_per_group):
                    continue
                sites.append(t.site)
                rows.append(transform(reduce(t.fractions, axis=0)))
    values = np.vstack(rows) if rows else np.empty((0, len(sample_keys)))
    return RollupResult(sites=sites, sample_keys=sample_keys, values=values, space=space)
