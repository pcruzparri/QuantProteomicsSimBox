"""Shared roll-up core: the result type, the aggregation space, the group-change estimator, and the
samples x peptides x serine-span indexing reused by the stoichiometry builders.

A "roll-up" turns observed `Sample`s into per-site, per-sample quantification (`RollupResult`); the
intensity and stoichiometry families live in sibling modules. ``group_site_change`` then collapses a
result into the estimated per-site between-group change.
"""

import bisect
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..observation import Sample
from ..protgen import Peptide


class Space(str, Enum):
    """The space a roll-up's site values live in — fixes how ``group_site_change`` forms the change.

    ``log2`` / ``logit`` are already log spaces, so the change is a difference of group means;
    ``linear`` / ``fraction`` are ratio spaces, so the change is the log2 of the group-mean ratio.
    A ``str`` enum, so members compare equal to their plain-string values.
    """

    LINEAR = "linear"
    LOG2 = "log2"
    FRACTION = "fraction"
    LOGIT = "logit"


# Spaces whose values are already logarithmic -> the between-group change is a difference.
_LOG_SPACES = frozenset({Space.LOG2, Space.LOGIT})


@dataclass
class RollupResult:
    """Site-level quantification from a roll-up.

    `values` is shape ``(n_sites, n_samples)``; row ``i`` is `sites[i]`, column ``j`` is
    `sample_keys[j]`. `space` records the value space (see ``Space``) so the between-group change is
    computed correctly downstream.
    """

    sites: list[int]
    sample_keys: list[tuple[int, int]]
    values: np.ndarray
    space: str = Space.LOG2


def group_site_change(result: RollupResult, group_a: int, group_b: int) -> dict[int, float]:
    """Estimated per-site between-group change (group B vs A) from rolled-up site values.

    Takes the nan-aware mean of each site's quantification over each group's samples, then forms the
    change according to the result's ``space``:
      - ``log2`` / ``logit`` (already a log space) -> ``mean_b - mean_a`` (a difference)
      - ``linear`` / ``fraction``                  -> ``log2(mean_b / mean_a)`` (a log2 ratio)
    Sites missing from a group yield nan/inf.

    Covers both the intensity FC (``linear``/``log2``) and the stoichiometry roll-up
    (``fraction``/``logit``); the value's meaning (log2 fold-change vs log-odds change) follows the space.
    """
    cols_a = [j for j, (g, _s) in enumerate(result.sample_keys) if g == group_a]
    cols_b = [j for j, (g, _s) in enumerate(result.sample_keys) if g == group_b]
    changes: dict[int, float] = {}
    with warnings.catch_warnings():
        # Degenerate sites (a group mean of 0 in a ratio space, or all-NaN) yield inf/nan by design;
        # the scorer filters those, so silence the expected reducer/division warnings.
        warnings.filterwarnings(
            "ignore", "(Mean of empty slice|divide by zero|invalid value)", RuntimeWarning
        )
        for i, site in enumerate(result.sites):
            mean_a = np.nanmean(result.values[i, cols_a]) if cols_a else np.nan
            mean_b = np.nanmean(result.values[i, cols_b]) if cols_b else np.nan
            if result.space in _LOG_SPACES:
                changes[site] = float(mean_b - mean_a)
            else:
                changes[site] = float(np.log2(mean_b / mean_a))
    return changes


def site_present(observed: np.ndarray, sample_keys: list[tuple[int, int]], min_per_group: int) -> bool:
    """True if the site is observed (``observed[j]`` True) in >= min_per_group samples of every group."""
    if min_per_group <= 0:
        return True
    for group in {g for g, _ in sample_keys}:
        cols = [j for j, (g2, _s) in enumerate(sample_keys) if g2 == group]
        if int(observed[cols].sum()) < min_per_group:
            return False
    return True


def serine_sites_and_keys(samples: list[Sample]) -> tuple[list[int], list[tuple[int, int]]]:
    """(serine positions, per-sample ``(group, subject)`` keys) for a single-protein sample set.

    Raises if the samples span more than one protein (site positions must share a coordinate system).
    """
    if len({s.protein_sequence for s in samples}) > 1:
        raise ValueError("expects samples from a single protein")
    serines = [i for i, aa in enumerate(samples[0].protein_sequence) if aa == "S"]
    sample_keys = [(s.group, s.subject) for s in samples]
    return serines, sample_keys


def iter_peptide_spans(
    samples: list[Sample], serines: list[int]
) -> Iterator[tuple[int, Peptide, list[int], set[int]]]:
    """Yield ``(sample_col, peptide, serines_in_span, mod_sites_set)`` for every spanned peptide.

    Skips peptides without span coordinates. `serines_in_span` are the serine positions inside the
    peptide's ``[start_index, end_index]`` (located by bisect on the sorted `serines`). Shared by the
    stoichiometry builders so the samples x peptides x serine-span walk lives in one place.
    """
    for col, sample in enumerate(samples):
        for pep in sample.peptides:
            if pep.start_index is None or pep.end_index is None:
                continue  # no span info -> cannot attribute to sites
            lo = bisect.bisect_left(serines, pep.start_index)
            hi = bisect.bisect_right(serines, pep.end_index)
            yield col, pep, serines[lo:hi], set(pep.mod_sites)
