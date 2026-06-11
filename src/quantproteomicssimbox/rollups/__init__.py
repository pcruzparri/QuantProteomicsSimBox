"""Peptide -> site roll-up methods.

Two families, plus a shared core:
- ``core``          — `RollupResult`, the `Space` enum, `group_site_change`, shared span indexing.
- ``intensity``     — the paper's scaling x aggregation intensity roll-up (`roll_up`).
- ``stoichiometry`` — the per-site modified-fraction roll-up, pooled or per-peptide (`roll_up_stoichiometry`).

Everything public is re-exported here, so ``from quantproteomicssimbox.rollups import X`` keeps working.
"""

from .core import RollupResult, Space, group_site_change, site_present
from .intensity import (
    AGGREGATIONS,
    SCALINGS,
    SiteTable,
    build_site_tables,
    roll_up,
    scale_rollup,
    scale_rrollup,
    scale_zrollup,
)
from .stoichiometry import (
    FRACTION_TRANSFORMS,
    STOICHIOMETRY_METHODS,
    PeptideFractionSite,
    StoichiometryMethod,
    StoichiometrySite,
    build_peptide_fraction_tables,
    build_stoichiometry_tables,
    roll_up_stoichiometry,
)

__all__ = [
    # core
    "RollupResult",
    "Space",
    "group_site_change",
    "site_present",
    # intensity
    "AGGREGATIONS",
    "SCALINGS",
    "SiteTable",
    "build_site_tables",
    "roll_up",
    "scale_rollup",
    "scale_rrollup",
    "scale_zrollup",
    # stoichiometry
    "FRACTION_TRANSFORMS",
    "STOICHIOMETRY_METHODS",
    "PeptideFractionSite",
    "StoichiometryMethod",
    "StoichiometrySite",
    "build_peptide_fraction_tables",
    "build_stoichiometry_tables",
    "roll_up_stoichiometry",
]
