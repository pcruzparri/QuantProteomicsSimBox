"""Tests for quantproteomicssimbox.methods — the QuantMethod strategy, factories, and registry."""

import numpy as np

from quantproteomicssimbox.experiment import Experiment
from quantproteomicssimbox.methods import (
    QUANT_METHODS,
    QuantMethod,
    intensity_method,
    stoichiometry_method,
)


def _exp() -> Experiment:
    return Experiment(
        n_proteins=2, protein_length=60, n_subjects=4, abundance=40,
        miscleavage_rate=0.25, var_subject=1.0, var_site=1.0,
        position_aware=True, rng=np.random.default_rng(0),
    )


def test_factories_build_quant_methods():
    intensity = intensity_method("rollup", "median", "log2")
    stoich = stoichiometry_method("peptide_mean")
    assert isinstance(intensity, QuantMethod) and "intensity" in intensity.name
    assert isinstance(stoich, QuantMethod) and "peptide_mean" in stoich.name


def test_registry_holds_intensity_and_stoichiometry_methods():
    assert {"int_median", "int_sum", "stoich_fraction", "stoich_pep_mean"} <= set(QUANT_METHODS)
    assert all(isinstance(m, QuantMethod) for m in QUANT_METHODS.values())


def test_score_accepts_every_registry_method():
    exp = _exp()
    exp.observe()
    for method in QUANT_METHODS.values():
        assert np.isfinite(exp.score(method))


def test_intensity_and_stoichiometry_are_distinct_estimands():
    exp = _exp()
    intensity = exp.score(intensity_method(aggregation="median"))
    fraction = exp.score(stoichiometry_method("fraction"))
    assert intensity != fraction  # different quantities -> generally different RMSE
