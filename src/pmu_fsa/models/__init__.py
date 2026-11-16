"""Regression, classification, and calibration models for frequency security assessment."""

from __future__ import annotations

from pmu_fsa.models.regressors import (
    BaseNadirRegressor,
    LinearNadirRegressor,
    DecisionTreeNadirRegressor,
    RandomForestNadirRegressor,
    GradientBoostingNadirRegressor,
    NadirRegressor,
    get_nadir_regressor,
    REGRESSOR_REGISTRY,
)
from pmu_fsa.models.classifiers import (
    BaseSecurityClassifier,
    BinarySecurityClassifier,
    LogisticSecurityClassifier,
    DecisionTreeSecurityClassifier,
    RandomForestSecurityClassifier,
    GradientBoostingSecurityClassifier,
    SecurityClassifier,
    get_security_classifier,
    CLASSIFIER_REGISTRY,
)
from pmu_fsa.models.calibrated_gb import CalibratedGBDTClassifier
from pmu_fsa.models.calibration_metrics import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_diagram,
    compute_calibration_metrics,
)

__all__ = [
    "BaseNadirRegressor",
    "LinearNadirRegressor",
    "DecisionTreeNadirRegressor",
    "RandomForestNadirRegressor",
    "GradientBoostingNadirRegressor",
    "NadirRegressor",
    "get_nadir_regressor",
    "REGRESSOR_REGISTRY",
    "BaseSecurityClassifier",
    "BinarySecurityClassifier",
    "LogisticSecurityClassifier",
    "DecisionTreeSecurityClassifier",
    "RandomForestSecurityClassifier",
    "GradientBoostingSecurityClassifier",
    "SecurityClassifier",
    "get_security_classifier",
    "CLASSIFIER_REGISTRY",
    "CalibratedGBDTClassifier",
    "brier_score",
    "expected_calibration_error",
    "maximum_calibration_error",
    "reliability_diagram",
    "compute_calibration_metrics",
]
