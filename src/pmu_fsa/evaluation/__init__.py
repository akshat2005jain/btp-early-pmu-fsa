"""Evaluation benchmarks for window sensitivity, OOD, noise, latency, statistics, and interpretability."""

from pmu_fsa.evaluation.sensitivity import WindowSensitivityBenchmark, split_scenarios_by_partition
from pmu_fsa.evaluation.ood_generalization import OODGeneralizationBenchmark
from pmu_fsa.evaluation.noise_stress import MeasurementNoiseBenchmark
from pmu_fsa.evaluation.latency import LatencyBenchmark, collect_hardware_software_info
from pmu_fsa.evaluation.statistical_tests import (
    paired_bootstrap_ci,
    paired_model_comparison_test,
    compare_all_models_statistical,
    LearningCurveEvaluator,
    MultiSeedStabilityEvaluator,
)
from pmu_fsa.evaluation.interpretability import (
    InterpretabilityAnalyzer,
    CANONICAL_14_FEATURES,
    FEATURE_GROUP_INDICES,
)

__all__ = [
    "WindowSensitivityBenchmark",
    "split_scenarios_by_partition",
    "OODGeneralizationBenchmark",
    "MeasurementNoiseBenchmark",
    "LatencyBenchmark",
    "collect_hardware_software_info",
    "paired_bootstrap_ci",
    "paired_model_comparison_test",
    "compare_all_models_statistical",
    "LearningCurveEvaluator",
    "MultiSeedStabilityEvaluator",
    "InterpretabilityAnalyzer",
    "CANONICAL_14_FEATURES",
    "FEATURE_GROUP_INDICES",
]
