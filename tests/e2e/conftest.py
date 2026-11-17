"""Shared 50 Hz E2E fixtures, production module bindings, and causal validation helpers."""

from __future__ import annotations

import copy
import math
from pathlib import Path
import random
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pytest

HAS_NUMPY = True
HAS_PYTEST = True

from pmu_fsa.evaluation.interpretability import InterpretabilityAnalyzer
from pmu_fsa.evaluation.latency import LatencyBenchmark
from pmu_fsa.evaluation.noise_stress import MeasurementNoiseBenchmark
from pmu_fsa.evaluation.ood_generalization import OODGeneralizationBenchmark
from pmu_fsa.evaluation.sensitivity import (
    WindowSensitivityBenchmark,
    split_scenarios_by_partition,
)
from pmu_fsa.evaluation.statistical_tests import (
    LearningCurveEvaluator,
    MultiSeedStabilityEvaluator,
    paired_bootstrap_ci,
    paired_model_comparison_test,
)
from pmu_fsa.features.extractor import (
    CANONICAL_FEATURE_NAMES,
    FeatureExtractor,
    extract_feature_vector,
    extract_features,
)
from pmu_fsa.features.window_slicer import CausalWindow, extract_causal_window
from pmu_fsa.models.calibrated_gb import CalibratedGBDTClassifier
from pmu_fsa.models.calibration_metrics import (
    brier_score,
    compute_calibration_metrics,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_diagram,
)
from pmu_fsa.models.classifiers import (
    BinarySecurityClassifier,
    DecisionTreeSecurityClassifier,
    GradientBoostingSecurityClassifier,
    LightGBMSecurityClassifier,
    LogisticSecurityClassifier,
    RandomForestSecurityClassifier,
    compute_binary_security_metrics,
    evaluate_probability_threshold_sweep,
)
from pmu_fsa.models.regressors import (
    AnalyticalSwingNadirRegressor,
    DecisionTreeNadirRegressor,
    GradientBoostingNadirRegressor,
    LightGBMNadirRegressor,
    LinearNadirRegressor,
    MeanNadirRegressor,
    RandomForestNadirRegressor,
)
from pmu_fsa.simulation.dataset_validator import (
    validate_dataset,
    write_dataset_quality_report,
)
from pmu_fsa.simulation.dynamic_engine import (
    DynamicSimulator,
    compute_kron_reduction,
    solve_power_flow,
)
from pmu_fsa.simulation.ieee39_data import (
    AREA_1_BUSES,
    AREA_2_BUSES,
    AREA_3_BUSES,
    BASE_MVA,
    BRANCH_DATA,
    GENERATOR_BUSES as IEEE39_GENERATOR_BUSES,
    NOMINAL_FREQ,
    PHYSICAL_FREQ_BOUNDS,
    SECURITY_THRESHOLD_HZ,
    get_ieee39_gen_data,
)
from pmu_fsa.simulation.pmu_recorder import (
    CANONICAL_13BUS_PMU_PLACEMENT,
    PMUNoiseConfig,
    PMURecorder,
    compute_causal_rocof,
    estimate_pmu_system_frequency,
)
from pmu_fsa.simulation.renewables import (
    DEFAULT_DISPLACED_BUSES,
    RenewablePenetrationConfig,
    apply_renewable_displacement,
    summarize_renewable_dispatch,
)
from pmu_fsa.simulation.scenario_generator import (
    ScenarioGenerator,
    assign_scenario_partition,
    classify_ood_regime,
)

# Canonical 50 Hz IEEE 39-bus generator table derived from production ieee39_data (macro_intertie=False)
IEEE39_GEN_PARAMS: Dict[int, Dict[str, Any]] = {
    int(g["bus"]): {
        "name": g["name"],
        "S_mva": float(g["S_mva"]),
        "P0_mw": float(g["P_mw"]),
        "Q0_mvar": float(g["Q_mvar"]),
        "V0": float(g["V_pu"]),
        "H": float(g["H"]),
        "D": float(g["D"]),
        "Xdp": float(g["Xdp"]),
        "R": float(g["R"]),
        "R_sys_pu": float(g["R_sys_pu"]),
        "Tg": float(g["Tg"]),
    }
    for g in get_ieee39_gen_data(macro_intertie=False)
}

IEEE39_BRANCHES: List[Tuple[int, int, float, float, float]] = [
    (int(b["from"]), int(b["to"]), float(b["r"]), float(b["x"]), float(b["b"]))
    for b in BRANCH_DATA
]

# Production-backed aliases so every E2E test exercises the real src/pmu_fsa classes
RefMeanNadirRegressor = MeanNadirRegressor
RefAnalyticalSwingNadirRegressor = AnalyticalSwingNadirRegressor
RefLinearNadirRegressor = LinearNadirRegressor
RefDecisionTreeNadirRegressor = DecisionTreeNadirRegressor
RefRandomForestNadirRegressor = RandomForestNadirRegressor
RefGradientBoostingNadirRegressor = GradientBoostingNadirRegressor
RefLightGBMNadirRegressor = LightGBMNadirRegressor
RefBinarySecurityClassifier = BinarySecurityClassifier
RefCalibratedGBDTClassifier = CalibratedGBDTClassifier
RefWindowSensitivityBenchmark = WindowSensitivityBenchmark
RefOODGeneralizationBenchmark = OODGeneralizationBenchmark
RefMeasurementNoiseBenchmark = MeasurementNoiseBenchmark
RefLatencyBenchmark = LatencyBenchmark

ref_extract_causal_window = extract_causal_window
ref_brier_score = brier_score
ref_expected_calibration_error = expected_calibration_error
ref_maximum_calibration_error = maximum_calibration_error
ref_reliability_diagram = reliability_diagram


def _to_float_list_1d(arr: Any) -> List[float]:
    if hasattr(arr, "tolist"):
        arr = arr.tolist()
    if isinstance(arr, (list, tuple)):
        flat: List[float] = []
        for item in arr:
            if isinstance(item, (list, tuple)):
                flat.append(float(item[0]))
            else:
                flat.append(float(item))
        return flat
    return [float(arr)]


def _to_float_list_2d(arr: Any) -> List[List[float]]:
    if hasattr(arr, "tolist"):
        arr = arr.tolist()
    if isinstance(arr, (list, tuple)):
        if len(arr) > 0 and isinstance(arr[0], (list, tuple)):
            return [[float(val) for val in row] for row in arr]
        return [[float(val)] for val in arr]
    return [[float(arr)]]


def _has_nan(a: Any) -> bool:
    arr = np.asarray(a, dtype=float)
    return bool(np.isnan(arr).any())


def _has_inf(a: Any) -> bool:
    arr = np.asarray(a, dtype=float)
    return bool(np.isinf(arr).any())


def generate_synthetic_pmu_stream(
    n_samples: int = 251,
    fps: int = 50,
    f_nominal: float = 50.0,
    f_nadir: Optional[float] = None,
    t_disturb: float = 1.0,
    disturbed_gen: int = 35,
    renewable_penetration: float = 0.20,
    active_loss_mw: float = 650.0,
    load_scale: float = 1.0,
    noise_std: float = 0.0,
    seed: int = 42,
    damping: Optional[float] = None,
    governor_Tg: Optional[float] = None,
    partition: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a physically structured 50 Hz 39-bus PMU stream for fast E2E testing."""
    if fps <= 0:
        raise ValueError(f"Reporting rate (fps) must be positive, got {fps}")
    if n_samples <= 0:
        raise ValueError(f"Sample count must be strictly positive, got {n_samples}")
    if disturbed_gen not in IEEE39_GENERATOR_BUSES:
        raise ValueError(f"disturbed_gen {disturbed_gen} not a valid generator bus in IEEE 39-bus system")
    if active_loss_mw < 0.0:
        raise ValueError(f"active_loss_mw must be non-negative, got {active_loss_mw}")
    if not (0.0 <= renewable_penetration <= 1.0):
        raise ValueError(f"renewable_penetration must be in [0.0, 1.0], got {renewable_penetration}")
    if noise_std < 0.0:
        raise ValueError(f"noise_std must be non-negative, got {noise_std}")
    if damping is not None and damping < 0.0:
        raise ValueError(f"Damping factor D must be non-negative, got {damping}")
    if governor_Tg is not None and governor_Tg <= 0.0:
        raise ValueError(f"Governor time constant Tg must be positive, got {governor_Tg}")

    base_H_sum = sum(p["H"] for p in IEEE39_GEN_PARAMS.values())
    effective_H_sum = max(25.0, base_H_sum * (1.0 - 0.72 * renewable_penetration))

    if f_nadir is None:
        if active_loss_mw == 0.0:
            target_nadir = f_nominal
        else:
            severity = (active_loss_mw / 650.0) * (base_H_sum / effective_H_sum) * (0.85 + 0.15 * load_scale)
            target_nadir = max(46.0, f_nominal - 0.52 * severity)
    else:
        target_nadir = float(f_nadir)

    if target_nadir < 40.0:
        raise ValueError(f"f_nadir {target_nadir} is below physical lower limit 40.0 Hz")

    rng = np.random.default_rng(seed)
    dt = 1.0 / float(fps)
    timestamps = np.round(np.arange(n_samples, dtype=float) * dt, 6)
    n_buses = 39

    delta_f_max = f_nominal - target_nadir
    tau_nadir = 2.35
    t_post = np.maximum(0.0, timestamps - t_disturb)
    post_mask = timestamps >= t_disturb

    # Combine inertial initial slope scaling with target nadir envelope
    inertia_boost = (base_H_sum / effective_H_sum) * 0.12
    surge = np.where(
        post_mask,
        (t_post / tau_nadir) * np.exp(1.0 - t_post / tau_nadir),
        0.0,
    )
    early_transient = np.where(
        post_mask,
        inertia_boost * delta_f_max * (1.0 - np.exp(-t_post / 0.18)) * np.exp(-t_post / 0.90),
        0.0,
    )
    delta_f_center = -(delta_f_max * surge + early_transient)
    if post_mask.any() and delta_f_max > 0.0:
        min_center = float(np.min(delta_f_center[post_mask]))
        if abs(min_center) > 1e-12:
            delta_f_center = delta_f_center * (-delta_f_max / min_center)

    osc = np.where(
        post_mask,
        0.03 * delta_f_max * np.exp(-0.45 * t_post) * np.sin(2.0 * math.pi * 0.8 * t_post),
        0.0,
    )

    bus_ids = np.arange(1, n_buses + 1, dtype=float)
    weights = np.exp(-np.abs(bus_ids - float(disturbed_gen)) / 8.0)

    bus_frequencies = (
        f_nominal
        + delta_f_center[:, None] * (1.0 + 0.16 * (weights[None, :] - 0.5))
        + osc[:, None] * np.cos(0.2 * bus_ids)[None, :]
    )
    voltage_magnitudes = 1.0 + 0.001 * np.sin(bus_ids)[None, :] - np.where(
        post_mask[:, None],
        0.045 * weights[None, :] * np.exp(-t_post[:, None] / 1.5),
        0.0,
    )
    voltage_magnitudes = np.maximum(0.82, voltage_magnitudes)

    # Angles in degrees around [-180, 180)
    base_ang_deg = 0.8 * (bus_ids - 19.0)
    swing_deg = np.where(
        post_mask[:, None],
        18.0 * (-delta_f_center[:, None]) * np.minimum(t_post[:, None], 2.5) * (weights[None, :] - 0.45),
        0.0,
    )
    voltage_angles = ((base_ang_deg[None, :] + swing_deg + 180.0) % 360.0) - 180.0

    if noise_std > 0.0:
        bus_frequencies = bus_frequencies + rng.normal(0.0, noise_std, size=bus_frequencies.shape)
        voltage_magnitudes = np.maximum(
            0.01,
            voltage_magnitudes + rng.normal(0.0, min(0.01, noise_std * 0.05), size=voltage_magnitudes.shape),
        )
        voltage_angles = (
            (voltage_angles + rng.normal(0.0, min(2.0, noise_std * 5.0), size=voltage_angles.shape) + 180.0) % 360.0
        ) - 180.0

    bus_rocof = compute_causal_rocof(bus_frequencies, fps=float(fps), window_length=5)
    pmu_sys_frequency = estimate_pmu_system_frequency(bus_frequencies)

    if post_mask.any():
        t_nadir_val = float(timestamps[post_mask][int(np.argmin(delta_f_center[post_mask]))])
    else:
        t_nadir_val = float(timestamps[-1])

    scen_id = f"SCEN_SYNTH_G{disturbed_gen}_L{int(round(active_loss_mw))}_R{int(round(renewable_penetration * 100))}_{seed}"
    ood_type = classify_ood_regime(renewable_penetration, load_scale)
    eff_part = partition if partition is not None else assign_scenario_partition(renewable_penetration, load_scale, scen_id)
    is_secure = 1 if target_nadir >= SECURITY_THRESHOLD_HZ else 0

    base_epoch = 1700000000
    total_secs = base_epoch + timestamps
    soc = np.floor(total_secs).astype(np.uint32)
    fracsec = np.round((total_secs - soc) * 1_000_000).astype(np.uint32) & np.uint32(0x00FFFFFF)

    meta: Dict[str, Any] = {
        "scenario_id": scen_id,
        "partition": eff_part,
        "ood_type": ood_type,
        "disturbed_gen": disturbed_gen,
        "loss_fraction": round(min(1.0, active_loss_mw / max(1.0, IEEE39_GEN_PARAMS[disturbed_gen]["P0_mw"])), 3),
        "active_loss_mw": float(active_loss_mw),
        "renewable_penetration": float(renewable_penetration),
        "p_ren_requested": float(renewable_penetration),
        "p_ren_realized": float(renewable_penetration),
        "load_scale": float(load_scale),
        "t_disturb": float(t_disturb),
        "t_sim": float(timestamps[-1]),
        "ground_truth_nadir": float(target_nadir),
        "ground_truth_t_nadir": round(t_nadir_val, 4),
        "bus_min_nadir": float(np.min(bus_frequencies[post_mask])) if post_mask.any() else float(target_nadir),
        "bus_min_t_nadir": round(t_nadir_val, 4),
        "is_secure": is_secure,
        "fps": int(fps),
        "f_nominal": float(f_nominal),
        "f_threshold_hz": SECURITY_THRESHOLD_HZ,
        "effective_H_sum": float(effective_H_sum),
    }
    if damping is not None:
        meta["damping"] = float(damping)
    if governor_Tg is not None:
        meta["governor_Tg"] = float(governor_Tg)

    return {
        "timestamps": timestamps,
        "soc": soc,
        "fracsec": fracsec,
        "bus_frequencies": bus_frequencies,
        "bus_rocof": bus_rocof,
        "voltage_magnitudes": voltage_magnitudes,
        "voltage_angles": voltage_angles,
        "pmu_sys_frequency": pmu_sys_frequency,
        "coi_frequency": pmu_sys_frequency.copy(),
        "metadata": meta,
    }


def assert_valid_pmu_trajectory(
    traj: Dict[str, Any],
    n_buses: int = 39,
    f_nominal: float = NOMINAL_FREQ,
) -> None:
    required_keys = [
        "timestamps",
        "bus_frequencies",
        "bus_rocof",
        "voltage_magnitudes",
        "voltage_angles",
        "coi_frequency",
        "metadata",
    ]
    for key in required_keys:
        assert key in traj, f"Missing required trajectory contract key: '{key}'"

    ts = traj["timestamps"]
    bf = traj["bus_frequencies"]
    br = traj["bus_rocof"]
    vm = traj["voltage_magnitudes"]
    va = traj["voltage_angles"]
    coi = traj["coi_frequency"]
    meta = traj["metadata"]

    n_samples = len(ts)
    assert n_samples > 0, "Trajectory timestamps array is empty"
    assert len(coi) == n_samples
    assert bf.shape == (n_samples, n_buses)
    assert br.shape == (n_samples, n_buses)
    assert vm.shape == (n_samples, n_buses)
    assert va.shape == (n_samples, n_buses)

    for name, arr in (
        ("timestamps", ts),
        ("bus_frequencies", bf),
        ("bus_rocof", br),
        ("voltage_magnitudes", vm),
        ("voltage_angles", va),
        ("coi_frequency", coi),
    ):
        assert not _has_nan(arr), f"NaN detected in {name}"
        assert not _has_inf(arr), f"Inf detected in {name}"

    ts_list = _to_float_list_1d(ts)
    for i in range(len(ts_list) - 1):
        assert ts_list[i + 1] > ts_list[i], (
            f"Non-monotonic or duplicate timestamp at index {i}: {ts_list[i]} >= {ts_list[i + 1]}"
        )

    min_f = float(np.min(bf))
    max_f = float(np.max(bf))
    low_b, high_b = PHYSICAL_FREQ_BOUNDS
    assert low_b <= min_f, f"Bus frequency drop exceeds physical bounds: {min_f} < {low_b} Hz"
    assert max_f <= f_nominal + 5.0, f"Bus frequency rise exceeds physical bounds: {max_f} > {f_nominal + 5.0} Hz"
    assert float(np.min(vm)) > 0.0, "Non-positive voltage magnitude detected"

    meta_keys = [
        "scenario_id",
        "disturbed_gen",
        "active_loss_mw",
        "renewable_penetration",
        "t_disturb",
        "ground_truth_nadir",
        "is_secure",
    ]
    for mk in meta_keys:
        assert mk in meta, f"Missing required metadata key: '{mk}'"

    assert isinstance(meta["scenario_id"], str) and len(meta["scenario_id"]) > 0, "scenario_id must be non-empty"
    assert meta["is_secure"] in (0, 1)
    assert low_b <= float(meta["ground_truth_nadir"]) <= high_b, (
        f"Ground truth nadir {meta['ground_truth_nadir']} outside [{low_b}, {high_b}] Hz"
    )


def assert_feature_matrix_integrity(feature_dict: Dict[str, Any]) -> None:
    for k in ("X", "y_nadir", "y_secure", "scenario_ids", "feature_names"):
        assert k in feature_dict, f"Missing required feature matrix key: '{k}'"

    X = np.asarray(feature_dict["X"], dtype=float)
    y_nadir = np.asarray(feature_dict["y_nadir"], dtype=float)
    y_secure = np.asarray(feature_dict["y_secure"], dtype=int)
    s_ids = feature_dict["scenario_ids"]
    f_names = list(feature_dict["feature_names"])

    n_scenarios = len(s_ids)
    assert n_scenarios > 0
    assert X.shape == (n_scenarios, len(f_names))
    assert len(f_names) == 14
    assert f_names == CANONICAL_FEATURE_NAMES
    assert len(y_nadir) == n_scenarios
    assert len(y_secure) == n_scenarios
    assert np.all(np.isfinite(X))
    assert np.all(np.isfinite(y_nadir))
    assert np.all((y_nadir >= 45.0) & (y_nadir <= 50.5))
    assert set(np.unique(y_secure)).issubset({0, 1})


def assert_strictly_causal(
    extractor_fn: Callable[[Dict[str, Any], float], Any],
    base_trajectory: Dict[str, Any],
    t_obs: float,
) -> None:
    t_disturb = float(base_trajectory["metadata"]["t_disturb"])
    cutoff_time = t_disturb + float(t_obs)

    features_clean = np.asarray(extractor_fn(base_trajectory, t_obs), dtype=float)

    corrupted_traj = copy.deepcopy(base_trajectory)
    ts = corrupted_traj["timestamps"]
    cutoff_idx = int(np.searchsorted(ts, cutoff_time + 1e-9, side="right"))
    assert cutoff_idx < len(ts), "Observation window covers entire trajectory; cannot test leakage"

    corrupted_traj["bus_frequencies"][cutoff_idx:, :] = 42.0
    corrupted_traj["bus_rocof"][cutoff_idx:, :] = -35.0
    corrupted_traj["coi_frequency"][cutoff_idx:] = 40.0
    if "pmu_sys_frequency" in corrupted_traj:
        corrupted_traj["pmu_sys_frequency"][cutoff_idx:] = 40.0
    corrupted_traj["voltage_angles"][cutoff_idx:, :] = 175.0

    features_corrupted = np.asarray(extractor_fn(corrupted_traj, t_obs), dtype=float)
    assert np.allclose(features_clean, features_corrupted, atol=1e-12, rtol=1e-12), (
        "Future data leakage detected when post-window samples were mutated!"
    )

    inside_idx = max(0, cutoff_idx - 2)
    corrupted_inside = copy.deepcopy(base_trajectory)
    corrupted_inside["bus_frequencies"][inside_idx, :] -= 1.5
    corrupted_inside["coi_frequency"][inside_idx] -= 1.5
    if "pmu_sys_frequency" in corrupted_inside:
        corrupted_inside["pmu_sys_frequency"][inside_idx] -= 1.5
    features_inside_mutated = np.asarray(extractor_fn(corrupted_inside, t_obs), dtype=float)
    assert not np.allclose(features_clean, features_inside_mutated, atol=1e-6), (
        "Sanity check failed: within-window mutation did not alter extracted features"
    )


_CACHED_REAL_PMU_TRAJECTORY: Optional[Dict[str, Any]] = None


def _get_real_pmu_trajectory() -> Dict[str, Any]:
    global _CACHED_REAL_PMU_TRAJECTORY
    if _CACHED_REAL_PMU_TRAJECTORY is None:
        gen = ScenarioGenerator(fps=50, t_span=(0.0, 6.0), t_disturb=1.0, seed=42)
        _CACHED_REAL_PMU_TRAJECTORY = gen.generate_single_scenario(
            tripped_bus=35,
            loss_fraction=1.0,
            renewable_penetration=0.20,
            load_scale=1.0,
        )
    return copy.deepcopy(_CACHED_REAL_PMU_TRAJECTORY)


@pytest.fixture
def real_pmu_trajectory() -> Dict[str, Any]:
    return _get_real_pmu_trajectory()


@pytest.fixture
def sample_pmu_trajectory() -> Dict[str, Any]:
    return generate_synthetic_pmu_stream(
        n_samples=301,
        fps=50,
        f_nominal=50.0,
        f_nadir=49.38,
        t_disturb=1.0,
        disturbed_gen=35,
        renewable_penetration=0.25,
        active_loss_mw=650.0,
        seed=101,
    )


@pytest.fixture
def ieee39_topology() -> Dict[str, Any]:
    return {
        "n_buses": 39,
        "n_generators": 10,
        "n_branches": 46,
        "base_mva": BASE_MVA,
        "nominal_freq": NOMINAL_FREQ,
        "security_threshold_hz": SECURITY_THRESHOLD_HZ,
        "physical_freq_bounds": PHYSICAL_FREQ_BOUNDS,
        "generator_buses": list(IEEE39_GENERATOR_BUSES),
        "generator_params": copy.deepcopy(IEEE39_GEN_PARAMS),
        "branches": list(IEEE39_BRANCHES),
        "area_partition": {
            "Area 1": list(AREA_1_BUSES),
            "Area 2": list(AREA_2_BUSES),
            "Area 3": list(AREA_3_BUSES),
        },
    }


def _build_synthetic_scenario_batch() -> List[Dict[str, Any]]:
    # 20 scenarios spanning all 10 generator buses, both security classes, and train/val/test/ood_test
    configs = [
        # ID train (6 scenarios: 3 secure >=49.50, 3 insecure <49.50)
        ("BATCH_SCEN_001", "train", 30, 250.0, 0.10, 1.00, 49.76),
        ("BATCH_SCEN_002", "train", 31, 400.0, 0.15, 0.95, 49.62),
        ("BATCH_SCEN_003", "train", 34, 355.0, 0.20, 1.00, 49.55),
        ("BATCH_SCEN_004", "train", 32, 650.0, 0.20, 1.05, 49.36),
        ("BATCH_SCEN_005", "train", 35, 650.0, 0.25, 1.00, 49.28),
        ("BATCH_SCEN_006", "train", 38, 830.0, 0.30, 1.10, 49.08),
        # ID val (4 scenarios: 2 secure, 2 insecure)
        ("BATCH_SCEN_007", "val", 30, 250.0, 0.20, 0.90, 49.71),
        ("BATCH_SCEN_008", "val", 37, 378.0, 0.15, 1.00, 49.56),
        ("BATCH_SCEN_009", "val", 33, 632.0, 0.25, 1.05, 49.31),
        ("BATCH_SCEN_010", "val", 39, 1000.0, 0.20, 1.00, 49.12),
        # ID test (4 scenarios: 2 secure, 2 insecure)
        ("BATCH_SCEN_011", "test", 36, 224.0, 0.10, 0.95, 49.74),
        ("BATCH_SCEN_012", "test", 34, 355.0, 0.20, 0.90, 49.57),
        ("BATCH_SCEN_013", "test", 35, 650.0, 0.30, 1.05, 49.24),
        ("BATCH_SCEN_014", "test", 38, 830.0, 0.25, 1.10, 49.11),
        # OOD test (6 scenarios: ood_res, ood_load, ood_combined; secure & insecure)
        ("BATCH_SCEN_015", "ood_test", 30, 250.0, 0.40, 1.00, 49.58),
        ("BATCH_SCEN_016", "ood_test", 36, 560.0, 0.45, 1.00, 49.26),
        ("BATCH_SCEN_017", "ood_test", 37, 216.0, 0.20, 0.80, 49.69),
        ("BATCH_SCEN_018", "ood_test", 33, 632.0, 0.25, 1.20, 49.22),
        ("BATCH_SCEN_019", "ood_test", 30, 250.0, 0.45, 0.80, 49.56),
        ("BATCH_SCEN_020", "ood_test", 38, 830.0, 0.50, 1.20, 48.92),
    ]
    scenarios: List[Dict[str, Any]] = []
    for idx, (sid, part, gen, loss, ren, ld, nadir) in enumerate(configs, start=1):
        scen = generate_synthetic_pmu_stream(
            n_samples=251,
            fps=50,
            f_nominal=50.0,
            f_nadir=nadir,
            t_disturb=1.0,
            disturbed_gen=gen,
            renewable_penetration=ren,
            active_loss_mw=loss,
            load_scale=ld,
            seed=200 + idx,
            partition=part,
        )
        scen["metadata"]["scenario_id"] = sid
        scenarios.append(scen)
    return scenarios


@pytest.fixture
def synthetic_scenario_batch() -> List[Dict[str, Any]]:
    return _build_synthetic_scenario_batch()


def _build_sample_causal_features() -> Dict[str, Any]:
    batch = _build_synthetic_scenario_batch()
    extractor = FeatureExtractor(t_obs=1.0, f_nominal=50.0)
    X_np, y_nadir, y_secure, meta_list = extractor.transform_batch(batch, as_dataframe=False)
    return {
        "X": X_np,
        "y_nadir": y_nadir,
        "y_secure": y_secure,
        "scenario_ids": [m["scenario_id"] for m in meta_list],
        "feature_names": list(CANONICAL_FEATURE_NAMES),
    }


@pytest.fixture
def sample_causal_features() -> Dict[str, Any]:
    return _build_sample_causal_features()


def _build_sample_regression_dataset(n_samples: int = 60) -> Dict[str, Any]:
    rng = np.random.default_rng(77)
    loss_mw = np.linspace(200.0, 980.0, n_samples)
    ren_pen = 0.10 + 0.35 * ((np.arange(n_samples) % 10) / 9.0)
    h_tot = 3.25 * (1.0 - 0.55 * ren_pen)
    p_pu = loss_mw / 100.0

    rocof_init = -50.0 * p_pu / (2.0 * h_tot * 100.0) + rng.normal(0.0, 0.008, size=n_samples)
    rocof_peak_neg = rocof_init * 1.15
    rocof_end = rocof_init * 0.45
    rocof_mean = rocof_init * 0.72
    freq_drop_end = rocof_mean * 1.0
    freq_drop_max = rocof_peak_neg * 0.85
    freq_dev_integral = 0.5 * freq_drop_end
    bus_drop_worst = freq_drop_max * 1.12
    area_freq_spread_max = 0.025 + 0.018 * p_pu
    bus_freq_std_mean = 0.008 + 0.005 * p_pu
    angle_divergence_max = 1.8 + 1.5 * p_pu
    tie_line_angle_excursion = 0.7 + 0.8 * p_pu
    freq_curvature_init = -0.35 * rocof_init
    freq_curvature_mean = -0.55 * rocof_init

    X = np.column_stack([
        rocof_init,
        rocof_peak_neg,
        rocof_end,
        rocof_mean,
        freq_drop_end,
        freq_drop_max,
        freq_dev_integral,
        bus_drop_worst,
        area_freq_spread_max,
        bus_freq_std_mean,
        angle_divergence_max,
        tie_line_angle_excursion,
        freq_curvature_init,
        freq_curvature_mean,
    ])
    y_nadir = np.clip(50.0 - 0.092 * (p_pu / (h_tot / 3.25)) + rng.normal(0.0, 0.01, size=n_samples), 48.0, 49.92)

    return {
        "X": X,
        "y_nadir": np.round(y_nadir, 4),
        "n_samples": n_samples,
        "n_features": 14,
        "feature_names": list(CANONICAL_FEATURE_NAMES),
    }


@pytest.fixture
def sample_regression_dataset() -> Dict[str, Any]:
    return _build_sample_regression_dataset()


def _build_sample_classification_dataset(reg_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if reg_data is None:
        reg_data = _build_sample_regression_dataset()
    y_nadir = np.asarray(reg_data["y_nadir"], dtype=float)
    y_secure = (y_nadir >= SECURITY_THRESHOLD_HZ).astype(int)
    return {
        "X": reg_data["X"],
        "y_nadir": y_nadir,
        "y_secure": y_secure,
        "threshold": SECURITY_THRESHOLD_HZ,
        "feature_names": list(CANONICAL_FEATURE_NAMES),
    }


@pytest.fixture
def sample_classification_dataset(sample_regression_dataset: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return _build_sample_classification_dataset(sample_regression_dataset)
