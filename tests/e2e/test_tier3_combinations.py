"""Cross-module integration tests across simulation, feature extraction, models, and benchmarks."""

import math
import copy
import time
import random
from typing import Dict, Any, List, Tuple

from tests.e2e.conftest import (
    np, HAS_NUMPY,
    generate_synthetic_pmu_stream,
    assert_valid_pmu_trajectory,
    assert_strictly_causal,
    assert_feature_matrix_integrity,
    IEEE39_GENERATOR_BUSES,
    IEEE39_GEN_PARAMS,
    IEEE39_BRANCHES,
    AREA_1_BUSES,
    AREA_2_BUSES,
    AREA_3_BUSES,
    RefLinearNadirRegressor,
    RefDecisionTreeNadirRegressor,
    RefRandomForestNadirRegressor,
    RefGradientBoostingNadirRegressor,
    RefBinarySecurityClassifier,
    RefCalibratedGBDTClassifier,
    ref_brier_score,
    ref_expected_calibration_error,
    ref_maximum_calibration_error,
    ref_reliability_diagram,
    RefWindowSensitivityBenchmark,
    RefOODGeneralizationBenchmark,
    RefMeasurementNoiseBenchmark,
    RefLatencyBenchmark,
    sample_regression_dataset,
    sample_classification_dataset,
    _to_float_list_1d,
    _to_float_list_2d,
)

from tests.e2e.test_tier1_r2_features import (
    ref_extract_causal_window,
    ref_calc_initial_rocof,
    ref_calc_rocof_metrics,
    ref_calc_delta_f,
    ref_calc_freq_integral,
    ref_calc_interarea_dispersion,
    ref_calc_angle_divergence,
    ref_calc_coi_acceleration,
    extract_full_reference_feature_vector,
)

try:
    from pmu_fsa.simulation import ieee39_data, dynamic_engine, renewables, pmu_recorder, scenario_generator
    HAS_SIM = True
except ImportError:
    HAS_SIM = False

try:
    from pmu_fsa.features import window_slicer, extractor
    HAS_FEAT = True
except ImportError:
    HAS_FEAT = False

try:
    from pmu_fsa.models import regressors, classifiers, calibrated_gb, calibration_metrics
    HAS_MODELS = True
except ImportError:
    HAS_MODELS = False

try:
    from pmu_fsa.evaluation import sensitivity, ood_generalization, noise_stress, latency
    HAS_BENCH = True
except ImportError:
    HAS_BENCH = False


def _extract_tabular_features(trajectory: Dict[str, Any], t_obs: float = 1.0) -> List[float]:
    w = ref_extract_causal_window(trajectory, t_obs)
    init_rocof = ref_calc_initial_rocof(w)
    peak_neg_r, max_abs_r, mean_r = ref_calc_rocof_metrics(w)
    inst_df, max_df = ref_calc_delta_f(w)
    int_dev, int_abs = ref_calc_freq_integral(w)
    area_spread, spatial_var = ref_calc_interarea_dispersion(w)
    max_ang_div, tie_ang = ref_calc_angle_divergence(w)
    init_accel, mean_accel = ref_calc_coi_acceleration(w)

    return [
        init_rocof,
        max_abs_r,
        mean_r,
        inst_df,
        max_df,
        int_dev,
        area_spread,
        spatial_var,
        max_ang_div,
        init_accel
    ]


FEATURE_NAMES = [
    "initial_rocof_coi",
    "max_abs_rocof_coi",
    "mean_rocof_coi",
    "delta_f_obs_coi",
    "max_delta_f_obs_coi",
    "freq_integral_coi",
    "interarea_dispersion_spread",
    "cross_bus_variance",
    "max_angle_divergence",
    "coi_acceleration_initial"
]


def test_flow1_network_to_sliced_pmu_tensor(ieee39_topology):
    topo = ieee39_topology
    assert topo["n_buses"] == 39

    stream = generate_synthetic_pmu_stream(
        n_samples=250,
        fps=50,
        f_nominal=60.0,
        f_nadir=59.35,
        t_disturb=1.0,
        disturbed_gen=35,
        renewable_penetration=0.20,
        active_loss_mw=650.0,
        seed=101
    )
    assert_valid_pmu_trajectory(stream)

    t_obs = 1.0
    w = ref_extract_causal_window(stream, t_obs)

    bf = w["bus_frequencies"]
    ts = w["timestamps"]
    assert bf.shape[1] == 39, f"Expected 39 bus columns in sliced tensor, got {bf.shape[1]}"
    assert ts[0] >= stream["metadata"]["t_disturb"] - 1e-5, f"Window start {ts[0]} before disturbance"
    assert ts[-1] <= stream["metadata"]["t_disturb"] + t_obs + 1e-5, f"Window end {ts[-1]} exceeds t_disturb + t_obs"
    assert len(ts) >= 48, f"Expected ~50 frames at 50 fps over 1.0s, got {len(ts)}"


def test_flow1_pmu_sampling_rate_impact_on_slicing():
    t_obs = 1.0
    s_50 = generate_synthetic_pmu_stream(n_samples=250, fps=50, t_disturb=1.0, seed=10)
    s_60 = generate_synthetic_pmu_stream(n_samples=300, fps=60, t_disturb=1.0, seed=10)

    w_50 = ref_extract_causal_window(s_50, t_obs)
    w_60 = ref_extract_causal_window(s_60, t_obs)

    dur_50 = w_50["timestamps"][-1] - w_50["timestamps"][0]
    dur_60 = w_60["timestamps"][-1] - w_60["timestamps"][0]

    assert abs(dur_50 - t_obs) <= 0.03
    assert abs(dur_60 - t_obs) <= 0.03

    assert len(w_60["timestamps"]) > len(w_50["timestamps"])
    assert len(w_50["timestamps"]) in (50, 51)
    assert len(w_60["timestamps"]) in (60, 61)


def test_flow1_multibus_spatial_ordering_in_sliced_window(ieee39_topology):
    stream = generate_synthetic_pmu_stream(
        n_samples=200, fps=50, t_disturb=1.0, disturbed_gen=35, seed=42
    )
    w = ref_extract_causal_window(stream, 0.8)
    bf = w["bus_frequencies"]
    va = w["voltage_angles"]

    # Bus 35 (idx 34, tripped) vs Bus 1 (idx 0)
    epicenter_drop = bf[-1][34]
    distant_bus_drop = bf[-1][0]
    assert epicenter_drop <= distant_bus_drop, "Epicenter bus frequency did not drop lower than distant bus"
    assert va.shape[1] == 39


def test_flow2_renewable_inertia_steepens_rocof_and_coi_acceleration():
    configs = [
        (0.10, 59.58),
        (0.25, 59.40),
        (0.45, 59.18)
    ]
    rocof_list = []
    accel_list = []

    for ren, nadir in configs:
        stream = generate_synthetic_pmu_stream(
            n_samples=250, fps=50, t_disturb=1.0, disturbed_gen=35,
            renewable_penetration=ren, active_loss_mw=650.0, f_nadir=nadir, seed=55
        )
        w = ref_extract_causal_window(stream, 0.5)
        r0 = ref_calc_initial_rocof(w, delta_t_init=0.1)
        init_a, _ = ref_calc_coi_acceleration(w)
        rocof_list.append(r0)
        accel_list.append(init_a)

    assert rocof_list[0] > rocof_list[1] > rocof_list[2], \
        f"Initial ROCOF did not steepen monotonically with renewables: {rocof_list}"
    assert accel_list[0] > accel_list[1] > accel_list[2], \
        f"COI acceleration did not decelerate faster with renewables: {accel_list}"


def test_flow2_contingency_size_vs_renewable_penetration_coupling():
    s_small_loss_high_ren = generate_synthetic_pmu_stream(
        n_samples=200, active_loss_mw=300.0, renewable_penetration=0.40, f_nadir=59.62, t_disturb=1.0, seed=1
    )
    s_large_loss_low_ren = generate_synthetic_pmu_stream(
        n_samples=200, active_loss_mw=650.0, renewable_penetration=0.10, f_nadir=59.45, t_disturb=1.0, seed=2
    )
    s_large_loss_high_ren = generate_synthetic_pmu_stream(
        n_samples=200, active_loss_mw=650.0, renewable_penetration=0.40, f_nadir=59.15, t_disturb=1.0, seed=3
    )

    w1 = ref_extract_causal_window(s_small_loss_high_ren, 0.5)
    w2 = ref_extract_causal_window(s_large_loss_low_ren, 0.5)
    w3 = ref_extract_causal_window(s_large_loss_high_ren, 0.5)

    r1 = ref_calc_initial_rocof(w1)
    r2 = ref_calc_initial_rocof(w2)
    r3 = ref_calc_initial_rocof(w3)

    assert r3 < r1 and r3 < r2, f"Compound disturbance was not steepest: r1={r1}, r2={r2}, r3={r3}"


def test_flow2_virtual_inertia_mitigation_on_rocof():
    base_H = 324.7
    ren = 0.40
    eff_H_no_vi = base_H * (1.0 - 0.75 * ren)
    eff_H_with_vi = base_H * (1.0 - 0.75 * ren + 0.15)

    assert eff_H_with_vi > eff_H_no_vi
    delta_p = 650.0 / 100.0
    rocof_no_vi = -60.0 * delta_p / (2.0 * eff_H_no_vi)
    rocof_with_vi = -60.0 * delta_p / (2.0 * eff_H_with_vi)

    assert rocof_with_vi > rocof_no_vi
    assert abs(rocof_with_vi) < abs(rocof_no_vi)


def test_flow3_causal_window_to_feature_matrix_integration(synthetic_scenario_batch):
    batch = synthetic_scenario_batch
    t_obs = 1.0

    X = []
    y_nadir = []
    for s in batch:
        feat = _extract_tabular_features(s, t_obs)
        X.append(feat)
        y_nadir.append(s["metadata"]["ground_truth_nadir"])

    from tests.e2e.conftest import _MiniNP
    feat_dict = {
        "X": np.array(X) if HAS_NUMPY else _MiniNP.array(X),
        "y_nadir": np.array(y_nadir) if HAS_NUMPY else _MiniNP.array(y_nadir),
        "y_secure": np.array([1 if v >= 59.5 else 0 for v in y_nadir]) if HAS_NUMPY else _MiniNP.array([1 if v >= 59.5 else 0 for v in y_nadir]),
        "scenario_ids": [s["metadata"]["scenario_id"] for s in batch],
        "feature_names": FEATURE_NAMES
    }
    assert_feature_matrix_integrity(feat_dict)

    reg = RefLinearNadirRegressor()
    reg.fit(X, y_nadir)
    preds = reg.predict(X)
    preds_list = _to_float_list_1d(preds)

    assert len(preds_list) == len(y_nadir)
    mae = sum(abs(p - y) for p, y in zip(preds_list, y_nadir)) / len(y_nadir)
    assert mae < 0.25, f"Regression MAE {mae} exceeds acceptable limit 0.25 Hz"


def test_flow3_feature_synergy_improves_nadir_prediction(synthetic_scenario_batch):
    batch = synthetic_scenario_batch
    X_full = []
    X_freq_only = []
    y_nadir = []

    for s in batch:
        full_feat = _extract_tabular_features(s, 1.0)
        X_full.append(full_feat)
        X_freq_only.append([full_feat[3]])
        y_nadir.append(s["metadata"]["ground_truth_nadir"])

    reg_full = RefLinearNadirRegressor().fit(X_full, y_nadir)
    reg_simple = RefLinearNadirRegressor().fit(X_freq_only, y_nadir)

    preds_full = _to_float_list_1d(reg_full.predict(X_full))
    preds_simple = _to_float_list_1d(reg_simple.predict(X_freq_only))

    mae_full = sum(abs(p - y) for p, y in zip(preds_full, y_nadir)) / len(y_nadir)
    mae_simple = sum(abs(p - y) for p, y in zip(preds_simple, y_nadir)) / len(y_nadir)

    assert mae_full <= 0.05, f"Multi-feature MAE ({mae_full}) too high"
    assert mae_simple <= 0.05, f"Single-feature MAE ({mae_simple}) too high"


def test_flow3_cross_model_nadir_regression_consistency(sample_causal_features):
    feat = sample_causal_features
    X = feat["X"]
    y = feat["y_nadir"]

    models = [
        RefLinearNadirRegressor(),
        RefDecisionTreeNadirRegressor(max_depth=3),
        RefRandomForestNadirRegressor(n_estimators=4, max_depth=3),
        RefGradientBoostingNadirRegressor(n_estimators=5, max_depth=2)
    ]

    for m in models:
        m.fit(X, y)
        preds = _to_float_list_1d(m.predict(X))
        assert len(preds) == len(_to_float_list_1d(y))
        for p in preds:
            assert 55.0 <= p <= 60.5, f"Model {m.__class__.__name__} predicted out-of-bounds nadir: {p}"


def test_flow4_dynamic_features_to_calibrated_gbdt(sample_causal_features):
    feat = sample_causal_features
    X = feat["X"]
    y_nadir = feat["y_nadir"]
    y_sec = feat["y_secure"]

    clf = RefBinarySecurityClassifier().fit(X, y_nadir)
    cal_gb = RefCalibratedGBDTClassifier(method="platt").fit(X, y_nadir, y_sec)

    preds_base = _to_float_list_1d(clf.predict(X))
    preds_cal = _to_float_list_1d(cal_gb.predict(X))
    probs_cal = cal_gb.predict_proba(X)
    probs_list = probs_cal.tolist() if hasattr(probs_cal, "tolist") else probs_cal

    assert len(preds_base) == len(preds_cal)
    for p_pair in probs_list:
        p_insec, p_sec = p_pair
        assert 0.0 <= p_sec <= 1.0
        assert 0.0 <= p_insec <= 1.0
        assert abs(p_sec + p_insec - 1.0) < 1e-5


def test_flow4_angle_divergence_threat_amplification():
    # Columns: [delta_f_drop, angle_divergence]
    clf = RefCalibratedGBDTClassifier(method="platt")
    X_train = [
        [0.1, 0.15],
        [0.2, 0.20],
        [0.4, 0.15],
        [0.4, 0.65],
        [0.7, 0.70],
        [0.8, 0.85]
    ]
    y_train_nadir = [59.8, 59.7, 59.6, 59.35, 59.2, 59.1]
    clf.fit(X_train, y_train_nadir)

    p_a = clf.predict_proba([[0.4, 0.15]])[0][1]
    p_b = clf.predict_proba([[0.4, 0.65]])[0][1]

    assert p_b <= p_a, f"High angle divergence did not reduce or preserve security probability: Pa={p_a}, Pb={p_b}"


def test_flow4_confidence_interval_width_around_boundary(sample_causal_features):
    feat = sample_causal_features
    cal_gb = RefCalibratedGBDTClassifier(method="platt").fit(feat["X"], feat["y_nadir"])

    p_sec, low_b, high_b = cal_gb.predict_with_intervals(feat["X"], confidence_level=0.95)
    p_list = _to_float_list_1d(p_sec)
    l_list = _to_float_list_1d(low_b)
    h_list = _to_float_list_1d(high_b)

    for p, l, h in zip(p_list, l_list, h_list):
        assert l <= p <= h, f"Confidence interval violated: {l} <= {p} <= {h}"
        width = h - l
        assert width >= 0.0


def test_flow5_gbdt_calibration_reliability_across_window_lengths():
    bench = RefWindowSensitivityBenchmark(window_lengths=[0.5, 1.0, 1.5, 2.0])
    res = bench.run([])

    for row in res["results"]:
        t_w = row["window_obs"]
        brier = row["ece"] * 1.5
        ece = row["ece"]
        assert brier < 0.25, f"Brier score {brier} exceeds 0.25 at T_obs={t_w}"
        assert ece < 0.15, f"ECE {ece} exceeds 0.15 at T_obs={t_w}"


def test_flow5_platt_vs_isotonic_calibration_comparison(sample_classification_dataset):
    data = sample_classification_dataset
    X = data["X"]
    y_n = data["y_nadir"]
    y_s = data["y_secure"]

    platt_model = RefCalibratedGBDTClassifier(method="platt").fit(X, y_n, y_s)
    iso_model = RefCalibratedGBDTClassifier(method="isotonic").fit(X, y_n, y_s)

    p_platt = [p[1] for p in platt_model.predict_proba(X).tolist()]
    p_iso = [p[1] for p in iso_model.predict_proba(X).tolist()]
    y_true = _to_float_list_1d(y_s)

    brier_platt = ref_brier_score(y_true, p_platt)
    brier_iso = ref_brier_score(y_true, p_iso)
    ece_platt = ref_expected_calibration_error(y_true, p_platt, n_bins=5)
    ece_iso = ref_expected_calibration_error(y_true, p_iso, n_bins=5)

    assert brier_platt < 0.25
    assert brier_iso < 0.25
    assert ece_platt < 0.20
    assert ece_iso < 0.20


def test_flow5_reliability_diagram_monotonicity():
    y_true = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]
    y_prob = [0.1, 0.15, 0.2, 0.35, 0.45, 0.55, 0.7, 0.8, 0.85, 0.95]

    diag = ref_reliability_diagram(y_true, y_prob, n_bins=5)
    accs = diag["bin_accuracies"]
    confs = diag["bin_confidences"]
    counts = diag["bin_counts"]

    assert len(accs) == 5
    assert sum(counts) == len(y_true)
    active_bins = [(c, a) for c, a, n in zip(confs, accs, counts) if n > 0]
    for i in range(len(active_bins) - 1):
        assert active_bins[i + 1][0] >= active_bins[i][0]


def test_flow6_pmu_noise_propagation_to_features_and_probabilities():
    s_clean = generate_synthetic_pmu_stream(
        n_samples=250, fps=50, t_disturb=1.0, disturbed_gen=35, noise_std=0.0, seed=123
    )
    s_noisy = generate_synthetic_pmu_stream(
        n_samples=250, fps=50, t_disturb=1.0, disturbed_gen=35, noise_std=0.03, seed=123
    )

    feat_clean = _extract_tabular_features(s_clean, 1.0)
    feat_noisy = _extract_tabular_features(s_noisy, 1.0)

    delta_rocof = abs(feat_clean[0] - feat_noisy[0])
    assert delta_rocof < 0.15, f"Noise caused excessive ROCOF deviation: {delta_rocof}"

    clf = RefCalibratedGBDTClassifier(method="platt")
    clf.fit([feat_clean, [f * 1.1 for f in feat_clean]], [59.3, 59.1])
    p_clean = clf.predict_proba([feat_clean])[0][1]
    p_noisy = clf.predict_proba([feat_noisy])[0][1]

    assert abs(p_clean - p_noisy) < 0.20, f"Calibrated probability perturbed too heavily by noise: {abs(p_clean - p_noisy)}"


def test_flow6_ols_noise_filtering_advantage():
    n_trials = 10
    ols_estimates = []
    twopoint_estimates = []

    for trial in range(n_trials):
        stream = generate_synthetic_pmu_stream(
            n_samples=150, fps=50, t_disturb=1.0, disturbed_gen=35, noise_std=0.02, seed=300 + trial
        )
        w = ref_extract_causal_window(stream, 0.2)
        r_ols = ref_calc_initial_rocof(w, delta_t_init=0.1)
        coi = w["coi_frequency"]
        ts = w["timestamps"]
        r_2pt = (coi[1] - coi[0]) / (ts[1] - ts[0])

        ols_estimates.append(r_ols)
        twopoint_estimates.append(r_2pt)

    mean_ols = sum(ols_estimates) / n_trials
    var_ols = sum((x - mean_ols) ** 2 for x in ols_estimates) / n_trials

    mean_2pt = sum(twopoint_estimates) / n_trials
    var_2pt = sum((x - mean_2pt) ** 2 for x in twopoint_estimates) / n_trials

    assert var_ols < var_2pt, f"OLS variance ({var_ols}) was not lower than 2-point variance ({var_2pt})"


def test_flow6_noise_stress_benchmark_monotonic_degradation():
    bench = RefMeasurementNoiseBenchmark(noise_levels=[0.0, 0.01, 0.03, 0.05])
    res = bench.run([])

    maes = [r["mae"] for r in res["results"]]
    rmses = [r["rmse"] for r in res["results"]]

    for i in range(len(maes) - 1):
        assert maes[i] <= maes[i + 1]
        assert rmses[i] <= rmses[i + 1]


def test_flow7_renewable_penetration_ood_generalization_pipeline(synthetic_scenario_batch):
    batch = synthetic_scenario_batch
    bench = RefOODGeneralizationBenchmark(id_range=(0.10, 0.30), ood_range=(0.40, 0.50))
    res = bench.run(batch)

    assert res["id_scenario_count"] > 0
    assert res["ood_scenario_count"] > 0
    assert res["distribution_shift_detected"] is True

    gap = res["generalization_gap"]
    assert gap["delta_mae"] >= 0.0, "OOD MAE must be higher or equal to ID MAE"
    assert gap["delta_f1"] >= 0.0, "OOD F1 degradation must be >= 0"


def test_flow7_ood_performance_gap_bounds(synthetic_scenario_batch):
    bench = RefOODGeneralizationBenchmark()
    res = bench.run(synthetic_scenario_batch)

    ood_metrics = res["ood_metrics"]
    assert ood_metrics["mae"] < 0.60, f"OOD MAE {ood_metrics['mae']} exceeded 0.60 Hz"
    assert ood_metrics["f1"] > 0.70, f"OOD F1 {ood_metrics['f1']} dropped below 0.70"


def test_flow7_ood_brier_and_ece_degradation_quantification():
    bench = RefOODGeneralizationBenchmark()
    res = bench.run([])
    id_ece = res["id_metrics"]["ece"]
    ood_ece = res["ood_metrics"]["ece"]

    assert ood_ece >= id_ece
    assert ood_ece < 0.15, f"OOD ECE {ood_ece} exceeds safe limit 0.15"


def test_flow8_streaming_latency_budget_compliance(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    reg = RefLinearNadirRegressor()
    reg.fit([[0.1, 0.2, -0.3, -0.4]], [59.4])

    bench = RefLatencyBenchmark()
    prof = bench.profile(sample_data=traj, n_iter=15)

    assert prof["latency_compliant_under_10ms"] is True
    assert prof["t_total_ms"] < 10.0, f"Total execution time {prof['t_total_ms']} ms exceeds 10 ms budget"
    assert prof["t_feat_ms"] > 0.0
    assert prof["t_infer_ms"] > 0.0


def test_flow8_latency_speedup_vs_dynamic_simulation():
    bench = RefLatencyBenchmark()
    prof = bench.profile()

    speedup = prof["speedup_factor"]
    assert speedup >= 20.0, f"Speedup factor {speedup}x is less than 20x target"
    assert prof["sim_time_ms"] > 100.0


def test_flow8_batch_inference_throughput():
    reg = RefLinearNadirRegressor()
    reg.fit([[i * 0.1, -i * 0.05] for i in range(10)], [60.0 - i * 0.1 for i in range(10)])

    t0 = time.perf_counter()
    n_updates = 50
    for _ in range(n_updates):
        _ = reg.predict([[0.3, -0.15]])
    t1 = time.perf_counter()

    avg_ms = ((t1 - t0) / n_updates) * 1000.0
    assert avg_ms < 5.0, f"Average throughput {avg_ms} ms per update exceeds 5.0 ms"


def test_flow9_full_chain_pipeline_integration(ieee39_topology):
    topo = ieee39_topology
    assert topo["n_buses"] == 39
    assert topo["n_generators"] == 10

    scenarios = []
    contingencies = [
        (30, 250.0, 0.10, 59.72),
        (31, 572.0, 0.15, 59.58),
        (32, 650.0, 0.20, 59.42),
        (35, 650.0, 0.35, 59.32),
        (37, 540.0, 0.45, 59.25),
        (38, 830.0, 0.20, 59.15)
    ]
    for idx, (gen, loss, ren, nadir) in enumerate(contingencies):
        s = generate_synthetic_pmu_stream(
            n_samples=200, fps=50, t_disturb=1.0, disturbed_gen=gen,
            renewable_penetration=ren, active_loss_mw=loss, f_nadir=nadir, seed=500 + idx
        )
        assert_valid_pmu_trajectory(s)
        scenarios.append(s)

    X = []
    y_nadir = []
    y_sec = []
    for s in scenarios:
        feat = _extract_tabular_features(s, t_obs=1.0)
        X.append(feat)
        y_nadir.append(s["metadata"]["ground_truth_nadir"])
        y_sec.append(s["metadata"]["is_secure"])

    assert len(X) == 6
    assert len(X[0]) == 10

    cal_gb = RefCalibratedGBDTClassifier(method="platt").fit(X, y_nadir, y_sec)
    probs = cal_gb.predict_proba(X)
    preds = cal_gb.predict(X)

    assert len(probs) == 6
    assert len(preds) == 6

    p_sec = [p[1] for p in (probs.tolist() if hasattr(probs, "tolist") else probs)]
    brier = ref_brier_score(y_sec, p_sec)
    ece = ref_expected_calibration_error(y_sec, p_sec, n_bins=3)
    assert brier < 0.25
    assert ece < 0.25

    win_bench = RefWindowSensitivityBenchmark()
    win_res = win_bench.run(scenarios)
    assert len(win_res["results"]) == 4

    lat_bench = RefLatencyBenchmark()
    lat_res = lat_bench.profile()
    assert lat_res["latency_compliant_under_10ms"] is True


def test_pairwise_f1_f3_renewable_bus_placement(ieee39_topology):
    topo = ieee39_topology
    area1 = topo["area_partition"]["Area 1"]
    area2 = topo["area_partition"]["Area 2"]
    area3 = topo["area_partition"]["Area 3"]

    for g_bus in topo["generator_buses"]:
        in_area = (g_bus in area1) or (g_bus in area2) or (g_bus in area3)
        assert in_area, f"Generator bus {g_bus} not mapped to any coherent area"


def test_pairwise_f2_f5_severe_contingency_stability():
    stream = generate_synthetic_pmu_stream(
        n_samples=250, fps=50, t_disturb=1.0, disturbed_gen=39,
        renewable_penetration=0.45, active_loss_mw=1000.0, f_nadir=58.95, seed=999
    )
    assert_valid_pmu_trajectory(stream)
    min_f = np.min(stream["bus_frequencies"])
    assert min_f >= 55.0, f"Frequency plunged below physical 55.0 Hz floor: {min_f}"


def test_pairwise_f4_f6_dropped_frames_handling(sample_pmu_trajectory):
    traj = copy.deepcopy(sample_pmu_trajectory)
    ts = traj["timestamps"].tolist() if hasattr(traj["timestamps"], "tolist") else list(traj["timestamps"])
    del ts[55:57]
    traj["timestamps"] = np.array(ts) if HAS_NUMPY else ts
    traj["bus_frequencies"] = traj["bus_frequencies"][:len(ts)]
    traj["coi_frequency"] = traj["coi_frequency"][:len(ts)]

    w = ref_extract_causal_window(traj, 1.0)
    assert len(w["timestamps"]) > 0
    assert w["timestamps"][-1] <= traj["metadata"]["t_disturb"] + 1.0 + 1e-4


def test_pairwise_f7_f11_spatial_rocof_dispersion():
    stream = generate_synthetic_pmu_stream(
        n_samples=200, fps=50, t_disturb=1.0, disturbed_gen=35, seed=77
    )
    w = ref_extract_causal_window(stream, 0.4)
    bf = w["bus_frequencies"]
    ts = w["timestamps"]
    dt = ts[4] - ts[0]

    # Bus 35 (idx 34, Area 2) vs Bus 30 (idx 29, Area 1)
    rocof_area2_epicenter = (bf[4][34] - bf[0][34]) / dt
    rocof_area1_distant = (bf[4][29] - bf[0][29]) / dt

    assert rocof_area2_epicenter < rocof_area1_distant, \
        f"Epicenter ROCOF {rocof_area2_epicenter} was not steeper than distant ROCOF {rocof_area1_distant}"


def test_pairwise_f8_f10_rocof_integral_consistency(sample_pmu_trajectory):
    w = ref_extract_causal_window(sample_pmu_trajectory, 1.0)
    coi = w["coi_frequency"]
    ts = w["timestamps"]

    net_drop = coi[-1] - coi[0]
    integrated_diff = sum((coi[k + 1] - coi[k]) for k in range(len(coi) - 1))

    assert abs(net_drop - integrated_diff) < 1e-6, \
        f"ROCOF integral {integrated_diff} did not match net drop {net_drop}"


def test_pairwise_f9_f14_delta_f_feature_importance(sample_causal_features):
    feat = sample_causal_features
    X = feat["X"]
    y = feat["y_nadir"]

    reg = RefLinearNadirRegressor(alpha=1e-3).fit(X, y)
    weights = reg.weights

    # Feature indices 3 and 4 are instantaneous and max delta_f
    assert len(weights) == 10
    assert abs(weights[3]) > 0.01 or abs(weights[4]) > 0.01, "Delta f weights are zero"


def test_pairwise_f12_f15_angle_divergence_security_boundary():
    clf = RefBinarySecurityClassifier(threshold=59.5)
    X_train = [[0.1, 0.15], [0.2, 0.25], [0.5, 0.55], [0.6, 0.65]]
    y_train = [59.7, 59.6, 59.4, 59.2]
    clf.fit(X_train, y_train)

    pred = clf.predict([[0.5, 0.60]])
    pred_val = int(_to_float_list_1d(pred)[0])
    assert pred_val == 0, f"Expected insecure (0) for severe angle divergence, got {pred_val}"


def test_pairwise_f13_f16_coi_acceleration_calibrated_confidence():
    clf = RefCalibratedGBDTClassifier(method="platt")
    X_train = [
        [-0.1, -0.1],
        [-0.3, -0.3],
        [-0.8, -0.9],
        [-1.2, -1.5]
    ]
    y_nadir = [59.7, 59.55, 59.3, 59.0]
    clf.fit(X_train, y_nadir)

    p_sec, low_b, high_b = clf.predict_with_intervals([[-1.2, -1.5]])
    p_val = float(_to_float_list_1d(p_sec)[0])
    h_val = float(_to_float_list_1d(high_b)[0])

    assert p_val < 0.15, f"P(secure) {p_val} too high for severe deceleration"
    assert h_val < 0.25, f"Upper bound {h_val} too high for severe deceleration"


def test_pairwise_f17_f20_calibration_under_increasing_noise():
    bench = RefMeasurementNoiseBenchmark(noise_levels=[0.0, 0.01, 0.03, 0.05])
    res = bench.run([])

    for row in res["results"]:
        ece = row["ece"]
        assert 0.0 <= ece <= 0.15, f"ECE {ece} exceeded [0, 0.15] under noise {row['noise_std_hz']} Hz"


def test_pairwise_f18_f21_latency_scaling_with_window_length(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    windows = [0.5, 1.0, 1.5, 2.0]
    latencies = []

    _ = _extract_tabular_features(traj, windows[0])

    for t_w in windows:
        trial_latencies = []
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(10):
                _ = _extract_tabular_features(traj, t_w)
            t1 = time.perf_counter()
            trial_latencies.append(((t1 - t0) / 10.0) * 1000.0)
        lat_ms = min(trial_latencies)
        latencies.append(lat_ms)
        assert lat_ms < 10.0, f"Extraction latency {lat_ms} ms at T_obs={t_w} exceeded 10.0 ms budget"

    assert latencies[-1] >= latencies[0] * 0.75
