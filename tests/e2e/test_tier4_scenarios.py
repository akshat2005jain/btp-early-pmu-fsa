"""End-to-end disturbance and grid security assessment scenario tests."""

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
)

from tests.e2e.test_tier3_combinations import (
    _extract_tabular_features,
    FEATURE_NAMES,
)


def test_scenario_1_severe_generator_outage_high_load():
    stream = generate_synthetic_pmu_stream(
        n_samples=300,
        fps=50,
        f_nominal=60.0,
        f_nadir=59.18,
        t_disturb=1.0,
        disturbed_gen=35,
        renewable_penetration=0.20,
        active_loss_mw=650.0,
        seed=101
    )
    assert_valid_pmu_trajectory(stream)
    assert stream["metadata"]["ground_truth_nadir"] < 59.20
    assert stream["metadata"]["is_secure"] == 0

    t_obs = 1.0
    w = ref_extract_causal_window(stream, t_obs)
    assert w["timestamps"][-1] <= 1.0 + t_obs + 1e-4

    t_start = time.perf_counter()
    init_rocof = ref_calc_initial_rocof(w)
    peak_neg_r, max_abs_r, mean_r = ref_calc_rocof_metrics(w)
    inst_df, max_df = ref_calc_delta_f(w)
    area_spread, _ = ref_calc_interarea_dispersion(w)
    feat_vector = _extract_tabular_features(stream, t_obs)
    t_feat_ms = (time.perf_counter() - t_start) * 1000.0

    assert init_rocof < -0.45, f"Initial ROCOF {init_rocof} was not steep enough for 650 MW trip"
    assert inst_df < -0.30, f"Instantaneous drop {inst_df} too shallow"

    training_scenarios = [
        ([-0.15, 0.20, -0.10, -0.12, -0.15, -0.05, 0.02, 0.001, 0.08, -0.12], 59.75),
        ([-0.30, 0.45, -0.22, -0.28, -0.32, -0.12, 0.04, 0.002, 0.15, -0.25], 59.58),
        ([-0.42, 0.65, -0.35, -0.45, -0.50, -0.20, 0.06, 0.004, 0.22, -0.40], 59.42),
        ([-0.65, 0.95, -0.52, -0.68, -0.75, -0.32, 0.09, 0.007, 0.35, -0.65], 59.18),
    ]
    X_train = [row[0] for row in training_scenarios]
    y_train = [row[1] for row in training_scenarios]

    reg = RefLinearNadirRegressor().fit(X_train, y_train)
    cal_gb = RefCalibratedGBDTClassifier(method="platt").fit(X_train, y_train)

    _ = _extract_tabular_features(stream, t_obs)
    _ = reg.predict([feat_vector])
    _ = cal_gb.predict_proba([feat_vector])

    timings = []
    for _ in range(5):
        t0 = time.perf_counter()
        fv = _extract_tabular_features(stream, t_obs)
        _ = reg.predict([fv])
        _ = cal_gb.predict_proba([fv])
        timings.append((time.perf_counter() - t0) * 1000.0)
    total_latency_ms = min(timings)

    pred_nadir = float(_to_float_list_1d(reg.predict([feat_vector]))[0])
    prob_pair = cal_gb.predict_proba([feat_vector])[0]
    p_sec = float(prob_pair[1] if hasattr(prob_pair, '__getitem__') else prob_pair)
    p_insec = float(prob_pair[0] if hasattr(prob_pair, '__getitem__') else 1.0 - p_sec)

    assert pred_nadir < 59.35, f"Predicted nadir {pred_nadir} failed to detect severe violation (<59.35 Hz)"
    assert p_sec < 0.15, f"Calibrated P(secure) {p_sec} is dangerously high for severe trip"
    assert p_insec > 0.85, f"Insecurity confidence {p_insec} below 85%"
    assert total_latency_ms < 10.0, f"Control-center alert latency {total_latency_ms} ms exceeded 10 ms budget"


def test_scenario_2_high_renewable_low_inertia_shift():
    ren_penetration = 0.45
    stream = generate_synthetic_pmu_stream(
        n_samples=250,
        fps=50,
        f_nominal=60.0,
        f_nadir=59.12,
        t_disturb=1.0,
        disturbed_gen=31,
        renewable_penetration=ren_penetration,
        active_loss_mw=572.0,
        seed=202
    )

    base_H_sum = sum(p["H"] for p in IEEE39_GEN_PARAMS.values())
    eff_H = stream["metadata"]["effective_H_sum"]
    assert eff_H < base_H_sum * 0.70, f"Effective inertia {eff_H} did not decrease to <70% of base {base_H_sum}"

    w = ref_extract_causal_window(stream, 1.0)
    init_rocof = ref_calc_initial_rocof(w)
    init_accel, mean_accel = ref_calc_coi_acceleration(w)
    feat_vec = _extract_tabular_features(stream, 1.0)

    assert init_accel < -0.50, f"COI deceleration {init_accel} was not severe enough for low-inertia shift"

    X_cal = [
        [-0.20, 0.25, -0.15, -0.15, -0.20, -0.06, 0.03, 0.001, 0.10, -0.15],
        [-0.38, 0.50, -0.28, -0.32, -0.38, -0.14, 0.05, 0.003, 0.18, -0.32],
        [-0.72, 1.05, -0.58, -0.75, -0.85, -0.36, 0.10, 0.008, 0.38, -0.75],
        [-0.90, 1.30, -0.70, -0.92, -1.05, -0.45, 0.14, 0.012, 0.48, -0.95],
    ]
    y_cal = [59.70, 59.55, 59.20, 59.05]
    cal_gb = RefCalibratedGBDTClassifier(method="platt").fit(X_cal, y_cal)

    p_sec, low_b, high_b = cal_gb.predict_with_intervals([feat_vec], confidence_level=0.95)
    p_val = float(_to_float_list_1d(p_sec)[0])
    h_val = float(_to_float_list_1d(high_b)[0])

    assert p_val < 0.10, f"Calibrated P(secure) {p_val} must be < 0.10 for severe low-inertia trip"
    assert h_val < 0.30, f"Confidence interval upper bound {h_val} exceeds 0.30"


def test_scenario_3_multi_area_boundary_tie_line_oscillation():
    # Gen 32 sits in Area 1 near the Area 2 boundary
    stream = generate_synthetic_pmu_stream(
        n_samples=250,
        fps=50,
        t_disturb=1.0,
        disturbed_gen=32,
        f_nadir=59.38,
        active_loss_mw=650.0,
        seed=303
    )

    w = ref_extract_causal_window(stream, 1.0)
    area_spread, spatial_var = ref_calc_interarea_dispersion(w)
    max_ang_div, tie_ang = ref_calc_angle_divergence(w)

    assert area_spread > 0.01, f"Area frequency spread {area_spread} too small for boundary disturbance"
    assert spatial_var > 0.0001, f"Cross-bus spatial variance {spatial_var} too small"
    assert max_ang_div > 0.30, f"Maximum angle divergence {max_ang_div} rad below expected 0.30 rad"
    assert tie_ang >= 0.0, "Tie-line angle difference must be non-negative"

    clf = RefBinarySecurityClassifier(threshold=59.5)
    X_tr = [[0.02, 0.10], [0.04, 0.20], [0.08, 0.35], [0.12, 0.50]]
    y_tr = [59.70, 59.58, 59.38, 59.20]
    clf.fit(X_tr, y_tr)

    pred = clf.predict([[area_spread, max_ang_div]])
    pred_label = int(_to_float_list_1d(pred)[0])
    assert pred_label == 0, "Model failed to classify severe inter-area divergence disturbance as insecure"


def test_scenario_4_marginal_disturbance_near_nadir_threshold():
    # 59.49 Hz falls 0.01 Hz below the 59.50 Hz security threshold
    stream = generate_synthetic_pmu_stream(
        n_samples=250,
        fps=50,
        f_nominal=60.0,
        f_nadir=59.49,
        t_disturb=1.0,
        disturbed_gen=30,
        renewable_penetration=0.15,
        active_loss_mw=180.0,
        seed=404
    )
    assert stream["metadata"]["ground_truth_nadir"] == 59.49
    assert stream["metadata"]["is_secure"] == 0

    feat_vec = _extract_tabular_features(stream, 1.0)

    nadirs = [59.70, 59.60, 59.50, 59.40, 59.30]
    X_cal = []
    for n_val in nadirs:
        s_tmp = generate_synthetic_pmu_stream(
            n_samples=200, f_nadir=n_val, t_disturb=1.0, disturbed_gen=30, renewable_penetration=0.15
        )
        X_cal.append(_extract_tabular_features(s_tmp, 1.0))
    cal_gb = RefCalibratedGBDTClassifier(method="platt").fit(X_cal, nadirs)

    probs = cal_gb.predict_proba([feat_vec])
    p_sec = float(probs[0][1])

    assert 0.35 <= p_sec <= 0.60, f"Calibrated probability {p_sec} is overconfident for marginal 59.49 Hz nadir"

    p_s, low_b, high_b = cal_gb.predict_with_intervals([feat_vec])
    ci_width = float(high_b[0] - low_b[0])
    assert ci_width >= 0.15, f"Confidence interval width {ci_width} too narrow for boundary case"


def test_scenario_5_noisy_pmu_stream_with_jitter_and_dropped_frames():
    s_clean = generate_synthetic_pmu_stream(
        n_samples=250, fps=50, t_disturb=1.0, disturbed_gen=35, active_loss_mw=650.0, f_nadir=59.35, noise_std=0.0, seed=505
    )
    s_noisy = generate_synthetic_pmu_stream(
        n_samples=250, fps=50, t_disturb=1.0, disturbed_gen=35, active_loss_mw=650.0, f_nadir=59.35, noise_std=0.03, seed=505
    )

    from tests.e2e.conftest import _MiniNP
    ts_list = _to_float_list_1d(s_noisy["timestamps"])
    drop_indices = [35, 36, 75, 110, 180]
    kept_indices = [i for i in range(len(ts_list)) if i not in drop_indices]

    s_jittered = copy.deepcopy(s_noisy)
    s_jittered["timestamps"] = np.array([ts_list[i] for i in kept_indices]) if HAS_NUMPY else _MiniNP.array([ts_list[i] for i in kept_indices])
    bf = s_jittered["bus_frequencies"]
    s_jittered["bus_frequencies"] = np.array([bf[i].tolist() if hasattr(bf[i], "tolist") else list(bf[i]) for i in kept_indices]) if HAS_NUMPY else _MiniNP.array([bf[i].tolist() if hasattr(bf[i], "tolist") else list(bf[i]) for i in kept_indices])
    coi = s_jittered["coi_frequency"]
    s_jittered["coi_frequency"] = np.array([coi[i] for i in kept_indices]) if HAS_NUMPY else _MiniNP.array([coi[i] for i in kept_indices])

    w_clean = ref_extract_causal_window(s_clean, 1.0)
    w_noisy = ref_extract_causal_window(s_jittered, 1.0)

    rocof_clean = ref_calc_initial_rocof(w_clean, delta_t_init=0.1)
    rocof_noisy = ref_calc_initial_rocof(w_noisy, delta_t_init=0.1)

    rocof_err = abs(rocof_clean - rocof_noisy)
    assert rocof_err < 0.12, f"Noise + dropped frames caused excessive OLS ROCOF deviation: {rocof_err} Hz/s"

    feat_clean = _extract_tabular_features(s_clean, 1.0)
    feat_noisy = _extract_tabular_features(s_jittered, 1.0)

    for v in feat_noisy:
        assert not math.isnan(v), "NaN found in noisy extracted features"
        assert not math.isinf(v), "Inf found in noisy extracted features"

    cal_gb = RefCalibratedGBDTClassifier(method="platt")
    cal_gb.fit([feat_clean, [f * 1.2 for f in feat_clean]], [59.35, 59.10])
    p_clean = cal_gb.predict_proba([feat_clean])[0][1]
    p_noisy = cal_gb.predict_proba([feat_noisy])[0][1]

    assert (p_clean >= 0.5) == (p_noisy >= 0.5), "Noise caused security classification flip!"


def test_scenario_6_ultra_early_0_5s_vs_standard_1_5s_comparison(synthetic_scenario_batch):
    batch = synthetic_scenario_batch
    y_true = [s["metadata"]["ground_truth_nadir"] for s in batch]
    y_sec = [s["metadata"]["is_secure"] for s in batch]

    X_05s = [_extract_tabular_features(s, 0.5) for s in batch]
    X_15s = [_extract_tabular_features(s, 1.5) for s in batch]

    reg_05 = RefLinearNadirRegressor().fit(X_05s, y_true)
    reg_15 = RefLinearNadirRegressor().fit(X_15s, y_true)

    preds_05 = _to_float_list_1d(reg_05.predict(X_05s))
    preds_15 = _to_float_list_1d(reg_15.predict(X_15s))

    mae_05 = sum(abs(p - y) for p, y in zip(preds_05, y_true)) / len(y_true)
    mae_15 = sum(abs(p - y) for p, y in zip(preds_15, y_true)) / len(y_true)

    assert mae_15 <= mae_05 + 1e-4, f"Standard 1.5s window MAE ({mae_15}) worse than 0.5s ({mae_05})"

    cal_05 = RefCalibratedGBDTClassifier(method="platt").fit(X_05s, y_true)
    cal_15 = RefCalibratedGBDTClassifier(method="platt").fit(X_15s, y_true)

    p_05 = [p[1] for p in cal_05.predict_proba(X_05s).tolist()]
    p_15 = [p[1] for p in cal_15.predict_proba(X_15s).tolist()]

    brier_05 = ref_brier_score(y_sec, p_05)
    brier_15 = ref_brier_score(y_sec, p_15)
    ece_05 = ref_expected_calibration_error(y_sec, p_05, n_bins=3)
    ece_15 = ref_expected_calibration_error(y_sec, p_15, n_bins=3)

    assert brier_05 < 0.20, f"Ultra-early Brier score {brier_05} exceeded 0.20"
    assert brier_15 < 0.20, f"Standard Brier score {brier_15} exceeded 0.20"
    assert ece_05 < 0.35, f"Ultra-early ECE {ece_05} exceeded 0.35"
    assert ece_15 < 0.30, f"Standard ECE {ece_15} exceeded 0.30"
    assert ece_15 <= ece_05 + 1e-4, f"ECE at 1.5s ({ece_15}) was not lower or comparable to 0.5s ({ece_05})"


def test_scenario_7_online_inference_latency_compliance_60fps(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    reg = RefLinearNadirRegressor()
    reg.fit([[0.1 * i, -0.05 * i] for i in range(10)], [60.0 - 0.1 * i for i in range(10)])

    _ = _extract_tabular_features(traj, 0.5)
    _ = reg.predict([[0.0, 0.0]])

    n_frames = 60
    latencies_ms = []

    for frame_idx in range(n_frames):
        frame_trials = []
        for _ in range(3):
            t0 = time.perf_counter()
            feat = _extract_tabular_features(traj, 0.5)
            _ = reg.predict([[feat[0], feat[3]]])
            t1 = time.perf_counter()
            frame_trials.append((t1 - t0) * 1000.0)
        latencies_ms.append(min(frame_trials))

    avg_latency = sum(latencies_ms) / len(latencies_ms)
    max_latency = max(latencies_ms)

    assert avg_latency < 5.0, f"Average stream processing latency {avg_latency:.2f} ms exceeded 5.0 ms target"
    assert max_latency < 10.0, f"Peak stream processing latency {max_latency:.2f} ms exceeded 10.0 ms budget"

    sim_time_ms = 350.0
    speedup = sim_time_ms / max(0.01, avg_latency)
    assert speedup > 30.0, f"Speedup factor {speedup:.1f}x below 30x requirement"


def test_scenario_8_full_end_to_end_pipeline_execution():
    configs = [
        (30, 250.0, 0.10, 59.72),
        (31, 572.0, 0.15, 59.58),
        (32, 650.0, 0.20, 59.42),
        (33, 632.0, 0.25, 59.38),
        (34, 508.0, 0.30, 59.51),
        (35, 650.0, 0.35, 59.32),
        (36, 560.0, 0.40, 59.35),
        (37, 540.0, 0.45, 59.28),
    ]

    scenarios = []
    for idx, (gen, loss, ren, nadir) in enumerate(configs):
        scen = generate_synthetic_pmu_stream(
            n_samples=250, fps=50, t_disturb=1.0, disturbed_gen=gen,
            renewable_penetration=ren, active_loss_mw=loss, f_nadir=nadir, seed=800 + idx
        )
        assert_valid_pmu_trajectory(scen)
        scenarios.append(scen)

    dataset_05 = [_extract_tabular_features(s, 0.5) for s in scenarios]
    dataset_10 = [_extract_tabular_features(s, 1.0) for s in scenarios]
    y_nadir = [s["metadata"]["ground_truth_nadir"] for s in scenarios]
    y_sec = [s["metadata"]["is_secure"] for s in scenarios]

    assert len(dataset_05) == len(configs)
    assert len(dataset_10) == len(configs)

    models = {
        "Linear": RefLinearNadirRegressor(),
        "DecisionTree": RefDecisionTreeNadirRegressor(max_depth=3),
        "RandomForest": RefRandomForestNadirRegressor(n_estimators=4, max_depth=3),
        "GradientBoosting": RefGradientBoostingNadirRegressor(n_estimators=5, max_depth=2)
    }

    regression_results = {}
    for name, m in models.items():
        m.fit(dataset_10, y_nadir)
        preds = _to_float_list_1d(m.predict(dataset_10))
        mae = sum(abs(p - y) for p, y in zip(preds, y_nadir)) / len(y_nadir)
        rmse = math.sqrt(sum((p - y) ** 2 for p, y in zip(preds, y_nadir)) / len(y_nadir))
        regression_results[name] = {"mae": round(mae, 4), "rmse": round(rmse, 4)}
        assert mae < 0.35, f"{name} MAE {mae} too high"

    cal_platt = RefCalibratedGBDTClassifier(method="platt").fit(dataset_10, y_nadir, y_sec)
    cal_iso = RefCalibratedGBDTClassifier(method="isotonic").fit(dataset_10, y_nadir, y_sec)

    p_platt = [p[1] for p in cal_platt.predict_proba(dataset_10).tolist()]
    p_iso = [p[1] for p in cal_iso.predict_proba(dataset_10).tolist()]

    brier_platt = ref_brier_score(y_sec, p_platt)
    brier_iso = ref_brier_score(y_sec, p_iso)
    ece_platt = ref_expected_calibration_error(y_sec, p_platt, n_bins=4)
    ece_iso = ref_expected_calibration_error(y_sec, p_iso, n_bins=4)

    assert brier_platt < 0.25
    assert brier_iso < 0.25
    assert ece_platt < 0.25
    assert ece_iso < 0.25

    sens_bench = RefWindowSensitivityBenchmark()
    sens_report = sens_bench.run(scenarios)
    assert len(sens_report["results"]) == 4

    ood_bench = RefOODGeneralizationBenchmark()
    ood_report = ood_bench.run(scenarios)
    assert ood_report["distribution_shift_detected"] is True

    noise_bench = RefMeasurementNoiseBenchmark()
    noise_report = noise_bench.run(scenarios)
    assert len(noise_report["results"]) == 5

    lat_bench = RefLatencyBenchmark()
    lat_report = lat_bench.profile()
    assert lat_report["latency_compliant_under_10ms"] is True


def test_scenario_9_under_frequency_load_shedding_readiness():
    stream = generate_synthetic_pmu_stream(
        n_samples=250,
        fps=50,
        f_nominal=60.0,
        f_nadir=59.12,
        t_disturb=1.0,
        disturbed_gen=38,
        renewable_penetration=0.35,
        active_loss_mw=830.0,
        seed=909
    )

    w = ref_extract_causal_window(stream, 0.8)
    init_rocof = ref_calc_initial_rocof(w)
    feat_vec = _extract_tabular_features(stream, 0.8)

    # UFLS Stage 1 threshold: 59.30 Hz
    ufls_thresh = 59.30
    reg = RefLinearNadirRegressor()
    reg.fit([
        [-0.20, -0.15],
        [-0.40, -0.32],
        [-0.75, -0.65],
        [-1.05, -0.95]
    ], [59.70, 59.50, 59.20, 59.05])

    pred_nadir = float(_to_float_list_1d(reg.predict([[feat_vec[0], feat_vec[3]]]))[0])

    assert pred_nadir < ufls_thresh, f"Predicted nadir {pred_nadir} failed to anticipate UFLS breach (<59.30 Hz)"
    ufls_armed = bool(pred_nadir < ufls_thresh)
    assert ufls_armed is True


def test_scenario_10_multi_area_boundary_islanding_prevention_alert():
    stream = generate_synthetic_pmu_stream(
        n_samples=250,
        fps=50,
        t_disturb=1.0,
        disturbed_gen=35,
        renewable_penetration=0.40,
        f_nadir=59.25,
        active_loss_mw=650.0,
        seed=1010
    )

    w = ref_extract_causal_window(stream, 1.0)
    max_ang_div, tie_ang = ref_calc_angle_divergence(w)

    assert max_ang_div > 0.35, f"Maximum angle divergence {max_ang_div} rad below expected 0.35 rad"

    out_of_step_thresh = 0.30
    out_of_step_alarm = bool(max_ang_div > out_of_step_thresh)
    assert out_of_step_alarm is True, "Failed to trigger out-of-step prevention alert"
