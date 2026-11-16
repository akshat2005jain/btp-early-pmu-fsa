"""Stress and numerical edge-case tests for simulation and feature extraction."""

import math
import copy
import time
import struct
import warnings
import pytest
import numpy as np

from pmu_fsa.simulation.ieee39_data import (
    BASE_MVA,
    NOMINAL_FREQ,
    N_BUSES,
    GENERATOR_BUSES,
    get_ieee39_bus_data,
    get_ieee39_branch_data,
    get_ieee39_gen_data,
)
from pmu_fsa.simulation.dynamic_engine import (
    DynamicSimulator,
    build_ybus,
    compute_kron_reduction,
)
from pmu_fsa.simulation.renewables import (
    RenewablePenetrationConfig,
    compute_effective_inertias,
    apply_renewable_displacement,
)
from pmu_fsa.simulation.pmu_recorder import (
    PMURecorder,
    PMUNoiseConfig,
    compute_crc_ccitt,
    compute_tve,
    compute_fir_rocof,
    apply_pmu_measurement_noise,
)
from pmu_fsa.simulation.scenario_generator import (
    ScenarioGenerator,
    assign_scenario_partition,
    create_scenario_id,
)
from pmu_fsa.features.window_slicer import (
    CausalWindow,
    extract_causal_window,
    slice_scenario_batch,
)
from pmu_fsa.features.extractor import (
    FeatureExtractor,
    extract_features,
    calc_initial_rocof,
    calc_rocof_metrics,
    calc_delta_f,
    calc_freq_integral,
    calc_interarea_dispersion,
    calc_angle_divergence,
    calc_coi_acceleration,
    wrap_to_pi,
    _validate_and_clean_nans,
    CORE_FEATURE_NAMES,
    DESCRIPTIVE_FEATURE_NAMES,
)


def test_t5_sim_kron_reduction_zero_load():
    Y_red, K, E_mag0, delta0, Pm0 = compute_kron_reduction(load_scaling=0.0, return_all=True)
    assert Y_red.shape == (10, 10)
    assert K.shape == (39, 10)
    cond_num = np.linalg.cond(Y_red)
    assert cond_num < 25.0, f"Y_red condition number {cond_num} too high at zero load"
    # Line charging and generator shunts keep diagonal conductance positive
    diag_g = np.real(np.diag(Y_red))
    assert np.all(diag_g > 0.05), f"Diagonal conductance non-positive at zero load: {diag_g}"
    assert not np.isnan(Y_red).any()
    assert not np.isnan(K).any()


def test_t5_sim_kron_reduction_heavy_load_5x():
    Y_red, K, E_mag0, delta0, Pm0 = compute_kron_reduction(load_scaling=5.0, return_all=True)
    assert Y_red.shape == (10, 10)
    cond_num = np.linalg.cond(Y_red)
    assert cond_num < 30.0, f"Y_red condition number {cond_num} too high at 5x load"
    assert not np.isnan(Y_red).any()


def test_t5_sim_zero_damping_oscillation():
    sim = DynamicSimulator(load_scaling=1.0)
    sim.D[:] = 0.0
    out = sim.simulate(t_span=(0.0, 2.0), t_disturb=0.5, tripped_bus=35, p_loss_mw=650.0, fps=50)
    bf = out["bus_frequencies"]
    assert not np.isnan(bf).any()
    assert not np.isinf(bf).any()
    nadir = out["metadata"]["ground_truth_nadir"]
    assert (NOMINAL_FREQ - 5.0) <= nadir <= NOMINAL_FREQ


def test_t5_sim_extreme_damping_stiff():
    sim = DynamicSimulator(load_scaling=1.0)
    sim.D[:] = 50.0
    out = sim.simulate(t_span=(0.0, 1.5), t_disturb=0.5, tripped_bus=35, p_loss_mw=650.0, fps=50)
    bf = out["bus_frequencies"]
    assert not np.isnan(bf).any()
    nadir = out["metadata"]["ground_truth_nadir"]
    assert nadir >= (NOMINAL_FREQ - 0.40)


def test_t5_sim_zero_inertia_singularity_detection():
    sim = DynamicSimulator(load_scaling=1.0)
    sim.H[0] = 0.0
    y0 = np.concatenate([sim.delta0, np.zeros(sim.n_gen), sim.Pm0])
    with pytest.raises(ValueError, match="strictly positive"):
        sim.rhs(0.0, y0, 1.0, 650.0, 0)


def test_t5_sim_ultra_low_positive_inertia_nadir():
    sim = DynamicSimulator(load_scaling=1.0)
    sim.H[:] = 0.1
    sim.H_tot = float(np.sum(sim.H))
    out = sim.simulate(t_span=(0.0, 1.0), t_disturb=0.2, tripped_bus=35, p_loss_mw=650.0, fps=50)
    nadir = out["metadata"]["ground_truth_nadir"]
    assert nadir < (NOMINAL_FREQ * (52.0 / 60.0)), f"Expected deep nadir < {NOMINAL_FREQ * (52.0 / 60.0):.2f} Hz under H=0.1s, got {nadir}"
    assert out["metadata"]["is_secure"] == 0
    assert not np.isnan(out["bus_frequencies"]).any()


def test_t5_sim_non_generator_bus_trip_metadata_gap():
    sim = DynamicSimulator()
    out = sim.simulate(t_span=(0.0, 1.0), t_disturb=0.2, tripped_bus=15, p_loss_mw=650.0, fps=50)
    # Non-generator bus 15 produces no trip dynamically while metadata still records bus 15
    nadir = out["metadata"]["ground_truth_nadir"]
    assert nadir == NOMINAL_FREQ, f"Expected flat {NOMINAL_FREQ} Hz for non-generator trip, got {nadir}"
    assert out["metadata"]["disturbed_gen"] == 15
    assert out["metadata"]["active_loss_mw"] == 650.0


def test_t5_sim_flat_run_identity():
    sim = DynamicSimulator()
    out = sim.simulate(t_span=(0.0, 1.0), t_disturb=0.2, tripped_bus=None, p_loss_mw=0.0, fps=50)
    bf = out["bus_frequencies"]
    assert np.allclose(bf, NOMINAL_FREQ, atol=1e-5)
    assert out["metadata"]["disturbed_gen"] == 0
    assert out["metadata"]["active_loss_mw"] == 0.0
    assert out["metadata"]["is_secure"] == 1


def test_t5_sim_overfrequency_generation_surplus():
    sim = DynamicSimulator()
    out = sim.simulate(t_span=(0.0, 1.0), t_disturb=0.2, tripped_bus=35, p_loss_mw=-500.0, fps=50)
    bf = out["bus_frequencies"]
    max_f = float(np.max(bf))
    assert max_f > (NOMINAL_FREQ + 0.04), f"Expected frequency surge > {NOMINAL_FREQ + 0.04} Hz, got max {max_f}"
    assert out["metadata"]["is_secure"] == 1


def test_t5_sim_disturbance_after_simulation_end():
    sim = DynamicSimulator()
    out = sim.simulate(t_span=(0.0, 1.0), t_disturb=5.0, tripped_bus=35, p_loss_mw=650.0, fps=50)
    nadir = out["metadata"]["ground_truth_nadir"]
    assert nadir == NOMINAL_FREQ


def test_t5_sim_reconstruct_zero_voltage_singularity():
    sim = DynamicSimulator()
    delta = np.zeros((1, 10))
    domega = np.ones((1, 10)) * 0.01
    sim_copy = copy.deepcopy(sim)
    sim_copy.K[0, :] = 0.0
    _, _, bus_freq, _ = sim_copy.reconstruct_bus_states(delta, domega)
    assert not np.isnan(bus_freq[0, 0]), "Bus 0 frequency was NaN under zero voltage phasor"
    assert not np.isinf(bus_freq[0, 0]), "Bus 0 frequency was Inf under zero voltage phasor"


def test_t5_sim_ode_solver_method_equivalences():
    sim = DynamicSimulator()
    out_rk = sim.simulate(t_span=(0.0, 1.0), t_disturb=0.2, tripped_bus=35, p_loss_mw=650.0, fps=50, method="RK45")
    out_rad = sim.simulate(t_span=(0.0, 1.0), t_disturb=0.2, tripped_bus=35, p_loss_mw=650.0, fps=50, method="Radau")
    nadir_rk = out_rk["metadata"]["ground_truth_nadir"]
    nadir_rad = out_rad["metadata"]["ground_truth_nadir"]
    assert abs(nadir_rk - nadir_rad) < 0.05, f"Solver nadir mismatch: RK45={nadir_rk}, Radau={nadir_rad}"


def test_t5_ren_displacement_saturation_plateau():
    gen_data = get_ieee39_gen_data()
    cfg_40 = RenewablePenetrationConfig(p_ren=0.40)
    cfg_45 = RenewablePenetrationConfig(p_ren=0.45)
    cfg_80 = RenewablePenetrationConfig(p_ren=0.80)

    h_40 = sum(compute_effective_inertias(gen_data, cfg_40).values())
    h_45 = sum(compute_effective_inertias(gen_data, cfg_45).values())
    h_80 = sum(compute_effective_inertias(gen_data, cfg_80).values())

    # Default displaced units saturate at alpha=1.0 once p_ren >= 0.414
    assert h_40 > h_45
    assert abs(h_45 - 662.00) < 1e-4
    assert abs(h_80 - 662.00) < 1e-4


def test_t5_ren_expanded_displacement_pool_deep_low_inertia():
    gen_data = get_ieee39_gen_data()
    all_gen_buses = [g["bus"] for g in gen_data]
    cfg_80_all = RenewablePenetrationConfig(p_ren=0.80, displaced_buses=all_gen_buses)
    h_80 = sum(compute_effective_inertias(gen_data, cfg_80_all).values())
    base_H = sum(g["H"] for g in gen_data)
    assert h_80 < base_H * 0.35, f"Effective inertia {h_80} did not scale down under expanded displacement pool"


def test_t5_ren_virtual_inertia_support_damping():
    gen_data = get_ieee39_gen_data()
    cfg_no_vi = RenewablePenetrationConfig(p_ren=0.40, h_vir=0.0)
    cfg_with_vi = RenewablePenetrationConfig(p_ren=0.40, h_vir=5.0)

    h_no = sum(compute_effective_inertias(gen_data, cfg_no_vi).values())
    h_vi = sum(compute_effective_inertias(gen_data, cfg_with_vi).values())
    assert h_vi > h_no
    assert h_vi - h_no == pytest.approx(4 * (0.40 * 6150.1 / 2548.0) * 5.0, abs=1.0)


def test_t5_ren_config_boundary_validation():
    with pytest.raises(ValueError, match="p_ren"):
        RenewablePenetrationConfig(p_ren=-0.1)
    with pytest.raises(ValueError, match="p_ren"):
        RenewablePenetrationConfig(p_ren=1.2)
    with pytest.raises(ValueError, match="t_inv"):
        RenewablePenetrationConfig(t_inv=0.0)
    with pytest.raises(ValueError, match="t_inv"):
        RenewablePenetrationConfig(t_inv=-0.05)


def test_t5_pmu_c37_118_reporting_rate_enforcement():
    for valid_fps in [10, 25, 30, 50, 60, 100, 120]:
        rec = PMURecorder(fps=valid_fps)
        assert rec.fps == valid_fps

    for invalid_fps in [-50, 0, 15, 45, 75, 200]:
        with pytest.raises(ValueError, match="reporting rate must be"):
            PMURecorder(fps=invalid_fps)


def test_t5_pmu_frame_parser_truncated_buffer_defect():
    for short_len in [0, 8, 15, 17]:
        with pytest.raises(ValueError, match="too short"):
            PMURecorder.parse_frame(bytes(short_len))

    # 16-byte header + 2-byte CRC with missing payload
    header = struct.pack(">HHHIIH", 0xAA01, 18, 1, 1700000000, 0, 0)
    crc = compute_crc_ccitt(header)
    frame_18 = header + struct.pack(">H", crc)
    with pytest.raises(ValueError, match="Corrupted or truncated PMU frame"):
        PMURecorder.parse_frame(frame_18)


def test_t5_pmu_fir_rocof_single_sample_gradient_crash():
    single_f = np.array([60.0])
    rocof = compute_fir_rocof(single_f, fps=50.0, window_length=5)
    assert rocof.shape == (1,)
    assert rocof[0] == 0.0


def test_t5_pmu_fir_rocof_exact_ramp_derivative():
    fps = 50.0
    dt = 1.0 / fps
    true_slope = -2.50
    t_ramp = np.arange(0.0, 1.0, dt)
    f_ramp = 60.0 + true_slope * t_ramp
    rocof = compute_fir_rocof(f_ramp, fps=fps, window_length=5)
    assert np.allclose(rocof[2:-2], true_slope, atol=1e-12)


def test_t5_pmu_tve_near_zero_voltage_protection():
    v_true_mag = np.array([0.0, 1e-8, 1.0])
    v_true_ang = np.array([0.0, 0.0, 0.0])
    v_meas_mag = np.array([0.01, 1e-8, 1.01])
    v_meas_ang = np.array([0.0, 0.0, 0.0])
    tve = compute_tve(v_true_mag, v_true_ang, v_meas_mag, v_meas_ang)
    assert not np.isnan(tve).any()
    assert not np.isinf(tve).any()


def test_t5_pmu_extreme_gaussian_noise_voltage_floor():
    cfg = PMUNoiseConfig(sigma_v=0.50, seed=123)
    f = np.full((100, 39), 60.0)
    rocof = np.zeros((100, 39))
    v_mag = np.full((100, 39), 0.02)
    v_ang = np.zeros((100, 39))

    _, _, v_noisy, _ = apply_pmu_measurement_noise(f, rocof, v_mag, v_ang, cfg)
    assert np.all(v_noisy >= 0.01), f"Noisy voltage dropped below 0.01 floor: min={np.min(v_noisy)}"


def test_t5_slice_sub_interval_sample_deficit_rejection():
    scen = {
        "timestamps": np.linspace(0.0, 2.0, 101),
        "bus_frequencies": np.full((101, 39), 60.0),
        "bus_rocof": np.zeros((101, 39)),
        "voltage_magnitudes": np.ones((101, 39)),
        "voltage_angles": np.zeros((101, 39)),
        "coi_frequency": np.full(101, 60.0),
        "metadata": {"t_disturb": 1.0, "fps": 50},
    }
    with pytest.raises(ValueError, match="insufficient samples"):
        extract_causal_window(scen, t_obs=0.001)


def test_t5_slice_exact_frame_quantization():
    scen = {
        "timestamps": np.linspace(0.0, 5.0, 251),
        "bus_frequencies": np.full((251, 39), 60.0),
        "bus_rocof": np.zeros((251, 39)),
        "voltage_magnitudes": np.ones((251, 39)),
        "voltage_angles": np.zeros((251, 39)),
        "coi_frequency": np.full(251, 60.0),
        "metadata": {"t_disturb": 1.0, "fps": 50},
    }
    w_002 = extract_causal_window(scen, t_obs=0.02)
    assert w_002.n_samples == 2
    w_010 = extract_causal_window(scen, t_obs=0.10)
    assert w_010.n_samples == 6
    w_050 = extract_causal_window(scen, t_obs=0.50)
    assert w_050.n_samples == 26
    w_100 = extract_causal_window(scen, t_obs=1.00)
    assert w_100.n_samples == 51


def test_t5_slice_trajectory_overrun_truncation_warning():
    scen = {
        "timestamps": np.linspace(0.0, 2.0, 101),
        "bus_frequencies": np.full((101, 39), 60.0),
        "bus_rocof": np.zeros((101, 39)),
        "voltage_magnitudes": np.ones((101, 39)),
        "voltage_angles": np.zeros((101, 39)),
        "coi_frequency": np.full(101, 60.0),
        "metadata": {"t_disturb": 1.0, "fps": 50},
    }
    with pytest.warns(UserWarning, match="exceeds trajectory duration"):
        w = extract_causal_window(scen, t_obs=3.0)
    assert w.metadata["is_truncated"] is True
    assert w.timestamps[-1] == 2.0


def test_t5_slice_strict_nadir_anti_leakage_exception():
    scen = {
        "timestamps": np.linspace(0.0, 5.0, 251),
        "bus_frequencies": np.full((251, 39), 60.0),
        "bus_rocof": np.zeros((251, 39)),
        "voltage_magnitudes": np.ones((251, 39)),
        "voltage_angles": np.zeros((251, 39)),
        "coi_frequency": np.full(251, 60.0),
        "metadata": {"t_disturb": 1.0, "ground_truth_t_nadir": 2.2, "fps": 50},
    }
    with pytest.raises(ValueError, match="Anti-leakage violation"):
        extract_causal_window(scen, t_obs=1.5, strict_nadir=True)


def test_t5_slice_clip_to_nadir_pre_nadir_truncation():
    scen = {
        "timestamps": np.linspace(0.0, 5.0, 251),
        "bus_frequencies": np.full((251, 39), 60.0),
        "bus_rocof": np.zeros((251, 39)),
        "voltage_magnitudes": np.ones((251, 39)),
        "voltage_angles": np.zeros((251, 39)),
        "coi_frequency": np.full(251, 60.0),
        "metadata": {"t_disturb": 1.0, "ground_truth_t_nadir": 2.2, "fps": 50},
    }
    with pytest.warns(UserWarning, match="reaches or exceeds nadir time"):
        w = extract_causal_window(scen, t_obs=2.0, clip_to_nadir=True)
    assert w.timestamps[-1] <= 2.2


def test_t5_slice_memory_isolation_invariance():
    scen = {
        "timestamps": np.linspace(0.0, 2.0, 101),
        "bus_frequencies": np.full((101, 39), 60.0),
        "bus_rocof": np.zeros((101, 39)),
        "voltage_magnitudes": np.ones((101, 39)),
        "voltage_angles": np.zeros((101, 39)),
        "coi_frequency": np.full(101, 60.0),
        "metadata": {"t_disturb": 1.0, "fps": 50},
    }
    w = extract_causal_window(scen, t_obs=0.5)
    w.bus_frequencies[0, 0] = 999.0
    assert scen["bus_frequencies"][50, 0] == 60.0


def test_t5_feat_steady_state_constant_zero_identity():
    w_flat = CausalWindow(
        timestamps=np.linspace(1.0, 2.0, 51),
        bus_frequencies=np.full((51, 39), 60.0),
        bus_rocof=np.zeros((51, 39)),
        voltage_magnitudes=np.ones((51, 39)),
        voltage_angles=np.zeros((51, 39)),
        coi_frequency=np.full(51, 60.0),
        metadata={"t_disturb": 1.0, "fps": 50},
    )
    feats = extract_features(w_flat)
    for k in CORE_FEATURE_NAMES + DESCRIPTIVE_FEATURE_NAMES:
        assert abs(feats[k]) < 1e-9, f"Feature {k} was non-zero on flat trajectory: {feats[k]}"


def test_t5_feat_analytical_linear_ramp_equivalence():
    dt = 0.02
    t_ramp = np.arange(1.0, 2.0 + 0.5 * dt, dt)
    slope = -2.50
    coi_ramp = 60.0 + slope * (t_ramp - 1.0)
    w_ramp = CausalWindow(
        timestamps=t_ramp,
        bus_frequencies=np.tile(coi_ramp[:, None], (1, 39)),
        bus_rocof=np.full((len(t_ramp), 39), slope),
        voltage_magnitudes=np.ones((len(t_ramp), 39)),
        voltage_angles=np.zeros((len(t_ramp), 39)),
        coi_frequency=coi_ramp,
        metadata={"t_disturb": 1.0, "fps": 50},
    )
    feats = extract_features(w_ramp)
    assert abs(feats["init_rocof"] - slope) < 1e-4
    assert abs(feats["pk_rocof"] - slope) < 1e-4
    assert abs(feats["mean_rocof"] - slope) < 1e-4
    assert abs(feats["inst_drop"] - slope * 1.0) < 1e-4


def test_t5_feat_zero_frequency_deviation_integral():
    t_sym = np.linspace(1.0, 3.0, 101)
    coi_sym = 60.0 + np.sin(2.0 * np.pi * (t_sym - 1.0))
    w_sym = CausalWindow(
        timestamps=t_sym,
        bus_frequencies=np.tile(coi_sym[:, None], (1, 39)),
        bus_rocof=np.zeros((101, 39)),
        voltage_magnitudes=np.ones((101, 39)),
        voltage_angles=np.zeros((101, 39)),
        coi_frequency=coi_sym,
        metadata={"t_disturb": 1.0, "fps": 50},
    )
    feats = extract_features(w_sym)
    assert abs(feats["freq_integral_dev_coi"]) < 1e-12
    assert feats["freq_integral_abs_coi"] > 1.0


def test_t5_feat_phase_angle_circular_wrap_around():
    # Bus 4 (idx 3) at +179.5 deg, Bus 5 (idx 4) at -179.5 deg -> 1.0 deg wrap
    angles = np.zeros((2, 39))
    angles[-1, :] = 179.5
    angles[-1, 4] = -179.5
    angles[-1, 20:] = -179.5
    w_wrap = CausalWindow(
        timestamps=np.array([1.0, 1.02]),
        bus_frequencies=np.full((2, 39), 60.0),
        bus_rocof=np.zeros((2, 39)),
        voltage_magnitudes=np.ones((2, 39)),
        voltage_angles=angles,
        coi_frequency=np.array([60.0, 60.0]),
        metadata={"t_disturb": 1.0, "fps": 50},
    )
    feats = extract_features(w_wrap, angle_unit="deg")
    assert abs(feats["angle_div_max"] - 1.0) < 1e-4
    assert abs(feats["tie_line_angle_diff"] - 1.0) < 1e-4


def test_t5_feat_nan_cleaning_physical_default_fallback():
    all_nan = np.full((10, 39), np.nan)
    cleaned = _validate_and_clean_nans(all_nan, "bus_frequencies", policy="ffill")
    assert not np.isnan(cleaned).any()
    assert np.allclose(cleaned, NOMINAL_FREQ)


def test_t5_feat_disconnected_empty_area_dispersion():
    empty_area_partition = {1: [1, 2, 3], 2: [4, 5, 6], 3: []}
    w = CausalWindow(
        timestamps=np.linspace(1.0, 1.2, 10),
        bus_frequencies=np.full((10, 39), 59.8),
        bus_rocof=np.zeros((10, 39)),
        voltage_magnitudes=np.ones((10, 39)),
        voltage_angles=np.zeros((10, 39)),
        coi_frequency=np.full(10, 59.8),
        metadata={"t_disturb": 1.0, "fps": 50},
    )
    feats = extract_features(w, area_partition=empty_area_partition)
    assert feats["area_spread_max"] == 0.0


def test_t5_feat_sub_2ms_throughput_guarantee():
    w = CausalWindow(
        timestamps=np.linspace(1.0, 2.0, 51),
        bus_frequencies=np.full((51, 39), 59.8),
        bus_rocof=np.full((51, 39), -0.2),
        voltage_magnitudes=np.ones((51, 39)),
        voltage_angles=np.zeros((51, 39)),
        coi_frequency=np.full(51, 59.8),
        metadata={"t_disturb": 1.0, "fps": 50},
    )
    _ = extract_features(w)

    timings = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(100):
            _ = extract_features(w)
        timings.append(((time.perf_counter() - t0) / 100) * 1000.0)

    elapsed_ms = min(timings)
    assert elapsed_ms < 2.0, f"Feature extraction time {elapsed_ms:.3f} ms exceeded 2.0 ms budget"
