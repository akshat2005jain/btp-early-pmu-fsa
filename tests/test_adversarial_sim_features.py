"""Adversarial and numerical stress tests for 50 Hz simulation and 14-feature extraction."""

from __future__ import annotations

import struct
import time
import warnings

import numpy as np
import pytest

from pmu_fsa.features.extractor import (
    CANONICAL_FEATURE_NAMES,
    FeatureExtractor,
    _validate_and_clean_nans,
    extract_features,
)
from pmu_fsa.features.window_slicer import CausalWindow, extract_causal_window
from pmu_fsa.simulation.dynamic_engine import DynamicSimulator, compute_kron_reduction
from pmu_fsa.simulation.ieee39_data import NOMINAL_FREQ, get_ieee39_gen_data
from pmu_fsa.simulation.pmu_recorder import (
    PMURecorder,
    compute_causal_rocof,
    compute_crc_ccitt,
    compute_fir_rocof,
)
from pmu_fsa.simulation.renewables import (
    RenewablePenetrationConfig,
    apply_renewable_displacement,
    compute_effective_inertias,
    summarize_renewable_dispatch,
)


def test_probe_kron_reduction_load_scaling_extremes():
    for scale in (0.0, 0.001, 0.5, 1.0, 2.0, 3.0, 5.0):
        Y_red, K, E_mag0, delta0, Pm0 = compute_kron_reduction(load_scaling=scale, return_all=True)
        assert Y_red.shape == (10, 10)
        assert K.shape == (39, 10)
        assert not np.isnan(Y_red).any()
        assert not np.isnan(K).any()
        assert np.all(np.isfinite(E_mag0))
        assert np.all(np.isfinite(delta0))
        assert np.all(np.isfinite(Pm0))
        assert np.linalg.cond(Y_red) < 50.0


def test_probe_simulator_boundary_damping():
    nadirs = {}
    for d_val in (0.0, 2.0, 20.0, 50.0):
        sim = DynamicSimulator(load_scaling=1.0)
        sim.D[:] = d_val
        out = sim.simulate(t_span=(0.0, 3.0), t_disturb=0.5, tripped_bus=35, loss_fraction=1.0, fps=50)
        assert not np.isnan(out["bus_frequencies"]).any()
        assert not np.isinf(out["bus_frequencies"]).any()
        nadirs[d_val] = out["metadata"]["ground_truth_nadir"]
        assert 45.0 <= nadirs[d_val] < NOMINAL_FREQ

    assert nadirs[50.0] > nadirs[2.0] > nadirs[0.0]


def test_probe_low_and_zero_inertia_boundaries():
    for h_factor in (1.0, 0.2, 0.05):
        sim = DynamicSimulator(load_scaling=1.0)
        sim.H = sim.H * h_factor
        sim.H_tot = float(np.sum(sim.H))
        out = sim.simulate(t_span=(0.0, 1.5), t_disturb=0.2, tripped_bus=35, loss_fraction=0.70, fps=50)
        assert not np.isnan(out["bus_frequencies"]).any()
        assert out["metadata"]["ground_truth_nadir"] < NOMINAL_FREQ

    sim_zero = DynamicSimulator(load_scaling=1.0)
    sim_zero.H[0] = 0.0
    y0 = np.concatenate([sim_zero.delta0, np.zeros(sim_zero.n_gen), sim_zero.Pm0])
    with pytest.raises(ValueError, match="strictly positive"):
        sim_zero.rhs(0.0, y0, 1.0, 650.0, 0)
    with pytest.raises(ValueError, match="strictly positive"):
        sim_zero.simulate(t_span=(0.0, 1.0), t_disturb=0.2, tripped_bus=35, fps=50)


def test_probe_tripped_bus_and_disturbance_extremes():
    sim = DynamicSimulator()

    out_flat = sim.simulate(t_span=(0.0, 2.0), t_disturb=0.5, tripped_bus=None, p_loss_mw=0.0, fps=50)
    assert out_flat["metadata"]["ground_truth_nadir"] == pytest.approx(50.0, abs=1e-9)
    assert out_flat["metadata"]["is_secure"] == 1
    assert out_flat["metadata"]["disturbed_gen"] == 0

    with pytest.warns(UserWarning, match="not a valid generator bus"):
        out_non_gen = sim.simulate(t_span=(0.0, 2.0), t_disturb=0.5, tripped_bus=15, p_loss_mw=650.0, fps=50)
    assert out_non_gen["metadata"]["ground_truth_nadir"] == pytest.approx(50.0, abs=1e-9)

    out_surplus = sim.simulate(t_span=(0.0, 2.0), t_disturb=0.5, tripped_bus=35, p_loss_mw=-500.0, fps=50)
    assert float(np.max(out_surplus["bus_frequencies"])) > 50.04
    assert out_surplus["metadata"]["is_secure"] == 1

    out_after = sim.simulate(t_span=(0.0, 2.0), t_disturb=5.0, tripped_bus=35, loss_fraction=1.0, fps=50)
    assert out_after["metadata"]["ground_truth_nadir"] == pytest.approx(50.0, abs=1e-9)


def test_probe_renewables_linear_displacement_and_floor():
    gen_data = get_ieee39_gen_data(macro_intertie=False)
    base_h = sum(g["H"] for g in gen_data)
    assert base_h == pytest.approx(324.7)

    realized_linear = []
    for p_ren in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60):
        cfg = RenewablePenetrationConfig(p_ren=p_ren)
        mod = apply_renewable_displacement(gen_data, cfg, load_scaling=1.0)
        summary = summarize_renewable_dispatch(mod, load_scaling=1.0)
        realized_linear.append(summary["p_ren_realized"])
        assert summary["h_eff_total"] < base_h

    assert np.max(np.abs(np.diff(realized_linear, n=2))) < 1e-12

    for p_high in (0.80, 1.00):
        eff_h = compute_effective_inertias(gen_data, RenewablePenetrationConfig(p_ren=p_high))
        assert all(v >= 0.5 for v in eff_h.values())


def test_probe_pmu_frame_parsing_and_truncation():
    rec = PMURecorder(fps=50, f_nominal=50.0)
    sim = DynamicSimulator()
    raw = sim.simulate(t_span=(0.0, 0.5), t_disturb=0.1, tripped_bus=35, loss_fraction=1.0, fps=50)
    pmu_stream = rec.record_trajectory(raw)

    valid_frame = rec.serialize_frame(0, pmu_stream)
    assert len(valid_frame) == 338
    parsed = PMURecorder.parse_frame(valid_frame, f_nominal=50.0)
    assert parsed["sync"] == 0xAA01
    assert parsed["crc_valid"] is True
    assert parsed["freq_hz"] == pytest.approx(50.0, abs=1e-3)

    corrupt_crc = bytearray(valid_frame)
    corrupt_crc[-1] ^= 0xFF
    with pytest.raises(ValueError, match="CRC-CCITT mismatch"):
        PMURecorder.parse_frame(bytes(corrupt_crc), f_nominal=50.0)

    for test_len in (0, 8, 15, 16, 17, 18, 20, 24, 26):
        with pytest.raises(ValueError):
            PMURecorder.parse_frame(bytes(test_len), f_nominal=50.0)

    header = struct.pack(">HHHIIH", 0xAA01, 18, 1, 1700000000, 0, 0)
    crc = compute_crc_ccitt(header)
    with pytest.raises(ValueError, match="Corrupted or truncated"):
        PMURecorder.parse_frame(header + struct.pack(">H", crc), f_nominal=50.0)


def test_probe_pmu_reporting_rate_validation():
    for fps in (10, 25, 30, 50, 60, 100, 120):
        rec = PMURecorder(fps=fps)
        assert rec.fps == fps

    for bad_fps in (-10, 0, 15, 45, 75, 200):
        with pytest.raises(ValueError, match="reporting rate"):
            PMURecorder(fps=bad_fps)


def test_probe_causal_rocof_short_windows_and_ramp():
    fps = 50.0
    dt = 1.0 / fps
    true_slope = -2.5
    t_ramp = np.arange(0.0, 1.0, dt)
    f_ramp = 50.0 + true_slope * t_ramp

    rocof_ramp = compute_fir_rocof(f_ramp, fps=fps, window_length=5)
    assert np.max(np.abs(rocof_ramp[1:] - true_slope)) < 1e-12

    for n in (1, 2, 3, 4, 5):
        sub = compute_causal_rocof(f_ramp[:n], fps=fps, window_length=5)
        assert sub.shape == (n,)
        assert not np.isnan(sub).any()


def test_probe_window_slicer_extremes():
    scenario_mock = {
        "timestamps": np.linspace(0.0, 5.0, 251),
        "bus_frequencies": np.full((251, 39), 50.0),
        "bus_rocof": np.zeros((251, 39)),
        "voltage_magnitudes": np.ones((251, 39)),
        "voltage_angles": np.zeros((251, 39)),
        "pmu_sys_frequency": np.full(251, 50.0),
        "metadata": {
            "t_disturb": 1.0,
            "ground_truth_t_nadir": 2.5,
            "ground_truth_nadir": 49.30,
            "fps": 50,
            "f_nominal": 50.0,
        },
    }

    for sub_dt in (0.0001, 0.01):
        with pytest.raises(ValueError, match="insufficient samples"):
            extract_causal_window(scenario_mock, t_obs=sub_dt)

    w_min = extract_causal_window(scenario_mock, t_obs=0.02)
    assert w_min.n_samples == 2

    with pytest.warns(UserWarning, match="exceeds trajectory duration"):
        w_trunc = extract_causal_window(scenario_mock, t_obs=6.0)
    assert w_trunc.metadata["is_truncated"] is True

    with pytest.raises(ValueError, match="Anti-leakage violation"):
        extract_causal_window(scenario_mock, t_obs=2.0, strict_nadir=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        w_clip = extract_causal_window(scenario_mock, t_obs=2.0, clip_to_nadir=True)
    assert float(w_clip.timestamps[-1]) <= 2.5


def test_probe_feature_identities_wrap_nan_and_throughput():
    w_flat = CausalWindow(
        timestamps=np.linspace(1.0, 2.0, 51),
        bus_frequencies=np.full((51, 39), 50.0),
        bus_rocof=np.zeros((51, 39)),
        voltage_magnitudes=np.ones((51, 39)),
        voltage_angles=np.zeros((51, 39)),
        pmu_sys_frequency=np.full(51, 50.0),
        metadata={"t_disturb": 1.0, "fps": 50, "f_nominal": 50.0},
    )
    f_flat = extract_features(w_flat, f_nominal=50.0)
    assert list(f_flat.keys()) == CANONICAL_FEATURE_NAMES
    assert all(abs(v) < 1e-9 for v in f_flat.values())

    t_sym = np.linspace(1.0, 3.0, 101)
    f_sym_wave = 50.0 + np.sin(2.0 * np.pi * (t_sym - 1.0))
    w_sym = CausalWindow(
        timestamps=t_sym,
        bus_frequencies=np.tile(f_sym_wave[:, None], (1, 39)),
        bus_rocof=np.zeros((101, 39)),
        voltage_magnitudes=np.ones((101, 39)),
        voltage_angles=np.zeros((101, 39)),
        metadata={"t_disturb": 1.0, "fps": 50, "f_nominal": 50.0},
    )
    f_sym = extract_features(w_sym, f_nominal=50.0)
    assert abs(f_sym["freq_dev_integral"]) < 1e-12

    angles_wrap = np.zeros((10, 39))
    angles_wrap[0, :] = 179.0
    angles_wrap[-1, :] = 179.0
    angles_wrap[-1, 4] = -179.0
    w_wrap = CausalWindow(
        timestamps=np.linspace(1.0, 1.2, 10),
        bus_frequencies=np.full((10, 39), 50.0),
        bus_rocof=np.zeros((10, 39)),
        voltage_magnitudes=np.ones((10, 39)),
        voltage_angles=angles_wrap,
        metadata={"t_disturb": 1.0, "fps": 50, "f_nominal": 50.0},
    )
    f_wrap = extract_features(w_wrap, f_nominal=50.0, angle_unit="deg")
    assert f_wrap["angle_divergence_max"] == pytest.approx(2.0, abs=1e-6)
    assert f_wrap["tie_line_angle_excursion"] == pytest.approx(2.0, abs=1e-6)

    arr_nan = np.array([50.0, np.nan, 49.8, np.nan, 49.5])
    with pytest.raises(ValueError, match="contains NaN or Inf"):
        _validate_and_clean_nans(arr_nan, "bus_frequencies", policy="raise", f_nominal=50.0)
    cleaned = _validate_and_clean_nans(arr_nan, "bus_frequencies", policy="ffill", f_nominal=50.0)
    np.testing.assert_allclose(cleaned, [50.0, 50.0, 49.8, 49.8, 49.5])

    all_nan = np.full(5, np.nan)
    cleaned_all = _validate_and_clean_nans(all_nan, "bus_frequencies", policy="ffill", f_nominal=50.0)
    np.testing.assert_allclose(cleaned_all, 50.0)

    empty_area = {1: [1, 2, 3], 2: [4, 5, 6], 3: []}
    f_empty_area = extract_features(w_flat, f_nominal=50.0, area_partition=empty_area)
    assert f_empty_area["area_freq_spread_max"] == 0.0

    extractor = FeatureExtractor(t_obs=1.0, f_nominal=50.0)
    for _ in range(5):
        _ = extractor.extract_features_dict(w_flat)
    timings = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(50):
            _ = extractor.extract_features_dict(w_flat)
        timings.append(((time.perf_counter() - t0) / 50.0) * 1000.0)
    assert min(timings) < 6.0


def run_all_probes():
    test_probe_kron_reduction_load_scaling_extremes()
    test_probe_simulator_boundary_damping()
    test_probe_low_and_zero_inertia_boundaries()
    test_probe_tripped_bus_and_disturbance_extremes()
    test_probe_renewables_linear_displacement_and_floor()
    test_probe_pmu_frame_parsing_and_truncation()
    test_probe_pmu_reporting_rate_validation()
    test_probe_causal_rocof_short_windows_and_ramp()
    test_probe_window_slicer_extremes()
    test_probe_feature_identities_wrap_nan_and_throughput()


if __name__ == "__main__":
    run_all_probes()
