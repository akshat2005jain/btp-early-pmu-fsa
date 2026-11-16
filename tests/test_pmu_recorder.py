"""Tests for the IEEE C37.118 PMU measurement recorder."""

import math
import numpy as np
import pytest

from pmu_fsa.simulation.pmu_recorder import (
    PMURecorder,
    PMUNoiseConfig,
    apply_pmu_measurement_noise,
    compute_tve,
    compute_fir_rocof,
    compute_crc_ccitt,
)
from pmu_fsa.simulation.dynamic_engine import DynamicSimulator


@pytest.fixture
def clean_simulation_trajectory():
    sim = DynamicSimulator(load_scaling=1.0)
    return sim.simulate(
        t_span=(0.0, 3.0),
        t_disturb=1.0,
        tripped_bus=35,
        p_loss_mw=650.0,
        fps=100,
    )


def test_sampling_rate_verification(clean_simulation_trajectory):
    raw_traj = clean_simulation_trajectory

    rec50 = PMURecorder(fps=50)
    stream50 = rec50.record_trajectory(raw_traj)
    t50 = stream50["timestamps"]
    diffs50 = np.diff(t50)
    expected_dt50 = 1.0 / 50.0

    assert np.allclose(diffs50, expected_dt50, atol=1e-8), (
        f"50 fps sampling delta mismatch: max error = {np.max(np.abs(diffs50 - expected_dt50))}"
    )
    assert len(t50) == int(round(3.0 * 50)) + 1
    assert stream50["metadata"]["fps"] == 50

    rec60 = PMURecorder(fps=60)
    stream60 = rec60.record_trajectory(raw_traj)
    t60 = stream60["timestamps"]
    diffs60 = np.diff(t60)
    expected_dt60 = 1.0 / 60.0

    assert np.allclose(diffs60, expected_dt60, atol=1e-8), (
        f"60 fps sampling delta mismatch: max error = {np.max(np.abs(diffs60 - expected_dt60))}"
    )
    assert len(t60) == int(round(3.0 * 60)) + 1
    assert stream60["metadata"]["fps"] == 60


def test_ieee_c37_118_noise_and_error_limits(clean_simulation_trajectory):
    raw_traj = clean_simulation_trajectory

    noise_cfg = PMUNoiseConfig(
        sigma_f=0.003,
        sigma_rocof=0.01,
        sigma_v=0.0002,
        sigma_theta_deg=0.03,
        seed=123,
    )

    rec = PMURecorder(fps=50, noise_config=noise_cfg)
    clean_stream = rec.record_trajectory(raw_traj, inject_noise=False)
    noisy_stream = rec.record_trajectory(raw_traj, inject_noise=True)

    v_true = clean_stream["voltage_magnitudes"]
    ang_true = clean_stream["voltage_angles"]
    v_meas = noisy_stream["voltage_magnitudes"]
    ang_meas = noisy_stream["voltage_angles"]

    tve = compute_tve(v_true, ang_true, v_meas, ang_meas)

    mean_tve = np.mean(tve)
    max_tve = np.max(tve)
    p99_tve = np.percentile(tve, 99)

    assert p99_tve < 0.010, f"99th percentile TVE ({p99_tve * 100:.3f}%) exceeds IEEE C37.118 1.0% limit"
    assert mean_tve < 0.005, f"Mean TVE ({mean_tve * 100:.3f}%) too high"

    f_true = clean_stream["bus_frequencies"]
    f_meas = noisy_stream["bus_frequencies"]
    fe = np.abs(f_meas - f_true)

    mean_fe = np.mean(fe)
    p95_fe = np.percentile(fe, 95)
    assert mean_fe < 0.005, f"Mean Frequency Error ({mean_fe:.4f} Hz) exceeds 0.005 Hz"


def test_binary_frame_serialization_and_crc(clean_simulation_trajectory):
    rec = PMURecorder(fps=50, id_code=42)
    pmu_stream = rec.record_trajectory(clean_simulation_trajectory)

    for frame_idx in [0, 10, 50, len(pmu_stream["timestamps"]) - 1]:
        frame_bytes = rec.serialize_frame(frame_idx, pmu_stream)

        assert len(frame_bytes) >= 16, "Serialized frame is shorter than header"
        # 39 * 8 (phasors) + 8 (freq/rocof) + 16 (header) + 2 (crc) = 338 bytes
        assert len(frame_bytes) == 338, f"Expected 338 bytes for 39-bus frame, got {len(frame_bytes)}"

        parsed = PMURecorder.parse_frame(frame_bytes)
        assert parsed["sync"] == 0xAA01, f"Invalid SYNC: {hex(parsed['sync'])}"
        assert parsed["frame_size"] == len(frame_bytes), "FRAMESIZE field mismatch"
        assert parsed["id_code"] == 42, f"IDCODE mismatch: {parsed['id_code']}"
        assert parsed["n_phasors"] == 39, f"Phasor count mismatch: {parsed['n_phasors']}"
        assert parsed["crc_valid"] is True, "CRC validation failed"

        expected_soc = int(pmu_stream["soc"][frame_idx])
        expected_fracsec = int(pmu_stream["fracsec"][frame_idx])
        assert parsed["soc"] == expected_soc
        assert parsed["fracsec"] == expected_fracsec

        expected_f = float(pmu_stream["coi_frequency"][frame_idx])
        assert abs(parsed["freq_hz"] - expected_f) < 1e-4

        corrupted_bytes = bytearray(frame_bytes)
        corrupted_bytes[10] ^= 0xFF
        with pytest.raises(ValueError, match="CRC-CCITT mismatch"):
            PMURecorder.parse_frame(bytes(corrupted_bytes))


def test_fir_filter_noise_resilient_rocof():
    fps = 50.0
    dt = 1.0 / fps
    t = np.arange(0.0, 4.0, dt)

    # f(t) = 60 - 0.5*t - 0.1*t^2  =>  df/dt = -0.5 - 0.2*t
    f_clean = 60.0 - 0.5 * t - 0.1 * (t ** 2)
    rocof_true = -0.5 - 0.2 * t

    rng = np.random.default_rng(999)
    noise = rng.normal(0.0, 0.01, size=len(t))
    f_noisy = f_clean + noise

    raw_rocof_noisy = np.gradient(f_noisy, dt)
    fir_rocof = compute_fir_rocof(f_noisy, fps=fps, window_length=5)

    fir_clean = compute_fir_rocof(f_clean, fps=fps, window_length=5)
    interior = slice(5, -5)
    max_clean_error = np.max(np.abs(fir_clean[interior] - rocof_true[interior]))
    assert max_clean_error < 1e-3, f"FIR filter introduced distortion on clean signal: {max_clean_error}"

    raw_rmse = np.sqrt(np.mean((raw_rocof_noisy[interior] - rocof_true[interior]) ** 2))
    fir_rmse = np.sqrt(np.mean((fir_rocof[interior] - rocof_true[interior]) ** 2))

    assert fir_rmse < raw_rmse, f"FIR filter failed to reduce ROCOF RMSE: fir={fir_rmse:.4f}, raw={raw_rmse:.4f}"
