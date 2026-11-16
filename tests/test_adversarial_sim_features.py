"""Stress probes for simulation and feature extraction edge cases."""

import sys
import math
import copy
import time
import struct
import warnings
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
)


def run_all_probes():
    results = {}
    print("=" * 70)
    print("STARTING EMPIRICAL ADVERSARIAL STRESS PROBES")
    print("=" * 70)

    print("\n--- Probe 1.1: Kron Reduction Under Load Scaling Extremes ---")
    scales = [0.0, 0.001, 0.5, 1.0, 2.0, 3.0, 5.0, -0.5]
    kron_results = {}
    for s in scales:
        try:
            Y_red, K, E_mag0, delta0, Pm0 = compute_kron_reduction(load_scaling=s, return_all=True)
            cond_yred = np.linalg.cond(Y_red)
            diag_g = np.real(np.diag(Y_red))
            min_g = float(np.min(diag_g))
            kron_results[s] = {
                "success": True,
                "cond_yred": float(cond_yred),
                "min_diag_conductance": min_g,
                "has_nan": bool(np.isnan(Y_red).any()),
                "pm0_sum": float(np.sum(Pm0)),
            }
        except Exception as e:
            kron_results[s] = {"success": False, "error": str(e)}
    results["kron_load_scaling"] = kron_results
    for s, res in kron_results.items():
        print(f"  load_scale={s:6.3f}: success={res['success']}, cond={res.get('cond_yred', 'N/A')}, min_G={res.get('min_diag_conductance', 'N/A')}")

    print("\n--- Probe 1.2: Simulator Boundary Damping (D=0.0 and D=50.0) ---")
    sim_damping = {}
    for d_val in [0.0, 2.0, 20.0, 50.0]:
        try:
            sim = DynamicSimulator(load_scaling=1.0)
            sim.D[:] = d_val
            out = sim.simulate(t_span=(0.0, 2.0), t_disturb=0.5, tripped_bus=35, p_loss_mw=650.0, fps=50)
            nadir = out["metadata"]["ground_truth_nadir"]
            has_nan = bool(np.isnan(out["bus_frequencies"]).any())
            sim_damping[d_val] = {"success": True, "nadir": nadir, "has_nan": has_nan}
        except Exception as e:
            sim_damping[d_val] = {"success": False, "error": str(e)}
    results["sim_damping"] = sim_damping
    for d_val, res in sim_damping.items():
        print(f"  D={d_val:5.1f}: success={res['success']}, nadir={res.get('nadir', 'N/A')}, has_nan={res.get('has_nan', 'N/A')}")

    print("\n--- Probe 1.3: Low and Zero Inertia Boundaries ---")
    inertia_results = {}
    for h_factor in [1.0, 0.2, 0.05, 0.01]:
        try:
            sim = DynamicSimulator(load_scaling=1.0)
            sim.H = sim.H * h_factor
            sim.H_tot = float(np.sum(sim.H))
            out = sim.simulate(t_span=(0.0, 1.0), t_disturb=0.2, tripped_bus=35, p_loss_mw=650.0, fps=50)
            inertia_results[h_factor] = {"success": True, "has_nan": bool(np.isnan(out["bus_frequencies"]).any()), "nadir": out["metadata"]["ground_truth_nadir"]}
        except Exception as e:
            inertia_results[h_factor] = {"success": False, "error": type(e).__name__ + ": " + str(e)}

    sim_zero = DynamicSimulator(load_scaling=1.0)
    sim_zero.H[0] = 0.0
    y0_zero = np.concatenate([sim_zero.delta0, np.zeros(sim_zero.n_gen), sim_zero.Pm0])
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        dydt_zero = sim_zero.rhs(0.0, y0_zero, 1.0, 650.0, 0)
    has_nan_zero = bool(np.isnan(dydt_zero).any())
    inertia_results["zero_H_rhs_nan"] = has_nan_zero
    inertia_results["zero_H_warning"] = len(w_list) > 0
    results["inertia_scaling"] = inertia_results
    for h_factor, res in inertia_results.items():
        print(f"  H_factor={h_factor}: {res}")

    print("\n--- Probe 1.4: Tripped Bus & Disturbance Extremes ---")
    disturb_cases = {
        "flat_run_none": {"bus": None, "loss": 0.0},
        "non_gen_bus_15": {"bus": 15, "loss": 650.0},
        "negative_loss_surplus": {"bus": 35, "loss": -500.0},
        "t_disturb_after_end": {"bus": 35, "loss": 650.0, "t_disturb": 5.0, "t_span": (0.0, 2.0)},
    }
    disturb_results = {}
    for name, cfg in disturb_cases.items():
        try:
            sim = DynamicSimulator()
            t_span = cfg.get("t_span", (0.0, 2.0))
            t_disturb = cfg.get("t_disturb", 0.5)
            out = sim.simulate(t_span=t_span, t_disturb=t_disturb, tripped_bus=cfg["bus"], p_loss_mw=cfg["loss"], fps=50)
            disturb_results[name] = {
                "success": True,
                "nadir": out["metadata"]["ground_truth_nadir"],
                "is_secure": out["metadata"]["is_secure"],
                "disturbed_gen_meta": out["metadata"]["disturbed_gen"],
                "active_loss_meta": out["metadata"]["active_loss_mw"],
            }
        except Exception as e:
            disturb_results[name] = {"success": False, "error": str(e)}
    results["disturb_cases"] = disturb_results
    for name, res in disturb_results.items():
        print(f"  {name}: success={res['success']}, nadir={res.get('nadir', 'N/A')}, meta_gen={res.get('disturbed_gen_meta', 'N/A')}")

    print("\n--- Probe 2.1: Renewables Displacement Saturation Curve ---")
    gen_data = get_ieee39_gen_data()
    base_H = sum(g["H"] for g in gen_data)
    ren_sweep = {}
    for p_ren in [0.0, 0.10, 0.20, 0.30, 0.40, 0.414, 0.45, 0.50, 0.60, 0.80, 1.00]:
        cfg = RenewablePenetrationConfig(p_ren=p_ren)
        eff_h = compute_effective_inertias(gen_data, cfg)
        tot_eff_h = sum(eff_h.values())
        ren_sweep[p_ren] = {
            "tot_H": tot_eff_h,
            "displaced_H": {b: eff_h[b] for b in [32, 34, 36, 38]},
        }
    results["ren_sweep"] = ren_sweep
    print(f"  Base H sum: {base_H:.2f}s")
    for p_ren, data in ren_sweep.items():
        print(f"  p_ren={p_ren:5.3f}: tot_H={data['tot_H']:.2f}s, bus32_H={data['displaced_H'][32]:.2f}s")

    print("\n--- Probe 3.1: PMU Recorder Frame Parsing Boundary Checks ---")
    rec = PMURecorder(fps=50)
    sim = DynamicSimulator()
    raw = sim.simulate(t_span=(0.0, 0.5), t_disturb=0.1, tripped_bus=35, p_loss_mw=650.0, fps=50)
    pmu_stream = rec.record_trajectory(raw)
    valid_frame = rec.serialize_frame(0, pmu_stream)
    print(f"  Valid serialized frame size: {len(valid_frame)} bytes")

    parsed = PMURecorder.parse_frame(valid_frame)
    print(f"  Valid parse: SYNC=0x{parsed['sync']:04X}, crc_valid={parsed['crc_valid']}, freq={parsed['freq_hz']:.3f} Hz")

    corrupt_crc_frame = bytearray(valid_frame)
    corrupt_crc_frame[-1] ^= 0xFF
    try:
        PMURecorder.parse_frame(bytes(corrupt_crc_frame))
        crc_test = "FAIL: Did not catch corrupt CRC"
    except ValueError as e:
        crc_test = f"PASS: Caught corrupt CRC ({e})"
    print(f"  CRC check: {crc_test}")

    trunc_results = {}
    for test_len in [0, 8, 15, 16, 17, 18, 20, 24, 26]:
        dummy_buf = bytes(test_len)
        try:
            PMURecorder.parse_frame(dummy_buf)
            trunc_results[test_len] = "Passed (Unexpected!)"
        except Exception as e:
            trunc_results[test_len] = f"{type(e).__name__}: {e}"
    results["frame_truncation"] = trunc_results
    for length, err in trunc_results.items():
        print(f"  Len {length:2d}: {err}")

    print("\n--- Probe 3.2: PMU Reporting Rate Rejections ---")
    fps_tests = {}
    for fps in [10, 25, 30, 50, 60, 100, 120]:
        try:
            r = PMURecorder(fps=fps)
            fps_tests[fps] = "Allowed"
        except ValueError as e:
            fps_tests[fps] = f"Rejected: {e}"
    results["fps_tests"] = fps_tests
    for fps, stat in fps_tests.items():
        print(f"  fps={fps:3d}: {stat}")

    print("\n--- Probe 3.3: FIR ROCOF Filter Boundary Cases ---")
    fir_tests = {}
    fps = 50.0
    dt = 1.0 / fps
    true_slope = -2.5
    t_ramp = np.arange(0.0, 1.0, dt)
    f_ramp = 60.0 + true_slope * t_ramp
    rocof_ramp = compute_fir_rocof(f_ramp, fps=fps, window_length=5)
    interior_err = np.max(np.abs(rocof_ramp[2:-2] - true_slope))
    fir_tests["ramp_interior_max_err"] = float(interior_err)
    print(f"  Ramp interior max error: {interior_err:.8e} (expected ~0)")

    for n in [1, 2, 3, 4, 5]:
        f_sub = f_ramp[:n]
        try:
            rocof_sub = compute_fir_rocof(f_sub, fps=fps, window_length=5)
            fir_tests[f"n_samples_{n}"] = {
                "success": True,
                "shape": rocof_sub.shape,
                "has_nan": bool(np.isnan(rocof_sub).any()),
            }
        except Exception as e:
            fir_tests[f"n_samples_{n}"] = {"success": False, "error": str(e)}
    results["fir_tests"] = fir_tests
    for k, v in fir_tests.items():
        print(f"  FIR {k}: {v}")

    print("\n--- Probe 4.1: Window Slicer Boundary Extremes ---")
    ws_results = {}
    scenario_mock = {
        "timestamps": np.linspace(0.0, 5.0, 251),
        "bus_frequencies": np.full((251, 39), 60.0),
        "bus_rocof": np.zeros((251, 39)),
        "voltage_magnitudes": np.ones((251, 39)),
        "voltage_angles": np.zeros((251, 39)),
        "coi_frequency": np.full(251, 60.0),
        "metadata": {
            "t_disturb": 1.0,
            "ground_truth_t_nadir": 2.5,
            "ground_truth_nadir": 59.3,
            "fps": 50,
        },
    }

    for tobs in [0.0001, 0.01, 0.02, 0.1, 1.0, 1.5, 4.0, 6.0]:
        try:
            w = extract_causal_window(scenario_mock, t_obs=tobs)
            ws_results[f"tobs_{tobs}"] = {
                "success": True,
                "n_samples": w.n_samples,
                "duration": w.duration,
                "truncated": w.metadata.get("is_truncated", False),
            }
        except Exception as e:
            ws_results[f"tobs_{tobs}"] = {"success": False, "error": f"{type(e).__name__}: {e}"}

    try:
        extract_causal_window(scenario_mock, t_obs=2.0, strict_nadir=True)
        ws_results["strict_nadir_fail"] = "Did not catch nadir violation"
    except ValueError as e:
        ws_results["strict_nadir_pass"] = f"Caught violation: {e}"

    w_clip = extract_causal_window(scenario_mock, t_obs=2.0, clip_to_nadir=True)
    ws_results["clip_to_nadir_samples"] = w_clip.n_samples
    ws_results["clip_to_nadir_t_end"] = float(w_clip.timestamps[-1])

    results["window_slicer"] = ws_results
    for k, v in ws_results.items():
        print(f"  Slicer {k}: {v}")

    print("\n--- Probe 5.1: Feature Extractor Mathematical & Physical Identities ---")
    feat_results = {}

    w_flat = CausalWindow(
        timestamps=np.linspace(1.0, 2.0, 51),
        bus_frequencies=np.full((51, 39), 60.0),
        bus_rocof=np.zeros((51, 39)),
        voltage_magnitudes=np.ones((51, 39)),
        voltage_angles=np.zeros((51, 39)),
        coi_frequency=np.full(51, 60.0),
        metadata={"t_disturb": 1.0, "fps": 50},
    )
    f_flat = extract_features(w_flat)
    flat_all_zero = all(abs(v) < 1e-9 for k, v in f_flat.items())
    feat_results["flat_all_zero"] = flat_all_zero
    print(f"  Flat trajectory all features zero: {flat_all_zero}")

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
    f_sym = extract_features(w_sym)
    integ_dev = f_sym["freq_integral_dev_coi"]
    integ_abs = f_sym["freq_integral_abs_coi"]
    feat_results["sym_integ_dev"] = float(integ_dev)
    feat_results["sym_integ_abs"] = float(integ_abs)
    print(f"  Symmetric swing: integ_dev={integ_dev:.6e} (expected ~0), integ_abs={integ_abs:.4f} (expected > 0)")

    angles_wrap = np.zeros((10, 39))
    angles_wrap[-1, 0] = 179.0
    angles_wrap[-1, 1] = -179.0
    angles_wrap[-1, 3] = 179.0
    angles_wrap[-1, 4] = -179.0
    w_wrap = CausalWindow(
        timestamps=np.linspace(1.0, 1.2, 10),
        bus_frequencies=np.full((10, 39), 60.0),
        bus_rocof=np.zeros((10, 39)),
        voltage_magnitudes=np.ones((10, 39)),
        voltage_angles=angles_wrap,
        coi_frequency=np.full(10, 60.0),
        metadata={"t_disturb": 1.0, "fps": 50},
    )
    f_wrap = extract_features(w_wrap, angle_unit="deg")
    max_ang = f_wrap["angle_div_max"]
    tie_ang = f_wrap["tie_line_angle_diff"]
    feat_results["wrap_max_ang"] = float(max_ang)
    feat_results["wrap_tie_ang"] = float(tie_ang)
    print(f"  Wrap around at +-179 deg: max_ang={max_ang:.4f} deg (expected 2.0 or <= 180), tie_ang={tie_ang:.4f} deg (expected 2.0)")

    arr_nan = np.array([60.0, np.nan, 59.8, np.nan, 59.5])
    try:
        _validate_and_clean_nans(arr_nan, "test", policy="raise")
        feat_results["nan_raise"] = "Failed to raise"
    except ValueError:
        feat_results["nan_raise"] = "Raised ValueError as expected"
    cleaned = _validate_and_clean_nans(arr_nan, "bus_frequencies", policy="ffill")
    feat_results["nan_cleaned"] = cleaned.tolist()
    print(f"  NaN clean result: {cleaned}")

    arr_all_nan = np.full(5, np.nan)
    cleaned_all_nan = _validate_and_clean_nans(arr_all_nan, "bus_frequencies", policy="ffill")
    feat_results["all_nan_cleaned"] = cleaned_all_nan.tolist()
    print(f"  All-NaN cleaned result: {cleaned_all_nan} (expected nominal 60.0)")

    empty_area_partition = {1: [1, 2, 3], 2: [4, 5, 6], 3: []}
    w_custom_area = CausalWindow(
        timestamps=np.linspace(1.0, 1.2, 10),
        bus_frequencies=np.full((10, 39), 59.8),
        bus_rocof=np.zeros((10, 39)),
        voltage_magnitudes=np.ones((10, 39)),
        voltage_angles=np.zeros((10, 39)),
        coi_frequency=np.full(10, 59.8),
        metadata={"t_disturb": 1.0, "fps": 50},
    )
    f_custom_area = extract_features(w_custom_area, area_partition=empty_area_partition)
    feat_results["empty_area_spread"] = f_custom_area["area_spread_max"]
    print(f"  Empty area partition spread: {f_custom_area['area_spread_max']}")

    extractor = FeatureExtractor(t_obs=1.0)
    n_runs = 500
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = extractor.extract_features_dict(w_flat)
    t_tot = time.perf_counter() - t0
    ms_per_eval = (t_tot / n_runs) * 1000.0
    feat_results["ms_per_feature_eval"] = ms_per_eval
    print(f"  Throughput benchmark: {ms_per_eval:.3f} ms / scenario (budget: < 2.0 ms)")

    results["features"] = feat_results

    print("\n" + "=" * 70)
    print("ALL EMPIRICAL PROBES COMPLETED SUCCESSFULLY")
    print("=" * 70)
    return results


if __name__ == "__main__":
    run_all_probes()
