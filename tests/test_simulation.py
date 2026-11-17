"""Tests for the 50 Hz IEEE 39-bus dynamic simulator, renewables, and dataset validator."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from pmu_fsa.features.extractor import CANONICAL_FEATURE_NAMES, FeatureExtractor
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
    AREA_PARTITION,
    BASE_MVA,
    BUS_TO_AREA,
    GENERATOR_BUSES,
    N_BRANCHES,
    N_BUSES,
    N_GENERATORS,
    NOMINAL_FREQ,
    PHYSICAL_FREQ_BOUNDS,
    SECURITY_THRESHOLD_HZ,
    compute_system_droop_pu,
    get_ieee39_branch_data,
    get_ieee39_bus_data,
    get_ieee39_gen_data,
    get_total_system_inertia,
)
from pmu_fsa.simulation.pmu_recorder import PMUNoiseConfig
from pmu_fsa.simulation.renewables import (
    DEFAULT_DISPLACED_BUSES,
    RenewablePenetrationConfig,
    apply_renewable_displacement,
    compute_effective_inertias,
    summarize_renewable_dispatch,
)
from pmu_fsa.simulation.scenario_generator import (
    ScenarioGenerator,
    assign_scenario_partition,
    classify_ood_regime,
    export_raw_and_pmu_scenario,
)


def test_ieee39_50hz_constants_and_system_droop():
    assert NOMINAL_FREQ == 50.0
    assert SECURITY_THRESHOLD_HZ == 49.50
    assert PHYSICAL_FREQ_BOUNDS == (45.0, 50.5)
    assert BASE_MVA == 100.0
    assert N_BUSES == 39
    assert N_GENERATORS == 10
    assert N_BRANCHES == 46
    assert len(get_ieee39_bus_data()) == 39
    assert len(get_ieee39_branch_data()) == 46
    assert len(BUS_TO_AREA) == 39

    gens = get_ieee39_gen_data()
    assert [g["bus"] for g in gens] == GENERATOR_BUSES
    for g in gens:
        assert g["S_mva"] > 0.0
        assert g["R"] == 0.05
        assert g["R_sys_pu"] == pytest.approx(compute_system_droop_pu(0.05, g["S_mva"]))
        assert g["Tg"] == pytest.approx(4.20)

    with pytest.raises(ValueError, match="positive"):
        compute_system_droop_pu(0.05, 0.0)


def test_macro_intertie_false_vs_true_inertia_and_dynamics():
    gens_reg = get_ieee39_gen_data(macro_intertie=False)
    gens_mac = get_ieee39_gen_data(macro_intertie=True)

    g39_reg = [g for g in gens_reg if g["bus"] == 39][0]
    g39_mac = [g for g in gens_mac if g["bus"] == 39][0]

    assert g39_reg["H"] == pytest.approx(42.0)
    assert g39_reg["Xdp"] == pytest.approx(0.0400)
    assert g39_mac["H"] == pytest.approx(500.0)
    assert g39_mac["Xdp"] == pytest.approx(0.0060)

    assert get_total_system_inertia(macro_intertie=False) == pytest.approx(324.7)
    assert get_total_system_inertia(macro_intertie=True) == pytest.approx(782.7)

    sim_reg = DynamicSimulator(load_scaling=1.0)
    assert sim_reg.macro_intertie is False
    assert sim_reg.H_tot == pytest.approx(324.7)

    sim_mac = DynamicSimulator(load_scaling=1.0, macro_intertie=True)
    assert sim_mac.macro_intertie is True
    assert sim_mac.H_tot == pytest.approx(782.7)

    res_reg = sim_reg.simulate(t_span=(0.0, 6.0), t_disturb=1.0, tripped_bus=35, loss_fraction=1.0, fps=50)
    res_mac = sim_mac.simulate(t_span=(0.0, 6.0), t_disturb=1.0, tripped_bus=35, loss_fraction=1.0, fps=50)

    assert res_reg["metadata"]["ground_truth_nadir"] < res_mac["metadata"]["ground_truth_nadir"]
    idx0 = int(np.searchsorted(res_reg["timestamps"], 1.0))
    dt = res_reg["timestamps"][idx0 + 1] - res_reg["timestamps"][idx0]
    rocof_reg = (res_reg["coi_frequency"][idx0 + 1] - res_reg["coi_frequency"][idx0]) / dt
    rocof_mac = (res_mac["coi_frequency"][idx0 + 1] - res_mac["coi_frequency"][idx0]) / dt
    assert rocof_reg < rocof_mac < 0.0


@pytest.mark.parametrize("load_scale", [0.80, 1.00, 1.20])
def test_nr_power_flow_steady_state_equilibrium_and_balance(load_scale: float):
    V0, p_gen_mw, q_gen_mvar, diag = solve_power_flow(load_scaling=load_scale)
    assert diag["converged"] is True
    assert diag["max_mismatch_pu"] < 1e-10
    assert len(V0) == N_BUSES
    assert len(p_gen_mw) == N_GENERATORS
    assert len(q_gen_mvar) == N_GENERATORS
    assert diag["total_gen_mw"] > diag["total_load_mw"]
    assert 10.0 <= diag["losses_mw"] <= 120.0

    sim = DynamicSimulator(load_scaling=load_scale)
    assert sim.pf_diagnostics["converged"] is True
    assert sim.max_pm0_mismatch_mw < 1e-6

    res = sim.simulate(t_span=(0.0, 2.0), tripped_bus=None, fps=50)
    bus_freq = res["bus_frequencies"]
    coi_freq = res["coi_frequency"]
    bus_rocof = res["bus_rocof"]

    assert bus_freq.shape == (len(res["timestamps"]), N_BUSES)
    assert np.max(np.abs(bus_freq - 50.0)) < 1e-9
    assert np.max(np.abs(coi_freq - 50.0)) < 1e-9
    assert np.max(np.abs(bus_rocof)) < 1e-9
    assert res["metadata"]["ground_truth_nadir"] == pytest.approx(50.0, abs=1e-9)
    assert res["metadata"]["is_secure"] == 1


def test_theoretical_initial_rocof_after_unit_disconnection():
    sim = DynamicSimulator(load_scaling=1.0)
    t_disturb = 1.0
    res = sim.simulate(
        t_span=(0.0, 5.0),
        t_disturb=t_disturb,
        tripped_bus=35,
        loss_fraction=1.0,
        fps=50,
    )

    t = res["timestamps"]
    f_coi = res["coi_frequency"]
    idx0 = int(np.searchsorted(t, t_disturb))
    dt = t[idx0 + 1] - t[idx0]
    sim_initial_rocof = (f_coi[idx0 + 1] - f_coi[idx0]) / dt

    h_post = res["metadata"]["effective_H_post"]
    p_loss_pu = res["metadata"]["active_loss_pu"]
    theo_rocof = -NOMINAL_FREQ * p_loss_pu / (2.0 * h_post)

    rel_error = abs(sim_initial_rocof - theo_rocof) / abs(theo_rocof)
    assert rel_error < 0.02, (
        f"Initial COI ROCOF mismatch: sim={sim_initial_rocof:.5f}, "
        f"theo={theo_rocof:.5f}, rel_error={rel_error:.4f}"
    )


@pytest.mark.parametrize("tripped_bus", [30, 31, 35, 38, 39])
def test_complete_generator_trip_zero_post_electrical_power(tripped_bus: int):
    sim = DynamicSimulator(load_scaling=1.0)
    trip_idx = list(sim.gen_buses).index(tripped_bus)
    res = sim.simulate(
        t_span=(0.0, 4.0),
        t_disturb=1.0,
        tripped_bus=tripped_bus,
        loss_fraction=1.0,
        fps=50,
    )

    meta = res["metadata"]
    assert meta["event_type"] == "complete_generator_trip"
    assert meta["loss_fraction"] == pytest.approx(1.0)
    assert meta["effective_H_post"] == pytest.approx(sim.H_tot - sim.H[trip_idx])

    post_mask = res["timestamps"] >= 1.0
    pe_tripped_post = res["generator_electrical_powers"][post_mask, trip_idx]
    pm_tripped_post = res["generator_powers"][post_mask, trip_idx]

    assert np.max(np.abs(pe_tripped_post)) == 0.0
    assert np.max(np.abs(pm_tripped_post)) == 0.0


def test_partial_loss_scaling_and_coi_vs_bus_nadir():
    sim = DynamicSimulator(load_scaling=1.0)
    trip_idx = list(sim.gen_buses).index(35)

    res_04 = sim.simulate(t_span=(0.0, 6.0), t_disturb=1.0, tripped_bus=35, loss_fraction=0.40, fps=50)
    res_07 = sim.simulate(t_span=(0.0, 6.0), t_disturb=1.0, tripped_bus=35, loss_fraction=0.70, fps=50)
    res_10 = sim.simulate(t_span=(0.0, 6.0), t_disturb=1.0, tripped_bus=35, loss_fraction=1.00, fps=50)

    assert res_04["metadata"]["event_type"] == "partial_generation_loss"
    assert res_07["metadata"]["event_type"] == "partial_generation_loss"
    assert res_04["metadata"]["effective_H_post"] == pytest.approx(sim.H_tot - 0.40 * sim.H[trip_idx])
    assert res_07["metadata"]["effective_H_post"] == pytest.approx(sim.H_tot - 0.70 * sim.H[trip_idx])

    post_mask = res_04["timestamps"] >= 1.0
    assert np.min(np.abs(res_04["generator_electrical_powers"][post_mask, trip_idx])) > 0.5

    n04 = res_04["metadata"]["ground_truth_nadir"]
    n07 = res_07["metadata"]["ground_truth_nadir"]
    n10 = res_10["metadata"]["ground_truth_nadir"]
    assert 50.0 > n04 > n07 > n10 >= 45.0

    coi_post_min = float(np.min(res_10["coi_frequency"][res_10["timestamps"] >= 1.0]))
    bus_post_min = float(np.min(res_10["bus_frequencies"][res_10["timestamps"] >= 1.0]))
    assert n10 == pytest.approx(coi_post_min, abs=1e-9)
    assert res_10["metadata"]["bus_min_nadir"] == pytest.approx(bus_post_min, abs=1e-9)
    assert res_10["metadata"]["bus_min_nadir"] <= n10


def test_renewable_penetration_linear_scaling_and_droop_option():
    assert DEFAULT_DISPLACED_BUSES == [30, 32, 33, 34, 35, 36, 37, 38, 39]
    gen_data = get_ieee39_gen_data(macro_intertie=False)

    penetrations = [0.10, 0.20, 0.30, 0.40, 0.50]
    realized = []
    inertias = []
    for p in penetrations:
        cfg = RenewablePenetrationConfig(p_ren=p)
        mod_gens = apply_renewable_displacement(gen_data, cfg, load_scaling=1.0)
        summary = summarize_renewable_dispatch(mod_gens, load_scaling=1.0)
        realized.append(summary["p_ren_realized"])
        inertias.append(summary["h_eff_total"])
        assert abs(summary["p_ren_realized"] - p) < 0.005

    second_diffs = np.diff(realized, n=2)
    assert np.max(np.abs(second_diffs)) < 1e-12
    assert all(inertias[i] > inertias[i + 1] for i in range(len(inertias) - 1))

    mod_10 = apply_renewable_displacement(gen_data, RenewablePenetrationConfig(p_ren=0.30), load_scaling=1.0)
    mod_12 = apply_renewable_displacement(gen_data, RenewablePenetrationConfig(p_ren=0.30), load_scaling=1.2)
    sum_10 = summarize_renewable_dispatch(mod_10, load_scaling=1.0)
    sum_12 = summarize_renewable_dispatch(mod_12, load_scaling=1.2)
    assert sum_12["p_ren_realized"] == pytest.approx(sum_10["p_ren_realized"], abs=1e-12)
    assert sum_12["displaced_mw"] == pytest.approx(1.2 * sum_10["displaced_mw"], rel=1e-9)

    sim_no_droop = DynamicSimulator(
        load_scaling=1.0,
        renewable_config=RenewablePenetrationConfig(p_ren=0.40, enable_droop=False),
    )
    sim_droop = DynamicSimulator(
        load_scaling=1.0,
        renewable_config=RenewablePenetrationConfig(p_ren=0.40, enable_droop=True, r_ren=0.05, t_inv=0.02),
    )
    assert sim_no_droop.has_ren_droop is False
    assert sim_droop.has_ren_droop is True

    r_no = sim_no_droop.simulate(t_span=(0.0, 6.0), t_disturb=1.0, tripped_bus=35, loss_fraction=1.0, fps=50)
    r_dr = sim_droop.simulate(t_span=(0.0, 6.0), t_disturb=1.0, tripped_bus=35, loss_fraction=1.0, fps=50)
    assert r_dr["metadata"]["ground_truth_nadir"] > r_no["metadata"]["ground_truth_nadir"]


def test_dynamic_generator_trip_across_all_3_areas_and_voltage_bounds():
    sim = DynamicSimulator(load_scaling=1.0)
    for area_id, gen_bus in [(1, 32), (2, 35), (3, 38)]:
        res = sim.simulate(
            t_span=(0.0, 6.0),
            t_disturb=1.0,
            tripped_bus=gen_bus,
            loss_fraction=1.0,
            fps=50,
        )
        bf = res["bus_frequencies"]
        vm = res["voltage_magnitudes"]
        assert np.all(np.isfinite(bf))
        assert np.all(np.isfinite(vm))
        assert np.all((bf >= 45.0) & (bf <= 50.5))
        assert np.all((vm >= 0.80) & (vm <= 1.25))

        nadir = res["metadata"]["ground_truth_nadir"]
        t_nadir = res["metadata"]["ground_truth_t_nadir"]
        assert 45.0 <= nadir < 50.0
        assert 2.5 < t_nadir < 6.0
        assert area_id in res["area_frequencies"]


def test_kron_reduction_shape_and_conditioning():
    Y_red, K, E_mag0, delta0, Pm0 = compute_kron_reduction(load_scaling=1.0, return_all=True)
    assert Y_red.shape == (10, 10)
    assert K.shape == (39, 10)
    assert len(E_mag0) == 10
    assert len(delta0) == 10
    assert len(Pm0) == 10
    assert np.linalg.cond(Y_red) < 30.0


def test_dataset_reproducibility_channel_separation_and_validator(tmp_path: Path):
    noise_cfg = PMUNoiseConfig(sigma_f=0.005, seed=101)
    gen_a = ScenarioGenerator(fps=50, t_span=(0.0, 6.0), noise_config=noise_cfg, seed=101)
    gen_b = ScenarioGenerator(fps=50, t_span=(0.0, 6.0), noise_config=noise_cfg, seed=101)

    pairs_a = gen_a.generate_scenario_batch(
        gen_buses=[30, 35, 38],
        loss_fractions=[1.0, 0.40],
        penetrations=[0.10, 0.30, 0.50],
        load_scales=[0.80, 1.00, 1.20],
        max_scenarios=8,
        inject_noise=True,
        return_raw=True,
    )
    pairs_b = gen_b.generate_scenario_batch(
        gen_buses=[30, 35, 38],
        loss_fractions=[1.0, 0.40],
        penetrations=[0.10, 0.30, 0.50],
        load_scales=[0.80, 1.00, 1.20],
        max_scenarios=8,
        inject_noise=True,
        return_raw=True,
    )

    assert len(pairs_a) == 8
    ids_a = [pmu["metadata"]["scenario_id"] for _, pmu in pairs_a]
    ids_b = [pmu["metadata"]["scenario_id"] for _, pmu in pairs_b]
    assert ids_a == ids_b
    assert len(set(ids_a)) == 8

    for (_, pmu_1), (_, pmu_2) in zip(pairs_a, pairs_b):
        np.testing.assert_allclose(pmu_1["bus_frequencies"], pmu_2["bus_frequencies"], rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(pmu_1["voltage_angles"], pmu_2["voltage_angles"], rtol=1e-11, atol=1e-11)

    raw_sim_dir = tmp_path / "raw_sim"
    raw_pmu_dir = tmp_path / "raw_pmu"
    proc_dir = tmp_path / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    pmu_scenarios = []
    for raw_traj, pmu_traj in pairs_a:
        sim_path, pmu_path = export_raw_and_pmu_scenario(raw_traj, pmu_traj, raw_sim_dir, raw_pmu_dir)
        with np.load(sim_path, allow_pickle=True) as s_npz:
            assert "generator_speeds" in s_npz.files
            assert "bus_frequencies_true" in s_npz.files
        with np.load(pmu_path, allow_pickle=True) as p_npz:
            assert "pmu_sys_frequency" in p_npz.files
            assert "soc" in p_npz.files
            for forbidden in ("generator_deltas", "generator_speeds", "generator_powers", "bus_frequencies_true"):
                assert forbidden not in p_npz.files
        pmu_scenarios.append(pmu_traj)

    extractor = FeatureExtractor(f_nominal=50.0)
    partitions_ref = None
    for w in (0.5, 1.0, 1.5, 2.0):
        X_df, y_nadir, y_sec, meta_df = extractor.transform_batch(pmu_scenarios, t_obs=w, as_dataframe=True)
        parts = meta_df["partition"].to_numpy(dtype=str)
        if partitions_ref is None:
            partitions_ref = parts
        else:
            np.testing.assert_array_equal(parts, partitions_ref)

        np.savez_compressed(
            proc_dir / f"features_t{w:.1f}s.npz",
            X=X_df.to_numpy(dtype=float),
            y_nadir=y_nadir,
            y_secure=y_sec,
            feature_names=np.asarray(CANONICAL_FEATURE_NAMES, dtype=str),
            scenario_ids=meta_df["scenario_id"].to_numpy(dtype=str),
            partitions=parts,
        )

    report = validate_dataset(
        raw_pmu_dir=raw_pmu_dir,
        processed_dir=proc_dir,
        windows=(0.5, 1.0, 1.5, 2.0),
        f_nominal=50.0,
        threshold=49.50,
        raw_sim_dir=raw_sim_dir,
        expected_n_scenarios=8,
    )
    assert report["all_passed"] is True, f"Failed checks: {[k for k, v in report['checks'].items() if not v['passed']]}"
    assert len(report["checks"]) == 14

    md_path, man_path = write_dataset_quality_report(
        report,
        md_path=tmp_path / "quality_report.md",
        manifest_path=tmp_path / "manifest.json",
    )
    assert Path(md_path).exists()
    assert Path(man_path).exists()

    # Verify negative detection when internal ODE state leaks into raw_pmu
    bad_pmu_dir = tmp_path / "bad_pmu"
    shutil.copytree(raw_pmu_dir, bad_pmu_dir)
    target_fp = sorted(bad_pmu_dir.glob("*.npz"))[0]
    with np.load(target_fp, allow_pickle=True) as orig:
        payload = {k: orig[k] for k in orig.files}
    payload["generator_speeds"] = np.zeros((10, 10))
    bad_meta = dict(payload["metadata"].item())
    bad_meta["is_secure"] = 1 - int(bad_meta["is_secure"])
    payload["metadata"] = bad_meta
    np.savez_compressed(target_fp, **payload)

    bad_report = validate_dataset(raw_pmu_dir=bad_pmu_dir, f_nominal=50.0, threshold=49.50)
    assert bad_report["all_passed"] is False
    assert bad_report["checks"]["4_nan_inf_and_channel_separation"]["passed"] is False
    assert bad_report["checks"]["9_security_label_consistency"]["passed"] is False


def test_ood_regime_and_partition_rules():
    assert classify_ood_regime(0.20, 1.00) == "id"
    assert classify_ood_regime(0.45, 1.00) == "ood_res"
    assert classify_ood_regime(0.20, 0.80) == "ood_load"
    assert classify_ood_regime(0.50, 1.20) == "ood_combined"

    assert assign_scenario_partition(0.45, 1.00, "scen_ood_1") == "ood_test"
    assert assign_scenario_partition(0.20, 1.20, "scen_ood_2") == "ood_test"

    id_parts = {assign_scenario_partition(0.20, 1.00, f"scen_id_{k:03d}") for k in range(80)}
    assert id_parts == {"train", "val", "test"}


def test_rhs_complete_trip_with_renewable_droop_zeros_fourth_state():
    ren_cfg = RenewablePenetrationConfig(p_ren=0.30, enable_droop=True, r_ren=0.04, t_inv=0.05)
    sim = DynamicSimulator(load_scaling=1.0, renewable_config=ren_cfg)
    assert sim.has_ren_droop is True

    m = sim.n_gen
    state = np.zeros(4 * m, dtype=float)
    state[:m] = sim.delta0
    state[3 * m :] = 0.25  # Non-zero converter droop state prior to trip

    trip_idx = 4
    deriv = sim.rhs(2.0, state, t_disturb=1.0, trip_gen_idx=trip_idx, p_loss_mw=500.0, loss_fraction=1.0)
    assert len(deriv) == 4 * m
    assert np.all(np.isfinite(deriv))
    assert deriv[trip_idx] == 0.0
    assert deriv[m + trip_idx] == 0.0
    assert deriv[2 * m + trip_idx] == 0.0
    assert deriv[3 * m + trip_idx] == 0.0

    traj = sim.simulate(t_span=(0.0, 4.0), t_disturb=1.0, tripped_bus=35, loss_fraction=1.0, fps=50)
    assert traj["metadata"]["ground_truth_nadir"] < 50.0
    assert np.all(np.isfinite(traj["coi_frequency"]))

