"""Contingency scenario generation, stratified partitioning, and raw/PMU NPZ serialization."""

from __future__ import annotations

import hashlib
import inspect
import os
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from pmu_fsa.simulation.ieee39_data import (
    AREA_PARTITION,
    BASE_MVA,
    GENERATOR_BUSES,
    GENERATOR_DATA,
    NOMINAL_FREQ,
    SECURITY_THRESHOLD_HZ,
)
from pmu_fsa.simulation.renewables import RenewablePenetrationConfig
from pmu_fsa.simulation.dynamic_engine import DynamicSimulator
from pmu_fsa.simulation.pmu_recorder import PMUNoiseConfig, PMURecorder


GEN_CAPACITIES_MW: Dict[int, float] = {
    int(g["bus"]): float(g["P_mw"]) for g in GENERATOR_DATA
}

DEFAULT_LOSS_FRACTIONS: List[float] = [1.0, 0.70, 0.40]
DEFAULT_PENETRATIONS: List[float] = [0.10, 0.20, 0.30, 0.40, 0.50]
DEFAULT_LOAD_SCALES: List[float] = [0.80, 0.90, 1.00, 1.10, 1.20]


def get_software_versions() -> Dict[str, str]:
    """Collect runtime versions of core scientific libraries for provenance metadata."""
    versions: Dict[str, str] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    for pkg_name, mod_name in (
        ("scipy", "scipy"),
        ("scikit_learn", "sklearn"),
        ("pandas", "pandas"),
    ):
        try:
            mod = __import__(mod_name)
            versions[pkg_name] = str(getattr(mod, "__version__", "unknown"))
        except ImportError:
            versions[pkg_name] = "not_installed"
    return versions


def _estimate_pmu_sys_frequency(bus_frequencies: np.ndarray) -> np.ndarray:
    """Compute wide-area PMU system frequency strictly from observed bus frequencies."""
    try:
        from pmu_fsa.simulation import pmu_recorder as _pmu_rec
        if hasattr(_pmu_rec, "estimate_pmu_system_frequency"):
            return np.asarray(_pmu_rec.estimate_pmu_system_frequency(bus_frequencies), dtype=float)
    except Exception:
        pass

    bf = np.asarray(bus_frequencies, dtype=float)
    if bf.ndim == 1:
        return bf.copy()

    n_samples, n_buses = bf.shape
    area_means: List[np.ndarray] = []
    area_weights: List[float] = []

    for area_id in sorted(AREA_PARTITION.keys()):
        cols = [b - 1 for b in AREA_PARTITION[area_id] if 0 <= b - 1 < n_buses]
        if not cols:
            continue
        sub = bf[:, cols]
        valid_counts = np.sum(np.isfinite(sub), axis=1)
        if np.any(valid_counts > 0):
            with np.errstate(invalid="ignore"):
                a_mean = np.nanmean(sub, axis=1)
            area_means.append(a_mean)
            area_weights.append(float(len(cols)))

    if len(area_means) == 3:
        stacked = np.column_stack(area_means)
        w = np.asarray(area_weights, dtype=float)
        valid = np.isfinite(stacked)
        w_mat = np.where(valid, w[None, :], 0.0)
        w_sum = np.sum(w_mat, axis=1)
        safe_vals = np.where(valid, stacked, 0.0)
        out = np.divide(
            np.sum(safe_vals * w_mat, axis=1),
            np.maximum(w_sum, 1e-12),
        )
        if np.all(w_sum > 0):
            return out

    with np.errstate(invalid="ignore"):
        fallback = np.nanmean(bf, axis=1)
    return np.where(np.isfinite(fallback), fallback, NOMINAL_FREQ)


def classify_ood_regime(renewable_penetration: float, load_scale: float) -> str:
    """Classify operating point into 'id', 'ood_res', 'ood_load', or 'ood_combined'."""
    p_ren = float(renewable_penetration)
    ld = float(load_scale)
    is_res_id = p_ren <= 0.32
    is_load_id = 0.88 <= ld <= 1.12

    if is_res_id and is_load_id:
        return "id"
    if (not is_res_id) and is_load_id:
        return "ood_res"
    if is_res_id and (not is_load_id):
        return "ood_load"
    return "ood_combined"


def assign_scenario_partition(
    renewable_penetration: float,
    load_scale: float,
    scenario_id: str,
    train_ratio: float = 0.55,
    val_ratio: float = 0.25,
    test_ratio: float = 0.20,
    return_ood_type: bool = False,
) -> Union[str, Tuple[str, str]]:
    """Assign scenario deterministically to 'train', 'val', 'test', or 'ood_test'."""
    ood_type = classify_ood_regime(renewable_penetration, load_scale)
    if ood_type != "id":
        return ("ood_test", ood_type) if return_ood_type else "ood_test"

    total = float(train_ratio + val_ratio + test_ratio)
    if total <= 0.0:
        raise ValueError("Partition ratios must sum to a positive value")
    tr = float(train_ratio) / total
    vr = float(val_ratio) / total

    digest = hashlib.md5(scenario_id.encode("utf-8")).hexdigest()
    hash_val = (int(digest, 16) % 10000) / 10000.0

    if hash_val < tr:
        partition = "train"
    elif hash_val < (tr + vr):
        partition = "val"
    else:
        partition = "test"

    return (partition, ood_type) if return_ood_type else partition


def create_scenario_id(
    gen_bus: int,
    loss_mw: float,
    p_ren: float,
    load_scale: float,
    rep_idx: int = 0,
) -> str:
    """Format deterministic scenario ID independent of observation window T_obs."""
    return (
        f"scen_gen{int(gen_bus)}"
        f"_loss{int(round(loss_mw))}"
        f"_ren{int(round(p_ren * 100))}"
        f"_ld{int(round(load_scale * 100))}"
        f"_{int(rep_idx):03d}"
    )


def _scenario_seed_offset(scenario_id: str) -> int:
    digest = hashlib.md5(scenario_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 1_000_000


def _order_by_distinct_buses(items: List[Tuple[Any, ...]], rng: np.random.Generator) -> List[Tuple[Any, ...]]:
    """Deterministically shuffle items while round-robin interleaving generator buses."""
    if len(items) <= 1:
        return list(items)
    perm = rng.permutation(len(items))
    shuffled = [items[int(i)] for i in perm]

    by_bus: Dict[int, List[Tuple[Any, ...]]] = {}
    bus_order: List[int] = []
    for entry in shuffled:
        bus = int(entry[0])
        if bus not in by_bus:
            by_bus[bus] = []
            bus_order.append(bus)
        by_bus[bus].append(entry)

    interleaved: List[Tuple[Any, ...]] = []
    idx = 0
    while len(interleaved) < len(shuffled):
        for bus in bus_order:
            bucket = by_bus[bus]
            if idx < len(bucket):
                interleaved.append(bucket[idx])
        idx += 1
    return interleaved


def _stratified_subsample_combinations(
    combos: List[Tuple[int, float, float, float, float, int, str, str, str]],
    max_scenarios: int,
    seed: int,
) -> List[Tuple[int, float, float, float, float, int, str, str, str]]:
    """Subsample grid combinations stratified across partitions, OOD regimes, loss fractions, and buses."""
    if max_scenarios >= len(combos):
        return list(combos)
    if max_scenarios <= 0:
        return []

    rng = np.random.default_rng(seed)

    # Group into regime/partition buckets and outage severity tiers so small batches span both classes
    regime_keys = ["train", "test", "val", "ood_res", "ood_load", "ood_combined"]
    sev_keys = ["light", "heavy", "mid"]
    strata: Dict[Tuple[str, str], List[Tuple[int, float, float, float, float, int, str, str, str]]] = {
        (rk, sk): [] for rk in regime_keys for sk in sev_keys
    }

    for entry in combos:
        bus, lf, p_loss_nom, p_ren, ld, rep_idx, scen_id, part, ood_type = entry
        rk = part if ood_type == "id" else ood_type
        if p_loss_nom <= 235.0:
            sk = "light"
        elif p_loss_nom >= 580.0:
            sk = "heavy"
        else:
            sk = "mid"
        strata.setdefault((rk, sk), []).append(entry)

    regime_queues: Dict[str, List[Tuple[int, float, float, float, float, int, str, str, str]]] = {}
    for rk in regime_keys:
        ordered_parts = [
            _order_by_distinct_buses(strata.get((rk, sk), []), rng)
            for sk in sev_keys
        ]
        merged: List[Tuple[int, float, float, float, float, int, str, str, str]] = []
        max_len = max((len(p) for p in ordered_parts), default=0)
        for i in range(max_len):
            for p in ordered_parts:
                if i < len(p):
                    merged.append(p[i])
        if merged:
            regime_queues[rk] = merged

    # Priority cycle ensures small n_scenarios (e.g. 12, 24, 30) populates train/val/test + all OOD types
    cycle_pattern = [
        "train", "ood_res", "test", "val",
        "train", "ood_load", "test", "val",
        "train", "ood_combined",
    ]
    cursors: Dict[str, int] = {rk: 0 for rk in regime_queues}
    selected: List[Tuple[int, float, float, float, float, int, str, str, str]] = []

    while len(selected) < max_scenarios:
        progressed = False
        for rk in cycle_pattern:
            if len(selected) >= max_scenarios:
                break
            q = regime_queues.get(rk)
            if q is not None and cursors[rk] < len(q):
                selected.append(q[cursors[rk]])
                cursors[rk] += 1
                progressed = True
        if not progressed:
            for rk, q in regime_queues.items():
                while cursors[rk] < len(q) and len(selected) < max_scenarios:
                    selected.append(q[cursors[rk]])
                    cursors[rk] += 1
                    progressed = True
            if not progressed:
                break

    return selected


class ScenarioGenerator:
    """Generates post-contingency trajectories across generator outages and operating points."""

    def __init__(
        self,
        fps: int = 50,
        t_span: Tuple[float, float] = (0.0, 10.0),
        t_disturb: float = 1.0,
        security_threshold_hz: float = SECURITY_THRESHOLD_HZ,
        macro_intertie: bool = False,
        f_nominal: float = NOMINAL_FREQ,
        noise_config: Optional[PMUNoiseConfig] = None,
        seed: int = 42,
    ):
        self.fps = int(fps)
        self.t_span = (float(t_span[0]), float(t_span[1]))
        self.t_disturb = float(t_disturb)
        self.security_threshold_hz = float(security_threshold_hz)
        self.macro_intertie = bool(macro_intertie)
        self.f_nominal = float(f_nominal)
        self.noise_config = noise_config
        self.seed = int(seed)
        self.pmu_recorder = PMURecorder(
            fps=self.fps,
            noise_config=self.noise_config,
            f_nominal=self.f_nominal,
        )
        self._software_versions = get_software_versions()

    def generate_single_scenario(
        self,
        tripped_bus: int,
        p_loss_mw: Optional[float] = None,
        renewable_penetration: float = 0.20,
        load_scale: float = 1.0,
        rep_idx: int = 0,
        inject_noise: bool = False,
        noise_config: Optional[PMUNoiseConfig] = None,
        loss_fraction: Optional[float] = None,
        seed_offset: Optional[int] = None,
        observed_buses: Optional[List[int]] = None,
        return_raw: bool = False,
    ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Simulate one generator outage scenario and return PMU trajectory (or (raw, pmu) if return_raw)."""
        nom_p = GEN_CAPACITIES_MW.get(int(tripped_bus), 500.0)
        if loss_fraction is None:
            if p_loss_mw is None:
                eff_lf = 1.0
                p_loss_nom = nom_p
            else:
                p_loss_nom = float(p_loss_mw)
                eff_lf = min(1.0, max(0.0, p_loss_nom / max(1e-6, nom_p)))
        else:
            eff_lf = float(loss_fraction)
            p_loss_nom = float(p_loss_mw) if p_loss_mw is not None else (nom_p * eff_lf)

        scenario_id = create_scenario_id(
            tripped_bus, p_loss_nom, renewable_penetration, load_scale, rep_idx
        )
        partition, ood_type = assign_scenario_partition(
            renewable_penetration, load_scale, scenario_id, return_ood_type=True
        )

        ren_cfg = RenewablePenetrationConfig(p_ren=float(renewable_penetration))
        sim = DynamicSimulator(
            load_scaling=float(load_scale),
            renewable_config=ren_cfg,
            macro_intertie=self.macro_intertie,
            f_nominal=self.f_nominal,
            security_threshold_hz=self.security_threshold_hz,
        )

        sim_kwargs: Dict[str, Any] = {
            "t_span": self.t_span,
            "t_disturb": self.t_disturb,
            "tripped_bus": int(tripped_bus),
            "p_loss_mw": float(p_loss_nom),
            "fps": self.fps,
        }
        sim_params = inspect.signature(sim.simulate).parameters
        if "loss_fraction" in sim_params and loss_fraction is not None:
            sim_kwargs["loss_fraction"] = eff_lf

        raw_traj = sim.simulate(**sim_kwargs)

        eff_cfg = noise_config if noise_config is not None else self.noise_config
        cfg_has_noise = eff_cfg is not None and (
            eff_cfg.sigma_f > 0.0
            or eff_cfg.sigma_rocof > 0.0
            or eff_cfg.sigma_v > 0.0
            or eff_cfg.sigma_theta_deg > 0.0
        )
        should_inject = bool(inject_noise or (noise_config is not None and cfg_has_noise) or (self.noise_config is not None and cfg_has_noise))

        eff_offset = int(seed_offset) if seed_offset is not None else _scenario_seed_offset(scenario_id)
        base_seed = int(eff_cfg.seed) if (eff_cfg is not None and eff_cfg.seed is not None) else self.seed
        scenario_seed = base_seed + eff_offset

        if should_inject and eff_cfg is None:
            eff_cfg = PMUNoiseConfig(sigma_f=0.005, seed=base_seed)

        rec_params = inspect.signature(PMURecorder.record_trajectory).parameters
        supports_seed_offset = "seed_offset" in rec_params
        supports_observed_buses = "observed_buses" in rec_params

        if eff_cfg is not None:
            rec_noise_cfg = PMUNoiseConfig(
                sigma_f=float(eff_cfg.sigma_f),
                sigma_rocof=float(eff_cfg.sigma_rocof),
                sigma_v=float(eff_cfg.sigma_v),
                sigma_theta_deg=float(eff_cfg.sigma_theta_deg),
                seed=base_seed if supports_seed_offset else scenario_seed,
            )
            recorder = PMURecorder(fps=self.fps, noise_config=rec_noise_cfg, f_nominal=self.f_nominal)
        else:
            rec_noise_cfg = None
            recorder = self.pmu_recorder

        rec_kwargs: Dict[str, Any] = {"inject_noise": should_inject}
        if supports_seed_offset:
            rec_kwargs["seed_offset"] = eff_offset
        if supports_observed_buses and observed_buses is not None:
            rec_kwargs["observed_buses"] = list(observed_buses)

        pmu_traj = recorder.record_trajectory(raw_traj, **rec_kwargs)

        if observed_buses is not None and not supports_observed_buses:
            obs_zero = {int(b) - 1 for b in observed_buses if 1 <= int(b) <= pmu_traj["bus_frequencies"].shape[1]}
            for ch in ("bus_frequencies", "bus_rocof", "voltage_magnitudes", "voltage_angles"):
                arr = np.array(pmu_traj[ch], dtype=float, copy=True)
                for col in range(arr.shape[1]):
                    if col not in obs_zero:
                        arr[:, col] = np.nan
                pmu_traj[ch] = arr

        pmu_sys_freq = pmu_traj.get("pmu_sys_frequency")
        if pmu_sys_freq is None:
            pmu_sys_freq = _estimate_pmu_sys_frequency(pmu_traj["bus_frequencies"])
        else:
            pmu_sys_freq = np.asarray(pmu_sys_freq, dtype=float)
        pmu_traj["pmu_sys_frequency"] = pmu_sys_freq
        # Keep dict-level alias pointing to PMU-estimated frequency (never internal simulator rotor speed)
        pmu_traj["coi_frequency"] = pmu_sys_freq.copy()

        raw_meta = dict(raw_traj.get("metadata", {}))
        t_raw = np.asarray(raw_traj["timestamps"], dtype=float)
        post_mask = t_raw >= self.t_disturb
        t_post = t_raw[post_mask] if np.any(post_mask) else t_raw
        if "bus_min_nadir" not in raw_meta:
            bf_post = raw_traj["bus_frequencies"][post_mask] if np.any(post_mask) else raw_traj["bus_frequencies"]
            flat_idx = np.unravel_index(int(np.argmin(bf_post)), bf_post.shape)
            raw_meta["bus_min_nadir"] = float(bf_post[flat_idx])
            raw_meta["bus_min_t_nadir"] = float(t_post[flat_idx[0]])
            coi_arr = np.asarray(raw_traj["coi_frequency"], dtype=float)
            coi_post = coi_arr[post_mask] if np.any(post_mask) else coi_arr
            coi_idx = int(np.argmin(coi_post))
            raw_meta["ground_truth_nadir"] = float(coi_post[coi_idx])
            raw_meta["ground_truth_t_nadir"] = float(t_post[coi_idx])

        gt_nadir = float(raw_meta["ground_truth_nadir"])
        gt_t_nadir = float(raw_meta["ground_truth_t_nadir"])
        is_sec = int(gt_nadir >= self.security_threshold_hz)
        act_loss_mw = float(raw_meta.get("active_loss_mw", p_loss_nom))
        act_loss_pu = float(raw_meta.get("active_loss_pu", act_loss_mw / BASE_MVA))
        p_ren_req = float(raw_meta.get("p_ren_requested", renewable_penetration))
        p_ren_real = float(raw_meta.get("p_ren_realized", renewable_penetration))
        eff_lf_final = float(raw_meta.get("loss_fraction", eff_lf))

        noise_meta = {
            "inject_noise": should_inject,
            "sigma_f": float(rec_noise_cfg.sigma_f) if (should_inject and rec_noise_cfg is not None) else 0.0,
            "sigma_rocof": float(rec_noise_cfg.sigma_rocof) if (should_inject and rec_noise_cfg is not None) else 0.0,
            "sigma_v": float(rec_noise_cfg.sigma_v) if (should_inject and rec_noise_cfg is not None) else 0.0,
            "sigma_theta_deg": float(rec_noise_cfg.sigma_theta_deg) if (should_inject and rec_noise_cfg is not None) else 0.0,
            "base_seed": base_seed,
            "seed_offset": eff_offset,
            "scenario_seed": scenario_seed,
        }

        shared_provenance = {
            "seed": scenario_seed,
            "base_seed": self.seed,
            "scenario_id": scenario_id,
            "partition": partition,
            "ood_type": ood_type,
            "disturbed_gen": int(tripped_bus),
            "loss_fraction": eff_lf_final,
            "nominal_loss_mw": float(p_loss_nom),
            "active_loss_mw": act_loss_mw,
            "active_loss_pu": act_loss_pu,
            "renewable_penetration": p_ren_req,
            "p_ren_requested": p_ren_req,
            "p_ren_realized": p_ren_real,
            "load_scale": float(load_scale),
            "t_disturb": self.t_disturb,
            "t_sim": float(self.t_span[1] - self.t_span[0]),
            "ground_truth_nadir": gt_nadir,
            "ground_truth_t_nadir": gt_t_nadir,
            "bus_min_nadir": float(raw_meta["bus_min_nadir"]),
            "bus_min_t_nadir": float(raw_meta["bus_min_t_nadir"]),
            "is_secure": is_sec,
            "fps": self.fps,
            "f_nominal": self.f_nominal,
            "f_threshold_hz": self.security_threshold_hz,
            "macro_intertie": self.macro_intertie,
            "software_versions": dict(self._software_versions),
        }

        raw_meta.update(shared_provenance)
        raw_traj["metadata"] = raw_meta

        pmu_meta = dict(pmu_traj.get("metadata", {}))
        pmu_meta.update(shared_provenance)
        pmu_meta["is_noisy"] = should_inject
        pmu_meta["noise_config"] = noise_meta
        pmu_meta["freq_estimator"] = pmu_meta.get("freq_estimator", "pmu_area_weighted_bus_mean")
        if observed_buses is not None:
            pmu_meta["observed_buses"] = [int(b) for b in observed_buses]
        # Strip internal inertia secrets from observable PMU metadata if desired while keeping audit provenance
        pmu_traj["metadata"] = pmu_meta

        if return_raw:
            return raw_traj, pmu_traj
        return pmu_traj

    def generate_scenario_batch(
        self,
        gen_buses: Optional[List[int]] = None,
        load_scales: Optional[List[float]] = None,
        penetrations: Optional[List[float]] = None,
        loss_fractions: Optional[List[float]] = None,
        max_scenarios: Optional[int] = None,
        inject_noise: bool = False,
        noise_config: Optional[PMUNoiseConfig] = None,
        seed: Optional[int] = None,
        observed_buses: Optional[List[int]] = None,
        return_raw: bool = False,
    ) -> List[Any]:
        """Generate scenarios over the combinatorial grid, stratifying when max_scenarios < grid size."""
        buses = list(gen_buses) if gen_buses is not None else list(GENERATOR_BUSES)
        loads = list(load_scales) if load_scales is not None else list(DEFAULT_LOAD_SCALES)
        rens = list(penetrations) if penetrations is not None else list(DEFAULT_PENETRATIONS)
        lfs = list(loss_fractions) if loss_fractions is not None else list(DEFAULT_LOSS_FRACTIONS)
        batch_seed = int(seed) if seed is not None else self.seed

        seen_counts: Dict[Tuple[int, int, int, int], int] = {}
        combos: List[Tuple[int, float, float, float, float, int, str, str, str]] = []

        for bus in buses:
            nom_p = GEN_CAPACITIES_MW.get(int(bus), 500.0)
            for lf in lfs:
                p_loss_nom = nom_p * float(lf)
                for p_ren in rens:
                    for ld in loads:
                        key = (
                            int(bus),
                            int(round(p_loss_nom)),
                            int(round(float(p_ren) * 100)),
                            int(round(float(ld) * 100)),
                        )
                        rep_idx = seen_counts.get(key, 0)
                        seen_counts[key] = rep_idx + 1
                        scen_id = create_scenario_id(bus, p_loss_nom, p_ren, ld, rep_idx)
                        part, ood_type = assign_scenario_partition(
                            p_ren, ld, scen_id, return_ood_type=True
                        )
                        combos.append((
                            int(bus),
                            float(lf),
                            float(p_loss_nom),
                            float(p_ren),
                            float(ld),
                            rep_idx,
                            scen_id,
                            part,
                            ood_type,
                        ))

        if max_scenarios is not None and max_scenarios < len(combos):
            selected = _stratified_subsample_combinations(combos, int(max_scenarios), batch_seed)
        elif max_scenarios is not None and max_scenarios > len(combos) and combos:
            # Extend deterministically with incremented rep_idx if caller asks for more than grid size
            selected = list(combos)
            extra_needed = int(max_scenarios) - len(combos)
            rng = np.random.default_rng(batch_seed + 1)
            for k in range(extra_needed):
                base = combos[int(rng.integers(0, len(combos)))]
                bus, lf, p_loss_nom, p_ren, ld, _, _, _, _ = base
                key = (int(bus), int(round(p_loss_nom)), int(round(p_ren * 100)), int(round(ld * 100)))
                rep_idx = seen_counts.get(key, 0)
                seen_counts[key] = rep_idx + 1
                scen_id = create_scenario_id(bus, p_loss_nom, p_ren, ld, rep_idx)
                part, ood_type = assign_scenario_partition(p_ren, ld, scen_id, return_ood_type=True)
                selected.append((bus, lf, p_loss_nom, p_ren, ld, rep_idx, scen_id, part, ood_type))
        else:
            selected = combos

        scenarios: List[Any] = []
        for idx, (bus, lf, p_loss_nom, p_ren, ld, rep_idx, scen_id, _, _) in enumerate(selected):
            offset = _scenario_seed_offset(scen_id) + idx
            scen_out = self.generate_single_scenario(
                tripped_bus=bus,
                p_loss_mw=p_loss_nom,
                renewable_penetration=p_ren,
                load_scale=ld,
                rep_idx=rep_idx,
                inject_noise=inject_noise,
                noise_config=noise_config,
                loss_fraction=lf,
                seed_offset=offset,
                observed_buses=observed_buses,
                return_raw=return_raw,
            )
            scenarios.append(scen_out)

        return scenarios


def export_raw_and_pmu_scenario(
    raw_traj: Dict[str, Any],
    pmu_traj: Dict[str, Any],
    raw_sim_dir: Union[str, Path],
    raw_pmu_dir: Union[str, Path],
) -> Tuple[str, str]:
    """Save internal ODE simulator states and observable PMU measurements to separate .npz directories."""
    sim_dir = Path(raw_sim_dir)
    pmu_dir = Path(raw_pmu_dir)
    sim_dir.mkdir(parents=True, exist_ok=True)
    pmu_dir.mkdir(parents=True, exist_ok=True)

    meta_pmu = dict(pmu_traj.get("metadata", {}))
    meta_raw = dict(raw_traj.get("metadata", {}))
    scen_id = str(meta_pmu.get("scenario_id") or meta_raw.get("scenario_id") or "scen_unknown")

    sim_path = str(sim_dir / f"{scen_id}.npz")
    pmu_path = str(pmu_dir / f"{scen_id}.npz")

    ts_raw = np.asarray(raw_traj["timestamps"], dtype=float)
    n_raw = len(ts_raw)
    np.savez_compressed(
        sim_path,
        timestamps=ts_raw,
        generator_deltas=np.asarray(raw_traj.get("generator_deltas", np.zeros((n_raw, 10))), dtype=float),
        generator_speeds=np.asarray(raw_traj.get("generator_speeds", np.zeros((n_raw, 10))), dtype=float),
        generator_powers=np.asarray(raw_traj.get("generator_powers", np.zeros((n_raw, 10))), dtype=float),
        coi_frequency=np.asarray(raw_traj["coi_frequency"], dtype=float),
        coi_rocof=np.asarray(raw_traj.get("coi_rocof", np.zeros(n_raw)), dtype=float),
        bus_frequencies_true=np.asarray(raw_traj["bus_frequencies"], dtype=float),
        voltage_magnitudes_true=np.asarray(raw_traj["voltage_magnitudes"], dtype=float),
        voltage_angles_true=np.asarray(raw_traj["voltage_angles"], dtype=float),
        metadata=meta_raw,
    )

    ts_pmu = np.asarray(pmu_traj["timestamps"], dtype=float)
    n_pmu = len(ts_pmu)
    bf_pmu = np.asarray(pmu_traj["bus_frequencies"], dtype=float)
    pmu_sys = pmu_traj.get("pmu_sys_frequency")
    if pmu_sys is None:
        pmu_sys = _estimate_pmu_sys_frequency(bf_pmu)
    else:
        pmu_sys = np.asarray(pmu_sys, dtype=float)

    soc = pmu_traj.get("soc")
    fracsec = pmu_traj.get("fracsec")
    if soc is None or fracsec is None:
        base_epoch = 1700000000
        soc = np.floor(base_epoch + ts_pmu).astype(np.uint32)
        fracsec = (np.round((base_epoch + ts_pmu - soc) * 1_000_000).astype(np.uint32) & 0x00FFFFFF)
    else:
        soc = np.asarray(soc, dtype=np.uint32)
        fracsec = np.asarray(fracsec, dtype=np.uint32)

    np.savez_compressed(
        pmu_path,
        timestamps=ts_pmu,
        soc=soc,
        fracsec=fracsec,
        bus_frequencies=bf_pmu,
        bus_rocof=np.asarray(pmu_traj.get("bus_rocof", np.zeros_like(bf_pmu)), dtype=float),
        voltage_magnitudes=np.asarray(pmu_traj["voltage_magnitudes"], dtype=float),
        voltage_angles=np.asarray(pmu_traj["voltage_angles"], dtype=float),
        pmu_sys_frequency=pmu_sys,
        metadata=meta_pmu,
    )

    return sim_path, pmu_path


def export_scenario_to_npz(scenario: Dict[str, Any], filepath: str) -> None:
    """Save observable PMU scenario trajectory to compressed .npz archive (backward-compatible)."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    ts = np.asarray(scenario["timestamps"], dtype=float)
    bf = np.asarray(scenario["bus_frequencies"], dtype=float)
    pmu_sys = scenario.get("pmu_sys_frequency")
    if pmu_sys is None:
        pmu_sys = _estimate_pmu_sys_frequency(bf)
    else:
        pmu_sys = np.asarray(pmu_sys, dtype=float)

    save_kwargs: Dict[str, Any] = {
        "timestamps": ts,
        "bus_frequencies": bf,
        "bus_rocof": np.asarray(scenario["bus_rocof"], dtype=float),
        "voltage_magnitudes": np.asarray(scenario["voltage_magnitudes"], dtype=float),
        "voltage_angles": np.asarray(scenario["voltage_angles"], dtype=float),
        "pmu_sys_frequency": pmu_sys,
        "coi_frequency": pmu_sys,
        "metadata": scenario["metadata"],
    }
    if "soc" in scenario:
        save_kwargs["soc"] = np.asarray(scenario["soc"], dtype=np.uint32)
    if "fracsec" in scenario:
        save_kwargs["fracsec"] = np.asarray(scenario["fracsec"], dtype=np.uint32)

    np.savez_compressed(filepath, **save_kwargs)


def load_scenario_from_npz(filepath: str) -> Dict[str, Any]:
    """Load PMU scenario trajectory from .npz file without leaking internal simulator states."""
    with np.load(filepath, allow_pickle=True) as data:
        files = set(data.files)
        raw_meta = data["metadata"]
        metadata = raw_meta.item() if hasattr(raw_meta, "item") else dict(raw_meta)

        bf = np.asarray(data["bus_frequencies"], dtype=float)
        if "pmu_sys_frequency" in files:
            pmu_sys = np.asarray(data["pmu_sys_frequency"], dtype=float)
        elif "coi_frequency" in files:
            pmu_sys = _estimate_pmu_sys_frequency(bf)
        else:
            pmu_sys = _estimate_pmu_sys_frequency(bf)

        out: Dict[str, Any] = {
            "timestamps": np.asarray(data["timestamps"], dtype=float),
            "bus_frequencies": bf,
            "bus_rocof": np.asarray(data["bus_rocof"], dtype=float),
            "voltage_magnitudes": np.asarray(data["voltage_magnitudes"], dtype=float),
            "voltage_angles": np.asarray(data["voltage_angles"], dtype=float),
            "pmu_sys_frequency": pmu_sys,
            # Alias points strictly to observable PMU system frequency for legacy callers
            "coi_frequency": pmu_sys.copy(),
            "metadata": metadata,
        }
        if "soc" in files:
            out["soc"] = np.asarray(data["soc"], dtype=np.uint32)
        if "fracsec" in files:
            out["fracsec"] = np.asarray(data["fracsec"], dtype=np.uint32)
        return out
