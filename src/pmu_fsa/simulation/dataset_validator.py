"""Phase 5 dataset quality validation, causality audit, and SHA-256 manifest generation."""

from __future__ import annotations

import datetime as dt_mod
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_VERSION = "2.0.0"
FEATURE_SCHEMA_VERSION = "2.0.0"


def _to_portable_path(p: Optional[Union[str, Path]]) -> Optional[str]:
    """Convert paths inside REPO_ROOT to portable POSIX repo-relative strings."""
    if p is None:
        return None
    path_obj = Path(p)
    try:
        resolved = path_obj.resolve()
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        return path_obj.as_posix()


def _get_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"

from pmu_fsa.simulation.ieee39_data import (
    NOMINAL_FREQ,
    PHYSICAL_FREQ_BOUNDS,
    SECURITY_THRESHOLD_HZ,
)
from pmu_fsa.simulation.scenario_generator import (
    classify_ood_regime,
    get_software_versions,
)


EXPECTED_CANONICAL_14_FEATURES: List[str] = [
    "rocof_init",
    "rocof_peak_neg",
    "rocof_end",
    "rocof_mean",
    "freq_drop_end",
    "freq_drop_max",
    "freq_dev_integral",
    "bus_drop_worst",
    "area_freq_spread_max",
    "bus_freq_std_mean",
    "angle_divergence_max",
    "tie_line_angle_excursion",
    "freq_curvature_init",
    "freq_curvature_mean",
]

VALID_PARTITIONS = ("train", "val", "test", "ood_test")
INTERNAL_SIM_FORBIDDEN_KEYS = ("generator_deltas", "generator_speeds", "generator_powers")


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Compute hex SHA-256 digest of a file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert NumPy scalars/arrays and Paths to native JSON-serializable types."""
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return _to_json_safe(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if math.isfinite(val) else None
    return obj


def _expected_window_samples(t_obs: float, fps: float) -> int:
    exact_frames = t_obs * fps
    if abs(exact_frames - round(exact_frames)) < 1e-6:
        return int(round(exact_frames)) + 1
    return int(math.floor(exact_frames + 1e-9)) + 1


def validate_dataset(
    raw_pmu_dir: Union[str, Path],
    processed_dir: Optional[Union[str, Path]] = None,
    windows: Sequence[float] = (0.5, 1.0, 1.5, 2.0),
    f_nominal: float = NOMINAL_FREQ,
    threshold: float = SECURITY_THRESHOLD_HZ,
    raw_sim_dir: Optional[Union[str, Path]] = None,
    expected_n_scenarios: Optional[int] = None,
) -> Dict[str, Any]:
    """Run all 14 Phase-5 dataset quality, leakage, unit, and causality checks."""
    pmu_dir = Path(raw_pmu_dir)
    proc_dir = Path(processed_dir) if processed_dir is not None else None
    sim_dir = Path(raw_sim_dir) if raw_sim_dir is not None else None
    win_list = [float(w) for w in windows]
    f_nom = float(f_nominal)
    f_thresh = float(threshold)
    low_bound, high_bound = PHYSICAL_FREQ_BOUNDS

    pmu_files = sorted(pmu_dir.glob("*.npz")) if pmu_dir.exists() else []
    sim_files = sorted(sim_dir.glob("*.npz")) if (sim_dir is not None and sim_dir.exists()) else []
    sim_map = {fp.stem: fp for fp in sim_files}

    checksums: Dict[str, Dict[str, str]] = {
        "raw_pmu": {},
        "raw_sim": {},
        "processed": {},
    }

    seen_ids: List[str] = []
    duplicate_ids: List[str] = []
    stem_mismatch_ids: List[str] = []
    nan_inf_errors: List[str] = []
    internal_leak_files: List[str] = []
    unit_errors: List[str] = []
    dt_errors: List[str] = []
    sample_count_errors: Dict[str, List[str]] = {f"{w:.1f}s": [] for w in win_list}
    nadir_label_errors: List[str] = []
    security_label_errors: List[str] = []
    ren_consistency_errors: List[str] = []
    physical_bound_errors: List[str] = []
    partition_rule_errors: List[str] = []

    raw_partition_by_id: Dict[str, str] = {}
    raw_ood_type_by_id: Dict[str, str] = {}
    raw_secure_by_id: Dict[str, int] = {}
    raw_nadir_by_id: Dict[str, float] = {}
    raw_t_nadir_by_id: Dict[str, float] = {}
    raw_t_disturb_by_id: Dict[str, float] = {}
    raw_ren_req_by_id: Dict[str, float] = {}
    raw_ren_real_by_id: Dict[str, float] = {}
    scenario_records_by_id: Dict[str, Dict[str, Any]] = {}
    raw_signatures: List[np.ndarray] = []

    id_set_seen: set[str] = set()

    for fp in pmu_files:
        checksums["raw_pmu"][fp.name] = compute_file_sha256(fp)
        with np.load(fp, allow_pickle=True) as data:
            keys = set(data.files)
            for bad_k in INTERNAL_SIM_FORBIDDEN_KEYS:
                if bad_k in keys:
                    internal_leak_files.append(f"{fp.name}:{bad_k}")

            raw_meta = data["metadata"]
            meta = raw_meta.item() if hasattr(raw_meta, "item") else dict(raw_meta)

            scen_id = str(meta.get("scenario_id", fp.stem))
            seen_ids.append(scen_id)
            if scen_id != fp.stem:
                stem_mismatch_ids.append(f"{fp.name}->{scen_id}")
            if scen_id in id_set_seen:
                duplicate_ids.append(scen_id)
            id_set_seen.add(scen_id)

            ts = np.asarray(data["timestamps"], dtype=float)
            bf = np.asarray(data["bus_frequencies"], dtype=float)
            br = np.asarray(data["bus_rocof"], dtype=float)
            vm = np.asarray(data["voltage_magnitudes"], dtype=float)
            va = np.asarray(data["voltage_angles"], dtype=float)
            psys = (
                np.asarray(data["pmu_sys_frequency"], dtype=float)
                if "pmu_sys_frequency" in keys
                else np.asarray(data["coi_frequency"], dtype=float)
            )

            obs_buses = meta.get("observed_buses")
            if obs_buses is not None and bf.ndim == 2:
                obs_idx = [int(b) - 1 for b in obs_buses if 1 <= int(b) <= bf.shape[1]]
                bf_chk = bf[:, obs_idx]
                br_chk = br[:, obs_idx]
                vm_chk = vm[:, obs_idx]
                va_chk = va[:, obs_idx]
            else:
                bf_chk, br_chk, vm_chk, va_chk = bf, br, vm, va

            for ch_name, arr in (
                ("timestamps", ts),
                ("bus_frequencies", bf_chk),
                ("bus_rocof", br_chk),
                ("voltage_magnitudes", vm_chk),
                ("voltage_angles", va_chk),
                ("pmu_sys_frequency", psys),
            ):
                if arr.size == 0 or not np.all(np.isfinite(arr)):
                    nan_inf_errors.append(f"{scen_id}:{ch_name}")

            # Raw trajectory signature for duplicate-trajectory check when processed_dir is absent
            if psys.size >= 5:
                idx_sig = np.linspace(0, psys.size - 1, 8, dtype=int)
                raw_signatures.append(np.concatenate([
                    psys[idx_sig],
                    [float(meta.get("ground_truth_nadir", 0.0)), float(meta.get("load_scale", 1.0))],
                ]))

            # Check 5: frequency (Hz), voltage magnitude (pu), angle (deg), ROCOF (Hz/s) units
            if psys.size > 0 and np.all(np.isfinite(psys)):
                if not (f_nom - 1.5 <= float(psys[0]) <= f_nom + 1.5):
                    unit_errors.append(f"{scen_id}:init_freq={psys[0]:.4f}")
                if float(np.min(bf_chk)) < 35.0 or float(np.max(bf_chk)) > 65.0:
                    unit_errors.append(f"{scen_id}:bus_freq_range=[{np.min(bf_chk):.2f},{np.max(bf_chk):.2f}]")
            if vm_chk.size > 0 and np.all(np.isfinite(vm_chk)):
                if float(np.min(vm_chk)) < 0.01 or float(np.max(vm_chk)) > 2.0:
                    unit_errors.append(f"{scen_id}:vmag_range=[{np.min(vm_chk):.3f},{np.max(vm_chk):.3f}]")
            if va_chk.size > 0 and np.all(np.isfinite(va_chk)):
                if float(np.max(np.abs(va_chk))) > 360.0 + 1e-3:
                    unit_errors.append(f"{scen_id}:vang_max_abs={np.max(np.abs(va_chk)):.2f}")
            if br_chk.size > 0 and np.all(np.isfinite(br_chk)):
                if float(np.max(np.abs(br_chk))) > 100.0:
                    unit_errors.append(f"{scen_id}:rocof_max_abs={np.max(np.abs(br_chk)):.2f}")

            # Check 6: uniform 1/fps sampling intervals
            fps = float(meta.get("fps", 50.0))
            expected_dt = 1.0 / fps
            if ts.size < 2:
                dt_errors.append(f"{scen_id}:len(ts)<2")
            else:
                diffs = np.diff(ts)
                if not np.allclose(diffs, expected_dt, atol=1e-6):
                    dt_errors.append(f"{scen_id}:max_dt_err={float(np.max(np.abs(diffs - expected_dt))):.2e}")

            # Check 7: expected sample counts per window
            t0 = float(meta.get("t_disturb", 1.0))
            for w in win_list:
                i0 = int(np.searchsorted(ts, t0 - 1e-9, side="left"))
                i1 = int(np.searchsorted(ts, t0 + w + 1e-9, side="right"))
                actual_n = i1 - i0
                exp_n = _expected_window_samples(w, fps)
                if actual_n != exp_n:
                    sample_count_errors[f"{w:.1f}s"].append(f"{scen_id}:{actual_n}!={exp_n}")

            # Check 8, 9, 13: COI nadir labels, security consistency, physical bounds
            if "ground_truth_nadir" not in meta or "ground_truth_t_nadir" not in meta:
                nadir_label_errors.append(f"{scen_id}:missing_nadir_keys")
                gt_nadir = float("nan")
                gt_t_nadir = float("nan")
            else:
                gt_nadir = float(meta["ground_truth_nadir"])
                gt_t_nadir = float(meta["ground_truth_t_nadir"])
                if not (math.isfinite(gt_nadir) and math.isfinite(gt_t_nadir)):
                    nadir_label_errors.append(f"{scen_id}:nonfinite_nadir")
                elif gt_t_nadir < t0 - 1e-6 or gt_nadir > f_nom + 0.05:
                    nadir_label_errors.append(f"{scen_id}:invalid_nadir=({gt_nadir:.4f}@{gt_t_nadir:.3f}s)")

                if scen_id in sim_map:
                    with np.load(sim_map[scen_id], allow_pickle=True) as sim_data:
                        t_s = np.asarray(sim_data["timestamps"], dtype=float)
                        coi_s = np.asarray(sim_data["coi_frequency"], dtype=float)
                        post_s = coi_s[t_s >= t0 - 1e-9]
                        if post_s.size > 0:
                            true_coi_min = float(np.min(post_s))
                            if abs(gt_nadir - true_coi_min) > 1e-3:
                                nadir_label_errors.append(
                                    f"{scen_id}:gt_nadir({gt_nadir:.4f})!=coi_min({true_coi_min:.4f})"
                                )

            if math.isfinite(gt_nadir):
                if not (low_bound <= gt_nadir <= high_bound):
                    physical_bound_errors.append(f"{scen_id}:{gt_nadir:.4f}")

                is_sec = int(meta.get("is_secure", -1))
                expected_sec = int(gt_nadir >= f_thresh)
                if is_sec != expected_sec:
                    security_label_errors.append(f"{scen_id}:is_secure={is_sec}!=expected={expected_sec}")
                raw_secure_by_id[scen_id] = is_sec
                raw_nadir_by_id[scen_id] = gt_nadir
                raw_t_nadir_by_id[scen_id] = gt_t_nadir
                raw_t_disturb_by_id[scen_id] = t0

            # Check 10: realized vs requested renewable consistency
            p_req = float(meta.get("p_ren_requested", meta.get("renewable_penetration", 0.0)))
            p_real = float(meta.get("p_ren_realized", p_req))
            raw_ren_req_by_id[scen_id] = p_req
            raw_ren_real_by_id[scen_id] = p_real
            if not (0.0 <= p_req <= 1.0 and 0.0 <= p_real <= 1.0):
                ren_consistency_errors.append(f"{scen_id}:out_of_range({p_req:.3f},{p_real:.3f})")
            elif abs(p_real - p_req) > 0.025:
                ren_consistency_errors.append(f"{scen_id}:req={p_req:.3f}_vs_real={p_real:.3f}")

            # Check 11: partition & OOD rules
            ld_scale = float(meta.get("load_scale", 1.0))
            part = str(meta.get("partition", "unknown"))
            expected_ood = classify_ood_regime(p_req, ld_scale)
            rec_ood = str(meta.get("ood_type", expected_ood))
            raw_partition_by_id[scen_id] = part
            raw_ood_type_by_id[scen_id] = rec_ood

            if part not in VALID_PARTITIONS:
                partition_rule_errors.append(f"{scen_id}:invalid_partition={part}")
            elif expected_ood != "id" and part != "ood_test":
                partition_rule_errors.append(f"{scen_id}:ood({expected_ood})_in_{part}")
            elif expected_ood == "id" and part == "ood_test":
                partition_rule_errors.append(f"{scen_id}:id_in_ood_test")
            if rec_ood != expected_ood:
                partition_rule_errors.append(f"{scen_id}:ood_type={rec_ood}!=expected={expected_ood}")

            scenario_records_by_id[scen_id] = {
                "scenario_id": scen_id,
                "partition": part,
                "ood_type": rec_ood,
                "disturbed_gen": int(meta.get("disturbed_gen", 0)),
                "loss_fraction": round(float(meta.get("loss_fraction", 0.0)), 4),
                "active_loss_mw": round(float(meta.get("active_loss_mw", 0.0)), 3),
                "p_ren_requested": round(p_req, 4),
                "p_ren_realized": round(p_real, 4),
                "load_scale": round(ld_scale, 4),
                "t_disturb_s": round(t0, 4),
                "ground_truth_nadir_hz": round(gt_nadir, 5) if math.isfinite(gt_nadir) else None,
                "ground_truth_t_nadir_s": round(gt_t_nadir, 4) if math.isfinite(gt_t_nadir) else None,
                "is_secure": int(meta.get("is_secure", -1)) if math.isfinite(gt_nadir) else None,
                "causal_windows": {
                    f"{w:.1f}s": bool(math.isfinite(gt_t_nadir) and (t0 + w < gt_t_nadir - 1e-6))
                    for w in win_list
                },
            }

    for fp in sim_files:
        checksums["raw_sim"][fp.name] = compute_file_sha256(fp)

    # Check 2: missing scenarios across raw_pmu / raw_sim / expected_n_scenarios / processed
    missing_errors: List[str] = []
    if len(pmu_files) == 0:
        missing_errors.append(f"No .npz scenario files found in {pmu_dir}")
    if expected_n_scenarios is not None and len(pmu_files) != int(expected_n_scenarios):
        missing_errors.append(f"raw_pmu count {len(pmu_files)} != expected {expected_n_scenarios}")
    if sim_dir is not None and sim_dir.exists() and len(sim_files) > 0:
        pmu_stems = {fp.stem for fp in pmu_files}
        sim_stems = {fp.stem for fp in sim_files}
        only_pmu = sorted(pmu_stems - sim_stems)
        only_sim = sorted(sim_stems - pmu_stems)
        if only_pmu:
            missing_errors.append(f"missing_in_raw_sim:{only_pmu[:5]}")
        if only_sim:
            missing_errors.append(f"missing_in_raw_pmu:{only_sim[:5]}")

    # Processed feature matrix checks (when processed_dir exists and has .npz files)
    processed_npz_files = sorted(proc_dir.glob("*.npz")) if (proc_dir is not None and proc_dir.exists()) else []
    processed_csv_files = sorted(proc_dir.glob("*.csv")) if (proc_dir is not None and proc_dir.exists()) else []
    for fp in processed_npz_files + processed_csv_files:
        checksums["processed"][fp.name] = compute_file_sha256(fp)

    duplicate_feature_row_errors: List[str] = []
    feature_order_errors: List[str] = []
    cross_window_partition_errors: List[str] = []
    processed_summary: Dict[str, Any] = {}

    if processed_npz_files:
        for p_fp in processed_npz_files:
            with np.load(p_fp, allow_pickle=True) as pdata:
                pkeys = set(pdata.files)
                X = np.asarray(pdata["X"], dtype=float)
                y_nadir = np.asarray(pdata["y_nadir"], dtype=float)
                y_sec = np.asarray(pdata["y_secure"], dtype=int)
                sids = [str(s) for s in pdata["scenario_ids"]]
                parts = [str(p) for p in pdata["partitions"]]
                fnames = [str(f) for f in pdata["feature_names"]] if "feature_names" in pkeys else []
                excl_ids = (
                    [str(s) for s in pdata["excluded_scenario_ids"]]
                    if "excluded_scenario_ids" in pkeys
                    else []
                )

                processed_summary[p_fp.name] = {
                    "shape": list(X.shape),
                    "n_features": int(X.shape[1]) if X.ndim == 2 else 0,
                    "n_excluded_strict_nadir": len(excl_ids),
                    "feature_names": fnames,
                }

                if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y_nadir)) and np.all(np.isfinite(y_sec))):
                    nan_inf_errors.append(f"processed:{p_fp.name}:NaN_or_Inf")

                if len(sids) != len(set(sids)):
                    duplicate_ids.append(f"processed:{p_fp.name}:duplicate_ids")

                # Account for raw scenarios vs processed + explicitly logged strict-nadir exclusions
                accounted = set(sids) | set(excl_ids)
                if id_set_seen and accounted != id_set_seen:
                    unaccounted = sorted(id_set_seen - accounted)
                    if unaccounted:
                        missing_errors.append(f"{p_fp.name}:missing_unlogged={unaccounted[:5]}")

                # Duplicate feature rows check
                if X.ndim == 2 and X.shape[0] > 1:
                    rounded_X = np.round(X, 10)
                    _, uniq_counts = np.unique(rounded_X, axis=0, return_counts=True)
                    n_dup_rows = int(np.sum(uniq_counts[uniq_counts > 1] - 1))
                    if n_dup_rows > 0:
                        duplicate_feature_row_errors.append(f"{p_fp.name}:{n_dup_rows}_duplicate_rows")

                # Canonical 14-feature ordering check
                if X.ndim != 2 or X.shape[1] != 14:
                    feature_order_errors.append(f"{p_fp.name}:shape={X.shape}!=14_cols")
                if fnames != EXPECTED_CANONICAL_14_FEATURES:
                    feature_order_errors.append(f"{p_fp.name}:feature_names={fnames}")

                # Security label consistency in processed file
                if not np.array_equal(y_sec, (y_nadir >= f_thresh).astype(int)):
                    security_label_errors.append(f"processed:{p_fp.name}:y_secure_mismatch")

                # Partition consistency across windows and vs raw metadata
                for sid, part in zip(sids, parts):
                    if sid in raw_partition_by_id and raw_partition_by_id[sid] != part:
                        cross_window_partition_errors.append(
                            f"{p_fp.name}:{sid}:{part}!=raw({raw_partition_by_id[sid]})"
                        )
    else:
        # When processed_dir is not yet generated, verify raw trajectory uniqueness & extractor schema
        if len(raw_signatures) > 1:
            sig_mat = np.round(np.vstack(raw_signatures), 8)
            _, uniq_counts = np.unique(sig_mat, axis=0, return_counts=True)
            n_dup_raw = int(np.sum(uniq_counts[uniq_counts > 1] - 1))
            if n_dup_raw > 0:
                duplicate_feature_row_errors.append(f"raw_trajectories:{n_dup_raw}_identical_trajectories")
        try:
            from pmu_fsa.features import extractor as _ext
            ext_names = list(getattr(_ext, "CANONICAL_FEATURE_NAMES", getattr(_ext, "DESCRIPTIVE_FEATURE_NAMES", [])))
            if len(ext_names) != 14 or ext_names != EXPECTED_CANONICAL_14_FEATURES:
                feature_order_errors.append(f"extractor_module_schema={ext_names}")
        except Exception as exc:
            feature_order_errors.append(f"extractor_import_error:{exc}")

    # Partition scenario sets & pairwise overlap check
    part_sets: Dict[str, set[str]] = {p: set() for p in VALID_PARTITIONS}
    for sid, part in raw_partition_by_id.items():
        if part in part_sets:
            part_sets[part].add(sid)

    overlap_errors: List[str] = []
    for i, p1 in enumerate(VALID_PARTITIONS):
        for p2 in VALID_PARTITIONS[i + 1 :]:
            inter = part_sets[p1] & part_sets[p2]
            if inter:
                overlap_errors.append(f"{p1}&{p2}:{sorted(inter)[:5]}")

    # Check 14: strict pre-nadir causality per window
    causality_by_window: Dict[str, Dict[str, Any]] = {}
    for w in win_list:
        w_key = f"{w:.1f}s"
        causal_ids: List[str] = []
        violating_ids: List[str] = []
        lead_times: List[float] = []

        for sid in sorted(raw_t_nadir_by_id.keys()):
            t0 = raw_t_disturb_by_id[sid]
            t_nad = raw_t_nadir_by_id[sid]
            t_end = t0 + w
            lead_s = t_nad - t_end
            lead_times.append(lead_s)
            if t_end < t_nad - 1e-6:
                causal_ids.append(sid)
            else:
                violating_ids.append(sid)

        n_tot = len(causal_ids) + len(violating_ids)
        lt_arr = np.asarray(lead_times, dtype=float) if lead_times else np.array([0.0])
        causality_by_window[w_key] = {
            "t_obs_s": w,
            "n_total": n_tot,
            "n_causal": len(causal_ids),
            "n_violating": len(violating_ids),
            "causal_fraction": round(len(causal_ids) / max(1, n_tot), 6),
            "violating_scenario_ids": violating_ids,
            "mean_lead_time_s": round(float(np.mean(lt_arr)), 4),
            "median_lead_time_s": round(float(np.median(lt_arr)), 4),
            "min_lead_time_s": round(float(np.min(lt_arr)), 4),
            "max_lead_time_s": round(float(np.max(lt_arr)), 4),
        }

    primary_win_key = "1.0s" if "1.0s" in causality_by_window else (f"{win_list[0]:.1f}s" if win_list else "1.0s")
    primary_causal_ok = (
        len(pmu_files) > 0
        and causality_by_window.get(primary_win_key, {}).get("n_causal", 0) > 0
    )

    checks: Dict[str, Dict[str, Any]] = {
        "1_duplicate_scenario_ids": {
            "passed": len(duplicate_ids) == 0 and len(stem_mismatch_ids) == 0,
            "duplicate_count": len(duplicate_ids),
            "stem_mismatch_count": len(stem_mismatch_ids),
            "errors": duplicate_ids + stem_mismatch_ids,
        },
        "2_missing_scenarios": {
            "passed": len(missing_errors) == 0,
            "raw_pmu_count": len(pmu_files),
            "raw_sim_count": len(sim_files),
            "errors": missing_errors,
        },
        "3_duplicate_feature_rows": {
            "passed": len(duplicate_feature_row_errors) == 0,
            "checked_processed_files": len(processed_npz_files),
            "errors": duplicate_feature_row_errors,
        },
        "4_nan_inf_and_channel_separation": {
            "passed": len(nan_inf_errors) == 0 and len(internal_leak_files) == 0,
            "nan_inf_count": len(nan_inf_errors),
            "internal_channel_leak_count": len(internal_leak_files),
            "errors": (nan_inf_errors + internal_leak_files)[:20],
        },
        "5_frequency_angle_rocof_units": {
            "passed": len(unit_errors) == 0,
            "error_count": len(unit_errors),
            "errors": unit_errors[:20],
        },
        "6_uniform_sampling_intervals": {
            "passed": len(dt_errors) == 0,
            "error_count": len(dt_errors),
            "errors": dt_errors[:20],
        },
        "7_expected_window_sample_counts": {
            "passed": all(len(v) == 0 for v in sample_count_errors.values()),
            "errors_by_window": {k: v[:10] for k, v in sample_count_errors.items() if v},
        },
        "8_valid_coi_nadir_labels": {
            "passed": len(nadir_label_errors) == 0,
            "error_count": len(nadir_label_errors),
            "errors": nadir_label_errors[:20],
        },
        "9_security_label_consistency": {
            "passed": len(security_label_errors) == 0,
            "threshold_hz": f_thresh,
            "error_count": len(security_label_errors),
            "errors": security_label_errors[:20],
        },
        "10_renewable_penetration_consistency": {
            "passed": len(ren_consistency_errors) == 0,
            "max_abs_diff": round(
                max(
                    (abs(raw_ren_real_by_id[s] - raw_ren_req_by_id[s]) for s in raw_ren_req_by_id),
                    default=0.0,
                ),
                6,
            ),
            "errors": ren_consistency_errors[:20],
        },
        "11_zero_partition_overlap_and_window_consistency": {
            "passed": (
                len(overlap_errors) == 0
                and len(partition_rule_errors) == 0
                and len(cross_window_partition_errors) == 0
            ),
            "overlap_errors": overlap_errors,
            "rule_errors": partition_rule_errors[:20],
            "cross_window_errors": cross_window_partition_errors[:20],
        },
        "12_canonical_14_feature_ordering": {
            "passed": len(feature_order_errors) == 0,
            "expected_features": EXPECTED_CANONICAL_14_FEATURES,
            "processed_files_checked": len(processed_npz_files),
            "errors": feature_order_errors,
        },
        "13_physical_target_bounds": {
            "passed": len(physical_bound_errors) == 0,
            "bounds_hz": [low_bound, high_bound],
            "min_nadir_hz": round(min(raw_nadir_by_id.values()), 4) if raw_nadir_by_id else None,
            "max_nadir_hz": round(max(raw_nadir_by_id.values()), 4) if raw_nadir_by_id else None,
            "errors": physical_bound_errors[:20],
        },
        "14_pre_nadir_causality_audit": {
            "passed": primary_causal_ok,
            "windows": causality_by_window,
        },
    }

    partition_counts = {p: len(part_sets[p]) for p in VALID_PARTITIONS}
    ood_types_list = ("id", "ood_res", "ood_load", "ood_combined")
    ood_type_counts = {
        ot: sum(1 for v in raw_ood_type_by_id.values() if v == ot)
        for ot in ood_types_list
    }

    class_counts_overall = {
        "secure_1": sum(1 for v in raw_secure_by_id.values() if v == 1),
        "insecure_0": sum(1 for v in raw_secure_by_id.values() if v == 0),
    }
    class_counts_by_partition: Dict[str, Dict[str, int]] = {}
    for p in VALID_PARTITIONS:
        sids_p = part_sets[p]
        class_counts_by_partition[p] = {
            "total": len(sids_p),
            "secure_1": sum(1 for s in sids_p if raw_secure_by_id.get(s) == 1),
            "insecure_0": sum(1 for s in sids_p if raw_secure_by_id.get(s) == 0),
        }

    class_counts_by_ood_type: Dict[str, Dict[str, int]] = {}
    for ot in ood_types_list:
        sids_o = [s for s, t in raw_ood_type_by_id.items() if t == ot]
        class_counts_by_ood_type[ot] = {
            "total": len(sids_o),
            "secure_1": sum(1 for s in sids_o if raw_secure_by_id.get(s) == 1),
            "insecure_0": sum(1 for s in sids_o if raw_secure_by_id.get(s) == 0),
        }

    nadirs_arr = np.asarray(list(raw_nadir_by_id.values()), dtype=float) if raw_nadir_by_id else np.array([])
    tnadirs_arr = np.asarray(list(raw_t_nadir_by_id.values()), dtype=float) if raw_t_nadir_by_id else np.array([])
    nadir_stats = {
        "min_hz": round(float(np.min(nadirs_arr)), 4) if nadirs_arr.size else None,
        "median_hz": round(float(np.median(nadirs_arr)), 4) if nadirs_arr.size else None,
        "mean_hz": round(float(np.mean(nadirs_arr)), 4) if nadirs_arr.size else None,
        "max_hz": round(float(np.max(nadirs_arr)), 4) if nadirs_arr.size else None,
        "t_nadir_min_s": round(float(np.min(tnadirs_arr)), 4) if tnadirs_arr.size else None,
        "t_nadir_median_s": round(float(np.median(tnadirs_arr)), 4) if tnadirs_arr.size else None,
        "t_nadir_max_s": round(float(np.max(tnadirs_arr)), 4) if tnadirs_arr.size else None,
    }

    agg_hasher = hashlib.sha256()
    for group_key in ("raw_pmu", "raw_sim", "processed"):
        for fname in sorted(checksums[group_key].keys()):
            agg_hasher.update(f"{group_key}/{fname}:{checksums[group_key][fname]}\n".encode("utf-8"))

    partition_hashes: Dict[str, Any] = {}
    for p in VALID_PARTITIONS:
        sorted_sids = sorted(part_sets[p])
        partition_hashes[p] = {
            "count": len(sorted_sids),
            "scenario_ids_sha256": hashlib.sha256("\n".join(sorted_sids).encode("utf-8")).hexdigest(),
        }
    partition_hashes["zero_overlap_verified"] = bool(
        checks["11_zero_partition_overlap_and_window_consistency"]["passed"]
    )

    all_passed = all(bool(c["passed"]) for c in checks.values())

    report: Dict[str, Any] = {
        "dataset_version": DATASET_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "git_commit": _get_git_commit(),
        "random_seed": 42,
        "generated_at_utc": dt_mod.datetime.now(dt_mod.timezone.utc).isoformat(),
        "raw_pmu_dir": _to_portable_path(pmu_dir),
        "raw_sim_dir": _to_portable_path(sim_dir) if sim_dir is not None else None,
        "processed_dir": _to_portable_path(proc_dir) if proc_dir is not None else None,
        "f_nominal_hz": f_nom,
        "security_threshold_hz": f_thresh,
        "windows_s": win_list,
        "simulation_config": {
            "network": "IEEE 39-Bus, 10-Machine (New England)",
            "base_mva": 100.0,
            "f_nominal_hz": f_nom,
            "security_threshold_hz": f_thresh,
            "pmu_fps": 50,
            "t_disturb_s": 1.0,
            "t_sim_s": 10.0,
            "macro_intertie": False,
            "bus39_h_system_base_s": 42.0,
            "synchronous_governor_droop_r_pu": 0.05,
            "synchronous_governor_tg_s": 4.20,
            "converter_fast_droop_enabled": False,
            "renewable_representation": (
                "Approach B: co-located constant-power GFL effective inertia, damping, "
                "and droop participation derating at unchanged terminal bus active-power dispatch"
            ),
            "pmu_filter": "Causal 5-point backward OLS ROCOF and area-weighted observed bus frequency reconstruction",
        },
        "total_scenarios": len(pmu_files),
        "all_passed": all_passed,
        "partition_counts": partition_counts,
        "partition_scenario_hashes": partition_hashes,
        "ood_type_counts": ood_type_counts,
        "class_counts_overall": class_counts_overall,
        "class_counts_by_partition": class_counts_by_partition,
        "class_counts_by_ood_type": class_counts_by_ood_type,
        "nadir_distribution": nadir_stats,
        "processed_files": processed_summary,
        "checks": checks,
        "dataset_aggregate_sha256": agg_hasher.hexdigest(),
        "scenario_records": [scenario_records_by_id[k] for k in sorted(scenario_records_by_id.keys())],
        "checksums_sha256": checksums,
        "software_versions": get_software_versions(),
    }
    return _to_json_safe(report)


def write_dataset_quality_report(
    report: Dict[str, Any],
    md_path: Union[str, Path],
    manifest_path: Union[str, Path],
) -> Tuple[str, str]:
    """Export JSON manifest (with SHA-256 digests) and Markdown dataset quality report."""
    md_p = Path(md_path)
    man_p = Path(manifest_path)
    md_p.parent.mkdir(parents=True, exist_ok=True)
    man_p.parent.mkdir(parents=True, exist_ok=True)

    safe_report = _to_json_safe(report)
    with open(man_p, "w", encoding="utf-8") as f:
        json.dump(safe_report, f, indent=2, sort_keys=False)

    checks = safe_report.get("checks", {})
    part_cls = safe_report.get("class_counts_by_partition", {})
    ood_cls = safe_report.get("class_counts_by_ood_type", {})
    nadir_dist = safe_report.get("nadir_distribution", {})
    causality = checks.get("14_pre_nadir_causality_audit", {}).get("windows", {})

    lines: List[str] = [
        "# IEEE 39-Bus PMU-FSA Dataset Quality & Causality Validation Report",
        "",
        f"- **Generated (UTC)**: `{safe_report.get('generated_at_utc')}`",
        f"- **Overall Validation Status**: **{'PASS' if safe_report.get('all_passed') else 'FAIL'}**",
        f"- **Total Scenarios**: `{safe_report.get('total_scenarios')}`",
        f"- **Nominal Frequency / Threshold**: `{safe_report.get('f_nominal_hz')} Hz` / `{safe_report.get('security_threshold_hz')} Hz`",
        f"- **Raw PMU Directory**: `{safe_report.get('raw_pmu_dir')}`",
        f"- **Raw Simulation Directory**: `{safe_report.get('raw_sim_dir')}`",
        f"- **Processed Feature Directory**: `{safe_report.get('processed_dir')}`",
        f"- **Aggregate Dataset SHA-256**: `{safe_report.get('dataset_aggregate_sha256')}`",
        "",
        "## 1. Scenario-Level Partition & Class Balance Summary",
        "",
        "| Partition | Total Scenarios | Secure (`>= 49.50 Hz`) | Insecure (`< 49.50 Hz`) | Secure Fraction |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]

    for p in VALID_PARTITIONS:
        row = part_cls.get(p, {"total": 0, "secure_1": 0, "insecure_0": 0})
        tot = int(row["total"])
        sec = int(row["secure_1"])
        insec = int(row["insecure_0"])
        frac = (sec / tot) if tot > 0 else 0.0
        lines.append(f"| `{p}` | {tot} | {sec} | {insec} | {frac:.3f} |")

    lines.extend([
        "",
        "### OOD Regime Breakdown",
        "",
        "| Regime (`ood_type`) | Total Scenarios | Secure (`1`) | Insecure (`0`) |",
        "| :--- | ---: | ---: | ---: |",
    ])
    for ot in ("id", "ood_res", "ood_load", "ood_combined"):
        row = ood_cls.get(ot, {"total": 0, "secure_1": 0, "insecure_0": 0})
        lines.append(f"| `{ot}` | {row['total']} | {row['secure_1']} | {row['insecure_0']} |")

    lines.extend([
        "",
        "## 2. Phase 5 Quality & Integrity Checklist (14 Automated Checks)",
        "",
        "| Check | Status | Summary |",
        "| :--- | :---: | :--- |",
    ])

    for chk_key, chk_val in checks.items():
        status_str = "PASS" if chk_val.get("passed") else "FAIL"
        errs = chk_val.get("errors") or chk_val.get("overlap_errors") or []
        detail = "0 violations" if not errs else f"{len(errs)} issue(s): {errs[:3]}"
        lines.append(f"| `{chk_key}` | **{status_str}** | {detail} |")

    lines.extend([
        "",
        "## 3. Pre-Nadir Causality & Lead-Time Audit (`t_0 + T_obs < t_nadir`)",
        "",
        "| Window `T_obs` (s) | Total | Strictly Causal | Violating | Causal Rate | Mean Lead (s) | Median Lead (s) | Min Lead (s) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])

    violating_details: List[str] = []
    for w_key, c_info in causality.items():
        lines.append(
            f"| {c_info['t_obs_s']:.1f} | {c_info['n_total']} | {c_info['n_causal']} | "
            f"{c_info['n_violating']} | {c_info['causal_fraction'] * 100:.2f}% | "
            f"{c_info['mean_lead_time_s']:.3f} | {c_info['median_lead_time_s']:.3f} | "
            f"{c_info['min_lead_time_s']:.3f} |"
        )
        v_ids = c_info.get("violating_scenario_ids", [])
        if v_ids:
            violating_details.append(f"- **`T_obs = {w_key}`** ({len(v_ids)} non-causal): `{', '.join(v_ids[:15])}`" + (" ..." if len(v_ids) > 15 else ""))

    if violating_details:
        lines.append("")
        lines.append("### Scenarios Violating Pre-Nadir Causality")
        lines.extend(violating_details)

    lines.extend([
        "",
        "## 4. COI Frequency Nadir Distribution",
        "",
        f"- **Nadir Range (Hz)**: `[{nadir_dist.get('min_hz')}, {nadir_dist.get('max_hz')}]` "
        f"(Mean: `{nadir_dist.get('mean_hz')} Hz`, Median: `{nadir_dist.get('median_hz')} Hz`)",
        f"- **Time-to-Nadir Range (s)**: `[{nadir_dist.get('t_nadir_min_s')}, {nadir_dist.get('t_nadir_max_s')}]` "
        f"(Median: `{nadir_dist.get('t_nadir_median_s')} s`)",
        f"- **Manifest File**: `{_to_portable_path(man_p)}`",
        "",
    ])

    md_p.write_text("\n".join(lines), encoding="utf-8")
    return str(md_p), str(man_p)
