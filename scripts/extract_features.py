#!/usr/bin/env python3
"""Extract canonical 14 causal PMU features and full metadata from scenario .npz files."""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from pmu_fsa.features import extractor as extractor_mod
from pmu_fsa.features.extractor import DESCRIPTIVE_FEATURE_NAMES, FeatureExtractor
from pmu_fsa.simulation.dataset_validator import (
    validate_dataset,
    write_dataset_quality_report,
)
from pmu_fsa.simulation.scenario_generator import (
    GEN_CAPACITIES_MW,
    _estimate_pmu_sys_frequency,
    assign_scenario_partition,
    classify_ood_regime,
    load_scenario_from_npz,
)

logger = logging.getLogger("extract_features")

METADATA_COLUMNS: List[str] = [
    "scenario_id",
    "partition",
    "ood_type",
    "disturbed_gen",
    "loss_fraction",
    "active_loss_mw",
    "renewable_penetration",
    "p_ren_realized",
    "load_scale",
    "ground_truth_nadir",
    "ground_truth_t_nadir",
    "lead_time_s",
    "is_prior_to_nadir",
    "is_secure",
]


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Causal PMU Feature Extractor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Directory with observable PMU .npz scenarios")
    parser.add_argument("--t-obs", type=float, default=1.0, help="Primary observation window length in seconds")
    parser.add_argument("--output-file", type=str, default="data/processed/features.npz", help="Primary output .npz path")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save extracted feature files")
    parser.add_argument("--all-windows", action="store_true", help="Extract across windows [0.5, 1.0, 1.5, 2.0]s")
    parser.add_argument("--f-nominal", type=float, default=50.0, help="Nominal power system frequency (Hz)")
    parser.add_argument("--threshold", type=float, default=49.50, help="Frequency security threshold (Hz)")
    parser.add_argument(
        "--strict-nadir",
        action="store_true",
        help="Enforce strict pre-nadir causality (t0 + T_obs < t_nadir) across all extracted files",
    )
    parser.add_argument(
        "--no-strict-nadir",
        action="store_true",
        help="Disable strict pre-nadir filtering on primary extraction",
    )
    parser.add_argument(
        "--observed-buses",
        type=int,
        nargs="+",
        default=None,
        help="Optional 1-based bus IDs with PMU coverage (masks unobserved buses to NaN)",
    )
    parser.add_argument("--save-csv", action="store_true", help="Also export tabular .csv alongside .npz")
    parser.add_argument("--validate", action="store_true", help="Run dataset validator over raw + processed directories")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")
    return parser.parse_args(args)


def _apply_partial_pmu_mask(
    scenarios: List[Dict[str, Any]],
    observed_buses: Sequence[int],
) -> List[Dict[str, Any]]:
    """Mask unobserved bus columns to NaN and recompute wide-area PMU system frequency."""
    obs_list = sorted({int(b) for b in observed_buses})
    masked_scenarios: List[Dict[str, Any]] = []

    for scen in scenarios:
        s_copy = copy.copy(scen)
        meta = dict(scen.get("metadata", {}))
        n_buses = scen["bus_frequencies"].shape[1]
        obs_zero = {b - 1 for b in obs_list if 1 <= b <= n_buses}

        for ch in ("bus_frequencies", "bus_rocof", "voltage_magnitudes", "voltage_angles"):
            if ch in scen:
                arr = np.array(scen[ch], dtype=float, copy=True)
                for col in range(arr.shape[1]):
                    if col not in obs_zero:
                        arr[:, col] = np.nan
                s_copy[ch] = arr

        pmu_sys = _estimate_pmu_sys_frequency(s_copy["bus_frequencies"])
        s_copy["pmu_sys_frequency"] = pmu_sys
        s_copy["coi_frequency"] = pmu_sys.copy()
        meta["observed_buses"] = obs_list
        s_copy["metadata"] = meta
        masked_scenarios.append(s_copy)

    return masked_scenarios


def _build_extractor(
    f_nominal: float,
    has_partial_pmu: bool,
    observed_buses: Optional[Sequence[int]] = None,
) -> FeatureExtractor:
    init_params = inspect.signature(FeatureExtractor.__init__).parameters
    kwargs: Dict[str, Any] = {"f_nominal": float(f_nominal)}
    if has_partial_pmu:
        if "nan_policy" in init_params:
            kwargs["nan_policy"] = "ignore_unobserved"
        if "observed_buses" in init_params and observed_buses is not None:
            kwargs["observed_buses"] = list(observed_buses)
        if "handle_nans" in init_params and "nan_policy" not in init_params:
            kwargs["handle_nans"] = True
    try:
        return FeatureExtractor(**kwargs)
    except ValueError:
        return FeatureExtractor(f_nominal=float(f_nominal), handle_nans=has_partial_pmu)


def _canonical_feature_columns(X_df: pd.DataFrame) -> List[str]:
    target_names = list(getattr(extractor_mod, "CANONICAL_FEATURE_NAMES", DESCRIPTIVE_FEATURE_NAMES))
    present = [c for c in target_names if c in X_df.columns]
    if len(present) == 14:
        return present
    return list(X_df.columns)[:14]


def extract_for_window(
    extractor: FeatureExtractor,
    scenarios: List[Dict[str, Any]],
    t_obs: float,
    output_path: Path,
    threshold: float = 49.50,
    strict_nadir: bool = False,
    observed_buses: Optional[Sequence[int]] = None,
    save_csv: bool = False,
) -> Dict[str, Any]:
    """Extract 14 canonical features + 14 metadata columns for window t_obs and save .npz/.csv."""
    retained_scenarios: List[Dict[str, Any]] = []
    excluded_ids: List[str] = []
    excluded_reasons: List[str] = []

    for idx, scen in enumerate(scenarios):
        raw_meta = scen.get("metadata", {})
        scen_id = str(raw_meta.get("scenario_id", f"scen_{idx:03d}"))
        t0 = float(raw_meta.get("t_disturb", 1.0))
        t_nadir_raw = raw_meta.get("ground_truth_t_nadir")
        t_end = t0 + float(t_obs)

        if strict_nadir and t_nadir_raw is not None and math.isfinite(float(t_nadir_raw)):
            t_nadir = float(t_nadir_raw)
            if t_end >= t_nadir - 1e-6:
                reason = f"pre_nadir_causality_violation: t_end={t_end:.4f}s >= t_nadir={t_nadir:.4f}s"
                excluded_ids.append(scen_id)
                excluded_reasons.append(reason)
                continue

        retained_scenarios.append(scen)

    if excluded_ids:
        logger.warning(
            "Window T_obs=%.2fs (strict_nadir=True): excluded %d/%d non-causal scenarios: %s",
            t_obs,
            len(excluded_ids),
            len(scenarios),
            excluded_ids[:10],
        )

    feat_names = list(getattr(extractor_mod, "CANONICAL_FEATURE_NAMES", DESCRIPTIVE_FEATURE_NAMES))[:14]

    if retained_scenarios:
        tb_params = inspect.signature(extractor.transform_batch).parameters
        tb_kwargs: Dict[str, Any] = {"t_obs": t_obs, "as_dataframe": True}
        if "observed_buses" in tb_params and observed_buses is not None:
            tb_kwargs["observed_buses"] = list(observed_buses)
        X_raw, y_nadir_raw, _, _ = extractor.transform_batch(retained_scenarios, **tb_kwargs)
        if isinstance(X_raw, pd.DataFrame):
            feat_names = _canonical_feature_columns(X_raw)
            X_df = X_raw[feat_names].copy()
        else:
            X_arr_tmp = np.asarray(X_raw, dtype=float)[:, :14]
            X_df = pd.DataFrame(X_arr_tmp, columns=feat_names)
    else:
        X_df = pd.DataFrame(columns=feat_names)
        y_nadir_raw = np.array([], dtype=float)

    meta_rows: List[Dict[str, Any]] = []
    y_nadir_list: List[float] = []
    y_secure_list: List[int] = []

    for idx, scen in enumerate(retained_scenarios):
        raw_meta = scen.get("metadata", {})
        scen_id = str(raw_meta.get("scenario_id", f"scen_{idx:03d}"))
        p_ren = float(raw_meta.get("p_ren_requested", raw_meta.get("renewable_penetration", 0.0)))
        p_ren_real = float(raw_meta.get("p_ren_realized", p_ren))
        ld_scale = float(raw_meta.get("load_scale", 1.0))
        ood_type = str(raw_meta.get("ood_type") or classify_ood_regime(p_ren, ld_scale))
        partition = str(raw_meta.get("partition") or assign_scenario_partition(p_ren, ld_scale, scen_id))
        disturbed_gen = int(raw_meta.get("disturbed_gen", 0))
        act_loss_mw = float(raw_meta.get("active_loss_mw", 0.0))
        nom_cap = GEN_CAPACITIES_MW.get(disturbed_gen, 500.0)
        loss_frac = float(
            raw_meta.get(
                "loss_fraction",
                (act_loss_mw / nom_cap) if (nom_cap > 0.0 and act_loss_mw > 0.0) else 0.0,
            )
        )
        t0 = float(raw_meta.get("t_disturb", 1.0))
        gt_nadir = float(raw_meta.get("ground_truth_nadir", y_nadir_raw[idx]))
        gt_t_nadir = float(raw_meta.get("ground_truth_t_nadir", float("nan")))
        t_end = t0 + float(t_obs)
        if math.isfinite(gt_t_nadir):
            lead_s = float(gt_t_nadir - t_end)
            prior_to_nadir = bool(t_end < gt_t_nadir - 1e-6)
        else:
            lead_s = float("nan")
            prior_to_nadir = True

        eff_thresh = float(raw_meta.get("f_threshold_hz", threshold))
        is_sec = int(gt_nadir >= eff_thresh)

        y_nadir_list.append(gt_nadir)
        y_secure_list.append(is_sec)
        meta_rows.append({
            "scenario_id": scen_id,
            "partition": partition,
            "ood_type": ood_type,
            "disturbed_gen": disturbed_gen,
            "loss_fraction": loss_frac,
            "active_loss_mw": act_loss_mw,
            "renewable_penetration": p_ren,
            "p_ren_realized": p_ren_real,
            "load_scale": ld_scale,
            "ground_truth_nadir": gt_nadir,
            "ground_truth_t_nadir": gt_t_nadir,
            "lead_time_s": lead_s,
            "is_prior_to_nadir": prior_to_nadir,
            "is_secure": is_sec,
        })

    meta_df = pd.DataFrame(meta_rows, columns=METADATA_COLUMNS)
    X_mat = X_df.to_numpy(dtype=float) if len(X_df) > 0 else np.empty((0, len(feat_names)), dtype=float)
    y_nadir = np.asarray(y_nadir_list, dtype=float)
    y_secure = np.asarray(y_secure_list, dtype=int)
    obs_arr = np.asarray(sorted(observed_buses) if observed_buses is not None else list(range(1, 40)), dtype=int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X_mat,
        y_nadir=y_nadir,
        y_secure=y_secure,
        feature_names=np.asarray(feat_names, dtype=str),
        scenario_ids=meta_df["scenario_id"].to_numpy(dtype=str),
        partitions=meta_df["partition"].to_numpy(dtype=str),
        ood_types=meta_df["ood_type"].to_numpy(dtype=str),
        scenario_id=meta_df["scenario_id"].to_numpy(dtype=str),
        partition=meta_df["partition"].to_numpy(dtype=str),
        ood_type=meta_df["ood_type"].to_numpy(dtype=str),
        disturbed_gen=meta_df["disturbed_gen"].to_numpy(dtype=int),
        loss_fraction=meta_df["loss_fraction"].to_numpy(dtype=float),
        active_loss_mw=meta_df["active_loss_mw"].to_numpy(dtype=float),
        renewable_penetration=meta_df["renewable_penetration"].to_numpy(dtype=float),
        p_ren_realized=meta_df["p_ren_realized"].to_numpy(dtype=float),
        load_scale=meta_df["load_scale"].to_numpy(dtype=float),
        ground_truth_nadir=y_nadir,
        ground_truth_t_nadir=meta_df["ground_truth_t_nadir"].to_numpy(dtype=float),
        lead_time_s=meta_df["lead_time_s"].to_numpy(dtype=float),
        is_prior_to_nadir=meta_df["is_prior_to_nadir"].to_numpy(dtype=bool),
        is_secure=y_secure,
        t_obs=float(t_obs),
        strict_nadir=bool(strict_nadir),
        observed_buses=obs_arr,
        excluded_scenario_ids=np.asarray(excluded_ids, dtype=str),
        excluded_reasons=np.asarray(excluded_reasons, dtype=str),
    )
    logger.info(
        "Saved feature dataset (N=%d, D=%d, excluded=%d) to %s",
        X_mat.shape[0],
        X_mat.shape[1],
        len(excluded_ids),
        output_path,
    )

    if save_csv:
        csv_path = output_path.with_suffix(".csv")
        df_combined = pd.concat([X_df.reset_index(drop=True), meta_df.reset_index(drop=True)], axis=1)
        df_combined.to_csv(csv_path, index=False)
        logger.info("Saved tabular CSV (%d rows, %d cols) to %s", len(df_combined), df_combined.shape[1], csv_path)

    return {
        "file": str(output_path),
        "t_obs_s": float(t_obs),
        "strict_nadir": bool(strict_nadir),
        "n_input": len(scenarios),
        "n_retained": int(X_mat.shape[0]),
        "n_features": int(X_mat.shape[1]),
        "n_causal": int(meta_df["is_prior_to_nadir"].sum()) if len(meta_df) else 0,
        "n_excluded": len(excluded_ids),
        "excluded_scenarios": [
            {"scenario_id": sid, "reason": rsn}
            for sid, rsn in zip(excluded_ids, excluded_reasons)
        ],
    }


def main(args=None) -> int:
    parsed = parse_args(args)
    logging.basicConfig(
        level=logging.WARNING if parsed.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    in_dir = Path(parsed.input_dir)
    if not in_dir.exists():
        logger.error("Input directory does not exist: %s", in_dir)
        return 1

    scenario_files = sorted(in_dir.glob("*.npz"))
    if not scenario_files:
        logger.error("No .npz scenario files found in %s", in_dir)
        return 1

    logger.info("Loading %d scenarios from %s...", len(scenario_files), in_dir)
    scenarios = [load_scenario_from_npz(str(fp)) for fp in scenario_files]

    if parsed.observed_buses is not None:
        scenarios = _apply_partial_pmu_mask(scenarios, parsed.observed_buses)

    out_file = Path(parsed.output_file)
    out_dir = Path(parsed.output_dir) if parsed.output_dir is not None else out_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    extractor = _build_extractor(
        f_nominal=parsed.f_nominal,
        has_partial_pmu=(parsed.observed_buses is not None),
        observed_buses=parsed.observed_buses,
    )

    strict_primary = bool(parsed.strict_nadir or (not parsed.no_strict_nadir))
    strict_sweep = bool(parsed.strict_nadir)

    extraction_records: List[Dict[str, Any]] = []
    windows = [0.5, 1.0, 1.5, 2.0] if parsed.all_windows else [float(parsed.t_obs)]

    for w in windows:
        if parsed.all_windows:
            w_out_file = out_dir / f"features_t{w:.1f}s.npz"
            rec = extract_for_window(
                extractor=extractor,
                scenarios=scenarios,
                t_obs=w,
                output_path=w_out_file,
                threshold=parsed.threshold,
                strict_nadir=strict_sweep,
                observed_buses=parsed.observed_buses,
                save_csv=parsed.save_csv,
            )
            extraction_records.append(rec)

            if abs(w - float(parsed.t_obs)) < 1e-6:
                primary_out = out_file if parsed.output_dir is None else (out_dir / out_file.name)
                if primary_out.resolve() != w_out_file.resolve():
                    rec_p = extract_for_window(
                        extractor=extractor,
                        scenarios=scenarios,
                        t_obs=w,
                        output_path=primary_out,
                        threshold=parsed.threshold,
                        strict_nadir=strict_primary,
                        observed_buses=parsed.observed_buses,
                        save_csv=parsed.save_csv,
                    )
                    extraction_records.append(rec_p)
        else:
            w_out_file = out_file if parsed.output_dir is None else (out_dir / out_file.name)
            rec = extract_for_window(
                extractor=extractor,
                scenarios=scenarios,
                t_obs=w,
                output_path=w_out_file,
                threshold=parsed.threshold,
                strict_nadir=strict_primary,
                observed_buses=parsed.observed_buses,
                save_csv=parsed.save_csv,
            )
            extraction_records.append(rec)

    summary_path = out_dir / "extraction_report.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dir": str(in_dir),
                "output_dir": str(out_dir),
                "observed_buses": list(parsed.observed_buses) if parsed.observed_buses is not None else "all_39",
                "files": extraction_records,
            },
            f,
            indent=2,
        )

    if parsed.validate:
        report = validate_dataset(
            raw_pmu_dir=in_dir,
            processed_dir=out_dir,
            f_nominal=parsed.f_nominal,
            threshold=parsed.threshold,
        )
        norm_out = str(out_dir).rstrip("/")
        is_default_proc = norm_out in ("data/processed", str(PROJECT_ROOT / "data" / "processed"))
        man_p = Path("data/manifest.json") if is_default_proc else (out_dir.parent / "manifest.json")
        md_p = Path("results/dataset_quality_report.md") if is_default_proc else (out_dir.parent / "dataset_quality_report.md")
        write_dataset_quality_report(report, md_path=md_p, manifest_path=man_p)

    return 0


if __name__ == "__main__":
    sys.exit(main())
