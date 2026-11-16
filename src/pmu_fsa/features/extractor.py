"""Feature extraction from post-disturbance PMU observation windows."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from pmu_fsa.features.window_slicer import CausalWindow, extract_causal_window


# 1-based bus numbers per coherent area (South, Central, North)
CANONICAL_AREA_PARTITION: Dict[int, List[int]] = {
    1: [1, 2, 3, 4, 18, 25, 30, 31, 32],
    2: [5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 33, 34, 35, 36],
    3: [9, 17, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 37, 38, 39],
}

# 0-based column indices corresponding to CANONICAL_AREA_PARTITION
CANONICAL_AREA_BUS_INDICES: Tuple[Tuple[int, ...], ...] = (
    (0, 1, 2, 3, 17, 24, 29, 30, 31),
    (4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35),
    (8, 16, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 36, 37, 38),
)
CANONICAL_AREA_1_IDX: np.ndarray = np.array([0, 1, 2, 3, 17, 24, 29, 30, 31], dtype=np.intp)
CANONICAL_AREA_2_IDX: np.ndarray = np.array([4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35], dtype=np.intp)
CANONICAL_AREA_3_IDX: np.ndarray = np.array([8, 16, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 36, 37, 38], dtype=np.intp)
CANONICAL_AREA_NP_INDICES: Tuple[np.ndarray, np.ndarray, np.ndarray] = (
    CANONICAL_AREA_1_IDX,
    CANONICAL_AREA_2_IDX,
    CANONICAL_AREA_3_IDX,
)

CORE_FEATURE_NAMES: List[str] = [
    "init_rocof",
    "pk_rocof",
    "max_abs_rocof",
    "mean_rocof",
    "inst_drop",
    "max_drop",
    "integ_dev",
    "integ_abs",
    "spread",
    "spatial_var",
    "max_ang",
    "tie_ang",
    "init_acc",
    "mean_acc",
]

DESCRIPTIVE_FEATURE_NAMES: List[str] = [
    "rocof_init_coi",
    "rocof_peak_neg_coi",
    "rocof_max_abs_coi",
    "rocof_mean_coi",
    "delta_f_tobs_coi",
    "delta_f_max_drop_coi",
    "freq_integral_dev_coi",
    "freq_integral_abs_coi",
    "area_spread_max",
    "bus_freq_var_mean",
    "angle_div_max",
    "tie_line_angle_diff",
    "coi_acc_init",
    "coi_acc_mean",
]


def wrap_to_pi(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Wrap angle(s) in radians to [-pi, pi)."""
    if isinstance(angle, (int, float)):
        return float(((angle + math.pi) % (2.0 * math.pi)) - math.pi)
    arr = np.asarray(angle)
    return ((arr + np.pi) % (2.0 * np.pi)) - np.pi


def _get_physics_default(name: str, f_nominal: float = 50.0) -> float:
    """Fallback channel value when all samples in a column are NaN/Inf."""
    lname = name.lower()
    if "freq" in lname:
        return float(f_nominal)
    elif "angle" in lname or "rocof" in lname:
        return 0.0
    elif "volt" in lname or "mag" in lname:
        return 1.0
    return float(f_nominal) if "f" in lname else 0.0


def _validate_and_clean_nans(
    arr: np.ndarray,
    name: str,
    policy: str = "raise",
    f_nominal: float = 50.0,
) -> np.ndarray:
    """Raise on NaN/Inf or forward-fill valid samples (backfilling leading NaNs)."""
    if arr.size > 0 and not np.isfinite(arr.sum()):
        has_nan = bool(np.isnan(arr).any())
        has_inf = bool(np.isinf(arr).any())

        if has_nan or has_inf:
            if policy == "raise":
                raise ValueError(f"Array '{name}' contains NaN or Inf values")
            elif policy in ("ffill", "locf", "impute"):
                cleaned = np.copy(arr)
                default_val = _get_physics_default(name, f_nominal=f_nominal)
                if cleaned.ndim == 1:
                    mask = np.isnan(cleaned) | np.isinf(cleaned)
                    valid_mask = ~mask
                    if not valid_mask.any():
                        cleaned[:] = default_val
                    else:
                        idx = np.where(valid_mask, np.arange(len(cleaned)), 0)
                        if mask[0]:
                            first_valid = int(np.argmax(valid_mask))
                            idx[:first_valid] = first_valid
                        np.maximum.accumulate(idx, out=idx)
                        cleaned = cleaned[idx]
                else:
                    for col in range(cleaned.shape[1]):
                        c = cleaned[:, col]
                        mask = np.isnan(c) | np.isinf(c)
                        if mask.any():
                            valid_mask = ~mask
                            if not valid_mask.any():
                                cleaned[:, col] = default_val
                            else:
                                idx = np.where(valid_mask, np.arange(len(c)), 0)
                                if mask[0]:
                                    first_valid = int(np.argmax(valid_mask))
                                    idx[:first_valid] = first_valid
                                np.maximum.accumulate(idx, out=idx)
                                cleaned[:, col] = c[idx]
                return cleaned
    return arr


def calc_initial_rocof(
    windowed_traj: Dict[str, Any],
    delta_t_init: float = 0.1,
    bus_idx: Optional[int] = None,
) -> float:
    """Estimate initial ROCOF over [t_0, t_0 + delta_t_init] via OLS linear slope."""
    ts = np.asarray(windowed_traj["timestamps"], dtype=float)
    if bus_idx is not None:
        y = np.asarray(windowed_traj["bus_frequencies"][:, bus_idx], dtype=float)
    else:
        y = np.asarray(windowed_traj["coi_frequency"], dtype=float)

    n = len(ts)
    if n < 2:
        return 0.0

    t_0 = ts[0]
    idx_end = int(np.searchsorted(ts, t_0 + delta_t_init + 1e-9, side="right"))
    idx_end = max(idx_end, 2)
    idx_end = min(idx_end, n)

    ts_init = ts[:idx_end] - t_0
    y_init = y[:idx_end]
    x_mean = float(np.mean(ts_init))
    y_mean = float(np.mean(y_init))

    # OLS slope: sum((t - t_bar)*(f - f_bar)) / sum((t - t_bar)^2)
    dx = ts_init - x_mean
    dy = y_init - y_mean
    ss_xx = float(np.dot(dx, dx))
    ss_xy = float(np.dot(dx, dy))

    if ss_xx <= 1e-12:
        return 0.0

    return float(ss_xy / ss_xx)


def calc_rocof_metrics(
    windowed_traj: Dict[str, Any],
    bus_idx: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Return (peak_negative, max_abs, mean) finite-difference ROCOF over the window in Hz/s."""
    ts = np.asarray(windowed_traj["timestamps"], dtype=float)
    if bus_idx is not None:
        y = np.asarray(windowed_traj["bus_frequencies"][:, bus_idx], dtype=float)
    else:
        y = np.asarray(windowed_traj["coi_frequency"], dtype=float)

    n = len(y)
    if n < 2:
        return 0.0, 0.0, 0.0

    dt = ts[1:] - ts[:-1]
    dt_safe = np.maximum(dt, 1e-6)
    diffs = (y[1:] - y[:-1]) / dt_safe

    peak_negative = float(diffs.min())
    max_abs = float(np.abs(diffs).max())
    total_dt = max(float(ts[-1] - ts[0]), 1e-6)
    mean_rocof = float((y[-1] - y[0]) / total_dt)

    return peak_negative, max_abs, mean_rocof


def calc_max_rocof(
    windowed_traj: Dict[str, Any],
    bus_idx: Optional[int] = None,
) -> Tuple[float, float]:
    pk_neg, max_abs, _ = calc_rocof_metrics(windowed_traj, bus_idx=bus_idx)
    return max_abs, pk_neg


def calc_mean_rocof(
    windowed_traj: Dict[str, Any],
    bus_idx: Optional[int] = None,
) -> float:
    _, _, mean_r = calc_rocof_metrics(windowed_traj, bus_idx=bus_idx)
    return mean_r


def calc_delta_f(
    windowed_traj: Dict[str, Any],
    f_nominal: float = 50.0,
    bus_idx: Optional[int] = None,
) -> Tuple[float, float]:
    """Return (end-minus-start frequency change, max drop relative to nominal) in Hz."""
    if bus_idx is not None:
        y = np.asarray(windowed_traj["bus_frequencies"][:, bus_idx], dtype=float)
    else:
        y = np.asarray(windowed_traj["coi_frequency"], dtype=float)

    if len(y) == 0:
        return 0.0, 0.0

    eff_f0 = f_nominal
    if f_nominal == 50.0:
        meta_f0 = windowed_traj.get("metadata", {}).get("f_nominal") if isinstance(windowed_traj, dict) else None
        if meta_f0 is not None:
            eff_f0 = float(meta_f0)
        elif np.isfinite(y[0]) and float(y[0]) > 55.0:
            eff_f0 = 60.0

    inst_drop = float(y[-1] - y[0])
    min_f = float(y.min())
    max_drop = float(min_f - eff_f0)

    return inst_drop, max_drop


def calc_freq_integral(
    windowed_traj: Dict[str, Any],
    f_nominal: float = 50.0,
    bus_idx: Optional[int] = None,
) -> Tuple[float, float]:
    """Return trapezoidal integrals of signed and absolute frequency deviation (Hz*s)."""
    ts = np.asarray(windowed_traj["timestamps"], dtype=float)
    if bus_idx is not None:
        y = np.asarray(windowed_traj["bus_frequencies"][:, bus_idx], dtype=float)
    else:
        y = np.asarray(windowed_traj["coi_frequency"], dtype=float)

    n = len(y)
    if n < 2:
        return 0.0, 0.0

    eff_f0 = f_nominal
    if f_nominal == 50.0:
        meta_f0 = windowed_traj.get("metadata", {}).get("f_nominal") if isinstance(windowed_traj, dict) else None
        if meta_f0 is not None:
            eff_f0 = float(meta_f0)
        elif np.isfinite(y[0]) and float(y[0]) > 55.0:
            eff_f0 = 60.0

    dt = ts[1:] - ts[:-1]
    dev = y - eff_f0

    integral_dev = 0.5 * float(np.dot(dev[:-1] + dev[1:], dt))
    integral_abs = 0.5 * float(np.dot(np.abs(dev[:-1]) + np.abs(dev[1:]), dt))

    return integral_dev, integral_abs


def calc_interarea_dispersion(
    windowed_traj: Dict[str, Any],
    area_partition: Optional[Dict[int, List[int]]] = None,
) -> Tuple[float, float]:
    """Return (max pairwise area mean spread in Hz, cross-bus frequency variance in Hz^2) at t_end."""
    bf = np.asarray(windowed_traj["bus_frequencies"], dtype=float)
    if len(bf) == 0:
        return 0.0, 0.0

    f_end = bf[-1]
    n_buses = len(f_end)

    if (area_partition is None or area_partition == CANONICAL_AREA_PARTITION) and n_buses >= 39:
        m0 = float(np.mean(f_end[CANONICAL_AREA_1_IDX]))
        m1 = float(np.mean(f_end[CANONICAL_AREA_2_IDX]))
        m2 = float(np.mean(f_end[CANONICAL_AREA_3_IDX]))
        spread = max(abs(m0 - m1), abs(m0 - m2), abs(m1 - m2))
    else:
        partition = area_partition or CANONICAL_AREA_PARTITION
        area_means: List[float] = []
        for area_id in sorted(partition.keys()):
            buses = partition[area_id]
            cols = [b - 1 for b in buses if 0 <= b - 1 < n_buses]
            if cols:
                area_means.append(float(np.mean(f_end[cols])))

        if len(area_means) >= 3:
            spread = max(
                abs(area_means[0] - area_means[1]),
                abs(area_means[0] - area_means[2]),
                abs(area_means[1] - area_means[2]),
            )
        elif len(area_means) == 2:
            spread = abs(area_means[0] - area_means[1])
        else:
            spread = 0.0

    spatial_var = float(np.var(f_end))

    return float(spread), float(spatial_var)


def calc_angle_divergence(
    windowed_traj: Dict[str, Any],
    tie_buses: Tuple[int, int] = (4, 5),
    angle_unit: str = "auto",
) -> Tuple[float, float]:
    """Return max bus angle spread and tie-line angle difference at t_end, accounting for wrap."""
    va = np.asarray(windowed_traj["voltage_angles"], dtype=float)
    if len(va) == 0:
        return 0.0, 0.0

    ang_end = va[-1]
    min_a = float(ang_end.min())
    max_a = float(ang_end.max())
    span = max_a - min_a

    is_deg = (angle_unit == "deg") or (
        angle_unit == "auto" and max(abs(min_a), abs(max_a)) > 2.0 * math.pi
    )
    half = 180.0 if is_deg else math.pi
    modulus = 360.0 if is_deg else 2.0 * math.pi

    if span <= half:
        max_div = span
    else:
        # Re-center around bus 1 before wrapping across +/-pi (or +/-180 deg)
        ref_diff = (ang_end - ang_end[0] + half) % modulus - half
        max_div = float(ref_diff.max() - ref_diff.min())

    # 1-based bus IDs -> 0-based column indices
    b1_col = tie_buses[0] - 1
    b2_col = tie_buses[1] - 1
    if 0 <= b1_col < len(ang_end) and 0 <= b2_col < len(ang_end):
        raw_tie = abs(ang_end[b1_col] - ang_end[b2_col])
        tie_ang = float(abs((raw_tie + half) % modulus - half))
    else:
        tie_ang = 0.0

    return float(max_div), float(tie_ang)


def calc_coi_acceleration(
    windowed_traj: Dict[str, Any],
    delta_t_init: float = 0.1,
) -> Tuple[float, float]:
    """Return initial (over delta_t_init) and full-window mean COI frequency slope in Hz/s."""
    ts = np.asarray(windowed_traj["timestamps"], dtype=float)
    coi = np.asarray(windowed_traj["coi_frequency"], dtype=float)

    n = len(coi)
    if n < 2:
        return 0.0, 0.0

    t_0 = ts[0]
    idx_end = int(np.searchsorted(ts, t_0 + delta_t_init + 1e-9, side="right"))
    idx_end = max(idx_end, 2)
    idx_end = min(idx_end, len(ts))

    dt_init = max(float(ts[idx_end - 1] - ts[0]), 1e-6)
    initial_accel = float((coi[idx_end - 1] - coi[0]) / dt_init)

    dt_total = max(float(ts[-1] - ts[0]), 1e-6)
    mean_accel = float((coi[-1] - coi[0]) / dt_total)

    return initial_accel, mean_accel


def extract_feature_vector(
    windowed_traj: Dict[str, Any],
    f_nominal: float = 50.0,
    area_partition: Optional[Dict[int, List[int]]] = None,
) -> List[float]:
    """Return the 14 features ordered according to CORE_FEATURE_NAMES."""
    feats = extract_features(
        windowed_traj,
        f_nominal=f_nominal,
        area_partition=area_partition,
    )
    return [feats[k] for k in CORE_FEATURE_NAMES]


def extract_features(
    scenario_or_window: Union[Dict[str, Any], CausalWindow, str, Path],
    t_obs: Optional[float] = None,
    f_nominal: float = 50.0,
    nan_policy: str = "raise",
    area_partition: Optional[Dict[int, List[int]]] = None,
    tie_buses: Tuple[int, int] = (4, 5),
    **kwargs: Any,
) -> Dict[str, float]:
    """Extract all 14 features (keyed by both descriptive and short names) from a window or scenario."""
    if t_obs is not None:
        w = extract_causal_window(scenario_or_window, t_obs=t_obs)
    elif isinstance(scenario_or_window, CausalWindow):
        w = scenario_or_window
    elif isinstance(scenario_or_window, dict) and "timestamps" in scenario_or_window:
        if "metadata" in scenario_or_window and "t_obs" in scenario_or_window["metadata"]:
            w = scenario_or_window
        elif "metadata" in scenario_or_window and "t_disturb" in scenario_or_window["metadata"]:
            default_tobs = float(kwargs.get("default_t_obs", 1.0))
            w = extract_causal_window(scenario_or_window, t_obs=default_tobs)
        else:
            w = scenario_or_window
    else:
        w = extract_causal_window(scenario_or_window, t_obs=1.0)

    ts = np.asarray(w["timestamps"], dtype=float)
    bf = np.asarray(w["bus_frequencies"], dtype=float)
    coi = np.asarray(w["coi_frequency"], dtype=float)
    va = np.asarray(w["voltage_angles"], dtype=float)

    # Infer 60 Hz nominal if metadata omits f_nominal and initial frequency > 55 Hz
    eff_f_nominal = float(f_nominal)
    if eff_f_nominal == 50.0:
        w_meta = w.get("metadata", {}) if isinstance(w, dict) else {}
        if "f_nominal" in w_meta:
            eff_f_nominal = float(w_meta["f_nominal"])
        else:
            finite_coi = coi[np.isfinite(coi)]
            if len(finite_coi) > 0 and float(finite_coi[0]) > 55.0:
                eff_f_nominal = 60.0

    bf = _validate_and_clean_nans(bf, "bus_frequencies", policy=nan_policy, f_nominal=eff_f_nominal)
    coi = _validate_and_clean_nans(coi, "coi_frequency", policy=nan_policy, f_nominal=eff_f_nominal)
    va = _validate_and_clean_nans(va, "voltage_angles", policy=nan_policy, f_nominal=eff_f_nominal)

    n = len(ts)
    if n == 0:
        zero_dict: Dict[str, float] = {}
        for k in DESCRIPTIVE_FEATURE_NAMES + CORE_FEATURE_NAMES:
            zero_dict[k] = 0.0
        return zero_dict

    delta_t_init = float(kwargs.get("delta_t_init", 0.1))

    if n < 2:
        init_rocof = 0.0
        pk_rocof = 0.0
        max_abs_r = 0.0
        mean_r = 0.0
        inst_drop = 0.0
        max_drop = float(float(coi[0]) - eff_f_nominal)
        integ_dev = 0.0
        integ_abs = 0.0
        init_acc = 0.0
        mean_acc = 0.0
    else:
        t_0 = ts[0]
        total_dt = max(float(ts[-1] - t_0), 1e-6)

        idx_end = int(np.searchsorted(ts, t_0 + delta_t_init + 1e-9, side="right"))
        idx_end = max(idx_end, 2)
        idx_end = min(idx_end, n)

        # OLS linear slope over [t_0, t_0 + delta_t_init]
        ts_init = ts[:idx_end] - t_0
        coi_init = coi[:idx_end]
        x_mean = float(np.mean(ts_init))
        y_mean = float(np.mean(coi_init))

        dx = ts_init - x_mean
        dy = coi_init - y_mean
        ss_xx = float(np.dot(dx, dx))
        ss_xy = float(np.dot(dx, dy))
        init_rocof = float(ss_xy / ss_xx) if ss_xx > 1e-12 else 0.0

        dt = ts[1:] - ts[:-1]
        dt_safe = np.maximum(dt, 1e-6)
        diffs = (coi[1:] - coi[:-1]) / dt_safe
        pk_rocof = float(diffs.min())
        max_abs_r = float(np.abs(diffs).max())
        mean_r = float((coi[-1] - coi[0]) / total_dt)

        inst_drop = float(coi[-1] - coi[0])
        max_drop = float(float(coi.min()) - eff_f_nominal)

        dev = coi - eff_f_nominal
        integ_dev = 0.5 * float(np.dot(dev[:-1] + dev[1:], dt))
        integ_abs = 0.5 * float(np.dot(np.abs(dev[:-1]) + np.abs(dev[1:]), dt))

        dt_init = max(float(ts[idx_end - 1] - t_0), 1e-6)
        init_acc = float((coi[idx_end - 1] - coi[0]) / dt_init)
        mean_acc = mean_r

    if len(bf) > 0:
        f_end = bf[-1]
        n_buses = len(f_end)
        if (area_partition is None or area_partition == CANONICAL_AREA_PARTITION) and n_buses >= 39:
            m0 = float(np.mean(f_end[CANONICAL_AREA_1_IDX]))
            m1 = float(np.mean(f_end[CANONICAL_AREA_2_IDX]))
            m2 = float(np.mean(f_end[CANONICAL_AREA_3_IDX]))
            spread = float(max(abs(m0 - m1), abs(m0 - m2), abs(m1 - m2)))
        else:
            partition = area_partition or CANONICAL_AREA_PARTITION
            area_means: List[float] = []
            for area_id in sorted(partition.keys()):
                buses = partition[area_id]
                cols = [b - 1 for b in buses if 0 <= b - 1 < n_buses]
                if cols:
                    area_means.append(float(np.mean(f_end[cols])))
            if len(area_means) >= 3:
                spread = float(max(
                    abs(area_means[0] - area_means[1]),
                    abs(area_means[0] - area_means[2]),
                    abs(area_means[1] - area_means[2]),
                ))
            elif len(area_means) == 2:
                spread = float(abs(area_means[0] - area_means[1]))
            else:
                spread = 0.0

        s_var = float(np.var(f_end))
    else:
        spread = 0.0
        s_var = 0.0

    if len(va) > 0:
        ang_end = va[-1]
        min_a = float(ang_end.min())
        max_a = float(ang_end.max())
        span = max_a - min_a
        angle_unit = kwargs.get("angle_unit", "auto")
        is_deg = (angle_unit == "deg") or (
            angle_unit == "auto" and max(abs(min_a), abs(max_a)) > 2.0 * math.pi
        )
        half = 180.0 if is_deg else math.pi
        modulus = 360.0 if is_deg else 2.0 * math.pi

        if span <= half:
            max_ang = span
        else:
            ref_diff = (ang_end - ang_end[0] + half) % modulus - half
            max_ang = float(ref_diff.max() - ref_diff.min())

        b1_col = tie_buses[0] - 1
        b2_col = tie_buses[1] - 1
        if 0 <= b1_col < len(ang_end) and 0 <= b2_col < len(ang_end):
            raw_tie = abs(ang_end[b1_col] - ang_end[b2_col])
            tie_ang = float(abs((raw_tie + half) % modulus - half))
        else:
            tie_ang = 0.0
    else:
        max_ang = 0.0
        tie_ang = 0.0

    return {
        "rocof_init_coi": init_rocof,
        "rocof_peak_neg_coi": pk_rocof,
        "rocof_max_abs_coi": max_abs_r,
        "rocof_mean_coi": mean_r,
        "delta_f_tobs_coi": inst_drop,
        "delta_f_max_drop_coi": max_drop,
        "freq_integral_dev_coi": integ_dev,
        "freq_integral_abs_coi": integ_abs,
        "area_spread_max": spread,
        "bus_freq_var_mean": s_var,
        "angle_div_max": max_ang,
        "tie_line_angle_diff": tie_ang,
        "coi_acc_init": init_acc,
        "coi_acc_mean": mean_acc,
        "init_rocof": init_rocof,
        "pk_rocof": pk_rocof,
        "max_abs_rocof": max_abs_r,
        "mean_rocof": mean_r,
        "inst_drop": inst_drop,
        "max_drop": max_drop,
        "integ_dev": integ_dev,
        "integ_abs": integ_abs,
        "spread": spread,
        "spatial_var": s_var,
        "max_ang": max_ang,
        "tie_ang": tie_ang,
        "init_acc": init_acc,
        "mean_acc": mean_acc,
    }


def extract_features_from_scenario(
    scenario: Union[Dict[str, Any], str, Path],
    t_obs: float = 1.0,
    **kwargs: Any,
) -> Dict[str, float]:
    return extract_features(scenario, t_obs=t_obs, **kwargs)


class FeatureExtractor:
    """Stateful wrapper around extract_features for single scenarios and dataset batches."""

    def __init__(
        self,
        t_obs: float = 1.0,
        f_nominal: float = 50.0,
        delta_t_init: float = 0.1,
        area_partition: Optional[Dict[int, List[int]]] = None,
        tie_buses: Tuple[int, int] = (4, 5),
        handle_nans: bool = False,
        nan_policy: str = "raise",
    ) -> None:
        self.t_obs = float(t_obs)
        self.f_nominal = float(f_nominal)
        self.delta_t_init = float(delta_t_init)
        self.area_partition = area_partition or CANONICAL_AREA_PARTITION
        self.tie_buses = tie_buses
        self.nan_policy = "ffill" if handle_nans else nan_policy

    def extract_features_dict(
        self,
        scenario: Union[Dict[str, Any], CausalWindow, str, Path],
        t_obs: Optional[float] = None,
    ) -> Dict[str, float]:
        """Extract feature dictionary using configured window and nominal parameters."""
        obs = self.t_obs if t_obs is None else float(t_obs)
        return extract_features(
            scenario,
            t_obs=obs,
            f_nominal=self.f_nominal,
            delta_t_init=self.delta_t_init,
            nan_policy=self.nan_policy,
            area_partition=self.area_partition,
            tie_buses=self.tie_buses,
        )

    def transform_single(
        self,
        scenario: Union[Dict[str, Any], CausalWindow, str, Path],
        t_obs: Optional[float] = None,
    ) -> Dict[str, float]:
        return self.extract_features_dict(scenario, t_obs=t_obs)

    def extract_scenario(
        self,
        scenario: Union[Dict[str, Any], CausalWindow, str, Path],
        t_obs: Optional[float] = None,
    ) -> Any:
        """Extract single-row feature DataFrame (or dict if pandas is unavailable)."""
        feats = self.extract_features_dict(scenario, t_obs=t_obs)
        if HAS_PANDAS:
            return pd.DataFrame([feats])
        return feats

    def transform_batch(
        self,
        scenarios: List[Union[Dict[str, Any], str, Path]],
        t_obs: Optional[float] = None,
        as_dataframe: bool = True,
    ) -> Tuple[Any, np.ndarray, np.ndarray, Any]:
        """Extract (X, y_nadir, y_secure, metadata) across a list of scenarios."""
        obs = self.t_obs if t_obs is None else float(t_obs)
        feature_rows: List[Dict[str, float]] = []
        y_nadir_list: List[float] = []
        y_secure_list: List[int] = []
        meta_list: List[Dict[str, Any]] = []

        for scen in scenarios:
            feats = self.extract_features_dict(scen, t_obs=obs)
            feature_rows.append(feats)

            raw_meta: Dict[str, Any] = {}
            if isinstance(scen, dict) and "metadata" in scen:
                raw_meta = dict(scen["metadata"])
            elif isinstance(scen, CausalWindow):
                raw_meta = dict(scen.metadata)

            nadir = float(raw_meta.get("ground_truth_nadir", self.f_nominal - 0.5))
            default_thresh = float(raw_meta.get("f_threshold_hz", 59.5 if nadir > 54.0 else (self.f_nominal - 0.5)))
            is_sec = int(raw_meta.get("is_secure", 1 if nadir >= default_thresh else 0))

            y_nadir_list.append(nadir)
            y_secure_list.append(is_sec)
            meta_list.append({
                "scenario_id": raw_meta.get("scenario_id", "unknown"),
                "partition": raw_meta.get("partition", "train"),
                "disturbed_gen": raw_meta.get("disturbed_gen", 0),
                "active_loss_mw": raw_meta.get("active_loss_mw", 0.0),
                "renewable_penetration": raw_meta.get("renewable_penetration", 0.0),
                "t_disturb": raw_meta.get("t_disturb", 1.0),
                "ground_truth_nadir": nadir,
                "is_secure": is_sec,
            })

        y_nadir = np.array(y_nadir_list, dtype=float)
        y_secure = np.array(y_secure_list, dtype=int)

        if as_dataframe and HAS_PANDAS:
            X = pd.DataFrame(feature_rows)
            meta_df = pd.DataFrame(meta_list)
            return X, y_nadir, y_secure, meta_df

        X_arr = np.array([[row[k] for k in DESCRIPTIVE_FEATURE_NAMES] for row in feature_rows], dtype=float)
        return X_arr, y_nadir, y_secure, meta_list


def extract_feature_dataset(
    scenarios: List[Union[Dict[str, Any], str, Path]],
    t_obs: float = 1.0,
    as_dataframe: bool = True,
) -> Tuple[Any, np.ndarray, np.ndarray, List[Any]]:
    """Extract (X, y_nadir, y_secure, scenario_ids) across a list of scenarios."""
    extractor = FeatureExtractor(t_obs=t_obs)
    X, y_nadir, y_secure, meta = extractor.transform_batch(scenarios, as_dataframe=as_dataframe)
    if isinstance(meta, pd.DataFrame) and "scenario_id" in meta.columns:
        ids = meta["scenario_id"].tolist()
    elif isinstance(meta, list):
        ids = [m.get("scenario_id", f"scen_{i}") for i, m in enumerate(meta)]
    else:
        ids = [f"scen_{i}" for i in range(len(scenarios))]
    return X, y_nadir, y_secure, ids
