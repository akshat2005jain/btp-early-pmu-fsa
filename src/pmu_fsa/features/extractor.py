"""Canonical 14-feature PMU-only causal feature extraction pipeline."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from pmu_fsa.features.window_slicer import CausalWindow, extract_causal_window
from pmu_fsa.simulation.pmu_recorder import (
    CANONICAL_13BUS_PMU_PLACEMENT,
    compute_causal_rocof,
    estimate_pmu_system_frequency,
)


# 1-based bus numbers per coherent area (South, Central, North)
CANONICAL_AREA_PARTITION: Dict[int, List[int]] = {
    1: [1, 2, 3, 4, 18, 25, 30, 31, 32],
    2: [5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 33, 34, 35, 36],
    3: [9, 17, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 37, 38, 39],
}

CANONICAL_AREA_BUS_INDICES: Tuple[Tuple[int, ...], ...] = (
    (0, 1, 2, 3, 17, 24, 29, 30, 31),
    (4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35),
    (8, 16, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 36, 37, 38),
)
CANONICAL_AREA_1_IDX: np.ndarray = np.array([0, 1, 2, 3, 17, 24, 29, 30, 31], dtype=np.intp)
CANONICAL_AREA_2_IDX: np.ndarray = np.array(
    [4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35], dtype=np.intp
)
CANONICAL_AREA_3_IDX: np.ndarray = np.array(
    [8, 16, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 36, 37, 38], dtype=np.intp
)
CANONICAL_AREA_NP_INDICES: Tuple[np.ndarray, np.ndarray, np.ndarray] = (
    CANONICAL_AREA_1_IDX,
    CANONICAL_AREA_2_IDX,
    CANONICAL_AREA_3_IDX,
)

PRIMARY_TIE_LINES: Tuple[Tuple[int, int], ...] = (
    (4, 5),
    (1, 39),
    (16, 17),
)
SECONDARY_INTERAREA_TIE_LINES: Tuple[Tuple[int, int], ...] = (
    (4, 14),
    (3, 18),
    (8, 9),
    (9, 39),
    (16, 19),
    (16, 21),
    (16, 24),
    (17, 18),
    (25, 26),
)

# Single canonical 14-feature schema (exported under all three standard constants)
CANONICAL_FEATURE_NAMES: List[str] = [
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
DESCRIPTIVE_FEATURE_NAMES: List[str] = list(CANONICAL_FEATURE_NAMES)
CORE_FEATURE_NAMES: List[str] = list(CANONICAL_FEATURE_NAMES)

FEATURE_SPECIFICATIONS: Dict[str, Dict[str, Any]] = {
    "rocof_init": {
        "name": "rocof_init",
        "meaning": "Initial post-disturbance wide-area PMU ROCOF over [t0, min(t0 + 0.10, t_end)]",
        "formula": "OLS_slope(t, f_sys_pmu) over [t0, min(t0 + 0.10, t_end)]",
        "units": "Hz/s",
        "required_pmu_channels": ["timestamps", "bus_frequencies"],
    },
    "rocof_peak_neg": {
        "name": "rocof_peak_neg",
        "meaning": "Most negative causal 5-point wide-area PMU ROCOF within [t0, t0 + T_obs]",
        "formula": "min_{t in W} rocof_sys_pmu(t)",
        "units": "Hz/s",
        "required_pmu_channels": ["timestamps", "bus_frequencies"],
    },
    "rocof_end": {
        "name": "rocof_end",
        "meaning": "Causal 5-point wide-area PMU ROCOF at window end t0 + T_obs",
        "formula": "rocof_sys_pmu(t0 + T_obs)",
        "units": "Hz/s",
        "required_pmu_channels": ["timestamps", "bus_frequencies"],
    },
    "rocof_mean": {
        "name": "rocof_mean",
        "meaning": "Noise-resilient window-wide OLS frequency slope over [t0, t0 + T_obs]",
        "formula": "OLS_slope(t, f_sys_pmu) over [t0, t0 + T_obs]",
        "units": "Hz/s",
        "required_pmu_channels": ["timestamps", "bus_frequencies"],
    },
    "freq_drop_end": {
        "name": "freq_drop_end",
        "meaning": "Wide-area PMU frequency change from disturbance onset to window end",
        "formula": "f_sys_pmu(t0 + T_obs) - f_sys_pmu(t0)",
        "units": "Hz",
        "required_pmu_channels": ["timestamps", "bus_frequencies"],
    },
    "freq_drop_max": {
        "name": "freq_drop_max",
        "meaning": "Worst wide-area PMU frequency excursion below nominal within [t0, t0 + T_obs]",
        "formula": "min_{t in W} f_sys_pmu(t) - f_nominal",
        "units": "Hz",
        "required_pmu_channels": ["timestamps", "bus_frequencies"],
    },
    "freq_dev_integral": {
        "name": "freq_dev_integral",
        "meaning": "Cumulative trapezoidal integral of wide-area PMU frequency deviation",
        "formula": "int_{t0}^{t0 + T_obs} (f_sys_pmu(t) - f_nominal) dt",
        "units": "Hz*s",
        "required_pmu_channels": ["timestamps", "bus_frequencies"],
    },
    "bus_drop_worst": {
        "name": "bus_drop_worst",
        "meaning": "Worst localized PMU bus frequency excursion relative to nominal across observed buses",
        "formula": "nanmin_{t in W, b in B_obs} (f_bus(t, b) - f_nominal)",
        "units": "Hz",
        "required_pmu_channels": ["timestamps", "bus_frequencies"],
    },
    "area_freq_spread_max": {
        "name": "area_freq_spread_max",
        "meaning": "Peak pairwise inter-area mean frequency separation over [t0, t0 + T_obs]",
        "formula": "max_{t in W} max_{a, b in {1,2,3}} |mean_Area_a(t) - mean_Area_b(t)|",
        "units": "Hz",
        "required_pmu_channels": ["timestamps", "bus_frequencies"],
    },
    "bus_freq_std_mean": {
        "name": "bus_freq_std_mean",
        "meaning": "Window-mean spatial standard deviation of observed PMU bus frequencies",
        "formula": "mean_{t in W} nanstd_{b in B_obs} f_bus(t, b)",
        "units": "Hz",
        "required_pmu_channels": ["timestamps", "bus_frequencies"],
    },
    "angle_divergence_max": {
        "name": "angle_divergence_max",
        "meaning": "Peak wrap-safe pairwise dynamic bus voltage phase-angle span relative to t0",
        "formula": "max_{t in W} max_{a, b in B_obs} |wrap_180((theta_a(t)-theta_a(0)) - (theta_b(t)-theta_b(0)))|",
        "units": "deg",
        "required_pmu_channels": ["timestamps", "voltage_angles"],
    },
    "tie_line_angle_excursion": {
        "name": "tie_line_angle_excursion",
        "meaning": "Peak wrap-safe dynamic phase-angle swing across primary inter-area tie-lines [(4,5), (1,39), (16,17)]",
        "formula": "max_{t in W} max_{(a,b) in Ties} |wrap_180((theta_a(t)-theta_b(t)) - (theta_a(0)-theta_b(0)))|",
        "units": "deg",
        "required_pmu_channels": ["timestamps", "voltage_angles"],
    },
    "freq_curvature_init": {
        "name": "freq_curvature_init",
        "meaning": "Initial frequency second derivative d^2f/dt^2 (governor/inertial deceleration curvature) over [t0, min(t0 + 0.20, t_end)]",
        "formula": "OLS_slope(tau, rocof_sys_pmu) over [t0, min(t0 + 0.20, t_end)]",
        "units": "Hz/s^2",
        "required_pmu_channels": ["timestamps", "bus_frequencies"],
    },
    "freq_curvature_mean": {
        "name": "freq_curvature_mean",
        "meaning": "Full-window mean frequency second derivative d^2f/dt^2 over [t0, t0 + T_obs]",
        "formula": "OLS_slope(tau, rocof_sys_pmu) over [t0, t0 + T_obs]",
        "units": "Hz/s^2",
        "required_pmu_channels": ["timestamps", "bus_frequencies"],
    },
}


def wrap_to_pi(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Wrap angle(s) in radians to [-pi, pi)."""
    if isinstance(angle, (int, float)):
        return float(((angle + math.pi) % (2.0 * math.pi)) - math.pi)
    arr = np.asarray(angle, dtype=float)
    return ((arr + math.pi) % (2.0 * math.pi)) - math.pi


def _wrap_deg_180(angle_deg: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Wrap angle(s) in degrees to [-180.0, 180.0)."""
    if isinstance(angle_deg, (int, float)):
        return float(((angle_deg + 180.0) % 360.0) - 180.0)
    arr = np.asarray(angle_deg, dtype=float)
    return ((arr + 180.0) % 360.0) - 180.0


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Closed-form OLS linear slope cov(x, y) / var(x)."""
    if len(x) < 2:
        return 0.0
    dx = x - float(np.sum(x) / len(x))
    dy = y - float(np.sum(y) / len(y))
    ss_xx = float(np.sum(dx * dx))
    if ss_xx <= 1e-15:
        return 0.0
    return float(np.sum(dx * dy) / ss_xx)


def _get_physics_default(name: str, f_nominal: float = 50.0) -> float:
    """Fallback channel value when all samples in an array are NaN/Inf."""
    lname = name.lower()
    if "freq" in lname:
        return float(f_nominal)
    if "angle" in lname or "rocof" in lname:
        return 0.0
    if "volt" in lname or "mag" in lname:
        return 1.0
    return float(f_nominal) if "f" in lname else 0.0


def _ffill_1d(col: np.ndarray, default_val: float) -> np.ndarray:
    """Causally forward-fill 1D array, backfilling leading NaN/Inf from first valid sample."""
    mask = np.isnan(col) | np.isinf(col)
    if not mask.any():
        return col
    valid_mask = ~mask
    cleaned = col.copy()
    if not valid_mask.any():
        cleaned[:] = default_val
        return cleaned
    idx = np.where(valid_mask, np.arange(len(cleaned)), 0)
    if mask[0]:
        first_valid = int(np.argmax(valid_mask))
        idx[:first_valid] = first_valid
    np.maximum.accumulate(idx, out=idx)
    return cleaned[idx]


def _validate_and_clean_nans(
    arr: np.ndarray,
    name: str,
    policy: str = "raise",
    f_nominal: float = 50.0,
) -> np.ndarray:
    """Validate or clean NaN/Inf values according to policy ('raise', 'ignore_unobserved', 'ffill', 'impute')."""
    if arr.size == 0 or np.isfinite(arr.sum()):
        return arr

    has_inf = bool(np.isinf(arr).any())
    has_nan = bool(np.isnan(arr).any())
    if not (has_nan or has_inf):
        return arr

    if policy == "raise":
        raise ValueError(f"Array '{name}' contains NaN or Inf values")

    default_val = _get_physics_default(name, f_nominal=f_nominal)

    if policy == "ignore_unobserved":
        if has_inf:
            raise ValueError(f"Array '{name}' contains NaN or Inf values")
        if arr.ndim == 2:
            all_nan_cols = np.all(np.isnan(arr), axis=0)
            obs_cols = ~all_nan_cols
            if not np.any(obs_cols):
                raise ValueError(f"Array '{name}' contains NaN or Inf values")
            if np.isnan(arr[:, obs_cols]).any():
                raise ValueError(f"Array '{name}' contains NaN or Inf values")
            return arr
        raise ValueError(f"Array '{name}' contains NaN or Inf values")

    if policy in ("ffill", "locf", "impute"):
        cleaned = np.copy(arr)
        if cleaned.ndim == 1:
            return _ffill_1d(cleaned, default_val)
        all_nan_cols = np.all(np.isnan(cleaned) | np.isinf(cleaned), axis=0)
        obs_cols = ~all_nan_cols
        if not np.any(obs_cols):
            cleaned[:] = default_val
            return cleaned
        for c_idx in np.flatnonzero(obs_cols):
            cleaned[:, c_idx] = _ffill_1d(cleaned[:, c_idx], default_val)
        if policy != "impute":
            for c_idx in np.flatnonzero(all_nan_cols):
                cleaned[:, c_idx] = default_val
        return cleaned

    raise ValueError(f"Unsupported nan_policy '{policy}'")


def _prepare_pmu_bus_channel(
    arr: np.ndarray,
    name: str,
    policy: str,
    f_nominal: float,
    observed_buses: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (cleaned_2d_array, observed_col_mask) preserving NaN on unobserved PMU columns."""
    mat = np.asarray(arr, dtype=float)
    if mat.ndim == 1:
        mat = mat[:, None]
    n_samples, n_buses = mat.shape
    if n_samples == 0 or n_buses == 0:
        return mat, np.zeros(n_buses, dtype=bool)

    if observed_buses is not None:
        mat = mat.copy()
        valid_zero = [int(b) - 1 for b in observed_buses if 1 <= int(b) <= n_buses]
        keep = np.zeros(n_buses, dtype=bool)
        if valid_zero:
            keep[valid_zero] = True
        mat[:, ~keep] = np.nan

    if np.isfinite(mat.sum()):
        return mat, np.ones(n_buses, dtype=bool)

    has_inf = bool(np.isinf(mat).any())
    all_invalid_cols = np.all(np.isnan(mat) | np.isinf(mat), axis=0)
    obs_col_mask = ~all_invalid_cols

    if policy == "raise":
        if observed_buses is None:
            raise ValueError(f"Array '{name}' contains NaN or Inf values")
        if has_inf or not np.any(obs_col_mask) or np.isnan(mat[:, obs_col_mask]).any():
            raise ValueError(f"Array '{name}' contains NaN or Inf values")
        return mat, obs_col_mask

    if policy == "ignore_unobserved":
        if has_inf or not np.any(obs_col_mask) or np.isnan(mat[:, obs_col_mask]).any():
            raise ValueError(f"Array '{name}' contains NaN or Inf values")
        return mat, obs_col_mask

    if policy in ("ffill", "locf", "impute"):
        cleaned = mat.copy()
        default_val = _get_physics_default(name, f_nominal=f_nominal)
        if not np.any(obs_col_mask):
            cleaned[:] = default_val
            return cleaned, np.ones(n_buses, dtype=bool)
        for c_idx in np.flatnonzero(obs_col_mask):
            cleaned[:, c_idx] = _ffill_1d(cleaned[:, c_idx], default_val)
        return cleaned, obs_col_mask

    raise ValueError(f"Unsupported nan_policy '{policy}'")


def _resolve_f_sys_from_traj(
    windowed_traj: Dict[str, Any],
    bus_idx: Optional[int] = None,
    area_partition: Optional[Dict[int, List[int]]] = None,
) -> np.ndarray:
    """Resolve PMU frequency series strictly from bus_frequencies unless missing."""
    if bus_idx is not None:
        return np.asarray(windowed_traj["bus_frequencies"][:, bus_idx], dtype=float)
    if "bus_frequencies" in windowed_traj:
        bf = np.asarray(windowed_traj["bus_frequencies"], dtype=float)
        if bf.size > 0:
            return estimate_pmu_system_frequency(bf, area_partition=area_partition)
    if "pmu_sys_frequency" in windowed_traj:
        return np.asarray(windowed_traj["pmu_sys_frequency"], dtype=float)
    return np.asarray(windowed_traj["coi_frequency"], dtype=float)


def _infer_fps(ts: np.ndarray, meta: Dict[str, Any]) -> float:
    if "fps" in meta and meta["fps"] is not None:
        return float(meta["fps"])
    if len(ts) >= 2:
        dt_med = float(np.median(np.diff(ts)))
        if dt_med > 1e-9:
            return 1.0 / dt_med
    return 50.0


def _compute_curvature_ols(
    ts: np.ndarray,
    rocof_sys: np.ndarray,
    idx_end: int,
    window_length: int = 5,
) -> float:
    """Closed-form OLS slope of causal rocof_sys over [t0, ts[idx_end-1]] in Hz/s^2."""
    end = min(idx_end, len(ts))
    if end < 3:
        return 0.0
    k = np.arange(1, end, dtype=np.intp)
    start_k = np.maximum(0, k - (window_length - 1))
    tau = 0.5 * (ts[start_k] + ts[k])
    return _ols_slope(tau - tau[0], rocof_sys[1:end])


def calc_initial_rocof(
    windowed_traj: Dict[str, Any],
    delta_t_init: float = 0.10,
    bus_idx: Optional[int] = None,
) -> float:
    """Estimate initial ROCOF over [t_0, min(t_0 + delta_t_init, t_end)] via OLS slope (Hz/s)."""
    ts = np.asarray(windowed_traj["timestamps"], dtype=float)
    y = _resolve_f_sys_from_traj(windowed_traj, bus_idx=bus_idx)
    n = len(ts)
    if n < 2:
        return 0.0
    t_0 = ts[0]
    idx_end = int(np.searchsorted(ts, t_0 + delta_t_init + 1e-9, side="right"))
    idx_end = min(max(idx_end, 2), n)
    return _ols_slope(ts[:idx_end] - t_0, y[:idx_end])


def calc_rocof_metrics(
    windowed_traj: Dict[str, Any],
    bus_idx: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Return (rocof_peak_neg, rocof_end, rocof_mean) in Hz/s from causal 5-point ROCOF and OLS."""
    ts = np.asarray(windowed_traj["timestamps"], dtype=float)
    y = _resolve_f_sys_from_traj(windowed_traj, bus_idx=bus_idx)
    n = len(y)
    if n < 2:
        return 0.0, 0.0, 0.0
    meta = windowed_traj.get("metadata", {}) if isinstance(windowed_traj, dict) else {}
    fps = _infer_fps(ts, meta)
    rocof_c = compute_causal_rocof(y, fps=fps, window_length=5)
    peak_neg = float(np.min(rocof_c[1:]))
    rocof_end = float(rocof_c[-1])
    mean_rocof = _ols_slope(ts - ts[0], y)
    return peak_neg, rocof_end, mean_rocof


def calc_max_rocof(
    windowed_traj: Dict[str, Any],
    bus_idx: Optional[int] = None,
) -> Tuple[float, float]:
    pk_neg, r_end, _ = calc_rocof_metrics(windowed_traj, bus_idx=bus_idx)
    return max(abs(pk_neg), abs(r_end)), pk_neg


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
    """Return (freq_drop_end, freq_drop_max) in Hz."""
    y = _resolve_f_sys_from_traj(windowed_traj, bus_idx=bus_idx)
    if len(y) == 0:
        return 0.0, 0.0
    eff_f0 = float(f_nominal)
    if eff_f0 == 50.0 and isinstance(windowed_traj, dict):
        meta_f0 = windowed_traj.get("metadata", {}).get("f_nominal")
        if meta_f0 is not None:
            eff_f0 = float(meta_f0)
    freq_drop_end = float(y[-1] - y[0]) if len(y) >= 2 else 0.0
    freq_drop_max = float(np.min(y) - eff_f0)
    return freq_drop_end, freq_drop_max


def calc_freq_integral(
    windowed_traj: Dict[str, Any],
    f_nominal: float = 50.0,
    bus_idx: Optional[int] = None,
) -> Tuple[float, float]:
    """Return (freq_dev_integral, abs_freq_dev_integral) in Hz*s."""
    ts = np.asarray(windowed_traj["timestamps"], dtype=float)
    y = _resolve_f_sys_from_traj(windowed_traj, bus_idx=bus_idx)
    if len(y) < 2:
        return 0.0, 0.0
    eff_f0 = float(f_nominal)
    if eff_f0 == 50.0 and isinstance(windowed_traj, dict):
        meta_f0 = windowed_traj.get("metadata", {}).get("f_nominal")
        if meta_f0 is not None:
            eff_f0 = float(meta_f0)
    dt = ts[1:] - ts[:-1]
    dev = y - eff_f0
    integral_dev = 0.5 * float(np.dot(dev[:-1] + dev[1:], dt))
    integral_abs = 0.5 * float(np.dot(np.abs(dev[:-1]) + np.abs(dev[1:]), dt))
    return integral_dev, integral_abs


def calc_interarea_dispersion(
    windowed_traj: Dict[str, Any],
    area_partition: Optional[Dict[int, List[int]]] = None,
) -> Tuple[float, float]:
    """Return (area_freq_spread_max, bus_freq_std_mean) in Hz across the full window."""
    bf = np.asarray(windowed_traj["bus_frequencies"], dtype=float)
    if bf.size == 0:
        return 0.0, 0.0
    if bf.ndim == 1:
        bf = bf[:, None]
    n_samples, n_buses = bf.shape

    if (
        n_buses == 39
        and (area_partition is None or area_partition is CANONICAL_AREA_PARTITION or area_partition == CANONICAL_AREA_PARTITION)
        and np.isfinite(np.sum(bf))
    ):
        m0 = np.sum(bf[:, CANONICAL_AREA_1_IDX], axis=1) * (1.0 / 9.0)
        m1 = np.sum(bf[:, CANONICAL_AREA_2_IDX], axis=1) * (1.0 / 15.0)
        m2 = np.sum(bf[:, CANONICAL_AREA_3_IDX], axis=1) * (1.0 / 15.0)
        spread_max = float(
            np.max(np.maximum(np.maximum(np.abs(m0 - m1), np.abs(m0 - m2)), np.abs(m1 - m2)))
        )
        bus_std_mean = float(np.mean(np.std(bf, axis=1)))
        return spread_max, bus_std_mean

    obs_mask = ~np.all(np.isnan(bf), axis=0)
    if not np.any(obs_mask):
        return 0.0, 0.0

    partition = area_partition or CANONICAL_AREA_PARTITION
    area_series: List[np.ndarray] = []
    for area_id in sorted(partition.keys()):
        cols = [b - 1 for b in partition[area_id] if 0 <= b - 1 < n_buses and obs_mask[b - 1]]
        if cols:
            sub = bf[:, cols]
            area_series.append(
                np.mean(sub, axis=1) if np.isfinite(sub.sum()) else np.nanmean(sub, axis=1)
            )

    if len(area_series) >= 3:
        s01 = np.abs(area_series[0] - area_series[1])
        s02 = np.abs(area_series[0] - area_series[2])
        s12 = np.abs(area_series[1] - area_series[2])
        spread_max = float(np.max(np.maximum(np.maximum(s01, s02), s12)))
    elif len(area_series) == 2:
        spread_max = float(np.max(np.abs(area_series[0] - area_series[1])))
    else:
        spread_max = 0.0

    bf_obs = bf[:, obs_mask]
    if bf_obs.shape[1] >= 2:
        std_t = np.std(bf_obs, axis=1) if np.isfinite(bf_obs.sum()) else np.nanstd(bf_obs, axis=1)
        bus_std_mean = float(np.mean(std_t))
    else:
        bus_std_mean = 0.0

    return spread_max, bus_std_mean


def calc_angle_divergence(
    windowed_traj: Dict[str, Any],
    tie_buses: Tuple[int, int] = (4, 5),
    angle_unit: str = "deg",
    tie_lines: Optional[Sequence[Tuple[int, int]]] = None,
) -> Tuple[float, float]:
    """Return (angle_divergence_max, tie_line_angle_excursion) in degrees across the window."""
    va_raw = np.asarray(windowed_traj["voltage_angles"], dtype=float)
    if va_raw.size == 0:
        return 0.0, 0.0
    if va_raw.ndim == 1:
        va_raw = va_raw[:, None]

    va_deg = np.degrees(va_raw) if angle_unit == "rad" else va_raw
    n_samples, n_buses = va_deg.shape
    if n_samples < 2 or n_buses == 0:
        return 0.0, 0.0

    all_finite_va = bool(np.isfinite(np.sum(va_deg)))
    if all_finite_va:
        obs_mask = np.ones(n_buses, dtype=bool)
        va_obs = va_deg
    else:
        obs_mask = ~np.all(np.isnan(va_deg), axis=0)
        if not np.any(obs_mask):
            return 0.0, 0.0
        va_obs = va_deg[:, obs_mask]
    if va_obs.shape[1] >= 2:
        # Wrap-safe dynamic rotor/bus phase shift relative to pre-disturbance/initial angle va[0]
        rel_deg = _wrap_deg_180(va_obs - va_obs[0:1, :])
        raw_spans = np.max(rel_deg, axis=1) - np.min(rel_deg, axis=1)
        over_mask = raw_spans > 180.0
        if np.any(over_mask):
            spans = raw_spans.copy()
            for t_idx in np.flatnonzero(over_mask):
                row = rel_deg[t_idx]
                pair_diff = _wrap_deg_180(row[:, None] - row[None, :])
                spans[t_idx] = float(np.max(np.abs(pair_diff)))
            angle_div_max = float(np.max(spans))
        else:
            angle_div_max = float(np.max(raw_spans))
    else:
        angle_div_max = 0.0

    if tie_lines is not None:
        candidate_ties = list(tie_lines)
    elif tie_buses != (4, 5):
        candidate_ties = [tie_buses] + [p for p in PRIMARY_TIE_LINES if p != tie_buses]
    else:
        candidate_ties = list(PRIMARY_TIE_LINES)

    valid_pairs: List[Tuple[int, int]] = [
        (a - 1, b - 1)
        for (a, b) in candidate_ties
        if 0 <= a - 1 < n_buses and 0 <= b - 1 < n_buses and obs_mask[a - 1] and obs_mask[b - 1]
    ]
    if not valid_pairs:
        valid_pairs = [
            (a - 1, b - 1)
            for (a, b) in SECONDARY_INTERAREA_TIE_LINES
            if 0 <= a - 1 < n_buses and 0 <= b - 1 < n_buses and obs_mask[a - 1] and obs_mask[b - 1]
        ]
    if not valid_pairs and np.sum(obs_mask) >= 2:
        # Fallback to inter-area observed PMU bus pairs when boundary tie endpoints are unobserved
        area_rep: List[int] = []
        for area_id in (1, 2, 3):
            cols = [b - 1 for b in CANONICAL_AREA_PARTITION[area_id] if 0 <= b - 1 < n_buses and obs_mask[b - 1]]
            if cols:
                area_rep.append(cols[0])
        if len(area_rep) >= 2:
            for i in range(len(area_rep)):
                for j in range(i + 1, len(area_rep)):
                    valid_pairs.append((area_rep[i], area_rep[j]))

    if valid_pairs:
        tie_max = 0.0
        for col_a, col_b in valid_pairs:
            diff_t = _wrap_deg_180(va_deg[:, col_a] - va_deg[:, col_b])
            swing_t = np.abs(_wrap_deg_180(diff_t - diff_t[0]))
            line_max = float(np.max(swing_t))
            if line_max > tie_max:
                tie_max = line_max
    else:
        tie_max = 0.0

    return angle_div_max, tie_max


def calc_coi_acceleration(
    windowed_traj: Dict[str, Any],
    delta_t_init: float = 0.20,
) -> Tuple[float, float]:
    """Return (freq_curvature_init, freq_curvature_mean) true d^2f/dt^2 in Hz/s^2."""
    ts = np.asarray(windowed_traj["timestamps"], dtype=float)
    f_sys = _resolve_f_sys_from_traj(windowed_traj)
    n = len(f_sys)
    if n < 3:
        return 0.0, 0.0
    meta = windowed_traj.get("metadata", {}) if isinstance(windowed_traj, dict) else {}
    fps = _infer_fps(ts, meta)
    rocof_sys = compute_causal_rocof(f_sys, fps=fps, window_length=5)
    idx_init = int(np.searchsorted(ts, ts[0] + delta_t_init + 1e-9, side="right"))
    idx_init = min(max(idx_init, 3), n)
    curv_init = _compute_curvature_ols(ts, rocof_sys, idx_init, window_length=5)
    curv_mean = _compute_curvature_ols(ts, rocof_sys, n, window_length=5)
    return curv_init, curv_mean


def extract_feature_vector(
    windowed_traj: Dict[str, Any],
    f_nominal: float = 50.0,
    area_partition: Optional[Dict[int, List[int]]] = None,
) -> List[float]:
    """Return the 14 canonical features ordered according to CANONICAL_FEATURE_NAMES."""
    feats = extract_features(
        windowed_traj,
        f_nominal=f_nominal,
        area_partition=area_partition,
    )
    return [feats[k] for k in CANONICAL_FEATURE_NAMES]


def extract_features(
    scenario_or_window: Union[Dict[str, Any], CausalWindow, str, Path],
    t_obs: Optional[float] = None,
    f_nominal: float = 50.0,
    nan_policy: str = "ignore_unobserved",
    area_partition: Optional[Dict[int, List[int]]] = None,
    tie_buses: Tuple[int, int] = (4, 5),
    observed_buses: Optional[Sequence[int]] = None,
    strict_nadir: bool = False,
    **kwargs: Any,
) -> Dict[str, float]:
    """Extract the 14 canonical PMU-only causal features from a scenario or CausalWindow."""
    if t_obs is not None:
        w: Union[CausalWindow, Dict[str, Any]] = extract_causal_window(
            scenario_or_window, t_obs=t_obs, strict_nadir=strict_nadir
        )
    elif isinstance(scenario_or_window, CausalWindow):
        w = scenario_or_window
    elif isinstance(scenario_or_window, dict) and "timestamps" in scenario_or_window:
        if "metadata" in scenario_or_window and "t_obs" in scenario_or_window["metadata"]:
            w = scenario_or_window
        elif "metadata" in scenario_or_window and "t_disturb" in scenario_or_window["metadata"]:
            default_tobs = float(kwargs.get("default_t_obs", 1.0))
            w = extract_causal_window(
                scenario_or_window, t_obs=default_tobs, strict_nadir=strict_nadir
            )
        else:
            w = scenario_or_window
    else:
        w = extract_causal_window(scenario_or_window, t_obs=1.0, strict_nadir=strict_nadir)

    ts = np.asarray(w["timestamps"], dtype=float)
    n = len(ts)
    if n == 0:
        return {k: 0.0 for k in CANONICAL_FEATURE_NAMES}

    w_meta = w.get("metadata", {}) if isinstance(w, dict) else {}
    eff_f_nominal = float(f_nominal)
    if eff_f_nominal == 50.0 and "f_nominal" in w_meta and w_meta["f_nominal"] is not None:
        eff_f_nominal = float(w_meta["f_nominal"])

    eff_fps = _infer_fps(ts, w_meta)

    has_bf = ("bus_frequencies" in w) and (np.asarray(w["bus_frequencies"]).size > 0)
    if has_bf:
        bf_clean, obs_bf_mask = _prepare_pmu_bus_channel(
            np.asarray(w["bus_frequencies"], dtype=float),
            "bus_frequencies",
            policy=nan_policy,
            f_nominal=eff_f_nominal,
            observed_buses=observed_buses,
        )
        obs_indices = np.flatnonzero(obs_bf_mask)
        f_sys_pmu = estimate_pmu_system_frequency(
            bf_clean,
            observed_bus_indices=obs_indices,
            area_partition=area_partition,
        )
    else:
        bf_clean = np.empty((n, 0), dtype=float)
        obs_bf_mask = np.zeros(0, dtype=bool)
        fallback_key = "pmu_sys_frequency" if "pmu_sys_frequency" in w else "coi_frequency"
        f_sys_raw = np.asarray(w[fallback_key], dtype=float)
        clean_policy = "ffill" if nan_policy in ("ffill", "locf", "impute") else "raise"
        f_sys_pmu = _validate_and_clean_nans(
            f_sys_raw, fallback_key, policy=clean_policy, f_nominal=eff_f_nominal
        )

    has_va = ("voltage_angles" in w) and (np.asarray(w["voltage_angles"]).size > 0)
    if has_va:
        va_clean, _ = _prepare_pmu_bus_channel(
            np.asarray(w["voltage_angles"], dtype=float),
            "voltage_angles",
            policy=nan_policy,
            f_nominal=eff_f_nominal,
            observed_buses=observed_buses,
        )
    else:
        va_clean = np.empty((n, 0), dtype=float)

    delta_t_init = float(kwargs.get("delta_t_init", 0.10))
    delta_t_curv = float(kwargs.get("delta_t_curv", 0.20))
    angle_unit = str(kwargs.get("angle_unit", "deg"))
    tie_lines = kwargs.get("tie_lines", None)

    if n < 2:
        rocof_init = 0.0
        rocof_peak_neg = 0.0
        rocof_end = 0.0
        rocof_mean = 0.0
        freq_drop_end = 0.0
        freq_drop_max = float(f_sys_pmu[0] - eff_f_nominal)
        freq_dev_integral = 0.0
        freq_curvature_init = 0.0
        freq_curvature_mean = 0.0
    else:
        t_0 = ts[0]
        idx_init = int(np.searchsorted(ts, t_0 + delta_t_init + 1e-9, side="right"))
        idx_init = min(max(idx_init, 2), n)
        rocof_init = _ols_slope(ts[:idx_init] - t_0, f_sys_pmu[:idx_init])

        rocof_sys_pmu = compute_causal_rocof(f_sys_pmu, fps=eff_fps, window_length=5)
        rocof_peak_neg = float(np.min(rocof_sys_pmu[1:]))
        rocof_end = float(rocof_sys_pmu[-1])
        rocof_mean = _ols_slope(ts - t_0, f_sys_pmu)

        freq_drop_end = float(f_sys_pmu[-1] - f_sys_pmu[0])
        freq_drop_max = float(np.min(f_sys_pmu) - eff_f_nominal)

        dt = ts[1:] - ts[:-1]
        dev = f_sys_pmu - eff_f_nominal
        freq_dev_integral = 0.5 * float(np.sum((dev[:-1] + dev[1:]) * dt))

        if n >= 3:
            idx_curv = int(np.searchsorted(ts, t_0 + delta_t_curv + 1e-9, side="right"))
            idx_curv = min(max(idx_curv, 3), n)
            freq_curvature_init = _compute_curvature_ols(ts, rocof_sys_pmu, idx_curv, window_length=5)
            freq_curvature_mean = _compute_curvature_ols(ts, rocof_sys_pmu, n, window_length=5)
        else:
            freq_curvature_init = 0.0
            freq_curvature_mean = 0.0

    if bf_clean.size > 0 and np.any(obs_bf_mask):
        bf_obs = bf_clean if np.all(obs_bf_mask) else bf_clean[:, obs_bf_mask]
        bus_drop_worst = float(np.min(bf_obs) - eff_f_nominal)
        area_freq_spread_max, bus_freq_std_mean = calc_interarea_dispersion(
            {"bus_frequencies": bf_clean}, area_partition=area_partition
        )
    else:
        bus_drop_worst = freq_drop_max
        area_freq_spread_max = 0.0
        bus_freq_std_mean = 0.0

    if va_clean.size > 0:
        angle_divergence_max, tie_line_angle_excursion = calc_angle_divergence(
            {"voltage_angles": va_clean},
            tie_buses=tie_buses,
            angle_unit=angle_unit,
            tie_lines=tie_lines,
        )
    else:
        angle_divergence_max = 0.0
        tie_line_angle_excursion = 0.0

    return {
        "rocof_init": rocof_init,
        "rocof_peak_neg": rocof_peak_neg,
        "rocof_end": rocof_end,
        "rocof_mean": rocof_mean,
        "freq_drop_end": freq_drop_end,
        "freq_drop_max": freq_drop_max,
        "freq_dev_integral": freq_dev_integral,
        "bus_drop_worst": bus_drop_worst,
        "area_freq_spread_max": area_freq_spread_max,
        "bus_freq_std_mean": bus_freq_std_mean,
        "angle_divergence_max": angle_divergence_max,
        "tie_line_angle_excursion": tie_line_angle_excursion,
        "freq_curvature_init": freq_curvature_init,
        "freq_curvature_mean": freq_curvature_mean,
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
        delta_t_init: float = 0.10,
        area_partition: Optional[Dict[int, List[int]]] = None,
        tie_buses: Tuple[int, int] = (4, 5),
        handle_nans: bool = False,
        nan_policy: str = "ignore_unobserved",
        strict_nadir: bool = False,
        observed_buses: Optional[Sequence[int]] = None,
    ) -> None:
        self.t_obs = float(t_obs)
        self.f_nominal = float(f_nominal)
        self.delta_t_init = float(delta_t_init)
        self.area_partition = area_partition or CANONICAL_AREA_PARTITION
        self.tie_buses = tie_buses
        self.nan_policy = "ffill" if handle_nans else nan_policy
        self.strict_nadir = bool(strict_nadir)
        self.observed_buses = list(observed_buses) if observed_buses is not None else None

    def extract_features_dict(
        self,
        scenario: Union[Dict[str, Any], CausalWindow, str, Path],
        t_obs: Optional[float] = None,
        strict_nadir: Optional[bool] = None,
        observed_buses: Optional[Sequence[int]] = None,
    ) -> Dict[str, float]:
        """Extract 14-feature dictionary using configured window and PMU parameters."""
        obs = self.t_obs if t_obs is None else float(t_obs)
        eff_strict = self.strict_nadir if strict_nadir is None else bool(strict_nadir)
        eff_obs = self.observed_buses if observed_buses is None else observed_buses

        if isinstance(scenario, CausalWindow) and t_obs is None and not eff_strict:
            return extract_features(
                scenario,
                t_obs=None,
                f_nominal=self.f_nominal,
                delta_t_init=self.delta_t_init,
                nan_policy=self.nan_policy,
                area_partition=self.area_partition,
                tie_buses=self.tie_buses,
                observed_buses=eff_obs,
            )

        return extract_features(
            scenario,
            t_obs=obs,
            f_nominal=self.f_nominal,
            delta_t_init=self.delta_t_init,
            nan_policy=self.nan_policy,
            area_partition=self.area_partition,
            tie_buses=self.tie_buses,
            observed_buses=eff_obs,
            strict_nadir=eff_strict,
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
        """Extract single-row 14-column feature DataFrame (or dict if pandas is unavailable)."""
        feats = self.extract_features_dict(scenario, t_obs=t_obs)
        if HAS_PANDAS:
            return pd.DataFrame([feats], columns=CANONICAL_FEATURE_NAMES, dtype=float)
        return feats

    def transform_batch(
        self,
        scenarios: List[Union[Dict[str, Any], CausalWindow, str, Path]],
        t_obs: Optional[float] = None,
        as_dataframe: bool = True,
        strict_nadir: Optional[bool] = None,
        observed_buses: Optional[Sequence[int]] = None,
    ) -> Tuple[Any, np.ndarray, np.ndarray, Any]:
        """Extract (X, y_nadir, y_secure, metadata) of shape (N, 14) across scenarios."""
        obs = self.t_obs if t_obs is None else float(t_obs)
        eff_strict = self.strict_nadir if strict_nadir is None else bool(strict_nadir)
        eff_obs = self.observed_buses if observed_buses is None else observed_buses

        feature_rows: List[Dict[str, float]] = []
        y_nadir_list: List[float] = []
        y_secure_list: List[int] = []
        meta_list: List[Dict[str, Any]] = []

        for scen in scenarios:
            if isinstance(scen, CausalWindow) and abs(scen.t_obs - obs) < 1e-6 and not eff_strict:
                w = scen
            else:
                w = extract_causal_window(scen, t_obs=obs, strict_nadir=eff_strict)

            feats = extract_features(
                w,
                t_obs=None,
                f_nominal=self.f_nominal,
                delta_t_init=self.delta_t_init,
                nan_policy=self.nan_policy,
                area_partition=self.area_partition,
                tie_buses=self.tie_buses,
                observed_buses=eff_obs,
            )
            feature_rows.append(feats)

            w_meta = dict(w.metadata)
            eff_f0 = float(w_meta.get("f_nominal", self.f_nominal))
            nadir = float(w_meta.get("ground_truth_nadir", eff_f0 - 0.5))
            default_thresh = float(w_meta.get("f_threshold_hz", eff_f0 - 0.5))
            is_sec = int(w_meta.get("is_secure", 1 if nadir >= default_thresh else 0))

            req_ren = float(
                w_meta.get("renewable_penetration", w_meta.get("p_ren_requested", 0.0))
            )
            realized_ren = float(w_meta.get("p_ren_realized", req_ren))
            partition_val = str(w_meta.get("partition", "train"))
            ood_type_val = str(
                w_meta.get("ood_type", "ood" if partition_val == "ood_test" else "id")
            )
            t_nadir_raw = w_meta.get("ground_truth_t_nadir", None)
            t_nadir_val = float(t_nadir_raw) if t_nadir_raw is not None else float("nan")

            y_nadir_list.append(nadir)
            y_secure_list.append(is_sec)
            meta_list.append({
                "scenario_id": w_meta.get("scenario_id", "unknown"),
                "partition": partition_val,
                "ood_type": ood_type_val,
                "disturbed_gen": w_meta.get("disturbed_gen", 0),
                "loss_fraction": float(w_meta.get("loss_fraction", 1.0)),
                "active_loss_mw": float(w_meta.get("active_loss_mw", 0.0)),
                "renewable_penetration": req_ren,
                "p_ren_realized": realized_ren,
                "load_scale": float(w_meta.get("load_scale", 1.0)),
                "t_disturb": float(w_meta.get("t_disturb", 1.0)),
                "t_obs": float(obs),
                "ground_truth_nadir": nadir,
                "ground_truth_t_nadir": t_nadir_val,
                "is_prior_to_nadir": bool(w_meta.get("is_prior_to_nadir", True)),
                "lead_time_s": float(w_meta.get("lead_time_s", float("nan"))),
                "is_secure": is_sec,
            })

        y_nadir = np.array(y_nadir_list, dtype=float)
        y_secure = np.array(y_secure_list, dtype=int)

        if feature_rows:
            X_arr = np.array(
                [[row[k] for k in CANONICAL_FEATURE_NAMES] for row in feature_rows],
                dtype=float,
            )
        else:
            X_arr = np.empty((0, len(CANONICAL_FEATURE_NAMES)), dtype=float)

        if as_dataframe and HAS_PANDAS:
            X_df = pd.DataFrame(X_arr, columns=CANONICAL_FEATURE_NAMES)
            meta_df = pd.DataFrame(meta_list)
            return X_df, y_nadir, y_secure, meta_df

        return X_arr, y_nadir, y_secure, meta_list


def extract_feature_dataset(
    scenarios: List[Union[Dict[str, Any], str, Path]],
    t_obs: float = 1.0,
    as_dataframe: bool = True,
) -> Tuple[Any, np.ndarray, np.ndarray, List[Any]]:
    """Extract (X, y_nadir, y_secure, scenario_ids) across a list of scenarios."""
    extractor = FeatureExtractor(t_obs=t_obs)
    X, y_nadir, y_secure, meta = extractor.transform_batch(scenarios, as_dataframe=as_dataframe)
    if HAS_PANDAS and isinstance(meta, pd.DataFrame) and "scenario_id" in meta.columns:
        ids = meta["scenario_id"].tolist()
    elif isinstance(meta, list):
        ids = [m.get("scenario_id", f"scen_{i}") for i, m in enumerate(meta)]
    else:
        ids = [f"scen_{i}" for i in range(len(scenarios))]
    return X, y_nadir, y_secure, ids
