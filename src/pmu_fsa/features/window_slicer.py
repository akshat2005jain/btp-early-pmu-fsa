"""Post-disturbance PMU observation window slicing and anti-leakage guardrails."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np

from pmu_fsa.simulation.pmu_recorder import compute_causal_rocof, estimate_pmu_system_frequency


FORBIDDEN_WINDOW_CHANNEL_KEYS: Set[str] = {
    "ground_truth_nadir",
    "ground_truth_t_nadir",
    "bus_min_nadir",
    "bus_min_t_nadir",
    "nadir_bus_id",
    "is_secure",
    "y_nadir",
    "y_secure",
    "generator_deltas",
    "generator_speeds",
    "generator_powers",
    "coi_rocof",
    "area_frequencies",
    "bus_frequencies_true",
    "voltage_magnitudes_true",
    "voltage_angles_true",
}


class CausalWindow(dict):
    """Dict-compatible container for a sliced [t_0, t_0 + t_obs] PMU observation window."""

    def __init__(
        self,
        timestamps: np.ndarray,
        bus_frequencies: np.ndarray,
        bus_rocof: np.ndarray,
        voltage_magnitudes: np.ndarray,
        voltage_angles: np.ndarray,
        coi_frequency: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
        pmu_sys_frequency: Optional[np.ndarray] = None,
        **extra_channels: Any,
    ) -> None:
        super().__init__()
        ts_arr = np.asarray(timestamps, dtype=float)
        bf_arr = np.asarray(bus_frequencies, dtype=float)
        br_arr = np.asarray(bus_rocof, dtype=float)
        vm_arr = np.asarray(voltage_magnitudes, dtype=float)
        va_arr = np.asarray(voltage_angles, dtype=float)

        if pmu_sys_frequency is not None:
            psys_arr = np.asarray(pmu_sys_frequency, dtype=float)
        elif bf_arr.size > 0:
            psys_arr = estimate_pmu_system_frequency(bf_arr)
        else:
            raise ValueError(
                "CausalWindow requires observable PMU 'bus_frequencies' or 'pmu_sys_frequency'; "
                "falling back to internal 'coi_frequency' is forbidden."
            )

        # Keep coi_frequency slot strictly as an observable PMU alias; never copy simulator COI.
        coi_arr = psys_arr.copy()

        self["timestamps"] = ts_arr
        self["bus_frequencies"] = bf_arr
        self["bus_rocof"] = br_arr
        self["voltage_magnitudes"] = vm_arr
        self["voltage_angles"] = va_arr
        self["pmu_sys_frequency"] = psys_arr
        self["coi_frequency"] = coi_arr
        self["metadata"] = dict(metadata or {})

        for k, v in extra_channels.items():
            if k in FORBIDDEN_WINDOW_CHANNEL_KEYS:
                continue
            self[k] = np.asarray(v) if isinstance(v, (list, tuple, np.ndarray)) else v

    @property
    def timestamps(self) -> np.ndarray:
        return self["timestamps"]

    @property
    def bus_frequencies(self) -> np.ndarray:
        return self["bus_frequencies"]

    @property
    def bus_rocof(self) -> np.ndarray:
        return self["bus_rocof"]

    @property
    def voltage_magnitudes(self) -> np.ndarray:
        return self["voltage_magnitudes"]

    @property
    def voltage_angles(self) -> np.ndarray:
        return self["voltage_angles"]

    @property
    def pmu_sys_frequency(self) -> np.ndarray:
        return self["pmu_sys_frequency"]

    @property
    def coi_frequency(self) -> np.ndarray:
        return self["coi_frequency"]

    @property
    def metadata(self) -> Dict[str, Any]:
        return self["metadata"]

    @property
    def n_samples(self) -> int:
        return len(self["timestamps"])

    @property
    def n_buses(self) -> int:
        bf = self["bus_frequencies"]
        return bf.shape[1] if bf.ndim > 1 else 39

    @property
    def duration(self) -> float:
        ts = self["timestamps"]
        return float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0

    @property
    def fps(self) -> float:
        return float(self["metadata"].get("fps", 50.0))

    @property
    def t_0(self) -> float:
        return float(self["metadata"].get("t_disturb", self["timestamps"][0]))

    @property
    def t_obs(self) -> float:
        return float(self["metadata"].get("t_obs", self.duration))

    @property
    def is_prior_to_nadir(self) -> bool:
        return bool(self["metadata"].get("is_prior_to_nadir", True))

    @property
    def lead_time_s(self) -> float:
        return float(self["metadata"].get("lead_time_s", float("nan")))

    def to_dict(self) -> Dict[str, Any]:
        return dict(self)

    def __repr__(self) -> str:
        return (
            f"CausalWindow(n_samples={self.n_samples}, n_buses={self.n_buses}, "
            f"duration={self.duration:.3f}s, t_0={self.t_0:.3f}s, t_obs={self.t_obs:.3f}s)"
        )


def _load_scenario_data(
    scenario_data: Union[Dict[str, Any], str, Path, np.lib.npyio.NpzFile],
) -> Dict[str, Any]:
    """Normalize scenario input from a dict, .npz path, or open NpzFile."""
    if isinstance(scenario_data, (str, Path)):
        with np.load(scenario_data, allow_pickle=True) as npz:
            data = {k: npz[k] for k in npz.files}
        if "metadata" in data and isinstance(data["metadata"], np.ndarray):
            data["metadata"] = data["metadata"].item()
        return data

    if isinstance(scenario_data, np.lib.npyio.NpzFile):
        data = {k: scenario_data[k] for k in scenario_data.files}
        if "metadata" in data and isinstance(data["metadata"], np.ndarray):
            data["metadata"] = data["metadata"].item()
        return data

    if isinstance(scenario_data, dict):
        return scenario_data

    raise TypeError(
        f"Unsupported scenario_data type: {type(scenario_data)}. "
        f"Expected dict, str, Path, or NpzFile."
    )


def extract_causal_window(
    scenario_data: Union[Dict[str, Any], str, Path, np.lib.npyio.NpzFile],
    t_obs: float,
    t_0: Optional[float] = None,
    strict_nadir: bool = False,
    clip_to_nadir: bool = False,
    require_pre_nadir: bool = False,
) -> CausalWindow:
    """Slice scenario PMU channels strictly to the causal window [t_0, t_0 + t_obs]."""
    if t_obs <= 0.0:
        raise ValueError(
            f"Observation window duration T_obs must be strictly positive (> 0), got {t_obs}"
        )

    data = _load_scenario_data(scenario_data)

    if "timestamps" not in data:
        raise KeyError("Scenario data missing required 'timestamps' array")

    timestamps = np.asarray(data["timestamps"], dtype=float)
    if len(timestamps) == 0:
        raise ValueError("Scenario timestamps array is empty")

    metadata = dict(data.get("metadata", {}))

    if t_0 is None:
        if "t_disturb" in metadata:
            t_0 = float(metadata["t_disturb"])
        else:
            raise ValueError(
                "Disturbance time t_0 not provided and 't_disturb' not found in metadata"
            )
    else:
        t_0 = float(t_0)

    if t_0 < timestamps[0] - 1e-6 or t_0 > timestamps[-1] + 1e-6:
        raise ValueError(
            f"Disturbance time t_0={t_0:.4f}s is outside trajectory bounds "
            f"[{timestamps[0]:.4f}s, {timestamps[-1]:.4f}s]"
        )

    t_end = t_0 + float(t_obs)

    idx_start = int(np.searchsorted(timestamps, t_0 - 1e-9, side="left"))
    idx_end = int(np.searchsorted(timestamps, t_end + 1e-9, side="right"))

    if idx_start >= len(timestamps) or idx_end <= idx_start:
        raise ValueError(f"No samples found within window [{t_0:.4f}s, {t_end:.4f}s]")

    if (idx_end - idx_start) < 2:
        raise ValueError(
            f"Window [{t_0:.4f}s, {t_end:.4f}s] contains insufficient samples "
            f"({idx_end - idx_start} < 2) for feature calculation"
        )

    is_truncated = False
    if t_end > timestamps[-1] + 1e-6:
        is_truncated = True
        warnings.warn(
            f"Observation window end {t_end:.4f}s exceeds trajectory duration "
            f"{timestamps[-1]:.4f}s; window truncated to available samples.",
            UserWarning,
            stacklevel=2,
        )

    is_prior_to_nadir = True
    lead_time_s = float("nan")
    t_nadir_raw = metadata.get("ground_truth_t_nadir", None)
    if t_nadir_raw is not None:
        t_nadir = float(t_nadir_raw)
        is_prior_to_nadir = bool(t_end < t_nadir - 1e-6)
        lead_time_s = float(t_nadir - t_end)
        if not is_prior_to_nadir:
            msg = (
                f"Observation window [{t_0:.4f}s, {t_end:.4f}s] reaches or exceeds "
                f"nadir time {t_nadir:.4f}s. Pre-nadir causality warning."
            )
            if strict_nadir or require_pre_nadir:
                raise ValueError(f"Anti-leakage violation: {msg}")
            warnings.warn(msg, UserWarning, stacklevel=2)

            if clip_to_nadir:
                idx_nadir = int(np.searchsorted(timestamps, t_nadir, side="left"))
                idx_end = min(idx_end, max(idx_start + 2, idx_nadir))
    elif require_pre_nadir:
        raise ValueError(
            "Anti-leakage violation: 'ground_truth_t_nadir' is required in metadata "
            "when require_pre_nadir=True"
        )

    ts_slice = timestamps[idx_start:idx_end].copy()

    required_bus_keys = ["bus_frequencies", "voltage_magnitudes", "voltage_angles"]
    sliced_arrays: Dict[str, np.ndarray] = {}
    for key in required_bus_keys:
        if key in data:
            sliced_arrays[key] = np.asarray(data[key], dtype=float)[idx_start:idx_end].copy()
        else:
            raise KeyError(f"Scenario data missing required array '{key}'")

    eff_fps = float(metadata.get("fps", 1.0 / max(1e-6, float(ts_slice[1] - ts_slice[0]))))
    if "bus_rocof" in data:
        sliced_arrays["bus_rocof"] = np.asarray(data["bus_rocof"], dtype=float)[idx_start:idx_end].copy()
    else:
        sliced_arrays["bus_rocof"] = compute_causal_rocof(
            sliced_arrays["bus_frequencies"], fps=eff_fps, window_length=5
        )

    pmu_sys_slice = estimate_pmu_system_frequency(sliced_arrays["bus_frequencies"])
    coi_slice = pmu_sys_slice.copy()

    extra_channels: Dict[str, Any] = {}
    handled_keys = (
        set(required_bus_keys)
        | {"timestamps", "metadata", "bus_rocof", "pmu_sys_frequency", "coi_frequency"}
        | FORBIDDEN_WINDOW_CHANNEL_KEYS
    )
    for key, val in data.items():
        if key in handled_keys:
            continue
        if isinstance(val, np.ndarray) and val.ndim >= 1 and len(val) == len(timestamps):
            extra_channels[key] = val[idx_start:idx_end].copy()
        else:
            extra_channels[key] = val

    sliced_meta = dict(metadata)
    sliced_meta["t_obs"] = float(t_obs)
    sliced_meta["t_0"] = float(t_0)
    sliced_meta["idx_start"] = idx_start
    sliced_meta["idx_end"] = idx_end
    sliced_meta["n_samples"] = idx_end - idx_start
    sliced_meta["is_prior_to_nadir"] = bool(is_prior_to_nadir)
    sliced_meta["lead_time_s"] = float(lead_time_s)
    sliced_meta["is_truncated"] = bool(is_truncated)

    return CausalWindow(
        timestamps=ts_slice,
        bus_frequencies=sliced_arrays["bus_frequencies"],
        bus_rocof=sliced_arrays["bus_rocof"],
        voltage_magnitudes=sliced_arrays["voltage_magnitudes"],
        voltage_angles=sliced_arrays["voltage_angles"],
        coi_frequency=coi_slice,
        pmu_sys_frequency=pmu_sys_slice,
        metadata=sliced_meta,
        **extra_channels,
    )


def slice_scenario_batch(
    scenarios: List[Union[Dict[str, Any], str, Path, np.lib.npyio.NpzFile]],
    t_obs: float,
    strict_nadir: bool = False,
    clip_to_nadir: bool = False,
    require_pre_nadir: bool = False,
) -> List[CausalWindow]:
    """Slice a list of scenarios to a uniform observation window duration t_obs."""
    windows: List[CausalWindow] = []
    for s in scenarios:
        w = extract_causal_window(
            s,
            t_obs=t_obs,
            strict_nadir=strict_nadir,
            clip_to_nadir=clip_to_nadir,
            require_pre_nadir=require_pre_nadir,
        )
        windows.append(w)
    return windows
