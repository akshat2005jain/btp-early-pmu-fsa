"""Post-disturbance PMU observation window slicing."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class CausalWindow(dict):
    """Dict-compatible container for a sliced [t_0, t_0 + t_obs] PMU observation window."""

    def __init__(
        self,
        timestamps: np.ndarray,
        bus_frequencies: np.ndarray,
        bus_rocof: np.ndarray,
        voltage_magnitudes: np.ndarray,
        voltage_angles: np.ndarray,
        coi_frequency: np.ndarray,
        metadata: Dict[str, Any],
        **extra_channels: Any,
    ) -> None:
        super().__init__()
        self["timestamps"] = np.asarray(timestamps)
        self["bus_frequencies"] = np.asarray(bus_frequencies)
        self["bus_rocof"] = np.asarray(bus_rocof)
        self["voltage_magnitudes"] = np.asarray(voltage_magnitudes)
        self["voltage_angles"] = np.asarray(voltage_angles)
        self["coi_frequency"] = np.asarray(coi_frequency)
        self["metadata"] = dict(metadata)
        for k, v in extra_channels.items():
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
        npz = np.load(scenario_data, allow_pickle=True)
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
) -> CausalWindow:
    """Slice scenario channels to the post-disturbance window [t_0, t_0 + t_obs]."""
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

    t_end = t_0 + t_obs

    # +/-1e-9 tolerance prevents floating-point boundary drops on uniform grids
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
    t_nadir = metadata.get("ground_truth_t_nadir", None)
    if t_nadir is not None:
        t_nadir = float(t_nadir)
        if t_end >= t_nadir - 1e-6:
            is_prior_to_nadir = False
            msg = (
                f"Observation window [{t_0:.4f}s, {t_end:.4f}s] reaches or exceeds "
                f"nadir time {t_nadir:.4f}s. Pre-nadir causality warning."
            )
            if strict_nadir:
                raise ValueError(f"Anti-leakage violation: {msg}")
            warnings.warn(msg, UserWarning, stacklevel=2)

            if clip_to_nadir:
                idx_nadir = int(np.searchsorted(timestamps, t_nadir, side="left"))
                idx_end = min(idx_end, max(idx_start + 2, idx_nadir))

    ts_slice = timestamps[idx_start:idx_end].copy()

    required_keys = [
        "bus_frequencies",
        "bus_rocof",
        "voltage_magnitudes",
        "voltage_angles",
        "coi_frequency",
    ]
    sliced_arrays: Dict[str, np.ndarray] = {}
    for key in required_keys:
        if key in data:
            sliced_arrays[key] = np.asarray(data[key])[idx_start:idx_end].copy()
        else:
            raise KeyError(f"Scenario data missing required array '{key}'")

    extra_channels: Dict[str, Any] = {}
    standard_keys = set(required_keys) | {"timestamps", "metadata"}
    for key, val in data.items():
        if key not in standard_keys:
            if isinstance(val, np.ndarray) and len(val) == len(timestamps):
                extra_channels[key] = val[idx_start:idx_end].copy()
            else:
                extra_channels[key] = val

    sliced_meta = dict(metadata)
    sliced_meta["t_obs"] = t_obs
    sliced_meta["t_0"] = t_0
    sliced_meta["idx_start"] = idx_start
    sliced_meta["idx_end"] = idx_end
    sliced_meta["n_samples"] = idx_end - idx_start
    sliced_meta["is_prior_to_nadir"] = is_prior_to_nadir
    sliced_meta["is_truncated"] = is_truncated

    return CausalWindow(
        timestamps=ts_slice,
        bus_frequencies=sliced_arrays["bus_frequencies"],
        bus_rocof=sliced_arrays["bus_rocof"],
        voltage_magnitudes=sliced_arrays["voltage_magnitudes"],
        voltage_angles=sliced_arrays["voltage_angles"],
        coi_frequency=sliced_arrays["coi_frequency"],
        metadata=sliced_meta,
        **extra_channels,
    )


def slice_scenario_batch(
    scenarios: List[Union[Dict[str, Any], str, Path, np.lib.npyio.NpzFile]],
    t_obs: float,
    strict_nadir: bool = False,
    clip_to_nadir: bool = False,
) -> List[CausalWindow]:
    """Slice a list of scenarios to a uniform observation window duration t_obs."""
    windows: List[CausalWindow] = []
    for s in scenarios:
        w = extract_causal_window(
            s,
            t_obs=t_obs,
            strict_nadir=strict_nadir,
            clip_to_nadir=clip_to_nadir,
        )
        windows.append(w)
    return windows
