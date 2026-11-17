"""Simplified IEEE C37.118 synchrophasor resampling, causal ROCOF, noise injection, and framing."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from pmu_fsa.simulation.ieee39_data import AREA_PARTITION


CANONICAL_13BUS_PMU_PLACEMENT: List[int] = [1, 2, 4, 5, 6, 9, 14, 16, 17, 19, 22, 25, 39]


def compute_crc_ccitt(data: bytes) -> int:
    """Compute CRC-CCITT checksum (poly 0x1021, init 0xFFFF) over input bytes."""
    crc = 0xFFFF
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


@dataclass
class PMUNoiseConfig:
    """Gaussian noise standard deviations for synthetic PMU measurement channels."""

    sigma_f: float = 0.005  # Hz
    sigma_rocof: float = 0.02  # Hz/s
    sigma_v: float = 0.0005  # pu
    sigma_theta_deg: float = 0.05  # deg
    seed: Optional[int] = None
    f_noise_std: Optional[float] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.f_noise_std is not None:
            self.sigma_f = float(self.f_noise_std)
        else:
            self.sigma_f = float(self.sigma_f)
        self.sigma_rocof = float(self.sigma_rocof)
        self.sigma_v = float(self.sigma_v)
        self.sigma_theta_deg = float(self.sigma_theta_deg)
        if (
            self.sigma_f < 0.0
            or self.sigma_rocof < 0.0
            or self.sigma_v < 0.0
            or self.sigma_theta_deg < 0.0
        ):
            raise ValueError("PMU noise standard deviations must be non-negative")
        self.f_noise_std = self.sigma_f

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sigma_f": self.sigma_f,
            "sigma_rocof": self.sigma_rocof,
            "sigma_v": self.sigma_v,
            "sigma_theta_deg": self.sigma_theta_deg,
            "seed": self.seed,
        }


def _wrap_deg_180(angles_deg: np.ndarray) -> np.ndarray:
    """Wrap angles in degrees to [-180.0, 180.0)."""
    return ((angles_deg + 180.0) % 360.0) - 180.0


def compute_causal_rocof(
    frequencies: np.ndarray,
    fps: float = 50.0,
    window_length: int = 5,
) -> np.ndarray:
    """Estimate ROCOF causally via closed-form OLS linear slope over past/present samples only."""
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")
    freq = np.asarray(frequencies, dtype=float)
    n_samples = freq.shape[0]
    rocof = np.zeros_like(freq, dtype=float)

    if n_samples == 0:
        return rocof

    if not np.isfinite(np.sum(freq[0])):
        rocof[0] = np.where(np.isnan(freq[0]), np.nan, 0.0)

    if n_samples == 1:
        return rocof

    inv_dt = float(fps)
    w_len = max(2, int(window_length))

    # n == 1: 2-sample backward difference (causal m=2 OLS slope)
    rocof[1] = (freq[1] - freq[0]) * inv_dt

    if w_len == 5:
        if n_samples >= 3:
            rocof[2] = (freq[2] - freq[0]) * (0.5 * inv_dt)
        if n_samples >= 4:
            rocof[3] = (-3.0 * freq[0] - freq[1] + freq[2] + 3.0 * freq[3]) * (0.1 * inv_dt)
        if n_samples >= 5:
            out = rocof[4:]
            np.subtract(freq[4:], freq[:-4], out=out)
            out *= 2.0
            out += freq[3:-1]
            out -= freq[1:-3]
            out *= 0.1 * inv_dt
        return rocof

    dt = 1.0 / inv_dt
    warmup_end = min(w_len - 1, n_samples)
    for n in range(2, warmup_end):
        m = n + 1
        k = np.arange(m, dtype=float)
        denom = (m * (m * m - 1.0) / 12.0) * dt
        weights = (k - 0.5 * (m - 1.0)) / denom
        if freq.ndim == 1:
            rocof[n] = float(np.dot(weights, freq[:m]))
        else:
            rocof[n] = weights @ freq[:m]

    if n_samples >= w_len:
        k = np.arange(w_len, dtype=float)
        denom = (w_len * (w_len * w_len - 1.0) / 12.0) * dt
        weights = (k - 0.5 * (w_len - 1.0)) / denom
        span = n_samples - w_len + 1
        acc = np.zeros_like(freq[w_len - 1 :], dtype=float)
        for j in range(w_len):
            acc = acc + weights[j] * freq[j : j + span]
        rocof[w_len - 1 :] = acc

    return rocof


def compute_fir_rocof(
    frequencies: np.ndarray,
    fps: float = 50.0,
    window_length: int = 5,
) -> np.ndarray:
    """Causal FIR OLS differentiator alias (zero future lookahead)."""
    return compute_causal_rocof(frequencies, fps=fps, window_length=window_length)


def _safe_nanmean_axis1(sub: np.ndarray) -> np.ndarray:
    """Warning-free row-wise mean ignoring NaNs (returns NaN on all-NaN rows)."""
    if np.isfinite(np.sum(sub)):
        return np.sum(sub, axis=1) / float(sub.shape[1])
    valid = ~np.isnan(sub)
    counts = np.sum(valid, axis=1)
    safe_vals = np.where(valid, sub, 0.0)
    return np.where(counts > 0, np.sum(safe_vals, axis=1) / np.maximum(counts, 1), np.nan)


def estimate_pmu_system_frequency(
    bus_frequencies: np.ndarray,
    observed_bus_indices: Optional[np.ndarray] = None,
    area_partition: Optional[Dict[int, List[int]]] = None,
) -> np.ndarray:
    """Reconstruct wide-area system frequency strictly from measured PMU bus frequencies."""
    bf = np.asarray(bus_frequencies, dtype=float)
    if bf.ndim == 1:
        return bf.copy()

    n_samples, n_buses = bf.shape
    if n_samples == 0 or n_buses == 0:
        return np.zeros(n_samples, dtype=float)

    # Fast exact path when all 39 canonical buses are observed and finite:
    # (9*mean(A1) + 15*mean(A2) + 15*mean(A3)) / 39 == sum(bf, axis=1) / 39
    if (
        n_buses == 39
        and (area_partition is None or area_partition is AREA_PARTITION)
        and (observed_bus_indices is None or len(observed_bus_indices) == 39)
        and np.isfinite(np.sum(bf))
    ):
        return np.sum(bf, axis=1) * (1.0 / 39.0)

    obs_mask = ~np.all(np.isnan(bf), axis=0)
    if observed_bus_indices is not None:
        idx_arr = np.asarray(observed_bus_indices, dtype=np.intp).ravel()
        explicit_mask = np.zeros(n_buses, dtype=bool)
        valid_idx = idx_arr[(idx_arr >= 0) & (idx_arr < n_buses)]
        explicit_mask[valid_idx] = True
        obs_mask = obs_mask & explicit_mask

    if not np.any(obs_mask):
        return np.full(n_samples, np.nan, dtype=float)

    partition = area_partition if area_partition is not None else AREA_PARTITION
    area_entries: List[Tuple[float, List[int]]] = []
    for area_id in sorted(partition.keys()):
        full_cols = [b - 1 for b in partition[area_id] if 0 <= b - 1 < n_buses]
        obs_cols = [c for c in full_cols if obs_mask[c]]
        if obs_cols:
            area_entries.append((float(len(full_cols)), obs_cols))

    if len(area_entries) >= 3:
        total_weight = sum(w for w, _ in area_entries)
        f_sys = np.zeros(n_samples, dtype=float)
        for weight, cols in area_entries:
            area_mean = _safe_nanmean_axis1(bf[:, cols])
            f_sys += (weight / total_weight) * area_mean
        return f_sys

    return _safe_nanmean_axis1(bf[:, obs_mask])


def apply_pmu_measurement_noise(
    bus_frequencies: np.ndarray,
    bus_rocof: np.ndarray,
    voltage_magnitudes: np.ndarray,
    voltage_angles: np.ndarray,
    noise_config: Optional[PMUNoiseConfig] = None,
    seed_offset: int = 0,
    fps: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Add zero-mean Gaussian noise to PMU bus frequency, ROCOF, voltage magnitude, and angle arrays."""
    cfg = noise_config if noise_config is not None else PMUNoiseConfig()
    eff_seed = None if cfg.seed is None else (int(cfg.seed) + int(seed_offset))
    rng = np.random.default_rng(eff_seed)

    bf = np.asarray(bus_frequencies, dtype=float)
    br = np.asarray(bus_rocof, dtype=float)
    vm = np.asarray(voltage_magnitudes, dtype=float)
    va = np.asarray(voltage_angles, dtype=float)

    f_noisy = bf + rng.normal(0.0, cfg.sigma_f, size=bf.shape) if cfg.sigma_f > 0.0 else bf.copy()
    v_noisy = vm + rng.normal(0.0, cfg.sigma_v, size=vm.shape) if cfg.sigma_v > 0.0 else vm.copy()
    v_noisy = np.where(np.isnan(v_noisy), np.nan, np.maximum(0.01, v_noisy))

    if cfg.sigma_theta_deg > 0.0:
        theta_noisy = _wrap_deg_180(va + rng.normal(0.0, cfg.sigma_theta_deg, size=va.shape))
    else:
        theta_noisy = va.copy()

    if fps is not None:
        rocof_noisy = compute_causal_rocof(f_noisy, fps=fps, window_length=5)
    elif cfg.sigma_rocof > 0.0:
        rocof_noisy = br + rng.normal(0.0, cfg.sigma_rocof, size=br.shape)
    else:
        rocof_noisy = br.copy()

    return f_noisy, rocof_noisy, v_noisy, theta_noisy


def compute_tve(
    v_mag_true: np.ndarray,
    v_ang_deg_true: np.ndarray,
    v_mag_meas: np.ndarray,
    v_ang_deg_meas: np.ndarray,
) -> np.ndarray:
    """Compute IEEE C37.118.1 Total Vector Error |V_meas - V_true| / |V_true| as a fraction."""
    ang_rad_true = np.radians(v_ang_deg_true)
    ang_rad_meas = np.radians(v_ang_deg_meas)

    v_true_complex = v_mag_true * np.exp(1j * ang_rad_true)
    v_meas_complex = v_mag_meas * np.exp(1j * ang_rad_meas)

    diff = np.abs(v_meas_complex - v_true_complex)
    mag_true = np.maximum(1e-6, np.abs(v_true_complex))
    return diff / mag_true


class PMURecorder:
    """Resamples dynamic trajectories into synchrophasor streams and simplified C37.118.2 data frames."""

    def __init__(
        self,
        fps: int = 50,
        noise_config: Optional[PMUNoiseConfig] = None,
        base_epoch_seconds: int = 1700000000,
        id_code: int = 1,
        f_nominal: float = 50.0,
    ) -> None:
        supported_fps = (10, 25, 30, 50, 60, 100, 120)
        if fps not in supported_fps:
            raise ValueError(f"PMU reporting rate must be one of {supported_fps} frames/s, got {fps}")
        self.fps = int(fps)
        self.dt = 1.0 / float(self.fps)
        self.noise_config = noise_config
        self.base_epoch_seconds = int(base_epoch_seconds)
        self.id_code = int(id_code)
        self.f_nominal = float(f_nominal)

    def record_trajectory(
        self,
        raw_trajectory: Dict[str, Any],
        inject_noise: bool = False,
        seed_offset: int = 0,
        observed_buses: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """Resample bus channels onto uniform fps timestamps, optionally add noise and PMU bus masking."""
        orig_t = np.asarray(raw_trajectory["timestamps"], dtype=float)
        t_start = float(orig_t[0])
        t_end = float(orig_t[-1])

        uniform_timestamps = np.arange(t_start, t_end + 0.5 * self.dt, self.dt)
        n_samples = len(uniform_timestamps)

        raw_bf = np.asarray(raw_trajectory["bus_frequencies"], dtype=float)
        raw_vm = np.asarray(raw_trajectory["voltage_magnitudes"], dtype=float)
        raw_va_deg = np.asarray(raw_trajectory["voltage_angles"], dtype=float)
        n_buses = raw_bf.shape[1]

        # Unwrap phase angles prior to linear interpolation so +/-180 deg crossings never glitch
        raw_va_unwrapped_deg = np.degrees(np.unwrap(np.radians(raw_va_deg), axis=0))

        bus_freq = np.empty((n_samples, n_buses), dtype=float)
        v_mag = np.empty((n_samples, n_buses), dtype=float)
        v_ang = np.empty((n_samples, n_buses), dtype=float)

        for b in range(n_buses):
            bus_freq[:, b] = np.interp(uniform_timestamps, orig_t, raw_bf[:, b])
            v_mag[:, b] = np.interp(uniform_timestamps, orig_t, raw_vm[:, b])
            v_ang[:, b] = np.interp(uniform_timestamps, orig_t, raw_va_unwrapped_deg[:, b])
        v_ang = _wrap_deg_180(v_ang)

        cfg: Optional[PMUNoiseConfig] = None
        if inject_noise:
            cfg = self.noise_config if self.noise_config is not None else PMUNoiseConfig()
            eff_seed = None if cfg.seed is None else (int(cfg.seed) + int(seed_offset))
            rng = np.random.default_rng(eff_seed)
            if cfg.sigma_f > 0.0:
                bus_freq = bus_freq + rng.normal(0.0, cfg.sigma_f, size=bus_freq.shape)
            if cfg.sigma_v > 0.0:
                v_mag = np.maximum(0.01, v_mag + rng.normal(0.0, cfg.sigma_v, size=v_mag.shape))
            if cfg.sigma_theta_deg > 0.0:
                v_ang = _wrap_deg_180(v_ang + rng.normal(0.0, cfg.sigma_theta_deg, size=v_ang.shape))

        # Identical causal 5-point OLS ROCOF estimator across both clean and noisy streams
        bus_rocof = compute_causal_rocof(bus_freq, fps=self.fps, window_length=5)

        observed_idx: Optional[np.ndarray] = None
        if observed_buses is not None:
            valid_zero_based = sorted({int(b) - 1 for b in observed_buses if 1 <= int(b) <= n_buses})
            observed_idx = np.array(valid_zero_based, dtype=np.intp)
            keep_mask = np.zeros(n_buses, dtype=bool)
            if len(observed_idx) > 0:
                keep_mask[observed_idx] = True
            unobs_mask = ~keep_mask
            bus_freq[:, unobs_mask] = np.nan
            bus_rocof[:, unobs_mask] = np.nan
            v_mag[:, unobs_mask] = np.nan
            v_ang[:, unobs_mask] = np.nan

        pmu_sys_frequency = estimate_pmu_system_frequency(
            bus_freq, observed_bus_indices=observed_idx
        )

        time_base = 1_000_000
        total_secs = self.base_epoch_seconds + uniform_timestamps
        soc_floor = np.floor(total_secs)
        soc_array = soc_floor.astype(np.uint32)
        frac_counts = np.round((total_secs - soc_floor) * time_base).astype(np.uint32)
        fracsec_array = frac_counts & np.uint32(0x00FFFFFF)

        metadata = dict(raw_trajectory.get("metadata", {}))
        metadata["fps"] = self.fps
        metadata["is_noisy"] = bool(inject_noise)
        metadata["freq_estimator"] = "pmu_area_weighted_bus_mean"
        if cfg is not None:
            metadata["noise_config"] = cfg.to_dict()
            metadata["noise_seed_offset"] = int(seed_offset)
        if observed_idx is not None:
            metadata["observed_buses"] = [int(i + 1) for i in observed_idx]

        return {
            "timestamps": uniform_timestamps,
            "soc": soc_array,
            "fracsec": fracsec_array,
            "bus_frequencies": bus_freq,
            "bus_rocof": bus_rocof,
            "voltage_magnitudes": v_mag,
            "voltage_angles": v_ang,
            "pmu_sys_frequency": pmu_sys_frequency,
            "coi_frequency": pmu_sys_frequency.copy(),
            "metadata": metadata,
        }

    def serialize_frame(
        self,
        frame_idx: int,
        pmu_stream: Dict[str, Any],
        id_code: Optional[int] = None,
    ) -> bytes:
        """Pack one timestamp into a simplified single-station IEEE C37.118.2-style binary data frame.

        Encodes the 16-byte C37.118.2 header (SYNC=0xAA01, FRAMESIZE, IDCODE, SOC, FRACSEC, STAT),
        32-bit float polar phasors (V_mag_pu, theta_rad), wide-area PMU FREQ deviation (Hz) and
        causal DFREQ (Hz/s), and a 16-bit CRC-CCITT trailer.
        """
        pmu_id = int(id_code if id_code is not None else self.id_code)
        soc = int(pmu_stream["soc"][frame_idx])
        fracsec = int(pmu_stream["fracsec"][frame_idx])
        stat = 0x0000

        meta = pmu_stream.get("metadata", {})
        f_nom = float(meta.get("f_nominal", self.f_nominal))
        eff_fps = float(meta.get("fps", self.fps))

        if "pmu_sys_frequency" in pmu_stream:
            f_sys_series = np.asarray(pmu_stream["pmu_sys_frequency"], dtype=float)
        elif "bus_frequencies" in pmu_stream:
            f_sys_series = estimate_pmu_system_frequency(pmu_stream["bus_frequencies"])
        else:
            f_sys_series = np.asarray(pmu_stream["coi_frequency"], dtype=float)

        freq_hz = float(f_sys_series[frame_idx])
        freq_dev = freq_hz - f_nom

        start_idx = max(0, frame_idx - 4)
        dfreq = float(
            compute_causal_rocof(
                f_sys_series[start_idx : frame_idx + 1],
                fps=eff_fps,
                window_length=5,
            )[-1]
        )

        header = struct.pack(">HHHIIH", 0xAA01, 0, pmu_id, soc, fracsec, stat)

        phasor_bytes = bytearray()
        n_buses = pmu_stream["voltage_magnitudes"].shape[1]
        for b in range(n_buses):
            v_m = float(pmu_stream["voltage_magnitudes"][frame_idx, b])
            v_a_rad = float(np.radians(pmu_stream["voltage_angles"][frame_idx, b]))
            phasor_bytes.extend(struct.pack(">ff", v_m, v_a_rad))

        freq_bytes = struct.pack(">ff", freq_dev, dfreq)

        frame_payload = bytearray(header) + phasor_bytes + freq_bytes
        total_frame_size = len(frame_payload) + 2
        struct.pack_into(">H", frame_payload, 2, total_frame_size)

        chk = compute_crc_ccitt(bytes(frame_payload))
        return bytes(frame_payload) + struct.pack(">H", chk)

    @staticmethod
    def parse_frame(frame_bytes: bytes, f_nominal: float = 50.0) -> Dict[str, Any]:
        """Unpack and CRC-verify a simplified single-station IEEE C37.118.2-style binary data frame."""
        if len(frame_bytes) < 18:
            raise ValueError(f"Frame length {len(frame_bytes)} too short for C37.118 header and CRC")

        payload = frame_bytes[:-2]
        if len(payload) < 24:
            raise ValueError("Corrupted or truncated PMU frame")

        try:
            received_chk = struct.unpack(">H", frame_bytes[-2:])[0]
            calculated_chk = compute_crc_ccitt(payload)
            if received_chk != calculated_chk:
                raise ValueError(
                    f"CRC-CCITT mismatch: received 0x{received_chk:04X}, computed 0x{calculated_chk:04X}"
                )

            sync, frame_size, id_code, soc, fracsec, stat = struct.unpack(">HHHIIH", payload[:16])
            if sync != 0xAA01:
                raise ValueError(f"Invalid SYNC word: 0x{sync:04X}, expected 0xAA01")

            offset = 16
            n_phasors = (len(payload) - 16 - 8) // 8
            phasors = []
            for _ in range(n_phasors):
                mag, ang = struct.unpack(">ff", payload[offset : offset + 8])
                phasors.append((mag, float(np.degrees(ang))))
                offset += 8

            if len(payload) < offset + 8:
                raise ValueError("Corrupted or truncated PMU frame")

            freq_dev, dfreq = struct.unpack(">ff", payload[offset : offset + 8])
            freq_hz = float(f_nominal) + freq_dev
        except struct.error as e:
            raise ValueError("Corrupted or truncated PMU frame") from e

        return {
            "sync": sync,
            "frame_size": frame_size,
            "id_code": id_code,
            "soc": soc,
            "fracsec": fracsec,
            "stat": stat,
            "n_phasors": n_phasors,
            "phasors": phasors,
            "freq_hz": freq_hz,
            "rocof": dfreq,
            "crc_valid": True,
        }
