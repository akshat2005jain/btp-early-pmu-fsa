#!/usr/bin/env python3
"""Generate all 13 publication figures (PNG + PDF) from real experimental outputs."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import numpy as np
import pandas as pd

from pmu_fsa.models.calibration_metrics import compute_calibration_metrics
from pmu_fsa.simulation.ieee39_data import (
    AREA_PARTITION,
    BUS_TO_AREA,
    LOAD_DATA,
    NOMINAL_FREQ,
    SECURITY_THRESHOLD_HZ,
    get_ieee39_branch_data,
    get_ieee39_bus_data,
    get_ieee39_gen_data,
    get_total_system_inertia,
)
from pmu_fsa.simulation.pmu_recorder import (
    CANONICAL_13BUS_PMU_PLACEMENT,
    PMUNoiseConfig,
    PMURecorder,
    compute_causal_rocof,
)
from pmu_fsa.simulation.scenario_generator import load_scenario_from_npz

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "mathtext.fontset": "dejavusans",
    "axes.edgecolor": "#94A3B8",
    "axes.linewidth": 0.9,
    "axes.titlesize": 10.4,
    "axes.labelsize": 9.2,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
    "grid.color": "#E2E8F0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PALETTE = {
    "navy": "#0F172A",
    "slate": "#475569",
    "muted": "#94A3B8",
    "blue": "#1D4ED8",
    "sky": "#0284C7",
    "teal": "#0D9488",
    "green": "#15803D",
    "emerald": "#059669",
    "amber": "#D97706",
    "orange": "#EA580C",
    "red": "#DC2626",
    "crimson": "#991B1B",
    "violet": "#7C3AED",
}

AREA_COLORS = {
    1: "#E11D48",  # Area 1 (South)
    2: "#2563EB",  # Area 2 (Central)
    3: "#059669",  # Area 3 (North)
}

FEATURE_LABELS = {
    "rocof_init": "Initial ROCOF (0–100 ms)",
    "rocof_peak_neg": "Peak Negative ROCOF",
    "rocof_end": "Window-End ROCOF",
    "rocof_mean": "Window-Mean ROCOF",
    "freq_drop_end": "Freq. Drop at T_obs",
    "freq_drop_max": "Max Window Freq. Drop",
    "freq_dev_integral": "Freq. Deviation Integral",
    "bus_drop_worst": "Worst Bus Freq. Drop",
    "area_freq_spread_max": "Max 3-Area Freq. Spread",
    "bus_freq_std_mean": "Mean Cross-Bus Freq. Std",
    "angle_divergence_max": "Max Angle Divergence",
    "tie_line_angle_excursion": "Tie-Line Angle Excursion",
    "freq_curvature_init": "Initial Freq. Curvature",
    "freq_curvature_mean": "Mean Freq. Curvature",
}

FEATURE_SHORT = {
    "rocof_init": "ROCOF_0",
    "rocof_peak_neg": "ROCOF_min",
    "rocof_end": "ROCOF_end",
    "rocof_mean": "ROCOF_avg",
    "freq_drop_end": "df_end",
    "freq_drop_max": "df_max",
    "freq_dev_integral": "int_df",
    "bus_drop_worst": "df_bus",
    "area_freq_spread_max": "dF_area",
    "bus_freq_std_mean": "std_f",
    "angle_divergence_max": "dAng_max",
    "tie_line_angle_excursion": "dAng_tie",
    "freq_curvature_init": "curv_0",
    "freq_curvature_mean": "curv_avg",
}

FEATURE_FAMILY = {
    "rocof_init": ("ROCOF Kinematics", "#2563EB"),
    "rocof_peak_neg": ("ROCOF Kinematics", "#2563EB"),
    "rocof_end": ("ROCOF Kinematics", "#2563EB"),
    "rocof_mean": ("ROCOF Kinematics", "#2563EB"),
    "freq_drop_end": ("Frequency Excursion", "#059669"),
    "freq_drop_max": ("Frequency Excursion", "#059669"),
    "freq_dev_integral": ("Frequency Excursion", "#059669"),
    "bus_drop_worst": ("Spatial & Angle", "#D97706"),
    "area_freq_spread_max": ("Spatial & Angle", "#D97706"),
    "bus_freq_std_mean": ("Spatial & Angle", "#D97706"),
    "angle_divergence_max": ("Spatial & Angle", "#D97706"),
    "tie_line_angle_excursion": ("Spatial & Angle", "#D97706"),
    "freq_curvature_init": ("2nd-Order Curvature", "#7C3AED"),
    "freq_curvature_mean": ("2nd-Order Curvature", "#7C3AED"),
}

IEEE39_COORDS: Dict[int, Tuple[float, float]] = {
    39: (1.0, 4.9),
    1: (1.7, 6.9),
    2: (3.2, 7.5),
    30: (2.0, 8.7),
    3: (3.7, 6.0),
    18: (5.2, 6.2),
    4: (4.2, 4.5),
    25: (4.9, 8.1),
    37: (4.3, 9.3),
    9: (1.8, 3.0),
    8: (3.1, 2.3),
    7: (3.0, 1.0),
    5: (4.3, 2.8),
    6: (4.5, 1.5),
    31: (4.5, 0.3),
    11: (5.8, 1.5),
    12: (6.6, 0.5),
    10: (7.2, 1.5),
    32: (8.4, 0.5),
    13: (7.5, 2.7),
    14: (6.1, 3.8),
    15: (7.4, 4.3),
    16: (8.3, 5.4),
    19: (9.9, 4.4),
    33: (11.4, 4.6),
    20: (10.2, 2.9),
    34: (11.6, 2.4),
    21: (9.8, 5.9),
    22: (11.1, 6.2),
    35: (12.5, 6.1),
    24: (9.2, 7.0),
    23: (10.7, 7.4),
    36: (12.2, 7.8),
    17: (6.7, 6.3),
    27: (7.5, 7.3),
    26: (6.8, 8.4),
    28: (8.4, 9.4),
    29: (9.6, 8.5),
    38: (11.2, 9.2),
}


def save_dual(fig: plt.Figure, out_dir: Path, stem: str, dpi: int = 240) -> Tuple[Path, Path]:
    png_p = out_dir / f"{stem}.png"
    pdf_p = out_dir / f"{stem}.pdf"
    fig.savefig(png_p, dpi=dpi, bbox_inches="tight", pad_inches=0.10, facecolor="white")
    fig.savefig(pdf_p, bbox_inches="tight", pad_inches=0.10, facecolor="white")
    plt.close(fig)
    return png_p, pdf_p


def _load_raw_sim_npz(fp: Path) -> Dict[str, Any]:
    with np.load(fp, allow_pickle=True) as data:
        raw_meta = data["metadata"]
        meta = raw_meta.item() if hasattr(raw_meta, "item") else dict(raw_meta)
        return {
            "timestamps": np.asarray(data["timestamps"], dtype=float),
            "coi_frequency": np.asarray(data["coi_frequency"], dtype=float),
            "coi_rocof": np.asarray(data["coi_rocof"], dtype=float) if "coi_rocof" in data.files else None,
            "bus_frequencies_true": np.asarray(data["bus_frequencies_true"], dtype=float),
            "voltage_angles_true": np.asarray(data["voltage_angles_true"], dtype=float),
            "metadata": meta,
        }


def _index_scenarios(raw_dir: Path, raw_sim_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for pmu_fp in sorted(raw_dir.glob("*.npz")):
        with np.load(pmu_fp, allow_pickle=True) as d:
            rm = d["metadata"]
            meta = rm.item() if hasattr(rm, "item") else dict(rm)
        scen_id = str(meta.get("scenario_id", pmu_fp.stem))
        sim_fp = raw_sim_dir / f"{scen_id}.npz"
        records.append({
            "scenario_id": scen_id,
            "pmu_path": pmu_fp,
            "sim_path": sim_fp if sim_fp.exists() else None,
            "disturbed_gen": int(meta.get("disturbed_gen", 0)),
            "loss_fraction": round(float(meta.get("loss_fraction", 1.0)), 2),
            "active_loss_mw": float(meta.get("active_loss_mw", 0.0)),
            "renewable_penetration": round(float(meta.get("renewable_penetration", 0.0)), 2),
            "p_ren_realized": float(meta.get("p_ren_realized", meta.get("renewable_penetration", 0.0))),
            "load_scale": round(float(meta.get("load_scale", 1.0)), 2),
            "nadir": float(meta.get("ground_truth_nadir", 50.0)),
            "t_nadir": float(meta.get("ground_truth_t_nadir", 3.2)),
            "is_secure": int(meta.get("is_secure", 1)),
            "partition": str(meta.get("partition", "train")),
        })
    return records


def _compute_area_frequencies(bus_freqs: np.ndarray) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for area_id, buses in AREA_PARTITION.items():
        cols = [b - 1 for b in buses if 1 <= b <= bus_freqs.shape[1]]
        out[area_id] = np.nanmean(bus_freqs[:, cols], axis=1)
    return out


def _parse_ci_string(val: Any) -> Tuple[float, float]:
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(val))
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return float("nan"), float("nan")


# ---------------------------------------------------------------------------
# Figure 1: IEEE 39-Bus 3-Area Topology + Causal PMU-to-Decision Architecture
# ---------------------------------------------------------------------------
def plot_fig01_system_and_architecture(
    out_dir: Path,
    feat_path: Path,
    models_dir: Path,
    tables_dir: Path,
    dpi: int = 240,
) -> Tuple[Path, Path]:
    buses = get_ieee39_bus_data()
    branches = get_ieee39_branch_data()
    gens = get_ieee39_gen_data(macro_intertie=False)
    gen_by_bus = {int(g["bus"]): g for g in gens}
    h_tot = get_total_system_inertia(macro_intertie=False)
    pmu_set = set(CANONICAL_13BUS_PMU_PLACEMENT)

    tm_path = models_dir / "training_metrics.json"
    tm = json.loads(tm_path.read_text(encoding="utf-8")) if tm_path.exists() else {}
    p_counts = tm.get("partition_counts", {})
    n_train = p_counts.get("train", 0)
    n_val = p_counts.get("val", 0)
    n_test = p_counts.get("test", 0)
    n_ood = p_counts.get("ood_test", 0)
    n_total = n_train + n_val + n_test + n_ood

    lat_path = tables_dir / "latency_profile.csv"
    if lat_path.exists():
        df_lat = pd.read_csv(lat_path)
        t_online_ms = float(df_lat["t_total_ms"].iloc[0])
        t_p99_ms = float(df_lat["t_total_p99_ms"].iloc[0])
        t_sim_ms = float(df_lat["sim_time_ms"].iloc[0])
        speedup = float(df_lat["speedup_factor"].iloc[0])
    else:
        t_online_ms, t_p99_ms, t_sim_ms, speedup = 0.0, 0.0, 0.0, 0.0

    fig, (ax_net, ax_arch) = plt.subplots(
        1, 2, figsize=(15.8, 7.5), gridspec_kw={"width_ratios": [1.18, 1.0]}
    )

    primary_ties = {(4, 5), (5, 4), (1, 39), (39, 1), (16, 17), (17, 16)}

    for br in branches:
        u, v = int(br["from"]), int(br["to"])
        x1, y1 = IEEE39_COORDS[u]
        x2, y2 = IEEE39_COORDS[v]
        is_primary_tie = (u, v) in primary_ties
        is_xfmr = float(br["b"]) == 0.0 and float(br["r"]) < 0.002
        cross_area = BUS_TO_AREA[u] != BUS_TO_AREA[v]

        if is_primary_tie:
            ax_net.plot([x1, x2], [y1, y2], color="#F59E0B", lw=3.8, alpha=0.35, zorder=2)
            ax_net.plot([x1, x2], [y1, y2], color="#B45309", lw=2.2, ls="--", zorder=3)
        elif cross_area:
            ax_net.plot([x1, x2], [y1, y2], color="#64748B", lw=1.5, ls="-.", zorder=2)
        elif is_xfmr:
            ax_net.plot([x1, x2], [y1, y2], color="#334155", lw=1.8, ls=":", zorder=2)
        else:
            ax_net.plot([x1, x2], [y1, y2], color="#94A3B8", lw=1.4, zorder=1)

    tie_annotations = [
        ((1, 39), "Tie 1–39", (-0.88, 0.15)),
        ((4, 5), "Tie 4–5", (-0.92, 0.0)),
        ((16, 17), "Tie 16–17", (0.02, 0.50)),
    ]
    for (u, v), lbl, (dx, dy) in tie_annotations:
        mx = 0.5 * (IEEE39_COORDS[u][0] + IEEE39_COORDS[v][0])
        my = 0.5 * (IEEE39_COORDS[u][1] + IEEE39_COORDS[v][1])
        ax_net.annotate(
            lbl,
            xy=(mx, my),
            xytext=(mx + dx, my + dy),
            fontsize=7.2,
            fontweight="bold",
            color="#92400E",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.20", fc="#FEF3C7", ec="#D97706", lw=0.9),
            arrowprops=dict(arrowstyle="-", color="#D97706", lw=0.9),
            zorder=7,
        )

    gen_tag_offsets = {
        30: (-1.05, 0.15),
        31: (-1.15, -0.05),
        32: (1.15, 0.05),
        33: (1.15, 0.08),
        34: (1.15, -0.10),
        35: (1.08, 0.0),
        36: (1.05, 0.20),
        37: (1.15, 0.25),
        38: (1.15, 0.20),
        39: (-0.05, -0.62),
    }

    for b_rec in buses:
        b_id = int(b_rec["bus_id"])
        area = int(b_rec["area"])
        bx, by = IEEE39_COORDS[b_id]
        c_area = AREA_COLORS[area]
        is_gen = b_id in gen_by_bus
        has_load = b_id in LOAD_DATA
        has_pmu = b_id in pmu_set

        if has_pmu:
            ax_net.scatter(
                [bx], [by], s=325 if is_gen else 230,
                facecolors="none", edgecolors="#0EA5E9", linewidths=1.5, ls="--", zorder=4,
            )

        if has_load and not is_gen:
            ax_net.scatter([bx + 0.23], [by - 0.22], s=25, marker="v", color="#475569", alpha=0.85, zorder=4)

        node_size = 235 if is_gen else 142
        ax_net.scatter(
            [bx], [by], s=node_size,
            color=c_area if is_gen else "white",
            edgecolors=c_area,
            linewidths=2.0 if is_gen else 1.6,
            zorder=5,
        )
        ax_net.text(
            bx, by, str(b_id),
            ha="center", va="center",
            fontsize=7.1 if is_gen else 6.8,
            fontweight="bold",
            color="white" if is_gen else c_area,
            zorder=6,
        )

        if is_gen:
            g = gen_by_bus[b_id]
            ox, oy = gen_tag_offsets[b_id]
            ax_net.text(
                bx + ox, by + oy,
                f"{g['name']} ({g['P_mw']:.0f} MW)",
                ha="center", va="center",
                fontsize=6.7, fontweight="bold", color=PALETTE["navy"],
                bbox=dict(boxstyle="round,pad=0.16", fc="#F8FAFC", ec=c_area, lw=0.75, alpha=0.96),
                zorder=6,
            )

    ax_net.text(
        -0.15, 10.05, "Area 1 (South/West — 9 Buses)",
        fontsize=8.0, fontweight="bold", color=AREA_COLORS[1], ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.22", fc="#FFF1F2", ec=AREA_COLORS[1], lw=0.8),
    )
    ax_net.text(
        -0.15, -0.52, "Area 2 (Central/East — 15 Buses)",
        fontsize=8.0, fontweight="bold", color=AREA_COLORS[2], ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.22", fc="#EFF6FF", ec=AREA_COLORS[2], lw=0.8),
    )
    ax_net.text(
        7.85, 10.05, "Area 3 (North/East — 15 Buses)",
        fontsize=8.0, fontweight="bold", color=AREA_COLORS[3], ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.22", fc="#ECFDF5", ec=AREA_COLORS[3], lw=0.8),
    )

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="Area 1 Bus / Gen", markerfacecolor=AREA_COLORS[1], markersize=7.5),
        Line2D([0], [0], marker="o", color="w", label="Area 2 Bus / Gen", markerfacecolor=AREA_COLORS[2], markersize=7.5),
        Line2D([0], [0], marker="o", color="w", label="Area 3 Bus / Gen", markerfacecolor=AREA_COLORS[3], markersize=7.5),
        Line2D([0], [0], color="#B45309", lw=2.2, ls="--", label="Primary Tie-Lines (1–39, 4–5, 16–17)"),
        Line2D([0], [0], color="#334155", lw=1.8, ls=":", label="Step-Up / Tap Transformers (12)"),
        Line2D([0], [0], marker="o", color="w", label="13-Bus Sparse PMU Subset", markerfacecolor="none", markeredgecolor="#0EA5E9", markersize=8.5),
        Line2D([0], [0], marker="v", color="w", label=f"PQ Load Buses ({len(LOAD_DATA)})", markerfacecolor="#475569", markersize=6.5),
    ]
    ax_net.legend(
        handles=legend_handles, loc="lower center", bbox_to_anchor=(0.50, 0.005),
        fontsize=6.9, framealpha=0.96, ncol=3,
    )
    ax_net.set_xlim(-0.45, 14.05)
    ax_net.set_ylim(-2.15, 10.65)
    ax_net.set_xticks([])
    ax_net.set_yticks([])
    for sp in ax_net.spines.values():
        sp.set_edgecolor("#CBD5E1")
    ax_net.set_title(
        f"(a) IEEE 39-Bus, 10-Machine, 46-Branch Network ({NOMINAL_FREQ:.1f} Hz, H_tot = {h_tot:.1f} s)",
        fontsize=10.4, fontweight="bold", pad=8,
    )

    # --- Panel (b): Causal PMU-to-Decision Architecture ---
    ax_arch.set_xlim(0.0, 10.0)
    ax_arch.set_ylim(0.0, 10.0)
    ax_arch.axis("off")
    ax_arch.set_title(
        "(b) Strictly Causal Wide-Area PMU-to-Decision Screening Architecture",
        fontsize=10.4, fontweight="bold", pad=8,
    )

    def _box(x: float, y: float, w: float, h: float, title: str, body: str, fc: str, ec: str, title_c: str = "#0F172A") -> None:
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.14,rounding_size=0.22",
            facecolor=fc, edgecolor=ec, linewidth=1.35, zorder=2,
        )
        ax_arch.add_patch(patch)
        ax_arch.text(x + 0.24, y + h - 0.30, title, fontsize=8.3, fontweight="bold", color=title_c, va="top", ha="left", zorder=3)
        ax_arch.text(x + 0.24, y + h - 0.64, body, fontsize=7.3, color="#334155", va="top", ha="left", linespacing=1.28, zorder=3)

    def _arrow(x1: float, y1: float, x2: float, y2: float, color: str = "#334155") -> None:
        arr = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>,head_length=5.5,head_width=3.2",
            lw=1.4, color=color, zorder=4,
        )
        ax_arch.add_patch(arr)

    _box(
        0.4, 8.35, 9.2, 1.32,
        f"1. Contingency & 50.0 Hz Multi-Machine Grid ({n_total} Disjoint Scenarios)",
        f"• NR AC power flow + Kron ODEs; complete/partial (100/70/40%) trips on Buses 30–39 (t0=1.0 s)\n"
        f"• RES displacement 10–50% & load scaling 0.80–1.20x  |  Splits: Train={n_train}, Val={n_val}, ID Test={n_test}, OOD={n_ood}",
        "#F8FAFC", "#64748B",
    )
    _arrow(5.0, 8.32, 5.0, 7.90)

    fw = FancyBboxPatch(
        (0.4, 6.28), 9.2, 1.56,
        boxstyle="round,pad=0.12,rounding_size=0.20",
        facecolor="#ECFDF5", edgecolor="#059669", linewidth=1.6, zorder=2,
    )
    ax_arch.add_patch(fw)
    ax_arch.text(
        0.64, 7.56,
        "2. IEEE C37.118.1/.2 PMU Stream & Pre-Nadir Causal Window [t0, t0 + T_obs]",
        fontsize=8.3, fontweight="bold", color="#065F46", va="top",
    )
    ax_arch.text(
        0.64, 7.20,
        "• 50 fps bus synchrophasors (f_bus, V, θ)  |  Primary T_obs = 1.0 s (sweep {0.5, 1.0, 1.5, 2.0} s)\n"
        "• STRICT CAUSAL FIREWALL: zero internal rotor ω/H_eff access; all samples t > t0 + T_obs masked\n"
        "• Reconstructs wide-area f_sys_pmu(t) & causal 5-point backward OLS ROCOF",
        fontsize=7.3, color="#1E293B", va="top", linespacing=1.26,
    )
    _arrow(5.0, 6.24, 5.0, 5.82, color="#059669")

    _box(
        0.4, 4.35, 9.2, 1.42,
        "3. Canonical 14-Feature Spatio-Temporal Extractor (Full Rank 14/14)",
        "• ROCOF kinematics (4): rocof_init, rocof_peak_neg, rocof_end, rocof_mean\n"
        "• Aggregate excursion (3) + Spatial/Tie-line angles (5) + 2nd-order arrest curvature d²f/dt² (2)",
        "#EFF6FF", "#2563EB", "#1E3A8A",
    )

    _arrow(2.7, 4.32, 2.5, 3.85, color="#1D4ED8")
    _arrow(7.3, 4.32, 7.5, 3.85, color="#7C3AED")

    _box(
        0.4, 2.15, 4.4, 1.65,
        "4A. Continuous Nadir Head",
        "• GBDT / Ridge / RF / LightGBM\n"
        "• Predicts continuous $\\hat{f}_{\\mathrm{nadir}}$ (Hz)\n"
        "• Quantile GBDT (α=0.025, 0.975) +\n"
        "  conformal & OOD distance intervals",
        "#EFF6FF", "#1D4ED8", "#1E3A8A",
    )

    _box(
        5.2, 2.15, 4.4, 1.65,
        "4B. Calibrated Probability Head",
        "• 5-fold cross-fit Platt sigmoid &\n"
        "  PAVA Isotonic step calibration\n"
        f"• Outputs $\\hat{{P}}(f_{{\\mathrm{{nadir}}}} \\geq {SECURITY_THRESHOLD_HZ:.2f}\\,\\mathrm{{Hz}})$\n"
        "• Validated via Brier, ECE & MCE",
        "#F5F3FF", "#7C3AED", "#4C1D95",
    )

    _arrow(2.6, 2.12, 4.1, 1.64, color="#1D4ED8")
    _arrow(7.4, 2.12, 5.9, 1.64, color="#7C3AED")

    _box(
        0.4, 0.22, 9.2, 1.38,
        "5. Safety-Asymmetric Control-Room Screening Decision",
        f"• Conservative threshold τ (e.g. τ = 0.70 eliminates missed insecure events) triggers FFR / fast-valving / adaptive UFLS\n"
        f"• Measured Hardware Latency: {t_online_ms:.2f} ms median ({t_p99_ms:.2f} ms p99) vs {t_sim_ms:.1f} ms 10-s ODE ({speedup:.1f}× speedup)",
        "#FEF2F2", "#DC2626", "#991B1B",
    )

    fig.tight_layout(pad=1.2)
    return save_dual(fig, out_dir, "fig01_system_and_architecture", dpi=dpi)


# ---------------------------------------------------------------------------
# Figure 2: Post-Outage Trajectories (Loss Fraction, RES Sweep, 3-Area Swings)
# ---------------------------------------------------------------------------
def plot_fig02_generation_loss_response(
    out_dir: Path,
    raw_dir: Path,
    raw_sim_dir: Path,
    dpi: int = 240,
) -> Tuple[Path, Path]:
    records = _index_scenarios(raw_dir, raw_sim_dir)
    if not records:
        raise FileNotFoundError(f"No scenario .npz files found in {raw_dir}")

    by_key = {
        (r["disturbed_gen"], r["loss_fraction"], r["renewable_penetration"], r["load_scale"]): r
        for r in records
    }

    chosen_bus_lf = 38
    for cand_bus in (38, 32, 35, 33, 39, 31):
        if all((cand_bus, lf, 0.30, 1.00) in by_key for lf in (0.40, 0.70, 1.00)):
            chosen_bus_lf = cand_bus
            break
    lf_recs = [by_key[(chosen_bus_lf, lf, 0.30, 1.00)] for lf in (0.40, 0.70, 1.00) if (chosen_bus_lf, lf, 0.30, 1.00) in by_key]
    if not lf_recs:
        lf_recs = sorted(records, key=lambda r: r["loss_fraction"])[:3]

    res_recs: List[Dict[str, Any]] = []
    for cand_lf in (0.70, 1.00, 0.40):
        for cand_bus in (34, 37, 36, 33, 31, 32, 35, 38):
            if all((cand_bus, cand_lf, p, 1.00) in by_key for p in (0.10, 0.30, 0.50)):
                trio = [by_key[(cand_bus, cand_lf, p, 1.00)] for p in (0.10, 0.30, 0.50)]
                if trio[0]["nadir"] >= SECURITY_THRESHOLD_HZ and trio[-1]["nadir"] < SECURITY_THRESHOLD_HZ:
                    res_recs = trio
                    break
        if res_recs:
            break
    if not res_recs:
        for cand_bus in range(30, 40):
            if all((cand_bus, 1.00, p, 1.00) in by_key for p in (0.10, 0.30, 0.50)):
                res_recs = [by_key[(cand_bus, 1.00, p, 1.00)] for p in (0.10, 0.30, 0.50)]
                break
    if not res_recs:
        res_recs = sorted(records, key=lambda r: r["renewable_penetration"])[:3]

    spatial_rec = by_key.get((32, 1.00, 0.30, 1.00)) or by_key.get((31, 1.00, 0.30, 1.00)) or lf_recs[-1]

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.7))
    ax1, ax2, ax3 = axes

    # Panel (a): Complete vs Partial Generator Loss
    lf_colors = {0.40: PALETTE["emerald"], 0.70: PALETTE["amber"], 1.00: PALETTE["red"]}
    y_min_a, y_max_a = 49.35, 50.05
    for rec in lf_recs:
        pmu_d = load_scenario_from_npz(str(rec["pmu_path"]))
        t_p = pmu_d["timestamps"]
        f_pmu = pmu_d["pmu_sys_frequency"]
        c = lf_colors.get(rec["loss_fraction"], PALETTE["blue"])
        lf_pct = int(round(rec["loss_fraction"] * 100))

        if rec["sim_path"] is not None:
            sim_d = _load_raw_sim_npz(rec["sim_path"])
            t_s = sim_d["timestamps"]
            f_coi = sim_d["coi_frequency"]
            ax1.plot(
                t_s, f_coi, color=c, lw=2.1,
                label=f"{lf_pct}% Outage ({rec['active_loss_mw']:.0f} MW, nadir {rec['nadir']:.2f} Hz)",
            )
            ax1.plot(t_p, f_pmu, color=c, lw=1.0, ls=":", alpha=0.8)
            y_min_a = min(y_min_a, float(np.min(f_coi)))
            y_max_a = max(y_max_a, float(np.max(f_coi)))
        else:
            ax1.plot(
                t_p, f_pmu, color=c, lw=2.1,
                label=f"{lf_pct}% Outage ({rec['active_loss_mw']:.0f} MW, nadir {rec['nadir']:.2f} Hz)",
            )
            y_min_a = min(y_min_a, float(np.min(f_pmu)))
            y_max_a = max(y_max_a, float(np.max(f_pmu)))

        ax1.scatter([rec["t_nadir"]], [rec["nadir"]], color=c, s=36, zorder=5, edgecolors="white", lw=0.7)

    ax1.plot([], [], color=PALETTE["slate"], lw=1.1, ls=":", label="Observable f_sys_pmu(t)")
    ax1.axhline(SECURITY_THRESHOLD_HZ, color=PALETTE["navy"], lw=1.4, ls="--", label="Threshold (49.50 Hz)")
    ax1.axvline(1.0, color=PALETTE["muted"], lw=1.0, ls="-.")
    ax1.set_xlim(0.5, 10.0)
    ax1.set_ylim(y_min_a - 0.08, y_max_a + 0.07)
    ax1.set_title(
        f"(a) Complete vs. Partial Outage (Bus {lf_recs[0]['disturbed_gen']}, 30% RES)",
        fontweight="bold",
    )
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Frequency (Hz)")
    ax1.grid(True)
    ax1.legend(loc="lower right", fontsize=7.0, framealpha=0.95)

    # Panel (b): Renewable Penetration Sweep (10% vs 30% vs 50%)
    res_colors = {0.10: PALETTE["blue"], 0.30: PALETTE["amber"], 0.50: PALETTE["red"]}
    y_min_b, y_max_b = 49.35, 50.05
    for rec in res_recs:
        pmu_d = load_scenario_from_npz(str(rec["pmu_path"]))
        t_p = pmu_d["timestamps"]
        f_pmu = pmu_d["pmu_sys_frequency"]
        c = res_colors.get(rec["renewable_penetration"], PALETTE["violet"])
        pen_pct = int(round(rec["renewable_penetration"] * 100))

        if rec["sim_path"] is not None:
            sim_d = _load_raw_sim_npz(rec["sim_path"])
            h_post = float(sim_d["metadata"].get("effective_H_post", 0.0))
            h_str = f", H={h_post:.0f}s" if h_post > 0 else ""
            ax2.plot(
                sim_d["timestamps"], sim_d["coi_frequency"], color=c, lw=2.1,
                label=f"{pen_pct}% RES{h_str} (nadir {rec['nadir']:.2f} Hz)",
            )
            ax2.plot(t_p, f_pmu, color=c, lw=1.0, ls=":", alpha=0.8)
            y_min_b = min(y_min_b, float(np.min(sim_d["coi_frequency"])))
            y_max_b = max(y_max_b, float(np.max(sim_d["coi_frequency"])))
        else:
            ax2.plot(
                t_p, f_pmu, color=c, lw=2.1,
                label=f"{pen_pct}% RES (nadir {rec['nadir']:.2f} Hz)",
            )
            y_min_b = min(y_min_b, float(np.min(f_pmu)))
            y_max_b = max(y_max_b, float(np.max(f_pmu)))

        ax2.scatter([rec["t_nadir"]], [rec["nadir"]], color=c, s=36, zorder=5, edgecolors="white", lw=0.7)

    ax2.axhline(SECURITY_THRESHOLD_HZ, color=PALETTE["navy"], lw=1.4, ls="--", label="Threshold (49.50 Hz)")
    ax2.axvline(1.0, color=PALETTE["muted"], lw=1.0, ls="-.")
    ax2.set_xlim(0.5, 10.0)
    ax2.set_ylim(y_min_b - 0.08, y_max_b + 0.06)
    ax2.set_title(
        f"(b) Inertia Displacement Sweep (Bus {res_recs[0]['disturbed_gen']}, {int(round(res_recs[0]['loss_fraction']*100))}% Loss)",
        fontweight="bold",
    )
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.grid(True)
    ax2.legend(loc="lower right", fontsize=7.0, framealpha=0.95)

    # Panel (c): True COI vs 3-Area Frequencies + Zoomed Inter-Area Swing Inset
    pmu_sp = load_scenario_from_npz(str(spatial_rec["pmu_path"]))
    if spatial_rec["sim_path"] is not None:
        sim_sp = _load_raw_sim_npz(spatial_rec["sim_path"])
        t_sp = sim_sp["timestamps"]
        area_f = _compute_area_frequencies(sim_sp["bus_frequencies_true"])
        f_coi_sp = sim_sp["coi_frequency"]
    else:
        t_sp = pmu_sp["timestamps"]
        area_f = _compute_area_frequencies(pmu_sp["bus_frequencies"])
        f_coi_sp = pmu_sp["pmu_sys_frequency"]

    ax3.axvspan(1.0, 2.0, color="#DCFCE7", alpha=0.65, label="Primary Window [1.0, 2.0] s")
    ax3.plot(t_sp, area_f[1], color=AREA_COLORS[1], lw=1.7, alpha=0.92, label="Area 1 Mean Freq. (South)")
    ax3.plot(t_sp, area_f[2], color=AREA_COLORS[2], lw=1.7, alpha=0.92, label="Area 2 Mean Freq. (Central)")
    ax3.plot(t_sp, area_f[3], color=AREA_COLORS[3], lw=1.7, alpha=0.92, label="Area 3 Mean Freq. (North)")
    ax3.plot(t_sp, f_coi_sp, color=PALETTE["navy"], lw=2.2, label="True Global COI Frequency")
    ax3.plot(pmu_sp["timestamps"], pmu_sp["pmu_sys_frequency"], color="#F59E0B", lw=1.4, ls="--", label="PMU Estimate f_sys_pmu(t)")
    ax3.axhline(SECURITY_THRESHOLD_HZ, color=PALETTE["red"], lw=1.3, ls=":", label="Threshold (49.50 Hz)")

    # Zoomed inset on post-outage inter-area oscillations around the nadir [2.8, 3.8] s
    ax_ins = ax3.inset_axes([0.36, 0.60, 0.59, 0.35])
    z_m = (t_sp >= 2.75) & (t_sp <= 3.85)
    ax_ins.plot(t_sp[z_m], area_f[1][z_m], color=AREA_COLORS[1], lw=1.4)
    ax_ins.plot(t_sp[z_m], area_f[2][z_m], color=AREA_COLORS[2], lw=1.4)
    ax_ins.plot(t_sp[z_m], area_f[3][z_m], color=AREA_COLORS[3], lw=1.4)
    ax_ins.plot(t_sp[z_m], f_coi_sp[z_m], color=PALETTE["navy"], lw=1.7)
    ax_ins.set_title("Nadir Inter-Area Zoom", fontsize=6.7, fontweight="bold", pad=2)
    ax_ins.tick_params(labelsize=6.2)
    ax_ins.grid(True, alpha=0.6)

    y_min_c = min(float(np.min(v)) for v in area_f.values())
    ax3.set_xlim(0.7, 6.5)
    ax3.set_ylim(min(y_min_c, float(np.min(f_coi_sp))) - 0.08, 50.12)
    ax3.set_title(
        f"(c) 3-Area Swings vs. COI ({spatial_rec['scenario_id']})",
        fontweight="bold",
    )
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Frequency (Hz)")
    ax3.grid(True)
    ax3.legend(loc="lower right", fontsize=6.7, framealpha=0.95)

    fig.tight_layout(pad=1.1)
    return save_dual(fig, out_dir, "fig02_generation_loss_response", dpi=dpi)


# ---------------------------------------------------------------------------
# Figure 3: Causal Observation Window Illustration & Lead-Time Mechanics
# ---------------------------------------------------------------------------
def plot_fig03_causal_window_illustration(
    out_dir: Path,
    raw_dir: Path,
    raw_sim_dir: Path,
    proc_dir: Path,
    dpi: int = 240,
) -> Tuple[Path, Path]:
    records = _index_scenarios(raw_dir, raw_sim_dir)
    cands = [
        r for r in records
        if r["partition"] == "test" and 49.05 <= r["nadir"] <= 49.38 and abs(r["load_scale"] - 1.0) < 0.05
    ]
    if not cands:
        cands = [r for r in records if r["is_secure"] == 0] or records
    rec = sorted(cands, key=lambda r: abs(r["nadir"] - 49.22))[0]

    pmu_scen = load_scenario_from_npz(str(rec["pmu_path"]))
    t = pmu_scen["timestamps"]
    bf = pmu_scen["bus_frequencies"]
    f_sys = pmu_scen["pmu_sys_frequency"]
    meta = pmu_scen["metadata"]
    t0 = float(meta.get("t_disturb", 1.0))
    t_nadir = float(meta.get("ground_truth_t_nadir", rec["t_nadir"]))
    f_nadir = float(meta.get("ground_truth_nadir", rec["nadir"]))

    if not bool(meta.get("is_noisy", False)):
        noisy_rec = PMURecorder(
            fps=int(meta.get("fps", 50)),
            noise_config=PMUNoiseConfig(sigma_f=0.005, sigma_rocof=0.02, seed=42),
            f_nominal=NOMINAL_FREQ,
        ).record_trajectory(pmu_scen, inject_noise=True)
        bf_noisy = noisy_rec["bus_frequencies"]
    else:
        bf_noisy = bf

    fig = plt.figure(figsize=(14.8, 5.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.48, 1.0], height_ratios=[1.0, 1.0], hspace=0.38, wspace=0.24)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_rocof = fig.add_subplot(gs[0, 1])
    ax_lead = fig.add_subplot(gs[1, 1])

    ax_main.axvspan(t0, t0 + 1.0, color="#DCFCE7", alpha=0.80, zorder=1, label="Primary Causal Window [1.0, 2.0] s (T_obs = 1.0 s)")
    ax_main.axvspan(t0 + 1.0, float(t[-1]), color="#FEF2F2", alpha=0.52, zorder=1, label="Hidden Future Trajectory (t > t0 + T_obs, Masked)")

    fault_bus_idx = max(0, min(bf.shape[1] - 1, int(rec["disturbed_gen"]) - 1))
    sample_buses = [0, 3, 15, 25, fault_bus_idx]
    for idx_b, b_col in enumerate(sample_buses):
        ax_main.plot(
            t, bf_noisy[:, b_col],
            color="#94A3B8" if b_col != fault_bus_idx else "#FDA4AF",
            lw=0.85, alpha=0.65, zorder=2,
            label="Noisy Bus PMU Channels (σ_f = 5 mHz)" if idx_b == 0 else None,
        )

    ax_main.plot(t, f_sys, color=PALETTE["blue"], lw=2.4, zorder=4, label="Reconstructed Wide-Area f_sys_pmu(t)")
    if rec["sim_path"] is not None:
        sim_d = _load_raw_sim_npz(rec["sim_path"])
        ax_main.plot(sim_d["timestamps"], sim_d["coi_frequency"], color=PALETTE["navy"], lw=1.4, ls="--", zorder=4, label="Ground-Truth COI Frequency (Label Only)")

    ax_main.axhline(SECURITY_THRESHOLD_HZ, color=PALETTE["amber"], lw=1.7, ls="--", zorder=3, label="Security Threshold (49.50 Hz)")
    ax_main.axvline(t0, color=PALETTE["navy"], lw=1.3, ls="-.", zorder=3)

    win_styles = [
        (0.5, "#0EA5E9", ":"),
        (1.0, "#15803D", "--"),
        (1.5, "#7C3AED", ":"),
        (2.0, "#EA580C", ":"),
    ]
    y_lo_main = min(float(np.min(bf_noisy)), f_nadir) - 0.14
    y_hi_main = 50.15
    for w_s, w_col, w_ls in win_styles:
        t_cut = t0 + w_s
        ax_main.axvline(t_cut, color=w_col, lw=1.4 if w_s == 1.0 else 1.1, ls=w_ls, zorder=3)
        ax_main.text(
            t_cut, 50.095, f"+{w_s:.1f}s",
            ha="center", va="bottom", fontsize=7.2, fontweight="bold", color=w_col,
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec=w_col, lw=0.7),
        )

    f_at_cut = float(np.interp(t0 + 1.0, t, f_sys))
    t_lead_1s = t_nadir - (t0 + 1.0)
    ax_main.scatter([t0, t0 + 1.0, t_nadir], [50.0, f_at_cut, f_nadir], color=[PALETTE["navy"], PALETTE["green"], PALETTE["red"]], s=48, zorder=6)

    arrow_y = f_nadir + 0.08
    ax_main.annotate(
        "", xy=(t_nadir, arrow_y), xytext=(t0 + 1.0, arrow_y),
        arrowprops=dict(arrowstyle="<->", color=PALETTE["green"], lw=1.8),
        zorder=6,
    )
    ax_main.text(
        0.5 * (t0 + 1.0 + t_nadir), arrow_y + 0.03,
        f"t_lead = {t_lead_1s:.2f} s",
        ha="center", va="bottom", fontsize=8.0, fontweight="bold", color=PALETTE["green"],
        bbox=dict(boxstyle="round,pad=0.18", fc="#DCFCE7", ec=PALETTE["green"], lw=0.8),
        zorder=6,
    )
    ax_main.annotate(
        f"Actual COI Nadir\n{f_nadir:.3f} Hz @ t = {t_nadir:.2f} s",
        xy=(t_nadir, f_nadir),
        xytext=(2.35, 49.68),
        fontsize=7.9, fontweight="bold", color=PALETTE["crimson"],
        bbox=dict(boxstyle="round,pad=0.24", fc="#FEE2E2", ec=PALETTE["red"], lw=0.85),
        arrowprops=dict(arrowstyle="->", color=PALETTE["red"], lw=1.15),
        zorder=6,
    )

    ax_main.set_xlim(0.2, 8.5)
    ax_main.set_ylim(y_lo_main, y_hi_main)
    ax_main.set_title(
        f"(a) Causal Windows & Lead Time on Real Trajectory ({rec['scenario_id']})",
        fontweight="bold", pad=10,
    )
    ax_main.set_xlabel("Time (s)")
    ax_main.set_ylabel("Frequency (Hz)")
    ax_main.grid(True)
    ax_main.legend(loc="lower right", fontsize=7.1, framealpha=0.95)

    fps_val = float(meta.get("fps", 50.0))
    dt = 1.0 / fps_val
    f_bus_sample = bf_noisy[:, fault_bus_idx]
    rocof_2pt = np.diff(f_bus_sample, prepend=f_bus_sample[0]) / dt
    rocof_ols_sys = compute_causal_rocof(f_sys, fps=fps_val, window_length=5)

    zoom_mask = (t >= 0.9) & (t <= 3.1)
    ax_rocof.plot(t[zoom_mask], rocof_2pt[zoom_mask], color="#CBD5E1", lw=0.95, label="Noisy Single-Bus 2-pt df/dt")
    ax_rocof.plot(t[zoom_mask], compute_causal_rocof(f_bus_sample, fps=fps_val, window_length=5)[zoom_mask], color="#38BDF8", lw=1.3, alpha=0.85, label="Single-Bus 5-pt Causal OLS")
    ax_rocof.plot(t[zoom_mask], rocof_ols_sys[zoom_mask], color=PALETTE["violet"], lw=2.0, label="Wide-Area f_sys_pmu Causal OLS")
    for w_s, w_col, w_ls in win_styles:
        ax_rocof.axvline(t0 + w_s, color=w_col, lw=1.0, ls=w_ls)
    ax_rocof.set_xlim(0.9, 3.1)
    ax_rocof.set_title("(b) Causal 5-Point OLS ROCOF Within Early Windows", fontweight="bold")
    ax_rocof.set_xlabel("Time (s)")
    ax_rocof.set_ylabel("ROCOF (Hz/s)")
    ax_rocof.grid(True)
    ax_rocof.legend(loc="lower right", fontsize=6.9, framealpha=0.95)

    win_vals = [0.5, 1.0, 1.5, 2.0]
    lead_arrays: List[np.ndarray] = []
    win_labels: List[str] = []
    for w_s in win_vals:
        csv_p = proc_dir / f"features_t{w_s:.1f}s.csv"
        npz_p = proc_dir / f"features_t{w_s:.1f}s.npz"
        if csv_p.exists():
            df_w = pd.read_csv(csv_p)
            arr = df_w["lead_time_s"].dropna().to_numpy(dtype=float)
        elif npz_p.exists():
            with np.load(npz_p, allow_pickle=True) as dw:
                arr = np.asarray(dw["lead_time_s"], dtype=float)
        else:
            arr = np.array([r["t_nadir"] - (1.0 + w_s) for r in records], dtype=float)
        lead_arrays.append(arr)
        win_labels.append(f"T_obs={w_s:.1f}s\n(N={len(arr)})")

    bp = ax_lead.boxplot(
        lead_arrays,
        tick_labels=win_labels,
        patch_artist=True,
        widths=0.46,
        medianprops=dict(color=PALETTE["navy"], lw=1.6),
        flierprops=dict(marker="o", markersize=3, alpha=0.45, markeredgecolor=PALETTE["slate"]),
    )
    box_cols = ["#BAE6FD", "#86EFAC", "#DDD6FE", "#FED7AA"]
    for patch, bc in zip(bp["boxes"], box_cols):
        patch.set_facecolor(bc)
        patch.set_edgecolor(PALETTE["slate"])
    for idx_w, arr in enumerate(lead_arrays, start=1):
        ax_lead.text(
            idx_w, float(np.max(arr)) + 0.06, f"μ={np.mean(arr):.2f}s",
            ha="center", va="bottom", fontsize=7.2, fontweight="bold", color=PALETTE["navy"],
        )
    ax_lead.axhline(0.0, color=PALETTE["red"], lw=1.1, ls="--")
    ax_lead.set_ylim(min(-0.15, min(float(np.min(a)) for a in lead_arrays) - 0.15), max(float(np.max(a)) for a in lead_arrays) + 0.38)
    ax_lead.set_title("(c) Dataset-Wide Pre-Nadir Lead Time t_lead", fontweight="bold")
    ax_lead.set_ylabel("Lead Time t_lead (s)")
    ax_lead.grid(True, axis="y")

    return save_dual(fig, out_dir, "fig03_causal_window_illustration", dpi=dpi)


# ---------------------------------------------------------------------------
# Figure 4: Empirical Feature Distributions & 14x14 Spearman Correlations
# ---------------------------------------------------------------------------
def plot_fig04_pmu_and_feature_distributions(
    out_dir: Path,
    proc_dir: Path,
    tables_dir: Path,
    dpi: int = 240,
) -> Tuple[Path, Path]:
    csv_p = proc_dir / "features_t1.0s.csv"
    if not csv_p.exists():
        csv_p = proc_dir / "features.csv"
    if csv_p.exists():
        df = pd.read_csv(csv_p)
    else:
        with np.load(proc_dir / "features.npz", allow_pickle=True) as d:
            cols = [str(c) for c in d["feature_names"]]
            df = pd.DataFrame(d["X"], columns=cols)
            df["is_secure"] = d["y_secure"]

    sec_df = df[df["is_secure"] == 1]
    insec_df = df[df["is_secure"] == 0]

    fig = plt.figure(figsize=(15.4, 6.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.38], wspace=0.28, hspace=0.35)

    key_specs = [
        ("rocof_init", "(a) Initial Wide-Area ROCOF", "rocof_init (Hz/s)", "upper left"),
        ("freq_dev_integral", "(b) Frequency Deviation Integral", "freq_dev_integral (Hz·s)", "upper left"),
        ("area_freq_spread_max", "(c) Max 3-Area Frequency Spread", "area_freq_spread_max (Hz)", "upper right"),
        ("freq_curvature_mean", "(d) Mean Arrest Curvature d²f/dt²", "freq_curvature_mean (Hz/s²)", "upper right"),
    ]

    for idx, (col, title, xlabel, leg_loc) in enumerate(key_specs):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        v_sec = sec_df[col].dropna().to_numpy(dtype=float)
        v_ins = insec_df[col].dropna().to_numpy(dtype=float)

        all_v = np.concatenate([v_sec, v_ins])
        bins = np.linspace(float(np.min(all_v)), float(np.max(all_v)), 24)
        n_s, _, _ = ax.hist(
            v_sec, bins=bins, density=True, alpha=0.52, color=PALETTE["blue"], edgecolor=PALETTE["blue"], lw=0.9,
            label=f"Secure (N={len(v_sec)}, med={np.median(v_sec):.3f})",
        )
        n_i, _, _ = ax.hist(
            v_ins, bins=bins, density=True, alpha=0.50, color=PALETTE["red"], edgecolor=PALETTE["red"], lw=0.9,
            label=f"Insecure (N={len(v_ins)}, med={np.median(v_ins):.3f})",
        )
        ax.axvline(float(np.median(v_sec)), color=PALETTE["blue"], lw=1.6, ls="--")
        ax.axvline(float(np.median(v_ins)), color=PALETTE["red"], lw=1.6, ls="--")
        ax.set_ylim(0.0, max(float(np.max(n_s)), float(np.max(n_i))) * 1.18)

        ax.set_title(title, fontweight="bold", fontsize=9.8)
        ax.set_xlabel(xlabel, fontsize=8.7)
        ax.set_ylabel("Empirical Density", fontsize=8.7)
        ax.grid(True)
        ax.legend(loc=leg_loc, fontsize=7.0, framealpha=0.95)

    ax_corr = fig.add_subplot(gs[:, 2])
    corr_csv = tables_dir / "feature_correlations_spearman.csv"
    if corr_csv.exists():
        corr_df = pd.read_csv(corr_csv, index_col=0)
    else:
        feat_cols = [c for c in FEATURE_SHORT if c in df.columns]
        corr_df = df[feat_cols].corr(method="spearman")

    c_mat = corr_df.to_numpy(dtype=float)
    short_names = [FEATURE_SHORT.get(str(c), str(c)) for c in corr_df.columns]

    im = ax_corr.imshow(c_mat, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    n_f = len(short_names)
    ax_corr.set_xticks(np.arange(n_f))
    ax_corr.set_yticks(np.arange(n_f))
    ax_corr.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7.4)
    ax_corr.set_yticklabels(short_names, fontsize=7.4)

    for i in range(n_f):
        for j in range(n_f):
            val = c_mat[i, j]
            if abs(val) >= 0.45 or i == j:
                txt_c = "white" if abs(val) >= 0.65 else PALETTE["navy"]
                ax_corr.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.8, color=txt_c, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax_corr, fraction=0.045, pad=0.03)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.set_label("Spearman ρ", fontsize=8.5)
    ax_corr.set_title("(e) 14-Feature Spearman Correlation Matrix (Train)", fontweight="bold", fontsize=9.9)

    return save_dual(fig, out_dir, "fig04_pmu_and_feature_distributions", dpi=dpi)


# ---------------------------------------------------------------------------
# Figure 5: Continuous Nadir Parity Scatter, Intervals & Residual Errors
# ---------------------------------------------------------------------------
def plot_fig05_nadir_parity_and_residuals(
    out_dir: Path,
    feat_path: Path,
    models_dir: Path,
    dpi: int = 240,
) -> Tuple[Path, Path]:
    with np.load(feat_path, allow_pickle=True) as data:
        X = np.asarray(data["X"], dtype=float)
        y_nadir = np.asarray(data["y_nadir"], dtype=float).ravel()
        parts = np.asarray([str(p) for p in data["partitions"]])
        ood_types = (
            np.asarray([str(o) for o in data["ood_types"]])
            if "ood_types" in data.files
            else np.array(["ood_all"] * len(y_nadir))
        )
        feat_names = [str(f) for f in data["feature_names"]] if "feature_names" in data.files else [f"f_{i}" for i in range(X.shape[1])]

    gbdt = joblib.load(models_dir / "gbdt.joblib")
    cal_platt = joblib.load(models_dir / "calibrated_gb_platt.joblib")

    id_mask = parts == "test"
    ood_mask = parts == "ood_test"
    if not np.any(id_mask):
        id_mask = parts != "ood_test"

    X_id_df = pd.DataFrame(X[id_mask], columns=feat_names)
    y_id = y_nadir[id_mask]
    pred_id = np.asarray(gbdt.predict(X_id_df), dtype=float).ravel()
    _, lo_id, hi_id = cal_platt.predict_nadir_intervals(X_id_df, confidence_level=0.95)
    res_id_mhz = (pred_id - y_id) * 1000.0
    cov_id = 100.0 * float(np.mean((y_id >= lo_id) & (y_id <= hi_id)))

    has_ood = bool(np.any(ood_mask))
    if has_ood:
        X_ood_df = pd.DataFrame(X[ood_mask], columns=feat_names)
        y_ood = y_nadir[ood_mask]
        ood_sub = ood_types[ood_mask]
        pred_ood = np.asarray(gbdt.predict(X_ood_df), dtype=float).ravel()
        _, lo_ood, hi_ood = cal_platt.predict_nadir_intervals(X_ood_df, confidence_level=0.95)
        res_ood_mhz = (pred_ood - y_ood) * 1000.0
        cov_ood = 100.0 * float(np.mean((y_ood >= lo_ood) & (y_ood <= hi_ood)))
    else:
        y_ood, pred_ood, lo_ood, hi_ood, ood_sub = y_id, pred_id, lo_id, hi_id, np.array(["id"] * len(y_id))
        res_ood_mhz, cov_ood = res_id_mhz, cov_id

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.7))

    lo_a = min(float(np.min(y_id)), float(np.min(lo_id))) - 0.06
    hi_a = max(float(np.max(y_id)), float(np.max(hi_id))) + 0.06
    ax1.plot([lo_a, hi_a], [lo_a, hi_a], color=PALETTE["navy"], lw=1.4, ls="--", label="1:1 Exact Parity")
    ax1.axvline(SECURITY_THRESHOLD_HZ, color=PALETTE["amber"], lw=1.1, ls=":")
    ax1.axhline(SECURITY_THRESHOLD_HZ, color=PALETTE["amber"], lw=1.1, ls=":", label="49.50 Hz Threshold")
    ax1.errorbar(
        y_id, pred_id,
        yerr=[np.maximum(0.0, pred_id - lo_id), np.maximum(0.0, hi_id - pred_id)],
        fmt="o", markersize=4.8, color=PALETTE["blue"], ecolor="#93C5FD", elinewidth=1.0, capsize=2.0, alpha=0.90,
        label=f"GBDT + 95% PI (Cov={cov_id:.1f}%)",
    )
    mae_id_mhz = float(np.mean(np.abs(res_id_mhz)))
    rmse_id_mhz = float(np.sqrt(np.mean(res_id_mhz ** 2)))
    r2_id = 1.0 - float(np.sum((y_id - pred_id) ** 2) / max(1e-12, np.sum((y_id - np.mean(y_id)) ** 2)))
    ax1.text(
        0.04, 0.96,
        f"Held-Out ID Test (N = {len(y_id)})\nMAE = {mae_id_mhz:.1f} mHz | RMSE = {rmse_id_mhz:.1f} mHz\nR² = {r2_id:.4f} | Mean PI = {np.mean(hi_id - lo_id):.3f} Hz",
        transform=ax1.transAxes, va="top", ha="left", fontsize=7.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", fc="#EFF6FF", ec=PALETTE["blue"], lw=0.85),
    )
    ax1.set_xlim(lo_a, hi_a)
    ax1.set_ylim(lo_a, hi_a)
    ax1.set_title("(a) Held-Out ID Test Nadir Parity & 95% PI", fontweight="bold")
    ax1.set_xlabel("Ground-Truth COI Nadir (Hz)")
    ax1.set_ylabel(r"Predicted Nadir $\hat{f}_{\mathrm{nadir}}$ (Hz)")
    ax1.grid(True)
    ax1.legend(loc="lower right", fontsize=7.2, framealpha=0.95)

    lo_b = min(float(np.min(y_ood)), float(np.min(pred_ood))) - 0.08
    hi_b = max(float(np.max(y_ood)), float(np.max(pred_ood))) + 0.08
    order_ood = np.argsort(y_ood)
    ax2.fill_between(
        y_ood[order_ood], lo_ood[order_ood], hi_ood[order_ood],
        color="#FED7AA", alpha=0.40, label=f"95% Quantile/OOD Band (Cov={cov_ood:.1f}%)",
    )
    ax2.plot([lo_b, hi_b], [lo_b, hi_b], color=PALETTE["navy"], lw=1.4, ls="--", label="1:1 Exact Parity")
    ax2.axvline(SECURITY_THRESHOLD_HZ, color=PALETTE["amber"], lw=1.1, ls=":")
    ax2.axhline(SECURITY_THRESHOLD_HZ, color=PALETTE["amber"], lw=1.1, ls=":")

    regime_styles = [
        ("ood_res", "OOD-RES (40–50%)", PALETTE["red"], "o"),
        ("ood_load", "OOD-Load (0.80x, 1.20x)", PALETTE["teal"], "s"),
        ("ood_combined", "OOD-Combined", PALETTE["violet"], "^"),
    ]
    plotted_any = False
    for r_key, r_lbl, r_col, r_mk in regime_styles:
        m_r = ood_sub == r_key
        if np.any(m_r):
            plotted_any = True
            r_mae = float(np.mean(np.abs(pred_ood[m_r] - y_ood[m_r]))) * 1000.0
            ax2.scatter(
                y_ood[m_r], pred_ood[m_r], s=22, marker=r_mk, color=r_col, alpha=0.80,
                label=f"{r_lbl} (N={int(np.sum(m_r))}, {r_mae:.1f} mHz)",
            )
    if not plotted_any:
        ax2.scatter(y_ood, pred_ood, s=24, color=PALETTE["orange"], alpha=0.8, label=f"OOD Test (N={len(y_ood)})")

    mae_ood_mhz = float(np.mean(np.abs(res_ood_mhz)))
    rmse_ood_mhz = float(np.sqrt(np.mean(res_ood_mhz ** 2)))
    r2_ood = 1.0 - float(np.sum((y_ood - pred_ood) ** 2) / max(1e-12, np.sum((y_ood - np.mean(y_ood)) ** 2)))
    ax2.text(
        0.04, 0.96,
        f"Held-Out OOD Test (N = {len(y_ood)})\nMAE = {mae_ood_mhz:.1f} mHz | RMSE = {rmse_ood_mhz:.1f} mHz\nR² = {r2_ood:.4f} | Mean PI = {np.mean(hi_ood - lo_ood):.3f} Hz",
        transform=ax2.transAxes, va="top", ha="left", fontsize=7.4, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", fc="#FFF7ED", ec=PALETTE["orange"], lw=0.85),
    )
    ax2.set_xlim(lo_b, hi_b)
    ax2.set_ylim(lo_b, hi_b)
    ax2.set_title("(b) Unseen OOD Test Nadir Parity by Regime", fontweight="bold")
    ax2.set_xlabel("Ground-Truth COI Nadir (Hz)")
    ax2.set_ylabel(r"Predicted Nadir $\hat{f}_{\mathrm{nadir}}$ (Hz)")
    ax2.grid(True)
    ax2.legend(loc="lower right", fontsize=6.9, framealpha=0.95)

    # Panel (c): Core residual distribution in [-60, +90] mHz (clipping extreme OOD tail into edge bin)
    lo_r, hi_r = -60.0, 90.0
    bins_r = np.linspace(lo_r, hi_r, 31)
    n_id_h, _, _ = ax3.hist(
        np.clip(res_id_mhz, lo_r, hi_r), bins=bins_r, density=True, alpha=0.58,
        color=PALETTE["blue"], edgecolor=PALETTE["blue"], lw=0.85,
        label=f"ID Test (Bias={np.mean(res_id_mhz):+.1f}, P95|e|={np.percentile(np.abs(res_id_mhz), 95):.1f} mHz)",
    )
    max_dens = float(np.max(n_id_h))
    if has_ood:
        n_oo_h, _, _ = ax3.hist(
            np.clip(res_ood_mhz, lo_r, hi_r), bins=bins_r, density=True, alpha=0.46,
            color=PALETTE["orange"], edgecolor=PALETTE["orange"], lw=0.85,
            label=f"OOD Test (Bias={np.mean(res_ood_mhz):+.1f}, P95|e|={np.percentile(np.abs(res_ood_mhz), 95):.1f} mHz)",
        )
        max_dens = max(max_dens, float(np.max(n_oo_h)))

    q95_conf_mhz = float(getattr(cal_platt, "conformal_q95_", 0.03)) * 1000.0
    ax3.axvline(0.0, color=PALETTE["navy"], lw=1.4, ls="--", label="Zero Bias")
    ax3.axvline(-q95_conf_mhz, color=PALETTE["green"], lw=1.2, ls=":", label=f"±OOF q95 ({q95_conf_mhz:.1f} mHz)")
    ax3.axvline(q95_conf_mhz, color=PALETTE["green"], lw=1.2, ls=":")
    ax3.set_xlim(lo_r, hi_r)
    ax3.set_ylim(0.0, max_dens * 1.28)
    ax3.set_title(r"(c) Residual Error ($\hat{f}_{\mathrm{nadir}} - f_{\mathrm{true}}$) Distribution", fontweight="bold")
    ax3.set_xlabel("Prediction Residual Error (mHz, tails clipped at [-60, +90])")
    ax3.set_ylabel("Empirical Density")
    ax3.grid(True)
    ax3.legend(loc="upper right", fontsize=6.9, framealpha=0.95)

    fig.tight_layout(pad=1.1)
    return save_dual(fig, out_dir, "fig05_nadir_parity_and_residuals", dpi=dpi)


# ---------------------------------------------------------------------------
# Figure 6: All 7 Regressors Comparison with 95% Paired Bootstrap CIs
# ---------------------------------------------------------------------------
def plot_fig06_regression_model_comparison(
    out_dir: Path,
    tables_dir: Path,
    dpi: int = 240,
) -> Tuple[Path, Path]:
    df_reg = pd.read_csv(tables_dir / "nadir_regression.csv")

    short_map = {
        "Constant Mean Baseline": "Constant Mean",
        "Analytical Swing-Equation Baseline": "Analytical Swing",
        "Ridge Linear Regression": "Ridge Linear",
        "Decision Tree (CART)": "Decision Tree (CART)",
        "Random Forest": "Random Forest",
        "Gradient Boosting (GBDT)": "Gradient Boosting (GBDT)",
    }

    def _clean_name(raw_name: str) -> str:
        if raw_name in short_map:
            return short_map[raw_name]
        if raw_name.startswith("LightGBM"):
            return "LightGBM"
        return raw_name

    df_reg = df_reg.copy()
    df_reg["ShortModel"] = df_reg["Model"].map(_clean_name)
    ci_bounds = df_reg["95% Bootstrap CI (Hz)"].map(_parse_ci_string)
    df_reg["ci_low_mhz"] = [b[0] * 1000.0 for b in ci_bounds]
    df_reg["ci_high_mhz"] = [b[1] * 1000.0 for b in ci_bounds]
    df_reg["mae_mhz"] = df_reg["MAE (Hz)"].astype(float) * 1000.0
    df_reg["rmse_mhz"] = df_reg["RMSE (Hz)"].astype(float) * 1000.0
    df_reg["p95_mhz"] = df_reg["P95 AE (Hz)"].astype(float) * 1000.0

    df_sorted = df_reg.sort_values("mae_mhz", ascending=True).reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.6, 5.1), gridspec_kw={"width_ratios": [1.10, 1.0]})

    y = np.arange(len(df_sorted))
    h = 0.36
    mae_vals = df_sorted["mae_mhz"].to_numpy()
    rmse_vals = df_sorted["rmse_mhz"].to_numpy()
    err_lo = np.maximum(0.0, mae_vals - df_sorted["ci_low_mhz"].to_numpy())
    err_hi = np.maximum(0.0, df_sorted["ci_high_mhz"].to_numpy() - mae_vals)

    ax1.barh(y + h / 2, mae_vals, height=h, color=PALETTE["blue"], edgecolor=PALETTE["navy"], lw=0.7, label="Held-Out Test MAE (mHz)")
    ax1.errorbar(
        mae_vals, y + h / 2, xerr=[err_lo, err_hi],
        fmt="none", ecolor=PALETTE["navy"], elinewidth=1.2, capsize=2.8, zorder=5, label="95% Bootstrap CI (n=1000)",
    )
    ax1.barh(y - h / 2, rmse_vals, height=h, color="#38BDF8", edgecolor=PALETTE["navy"], lw=0.7, label="Held-Out Test RMSE (mHz)")

    ax1.set_xscale("log")
    max_x = float(np.max(rmse_vals))
    ax1.set_xlim(1.8, max_x * 16.0)
    for idx_r, row in df_sorted.iterrows():
        right_edge = max(row["rmse_mhz"], row["ci_high_mhz"])
        ax1.text(
            right_edge * 1.12, idx_r,
            f"MAE {row['mae_mhz']:.1f} [{row['ci_low_mhz']:.1f}, {row['ci_high_mhz']:.1f}] | R²={row['R2']:.4f}",
            va="center", ha="left", fontsize=7.1, fontweight="bold", color=PALETTE["navy"],
        )

    ax1.set_yticks(y)
    ax1.set_yticklabels(df_sorted["ShortModel"], fontsize=8.5, fontweight="bold")
    ax1.set_title("(a) All 7 Regressors: Held-Out Test Error & 95% CIs", fontweight="bold")
    ax1.set_xlabel("Prediction Error (mHz, log scale)")
    ax1.grid(True, axis="x", which="both")
    ax1.legend(loc="lower right", fontsize=7.3, framealpha=0.95)

    df_zoom = df_sorted[df_sorted["ShortModel"] != "Constant Mean"].reset_index(drop=True)
    x_z = np.arange(len(df_zoom))
    w_z = 0.26
    z_mae = df_zoom["mae_mhz"].to_numpy()
    z_rmse = df_zoom["rmse_mhz"].to_numpy()
    z_p95 = df_zoom["p95_mhz"].to_numpy()
    z_ci_hi = df_zoom["ci_high_mhz"].to_numpy()
    z_elo = np.maximum(0.0, z_mae - df_zoom["ci_low_mhz"].to_numpy())
    z_ehi = np.maximum(0.0, z_ci_hi - z_mae)

    b1 = ax2.bar(x_z - w_z, z_mae, width=w_z, color=PALETTE["blue"], edgecolor=PALETTE["navy"], lw=0.7, label="MAE (mHz) + 95% CI")
    ax2.errorbar(x_z - w_z, z_mae, yerr=[z_elo, z_ehi], fmt="none", ecolor=PALETTE["navy"], elinewidth=1.2, capsize=2.8, zorder=5)
    b2 = ax2.bar(x_z, z_rmse, width=w_z, color="#38BDF8", edgecolor=PALETTE["navy"], lw=0.7, label="RMSE (mHz)")
    b3 = ax2.bar(x_z + w_z, z_p95, width=w_z, color=PALETTE["amber"], edgecolor=PALETTE["navy"], lw=0.7, label="P95 |Error| (mHz)")

    for bar, ci_top in zip(b1, z_ci_hi):
        hv = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, max(hv, ci_top) + 0.55, f"{hv:.1f}", ha="center", va="bottom", fontsize=6.9, fontweight="bold", color=PALETTE["blue"])
    for bar in list(b2) + list(b3):
        hv = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, hv + 0.55, f"{hv:.1f}", ha="center", va="bottom", fontsize=6.9, fontweight="bold", color=PALETTE["navy"])

    ax2.set_xticks(x_z)
    ax2.set_xticklabels(df_zoom["ShortModel"], rotation=20, ha="right", fontsize=8.2, fontweight="bold")
    ax2.set_ylim(0.0, float(np.max(z_p95)) * 1.25)
    ax2.set_title("(b) Zoomed Physics & ML Regressors (Linear mHz Scale)", fontweight="bold")
    ax2.set_ylabel("Held-Out ID Test Error (mHz)")
    ax2.grid(True, axis="y")
    ax2.legend(loc="upper left", fontsize=7.4, framealpha=0.95)

    fig.tight_layout(pad=1.1)
    return save_dual(fig, out_dir, "fig06_regression_model_comparison", dpi=dpi)


# ---------------------------------------------------------------------------
# Figure 7: Classification Confusion Matrix, Model Comparison & Safety Sweep
# ---------------------------------------------------------------------------
def plot_fig07_classification_and_safety(
    out_dir: Path,
    tables_dir: Path,
    dpi: int = 240,
) -> Tuple[Path, Path]:
    df_clf = pd.read_csv(tables_dir / "security_classification.csv")
    thr_path = tables_dir / "threshold_sweep.csv"
    if not thr_path.exists():
        thr_path = tables_dir / "threshold_sweep_safety.csv"
    df_thr = pd.read_csv(thr_path)

    row_05 = df_thr.iloc[(df_thr["proba_threshold"] - 0.5).abs().argsort()[:1]].iloc[0]
    tn, fp, fn, tp = int(row_05["tn"]), int(row_05["fp"]), int(row_05["fn"]), int(row_05["tp"])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 4.9), gridspec_kw={"width_ratios": [0.88, 1.18, 1.04]})

    cm_counts = np.array([[tn, fp], [fn, tp]], dtype=int)
    cell_colors = [
        ["#DCFCE7", "#FEE2E2"],
        ["#FEF3C7", "#DBEAFE"],
    ]
    cell_titles = [
        ["True Insecure (TN)\nCorrectly Flagged", "Missed Insecure (FP)\nSafety-Critical!"],
        ["False Alarm (FN)\nConservative", "True Secure (TP)\nCorrectly Cleared"],
    ]

    ax1.set_xlim(0, 2)
    ax1.set_ylim(0, 2)
    for r_i in range(2):
        row_tot = max(1, int(np.sum(cm_counts[r_i])))
        for c_i in range(2):
            y_pos = 1 - r_i
            rect = FancyBboxPatch(
                (c_i + 0.06, y_pos + 0.06), 0.88, 0.88,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=cell_colors[r_i][c_i],
                edgecolor="#DC2626" if (r_i == 0 and c_i == 1) else "#64748B",
                linewidth=1.8 if (r_i == 0 and c_i == 1) else 1.1,
            )
            ax1.add_patch(rect)
            cnt = cm_counts[r_i, c_i]
            pct = 100.0 * cnt / row_tot
            ax1.text(
                c_i + 0.5, y_pos + 0.60, f"{cnt}",
                ha="center", va="center", fontsize=16, fontweight="bold",
                color=PALETTE["crimson"] if (r_i == 0 and c_i == 1) else PALETTE["navy"],
            )
            ax1.text(
                c_i + 0.5, y_pos + 0.28, f"({pct:.1f}%)\n{cell_titles[r_i][c_i]}",
                ha="center", va="center", fontsize=7.2, fontweight="bold", color="#334155",
            )

    ax1.set_xticks([0.5, 1.5])
    ax1.set_xticklabels(["Pred. Insecure (0)", "Pred. Secure (1)"], fontweight="bold", fontsize=8.2)
    ax1.set_yticks([0.5, 1.5])
    ax1.set_yticklabels(["Actual Secure (1)", "Actual Insecure (0)"], fontweight="bold", fontsize=8.2, rotation=90, va="center")
    for sp in ax1.spines.values():
        sp.set_visible(False)
    ax1.tick_params(length=0)
    ax1.set_title(f"(a) Held-Out Test Confusion (τ=0.50, N={tn+fp+fn+tp})", fontweight="bold")

    short_clf = (
        df_clf["Model"]
        .str.replace("Proposed Calibrated GBDT ", "Cal. GBDT ", regex=False)
        .str.replace(" Classifier", "", regex=False)
    )
    y_c = np.arange(len(df_clf))
    hc = 0.36
    f1_v = df_clf["F1"].to_numpy(dtype=float)
    pr_v = df_clf["PR-AUC"].to_numpy(dtype=float)
    missed_v = df_clf["Missed_Insecure"].to_numpy(dtype=int)

    ax2.barh(y_c + hc / 2, f1_v, height=hc, color=PALETTE["blue"], edgecolor=PALETTE["navy"], lw=0.7, label="Security F1-Score")
    ax2.barh(y_c - hc / 2, pr_v, height=hc, color=PALETTE["emerald"], edgecolor=PALETTE["navy"], lw=0.7, label="PR-AUC")

    lo_x2 = max(0.0, min(float(np.min(f1_v)), float(np.min(pr_v))) - 0.08)
    ax2.set_xlim(lo_x2, 1.095)
    ax2.set_ylim(-0.6, len(df_clf) + 0.38)
    for idx_m in range(len(df_clf)):
        rm = max(f1_v[idx_m], pr_v[idx_m])
        ax2.text(
            rm + 0.004, idx_m,
            f"F1={f1_v[idx_m]:.4f} | PR={pr_v[idx_m]:.4f} (FP={missed_v[idx_m]})",
            va="center", ha="left", fontsize=6.8, fontweight="bold", color=PALETTE["navy"],
        )
    ax2.set_yticks(y_c)
    ax2.set_yticklabels(short_clf, fontsize=8.0, fontweight="bold")
    ax2.set_title("(b) All 7 Classifiers: F1, PR-AUC & Missed Insecure", fontweight="bold")
    ax2.set_xlabel("Score on Held-Out ID Test")
    ax2.grid(True, axis="x")
    ax2.legend(loc="upper right", fontsize=7.2, ncol=2, framealpha=0.95)

    taus = df_thr["proba_threshold"].to_numpy(dtype=float)
    fnr_ins_pct = df_thr["fnr_insecure"].to_numpy(dtype=float) * 100.0
    fpr_sec_pct = df_thr["fpr_secure"].to_numpy(dtype=float) * 100.0
    f1_thr_pct = df_thr["f1"].to_numpy(dtype=float) * 100.0

    ax3_r = ax3.twinx()
    l1 = ax3.plot(taus, fnr_ins_pct, "o-", color=PALETTE["red"], lw=2.2, markersize=6, label="FNR_insecure (Missed Insecure %)")
    l2 = ax3.plot(taus, fpr_sec_pct, "s--", color=PALETTE["amber"], lw=2.0, markersize=5.8, label="FPR_secure (False Alarm %)")
    l3 = ax3_r.plot(taus, f1_thr_pct, "^:", color=PALETTE["blue"], lw=1.9, markersize=5.8, label="Security F1 (%) [Right]")

    zero_idx = np.where(fnr_ins_pct <= 1e-6)[0]
    if len(zero_idx) > 0:
        tau_safe = float(taus[zero_idx[0]])
        ax3.axvline(tau_safe, color=PALETTE["green"], lw=1.3, ls="-.")
        ax3.annotate(
            f"τ = {tau_safe:.2f}: 0 Missed Insecure\n(FNR=0.0%, FPR={fpr_sec_pct[zero_idx[0]]:.1f}%)",
            xy=(tau_safe, fnr_ins_pct[zero_idx[0]]),
            xytext=(max(0.31, tau_safe - 0.37), max(fnr_ins_pct.max(), fpr_sec_pct.max()) * 0.58),
            fontsize=7.2, fontweight="bold", color=PALETTE["green"],
            bbox=dict(boxstyle="round,pad=0.22", fc="#DCFCE7", ec=PALETTE["green"], lw=0.8),
            arrowprops=dict(arrowstyle="->", color=PALETTE["green"], lw=1.1),
        )

    ax3.set_xlim(float(taus.min()) - 0.04, float(taus.max()) + 0.04)
    ax3.set_ylim(-0.6, max(12.5, float(max(fnr_ins_pct.max(), fpr_sec_pct.max())) * 1.35))
    ax3_r.set_ylim(max(80.0, float(f1_thr_pct.min()) - 4.0), 101.2)
    ax3.set_title("(c) Safety Decision-Threshold Sweep (Cal. GBDT)", fontweight="bold")
    ax3.set_xlabel(r"Decision Threshold $\tau$ on $\hat{P}(\mathrm{Secure})$")
    ax3.set_ylabel("Error Rate (%)", color=PALETTE["crimson"], fontweight="bold")
    ax3_r.set_ylabel("F1-Score (%)", color=PALETTE["blue"], fontweight="bold")
    ax3.grid(True)
    lns = l1 + l2 + l3
    ax3.legend(lns, [ln.get_label() for ln in lns], loc="upper left", fontsize=6.9, framealpha=0.95)

    fig.tight_layout(pad=1.1)
    return save_dual(fig, out_dir, "fig07_classification_and_safety", dpi=dpi)


# ---------------------------------------------------------------------------
# Figure 8: 100% Real Empirical Calibration Curves & Reliability Diagram
# ---------------------------------------------------------------------------
def plot_fig08_calibration_reliability(
    out_dir: Path,
    feat_path: Path,
    models_dir: Path,
    tables_dir: Path,
    dpi: int = 240,
) -> Tuple[Path, Path]:
    with np.load(feat_path, allow_pickle=True) as data:
        X = np.asarray(data["X"], dtype=float)
        y_sec = np.asarray(data["y_secure"], dtype=int).ravel()
        parts = np.asarray([str(p) for p in data["partitions"]])
        feat_names = [str(f) for f in data["feature_names"]] if "feature_names" in data.files else [f"f_{i}" for i in range(X.shape[1])]

    tm = json.loads((models_dir / "training_metrics.json").read_text(encoding="utf-8"))
    uncal_clf = joblib.load(models_dir / "gbdt_clf.joblib")
    cal_platt = joblib.load(models_dir / "calibrated_gb_platt.joblib")
    cal_iso = joblib.load(models_dir / "calibrated_gb_iso.joblib")

    test_mask = parts == "test"
    ood_mask = parts == "ood_test"
    held_mask = np.isin(parts, ["test", "ood_test"]) if np.any(ood_mask) else test_mask

    X_test_df = pd.DataFrame(X[test_mask], columns=feat_names)
    y_test_sec = y_sec[test_mask]
    X_held_df = pd.DataFrame(X[held_mask], columns=feat_names)
    y_held_sec = y_sec[held_mask]

    nadir_scores_test = cal_platt.predict_nadir(X_test_df)
    nadir_scores_held = cal_platt.predict_nadir(X_held_df)
    s_lo = min(float(np.min(nadir_scores_held)), SECURITY_THRESHOLD_HZ - 0.45)
    s_hi = max(float(np.max(nadir_scores_held)), SECURITY_THRESHOLD_HZ + 0.40)
    score_grid = np.linspace(s_lo, s_hi, 400)
    p_grid_platt = cal_platt.calibrate_scores(score_grid)[:, 1]
    p_grid_iso = cal_iso.calibrate_scores(score_grid)[:, 1]

    fig = plt.figure(figsize=(15.8, 5.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.12, 1.0], height_ratios=[1.0, 1.0], wspace=0.31, hspace=0.40)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_rel = fig.add_subplot(gs[:, 1])
    ax_hist = fig.add_subplot(gs[0, 2])
    ax_bar = fig.add_subplot(gs[1, 2])

    ax_map.scatter(
        nadir_scores_held, y_held_sec, s=22, alpha=0.28, color=PALETTE["slate"],
        label=f"Held-Out Outcomes (ID+OOD, N={len(y_held_sec)})",
    )
    ax_map.scatter(
        nadir_scores_test, y_test_sec, s=36, alpha=0.85, facecolors="none", edgecolors=PALETTE["navy"], lw=1.0,
        label=f"ID Test Outcomes (N={len(y_test_sec)})",
    )
    ax_map.plot(
        score_grid, p_grid_platt, color=PALETTE["blue"], lw=2.3,
        label=f"Platt Sigmoid (A={cal_platt.platt_a:.1f}, N_cal={cal_platt.n_cal_})",
    )
    ax_map.plot(
        score_grid, p_grid_iso, color=PALETTE["emerald"], lw=2.1,
        label=f"PAVA Isotonic ({len(cal_iso.isotonic_x)} knots, {cal_iso.effective_method_})",
    )
    ax_map.axvline(SECURITY_THRESHOLD_HZ, color=PALETTE["amber"], lw=1.4, ls="--", label="49.50 Hz Threshold")
    ax_map.axhline(0.50, color=PALETTE["muted"], lw=0.9, ls=":")
    ax_map.set_xlim(max(s_lo, 48.85), min(s_hi, 50.00))
    ax_map.set_ylim(-0.05, 1.05)
    ax_map.set_title("(a) Nadir-to-Probability Calibration Curves", fontweight="bold")
    ax_map.set_xlabel(r"Predicted Continuous Nadir Score $\hat{f}_{\mathrm{nadir}}$ (Hz)")
    ax_map.set_ylabel(r"Calibrated Probability $\hat{P}(\mathrm{Secure})$")
    ax_map.grid(True)
    ax_map.legend(loc="upper left", fontsize=7.0, framealpha=0.95)

    model_specs = [
        ("Uncalibrated GBDT", uncal_clf, PALETTE["red"], "o", "--"),
        ("Calibrated GBDT (Platt)", cal_platt, PALETTE["blue"], "s", "-"),
        ("Calibrated GBDT (Isotonic)", cal_iso, PALETTE["emerald"], "^", "-"),
    ]
    ax_rel.plot([0.0, 1.0], [0.0, 1.0], color=PALETTE["navy"], lw=1.3, ls=":", label="Ideal 1:1 Calibration")

    bin_counts_dict: Dict[str, List[int]] = {}
    for m_lbl, m_obj, m_col, m_mk, m_ls in model_specs:
        probs_te = m_obj.predict_proba(X_test_df)[:, 1]
        cm_te = compute_calibration_metrics(y_test_sec, probs_te, n_bins=10, threshold=SECURITY_THRESHOLD_HZ)

        probs_all = m_obj.predict_proba(X_held_df)[:, 1]
        cm_all = compute_calibration_metrics(y_held_sec, probs_all, n_bins=10, threshold=SECURITY_THRESHOLD_HZ)
        bin_counts_dict[m_lbl] = cm_all["bin_counts"]

        non_empty = [i for i, c in enumerate(cm_all["bin_counts"]) if c > 0]
        confs = [cm_all["bin_confidences"][i] for i in non_empty]
        accs = [cm_all["bin_accuracies"][i] for i in non_empty]
        cnts = [cm_all["bin_counts"][i] for i in non_empty]

        ax_rel.plot(
            confs, accs, marker=m_mk, ls=m_ls, color=m_col, lw=1.9, markersize=5.8,
            label=f"{m_lbl}\n(ID Brier={cm_te['brier_score']:.4f}, Held-Out ECE={cm_all['ece']:.4f})",
        )
        if "Platt" in m_lbl:
            for cx, cy, cn in zip(confs, accs, cnts):
                if cx < 0.05:
                    tx, ty, ha_val = cx + 0.02, cy + 0.04, "left"
                elif cx > 0.92:
                    tx, ty, ha_val = cx - 0.03, cy - 0.08, "right"
                elif 0.58 <= cx <= 0.68 and cy > 0.75:
                    tx, ty, ha_val = cx + 0.035, cy - 0.03, "left"
                elif cy > 0.88:
                    tx, ty, ha_val = cx, cy - 0.068, "center"
                else:
                    tx, ty, ha_val = cx, cy + 0.042, "center"
                ax_rel.annotate(
                    f"n={cn}", xy=(cx, cy), xytext=(tx, ty),
                    ha=ha_val, fontsize=6.6, fontweight="bold", color=m_col,
                )

    ax_rel.set_xlim(-0.02, 1.02)
    ax_rel.set_ylim(-0.02, 1.05)
    ax_rel.set_title(f"(b) 10-Bin Reliability Diagram (Held-Out N={len(y_held_sec)})", fontweight="bold")
    ax_rel.set_xlabel(r"Mean Bin Predicted Probability $\hat{P}(\mathrm{Secure})$")
    ax_rel.set_ylabel("Empirical Fraction Secure")
    ax_rel.grid(True)
    ax_rel.legend(loc="upper left", fontsize=6.7, framealpha=0.95)

    bins_x = np.arange(10)
    bw = 0.27
    ax_hist.bar(bins_x - bw, bin_counts_dict["Uncalibrated GBDT"], width=bw, color=PALETTE["red"], alpha=0.82, label="Uncal. GBDT")
    ax_hist.bar(bins_x, bin_counts_dict["Calibrated GBDT (Platt)"], width=bw, color=PALETTE["blue"], alpha=0.85, label="Platt")
    ax_hist.bar(bins_x + bw, bin_counts_dict["Calibrated GBDT (Isotonic)"], width=bw, color=PALETTE["emerald"], alpha=0.85, label="Isotonic")
    ax_hist.set_yscale("log")
    ax_hist.set_xticks(bins_x)
    ax_hist.set_xticklabels([f".{i}–.{i+1}" if i < 9 else ".9–1.0" for i in range(10)], rotation=28, ha="right", fontsize=6.9)
    ax_hist.set_title("(c) Bin Sample Counts (Log Scale)", fontweight="bold", fontsize=9.5)
    ax_hist.set_ylabel("Count")
    ax_hist.grid(True, axis="y")
    ax_hist.legend(loc="upper center", fontsize=6.5, ncol=3, framealpha=0.92)

    val_c = tm.get("val_metrics", {}).get("classification", {})
    test_c = tm.get("test_metrics", {}).get("classification", {})
    keys_m = [("gbdt_clf", "Uncal.\nGBDT"), ("calibrated_gb_platt", "Platt\nGBDT"), ("calibrated_gb_iso", "Isotonic\nGBDT")]
    xm = np.arange(len(keys_m))
    wm = 0.20
    v_brier = [float(val_c.get(k, {}).get("brier_score", 0.0)) for k, _ in keys_m]
    t_brier = [float(test_c.get(k, {}).get("Brier", 0.0)) for k, _ in keys_m]
    t_ece = [float(test_c.get(k, {}).get("ECE", 0.0)) for k, _ in keys_m]

    b_vb = ax_bar.bar(xm - wm, v_brier, width=wm, color="#93C5FD", edgecolor=PALETTE["navy"], lw=0.6, label="Val Brier")
    b_tb = ax_bar.bar(xm, t_brier, width=wm, color=PALETTE["blue"], edgecolor=PALETTE["navy"], lw=0.6, label="Test Brier")
    b_te = ax_bar.bar(xm + wm, t_ece, width=wm, color=PALETTE["violet"], edgecolor=PALETTE["navy"], lw=0.6, label="Test ECE")
    for bar in list(b_vb) + list(b_tb) + list(b_te):
        hv = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width() / 2.0, hv + 0.0008, f"{hv:.3f}", ha="center", va="bottom", fontsize=6.1, fontweight="bold")
    ax_bar.set_xticks(xm)
    ax_bar.set_xticklabels([lbl for _, lbl in keys_m], fontsize=7.8, fontweight="bold")
    ax_bar.set_ylim(0.0, max(max(v_brier), max(t_brier), max(t_ece)) * 1.34 + 0.004)
    ax_bar.set_title("(d) Val vs. ID Test Brier & ECE", fontweight="bold", fontsize=9.5)
    ax_bar.set_ylabel("Score")
    ax_bar.grid(True, axis="y")
    ax_bar.legend(loc="upper right", fontsize=6.5, ncol=3, framealpha=0.92)

    return save_dual(fig, out_dir, "fig08_calibration_reliability", dpi=dpi)


# ---------------------------------------------------------------------------
# Figure 9: Unsmoothed Observation Window Sensitivity (0.5 s – 2.0 s)
# ---------------------------------------------------------------------------
def plot_fig09_window_sensitivity(
    out_dir: Path,
    tables_dir: Path,
    dpi: int = 240,
) -> Tuple[Path, Path]:
    df_id = pd.read_csv(tables_dir / "window_sensitivity.csv")
    ood_p = tables_dir / "window_sensitivity_ood.csv"
    df_ood = pd.read_csv(ood_p) if ood_p.exists() else pd.DataFrame()

    w = df_id["window_obs"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.6))
    ax1, ax2, ax3, ax4 = axes.ravel()

    id_mae = df_id["mae"].to_numpy(dtype=float) * 1000.0
    id_rmse = df_id["rmse"].to_numpy(dtype=float) * 1000.0
    ax1.plot(w, id_mae, "o-", color=PALETTE["blue"], lw=2.2, markersize=6.5, label="ID Test MAE (mHz)")
    ax1.plot(w, id_rmse, "s--", color="#38BDF8", lw=1.9, markersize=5.8, label="ID Test RMSE (mHz)")
    if not df_ood.empty:
        ood_mae = df_ood["mae"].to_numpy(dtype=float) * 1000.0
        ax1.plot(df_ood["window_obs"], ood_mae, "^-.", color=PALETTE["orange"], lw=2.0, markersize=6.0, label="OOD Test MAE (mHz)")
        for wx, vy in zip(df_ood["window_obs"], ood_mae):
            ax1.text(wx, vy + 0.9, f"{vy:.1f}", ha="center", va="bottom", fontsize=7.2, fontweight="bold", color=PALETTE["orange"])
    for wx, vy in zip(w, id_mae):
        ax1.text(wx, vy - 1.8, f"{vy:.1f}", ha="center", va="top", fontsize=7.3, fontweight="bold", color=PALETTE["blue"])
    ax1.axvline(1.0, color=PALETTE["green"], lw=1.2, ls=":", label="Primary T_obs = 1.0 s")
    top_a = max(float(np.max(id_rmse)), float(np.max(df_ood["mae"]) * 1000.0) if not df_ood.empty else 0.0)
    ax1.set_xticks(w)
    ax1.set_ylim(max(0.0, float(np.min(id_mae)) - 4.0), top_a * 1.18)
    ax1.set_title("(a) Unsmoothed Nadir MAE & RMSE vs. T_obs", fontweight="bold")
    ax1.set_xlabel("Causal Observation Window T_obs (s)")
    ax1.set_ylabel("Nadir Error (mHz)")
    ax1.grid(True)
    ax1.legend(loc="upper right", fontsize=7.3, framealpha=0.95)

    id_f1 = df_id["f1"].to_numpy(dtype=float) * 100.0
    ax2.plot(w, id_f1, "o-", color=PALETTE["blue"], lw=2.2, markersize=6.5, label="Held-Out ID Test F1 (%)")
    if not df_ood.empty:
        ood_f1 = df_ood["f1"].to_numpy(dtype=float) * 100.0
        ax2.plot(df_ood["window_obs"], ood_f1, "^-.", color=PALETTE["orange"], lw=2.0, markersize=6.0, label="Held-Out OOD Test F1 (%)")
        for wx, vy_id, vy_oo in zip(w, id_f1, ood_f1):
            if vy_id >= vy_oo:
                ax2.text(wx, vy_id + 0.22, f"{vy_id:.2f}%", ha="center", va="bottom", fontsize=7.2, fontweight="bold", color=PALETTE["blue"])
                ax2.text(wx, vy_oo - 0.28, f"{vy_oo:.2f}%", ha="center", va="top", fontsize=7.2, fontweight="bold", color=PALETTE["orange"])
            else:
                ax2.text(wx, vy_oo + 0.22, f"{vy_oo:.2f}%", ha="center", va="bottom", fontsize=7.2, fontweight="bold", color=PALETTE["orange"])
                ax2.text(wx, vy_id - 0.28, f"{vy_id:.2f}%", ha="center", va="top", fontsize=7.2, fontweight="bold", color=PALETTE["blue"])
    else:
        for wx, vy in zip(w, id_f1):
            ax2.text(wx, vy + 0.22, f"{vy:.2f}%", ha="center", va="bottom", fontsize=7.3, fontweight="bold", color=PALETTE["blue"])
    ax2.axvline(1.0, color=PALETTE["green"], lw=1.2, ls=":")
    min_f1 = min(float(np.min(id_f1)), float(np.min(df_ood["f1"]) * 100.0) if not df_ood.empty else 100.0)
    ax2.set_xticks(w)
    ax2.set_ylim(max(85.0, min_f1 - 1.5), 100.9)
    ax2.set_title("(b) Unsmoothed Binary Security F1-Score vs. T_obs", fontweight="bold")
    ax2.set_xlabel("Causal Observation Window T_obs (s)")
    ax2.set_ylabel("Security F1-Score (%)")
    ax2.grid(True)
    ax2.legend(loc="lower right", fontsize=7.4, framealpha=0.95)

    id_ece = df_id["ece"].to_numpy(dtype=float)
    id_brier = df_id["brier_score"].to_numpy(dtype=float)
    xw = np.arange(len(w))
    bw = 0.25
    b_e = ax3.bar(xw - bw, id_ece, width=bw, color=PALETTE["violet"], edgecolor=PALETTE["navy"], lw=0.7, label="ID Test ECE")
    b_b = ax3.bar(xw, id_brier, width=bw, color=PALETTE["teal"], edgecolor=PALETTE["navy"], lw=0.7, label="ID Test Brier")
    bars_to_label = list(b_e) + list(b_b)
    max_c = max(float(np.max(id_ece)), float(np.max(id_brier)))
    if not df_ood.empty:
        ood_ece = df_ood["ece"].to_numpy(dtype=float)
        b_oe = ax3.bar(xw + bw, ood_ece, width=bw, color=PALETTE["orange"], edgecolor=PALETTE["navy"], lw=0.7, label="OOD Test ECE")
        bars_to_label.extend(list(b_oe))
        max_c = max(max_c, float(np.max(ood_ece)))
    for bar in bars_to_label:
        hv = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2.0, hv + 0.001, f"{hv:.3f}", ha="center", va="bottom", fontsize=6.6, fontweight="bold")
    ax3.set_xticks(xw)
    ax3.set_xticklabels([f"{val:.1f} s" for val in w])
    ax3.set_ylim(0.0, max_c * 1.26)
    ax3.set_title("(c) Unsmoothed Calibration ECE & Brier Across Windows", fontweight="bold")
    ax3.set_xlabel("Causal Observation Window T_obs")
    ax3.set_ylabel("Calibration Metric")
    ax3.grid(True, axis="y")
    ax3.legend(loc="upper right", fontsize=7.3, framealpha=0.95)

    lead_mean = df_id["t_lead"].to_numpy(dtype=float)
    lead_min = df_id["t_lead_min"].to_numpy(dtype=float)
    ax4.fill_between(w, lead_min, lead_mean, color="#BBF7D0", alpha=0.65, label="ID [Min, Mean] t_lead Range")
    ax4.plot(w, lead_mean, "o-", color=PALETTE["green"], lw=2.3, markersize=6.5, label="ID Mean Lead Time t_lead (s)")
    ax4.plot(w, lead_min, "s--", color="#047857", lw=1.7, markersize=5.5, label="ID Minimum Lead Time (s)")
    if not df_ood.empty and "t_lead_min" in df_ood.columns:
        ood_lmin = df_ood["t_lead_min"].to_numpy(dtype=float)
        ood_cpct = df_ood["causal_valid_ratio"].to_numpy(dtype=float) * 100.0
        ax4.plot(df_ood["window_obs"], ood_lmin, "v:", color=PALETTE["red"], lw=1.7, markersize=5.8, label="OOD Minimum Lead Time (s)")
        for wx, vy, cp in zip(df_ood["window_obs"], ood_lmin, ood_cpct):
            if cp < 99.9:
                ax4.annotate(
                    f"{cp:.1f}% causal\n(min {vy:+.2f}s)",
                    xy=(wx, vy), xytext=(wx - 0.45, vy + 0.25),
                    fontsize=7.0, fontweight="bold", color=PALETTE["crimson"],
                    bbox=dict(boxstyle="round,pad=0.18", fc="#FEE2E2", ec=PALETTE["red"], lw=0.75),
                    arrowprops=dict(arrowstyle="->", color=PALETTE["red"], lw=0.9),
                )

    for wx, vm, vmin in zip(w, lead_mean, lead_min):
        ax4.text(wx, vm + 0.06, f"{vm:.2f}s\n(min {vmin:.2f}s)", ha="center", va="bottom", fontsize=7.0, fontweight="bold", color=PALETTE["navy"])

    ax4.axhline(0.0, color=PALETTE["red"], lw=1.1, ls="--")
    ax4.axvline(1.0, color=PALETTE["green"], lw=1.2, ls=":")
    lo_d = min(0.0, float(np.min(lead_min)), float(np.min(df_ood["t_lead_min"])) if not df_ood.empty else 0.0) - 0.18
    ax4.set_xticks(w)
    ax4.set_ylim(lo_d, float(np.max(lead_mean)) + 0.42)
    ax4.set_title("(d) Measured Pre-Nadir Warning Lead Time t_lead", fontweight="bold")
    ax4.set_xlabel("Causal Observation Window T_obs (s)")
    ax4.set_ylabel("Lead Time Before Nadir (s)")
    ax4.grid(True)
    ax4.legend(loc="upper right", fontsize=7.1, framealpha=0.95)

    fig.tight_layout(pad=1.15)
    return save_dual(fig, out_dir, "fig09_window_sensitivity", dpi=dpi)


# ---------------------------------------------------------------------------
# Figure 10: Out-of-Distribution Generalization Across Regimes
# ---------------------------------------------------------------------------
def plot_fig10_ood_generalization(
    out_dir: Path,
    tables_dir: Path,
    dpi: int = 240,
) -> Tuple[Path, Path]:
    reg_path = tables_dir / "ood_regime_breakdown.csv"
    sum_path = tables_dir / "ood_generalization.csv"
    if reg_path.exists():
        df_r = pd.read_csv(reg_path)
    else:
        df_s = pd.read_csv(sum_path).set_index("Metric")
        id_c = [c for c in df_s.columns if c.startswith("ID")][0]
        ood_c = [c for c in df_s.columns if c.startswith("OOD")][0]
        df_r = pd.DataFrame([
            {
                "Regime": "ID Held-Out Test", "N": 0,
                "MAE (Hz)": df_s.loc["MAE", id_c], "P95 AE (Hz)": df_s.loc["P95_AE", id_c],
                "F1": df_s.loc["F1", id_c], "FNR_insecure": df_s.loc["FNR_INSECURE", id_c],
                "ECE": df_s.loc["ECE", id_c], "Brier": df_s.loc["BRIER_SCORE", id_c], "Interval_Hz": 0.0,
            },
            {
                "Regime": "OOD-All", "N": 0,
                "MAE (Hz)": df_s.loc["MAE", ood_c], "P95 AE (Hz)": df_s.loc["P95_AE", ood_c],
                "F1": df_s.loc["F1", ood_c], "FNR_insecure": df_s.loc["FNR_INSECURE", ood_c],
                "ECE": df_s.loc["ECE", ood_c], "Brier": df_s.loc["BRIER_SCORE", ood_c], "Interval_Hz": 0.0,
            },
        ])

    short_regimes = []
    for raw_r, n_cnt in zip(df_r["Regime"], df_r["N"]):
        s = (
            str(raw_r)
            .replace("ID Held-Out Test", "ID Test")
            .replace("OOD-RES (40-50%)", "OOD-RES\n(40–50%)")
            .replace("OOD-Load (0.80, 1.20)", "OOD-Load\n(0.80, 1.20)")
            .replace("OOD-Combined (RES + Load)", "OOD-\nCombined")
            .replace("OOD-All", "OOD-All\nPooled")
        )
        short_regimes.append(f"{s}\n(N={int(n_cnt)})" if int(n_cnt) > 0 else s)

    x = np.arange(len(df_r))
    bar_cols = [PALETTE["blue"], PALETTE["red"], PALETTE["teal"], PALETTE["violet"], PALETTE["orange"]][: len(df_r)]

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.6))
    ax1, ax2, ax3, ax4 = axes.ravel()

    mae_mhz = df_r["MAE (Hz)"].to_numpy(dtype=float) * 1000.0
    p95_mhz = df_r["P95 AE (Hz)"].to_numpy(dtype=float) * 1000.0
    w = 0.35
    b1 = ax1.bar(x - w / 2, mae_mhz, width=w, color=PALETTE["blue"], edgecolor=PALETTE["navy"], lw=0.7, label="MAE (mHz)")
    b2 = ax1.bar(x + w / 2, p95_mhz, width=w, color="#F59E0B", edgecolor=PALETTE["navy"], lw=0.7, label="P95 |Error| (mHz)")
    for bar in list(b1) + list(b2):
        hv = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, hv + 1.5, f"{hv:.1f}", ha="center", va="bottom", fontsize=7.0, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_regimes, fontsize=7.9, fontweight="bold")
    ax1.set_ylim(0.0, float(np.max(p95_mhz)) * 1.22)
    ax1.set_title("(a) Continuous Nadir Error Across ID & OOD Regimes", fontweight="bold")
    ax1.set_ylabel("Error (mHz)")
    ax1.grid(True, axis="y")
    ax1.legend(loc="upper right", fontsize=7.4, framealpha=0.95)

    f1_pct = df_r["F1"].to_numpy(dtype=float) * 100.0
    fnr_pct = df_r["FNR_insecure"].to_numpy(dtype=float) * 100.0
    ax2_r = ax2.twinx()
    bf1 = ax2.bar(x - w / 2, f1_pct, width=w, color=PALETTE["emerald"], edgecolor=PALETTE["navy"], lw=0.7, label="Security F1 (%) [Left]")
    bfnr = ax2_r.bar(x + w / 2, fnr_pct, width=w, color=PALETTE["red"], alpha=0.82, edgecolor=PALETTE["navy"], lw=0.7, label="FNR_insecure (%) [Right]")
    for bar in bf1:
        hv = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, hv + 0.22, f"{hv:.1f}%", ha="center", va="bottom", fontsize=6.9, fontweight="bold", color="#065F46")
    for bar in bfnr:
        hv = bar.get_height()
        ax2_r.text(bar.get_x() + bar.get_width() / 2.0, hv + 0.15, f"{hv:.1f}%", ha="center", va="bottom", fontsize=6.9, fontweight="bold", color=PALETTE["crimson"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_regimes, fontsize=7.9, fontweight="bold")
    ax2.set_ylim(max(85.0, float(np.min(f1_pct)) - 3.5), 103.4)
    ax2_r.set_ylim(0.0, max(6.5, float(np.max(fnr_pct)) * 1.65))
    ax2.set_title("(b) Security F1 & Safety Missed-Insecure Rate", fontweight="bold")
    ax2.set_ylabel("F1-Score (%)", color="#065F46", fontweight="bold")
    ax2_r.set_ylabel("FNR_insecure (%)", color=PALETTE["crimson"], fontweight="bold")
    ax2.grid(True, axis="y")
    h_leg = [Patch(facecolor=PALETTE["emerald"], edgecolor=PALETTE["navy"], label="F1 (%) [Left]"),
             Patch(facecolor=PALETTE["red"], edgecolor=PALETTE["navy"], label="FNR_insecure (%) [Right]")]
    ax2.legend(handles=h_leg, loc="upper right", fontsize=7.1, ncol=2, framealpha=0.95)

    ece_v = df_r["ECE"].to_numpy(dtype=float)
    brier_v = df_r["Brier"].to_numpy(dtype=float)
    be1 = ax3.bar(x - w / 2, ece_v, width=w, color=PALETTE["violet"], edgecolor=PALETTE["navy"], lw=0.7, label="ECE (10-bin)")
    be2 = ax3.bar(x + w / 2, brier_v, width=w, color=PALETTE["sky"], edgecolor=PALETTE["navy"], lw=0.7, label="Brier Score")
    for bar in list(be1) + list(be2):
        hv = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2.0, hv + 0.0007, f"{hv:.4f}", ha="center", va="bottom", fontsize=6.6, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(short_regimes, fontsize=7.9, fontweight="bold")
    ax3.set_ylim(0.0, max(float(np.max(ece_v)), float(np.max(brier_v))) * 1.28)
    ax3.set_title("(c) Probability Calibration Stability Under Domain Shift", fontweight="bold")
    ax3.set_ylabel("Calibration Error")
    ax3.grid(True, axis="y")
    ax3.legend(loc="upper right", fontsize=7.4, framealpha=0.95)

    iv_hz = df_r["Interval_Hz"].to_numpy(dtype=float)
    b_iv = ax4.bar(x, iv_hz, width=0.50, color=bar_cols, edgecolor=PALETTE["navy"], lw=0.8)
    id_iv = iv_hz[0] if len(iv_hz) > 0 and iv_hz[0] > 0 else 1.0
    ax4.axhline(id_iv, color=PALETTE["navy"], lw=1.2, ls="--", label=f"ID Baseline Width ({id_iv:.3f} Hz)")
    for bar, val in zip(b_iv, iv_hz):
        ratio_pct = (val / id_iv - 1.0) * 100.0
        tag = f"{val:.3f} Hz" if abs(ratio_pct) < 0.05 else f"{val:.3f} Hz\n({ratio_pct:+.1f}%)"
        ax4.text(bar.get_x() + bar.get_width() / 2.0, val + 0.006, tag, ha="center", va="bottom", fontsize=7.0, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(short_regimes, fontsize=7.9, fontweight="bold")
    lo_iv = max(0.0, float(np.min(iv_hz)) * 0.82)
    ax4.set_ylim(lo_iv, float(np.max(iv_hz)) * 1.12)
    ax4.set_title("(d) 95% Quantile/OOD Uncertainty Interval Width", fontweight="bold")
    ax4.set_ylabel("Mean 95% Nadir PI Width (Hz)")
    ax4.grid(True, axis="y")
    ax4.legend(loc="lower right", fontsize=7.3, framealpha=0.95)

    fig.tight_layout(pad=1.15)
    return save_dual(fig, out_dir, "fig10_ood_generalization", dpi=dpi)


# ---------------------------------------------------------------------------
# Figure 11: Unsmoothed PMU Measurement Noise Robustness (0–50 mHz)
# ---------------------------------------------------------------------------
def plot_fig11_pmu_noise_robustness(
    out_dir: Path,
    tables_dir: Path,
    dpi: int = 240,
) -> Tuple[Path, Path]:
    df_n = pd.read_csv(tables_dir / "measurement_noise.csv")
    sig_mhz = df_n["noise_std_hz"].to_numpy(dtype=float) * 1000.0

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.4))
    ax1, ax2, ax3, ax4 = axes.ravel()

    mae_v = df_n["mae"].to_numpy(dtype=float) * 1000.0
    rmse_v = df_n["rmse"].to_numpy(dtype=float) * 1000.0
    p95_v = df_n["p95_ae"].to_numpy(dtype=float) * 1000.0
    ax1.axvspan(0.0, 5.0, color="#DCFCE7", alpha=0.65, label="IEEE C37.118 Nominal Band (≤ 5 mHz)")
    ax1.plot(sig_mhz, mae_v, "o-", color=PALETTE["blue"], lw=2.2, markersize=6.0, label="Nadir MAE (mHz)")
    ax1.plot(sig_mhz, rmse_v, "s--", color="#38BDF8", lw=1.9, markersize=5.6, label="Nadir RMSE (mHz)")
    ax1.plot(sig_mhz, p95_v, "^:", color=PALETTE["amber"], lw=1.8, markersize=5.6, label="P95 |Error| (mHz)")
    for sx, my in zip(sig_mhz, mae_v):
        ax1.text(sx, my - 1.3, f"{my:.1f}", ha="center", va="top", fontsize=7.0, fontweight="bold", color=PALETTE["blue"])
    ax1.set_ylim(0.0, float(np.max(p95_v)) * 1.18)
    ax1.set_title("(a) Unsmoothed Nadir Error vs. PMU Frequency Noise", fontweight="bold")
    ax1.set_xlabel("PMU Frequency Noise Std Dev σ_f (mHz)")
    ax1.set_ylabel("Nadir Error (mHz)")
    ax1.grid(True)
    ax1.legend(loc="upper left", fontsize=7.2, framealpha=0.95)

    f1_pct = df_n["f1"].to_numpy(dtype=float) * 100.0
    fnr_pct = df_n["fnr_insecure"].to_numpy(dtype=float) * 100.0
    ax2_r = ax2.twinx()
    ax2.axvspan(0.0, 5.0, color="#DCFCE7", alpha=0.65)
    l1 = ax2.plot(sig_mhz, f1_pct, "o-", color=PALETTE["emerald"], lw=2.2, markersize=6.0, label="Security F1 (%) [Left]")
    l2 = ax2_r.plot(sig_mhz, fnr_pct, "s--", color=PALETTE["red"], lw=2.0, markersize=5.8, label="FNR_insecure (%) [Right]")
    ax2.set_ylim(max(85.0, float(np.min(f1_pct)) - 3.5), 100.8)
    ax2_r.set_ylim(0.0, max(8.0, float(np.max(fnr_pct)) * 1.6))
    ax2.set_title("(b) Security F1 & Missed-Insecure Rate Under Noise", fontweight="bold")
    ax2.set_xlabel("PMU Frequency Noise Std Dev σ_f (mHz)")
    ax2.set_ylabel("Security F1 (%)", color=PALETTE["emerald"], fontweight="bold")
    ax2_r.set_ylabel("FNR_insecure (%)", color=PALETTE["red"], fontweight="bold")
    ax2.grid(True)
    lns = l1 + l2
    ax2.legend(lns, [ln.get_label() for ln in lns], loc="center right", fontsize=7.3, framealpha=0.95)

    ece_v = df_n["ece"].to_numpy(dtype=float)
    brier_v = df_n["brier_score"].to_numpy(dtype=float)
    ax3.axvspan(0.0, 5.0, color="#DCFCE7", alpha=0.65, label="IEEE C37.118 Nominal (≤ 5 mHz)")
    ax3.plot(sig_mhz, ece_v, "o-", color=PALETTE["violet"], lw=2.2, markersize=6.0, label="Expected Calibration Error (ECE)")
    ax3.plot(sig_mhz, brier_v, "s--", color=PALETTE["teal"], lw=2.0, markersize=5.8, label="Brier Score")
    for idx_n, (sx, ey) in enumerate(zip(sig_mhz, ece_v)):
        if idx_n == 0:
            ax3.text(sx, ey - 0.0020, f"{ey:.3f}", ha="center", va="top", fontsize=7.0, fontweight="bold", color=PALETTE["violet"])
        else:
            ax3.text(sx, ey + 0.0014, f"{ey:.3f}", ha="center", va="bottom", fontsize=7.0, fontweight="bold", color=PALETTE["violet"])
    ax3.set_ylim(0.0, max(float(np.max(ece_v)), float(np.max(brier_v))) * 1.26)
    ax3.set_title("(c) Unsmoothed Calibration ECE & Brier vs. PMU Noise", fontweight="bold")
    ax3.set_xlabel("PMU Frequency Noise Std Dev σ_f (mHz)")
    ax3.set_ylabel("Calibration Metric")
    ax3.grid(True)
    ax3.legend(loc="upper left", fontsize=7.2, framealpha=0.95)

    mad_v = df_n["prob_stability_mad"].to_numpy(dtype=float)
    ax4.axvspan(0.0, 5.0, color="#DCFCE7", alpha=0.65, label="IEEE C37.118 Nominal (≤ 5 mHz)")
    ax4.plot(sig_mhz, mad_v, "D-", color=PALETTE["orange"], lw=2.2, markersize=6.0, label=r"Probability Drift MAD $|\hat{P}_{\mathrm{noisy}} - \hat{P}_{\mathrm{clean}}|$")
    for sx, mv in zip(sig_mhz, mad_v):
        ax4.text(sx, mv + max(0.0005, float(np.max(mad_v)) * 0.04), f"{mv:.4f}", ha="center", va="bottom", fontsize=7.1, fontweight="bold", color=PALETTE["orange"])
    ax4.set_ylim(-0.0008, max(0.005, float(np.max(mad_v)) * 1.26))
    ax4.set_title("(d) Predicted Probability Stability Across Noise Levels", fontweight="bold")
    ax4.set_xlabel("PMU Frequency Noise Std Dev σ_f (mHz)")
    ax4.set_ylabel("Mean Abs. Probability Shift")
    ax4.grid(True)
    ax4.legend(loc="upper left", fontsize=7.2, framealpha=0.95)

    fig.tight_layout(pad=1.15)
    return save_dual(fig, out_dir, "fig11_pmu_noise_robustness", dpi=dpi)


# ---------------------------------------------------------------------------
# Figure 12: Deduplicated 14-Feature Permutation Importance & Ablations
# ---------------------------------------------------------------------------
def plot_fig12_feature_importance_and_ablation(
    out_dir: Path,
    tables_dir: Path,
    dpi: int = 240,
) -> Tuple[Path, Path]:
    df_imp = pd.read_csv(tables_dir / "feature_importance.csv")
    df_ab = pd.read_csv(tables_dir / "feature_ablations.csv")

    df_imp_s = df_imp.sort_values("importance_mean", ascending=True).reset_index(drop=True)

    fig = plt.figure(figsize=(15.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.16, 1.0], height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.36)
    ax_imp = fig.add_subplot(gs[:, 0])
    ax_ab_mae = fig.add_subplot(gs[0, 1])
    ax_ab_f1 = fig.add_subplot(gs[1, 1])

    yf = np.arange(len(df_imp_s))
    means_mhz = df_imp_s["importance_mean"].to_numpy(dtype=float) * 1000.0
    stds_mhz = df_imp_s["importance_std"].to_numpy(dtype=float) * 1000.0
    imp_pct = df_imp_s["impurity_importance"].to_numpy(dtype=float) * 100.0
    f_cols = [FEATURE_FAMILY.get(str(f), ("Other", PALETTE["blue"]))[1] for f in df_imp_s["feature"]]
    f_lbls = [FEATURE_LABELS.get(str(f), str(f)) for f in df_imp_s["feature"]]

    ax_imp.barh(
        yf, np.maximum(0.0, means_mhz), xerr=stds_mhz, height=0.64,
        color=f_cols, edgecolor=PALETTE["navy"], lw=0.65,
        error_kw=dict(ecolor=PALETTE["navy"], elinewidth=1.0, capsize=2.4),
    )
    ax_imp.set_xscale("symlog", linthresh=0.5)
    max_imp_edge = float(np.max(means_mhz + stds_mhz))
    ax_imp.set_xlim(-0.08, max_imp_edge * 26.0)

    for idx_f, (m_v, s_v, g_pct) in enumerate(zip(means_mhz, stds_mhz, imp_pct)):
        anchor = max(0.04, m_v + s_v)
        offset = anchor * 1.15 if anchor >= 0.5 else anchor + 0.06
        ax_imp.text(
            offset, idx_f,
            f"{m_v:+.2f} ± {s_v:.2f} mHz (Gini {g_pct:.1f}%)",
            va="center", ha="left", fontsize=6.8, fontweight="bold", color=PALETTE["navy"],
        )

    ax_imp.set_yticks(yf)
    ax_imp.set_yticklabels(f_lbls, fontsize=8.1, fontweight="bold")
    ax_imp.axvline(0.0, color=PALETTE["slate"], lw=0.8)
    ax_imp.set_title("(a) All 14 Deduplicated Features: Permutation ΔMAE (±1σ)", fontweight="bold")
    ax_imp.set_xlabel("Permutation MAE Increase ΔMAE (mHz, symlog scale)")
    ax_imp.grid(True, axis="x")

    fam_handles = []
    seen_fams = set()
    for fam_name, fam_c in FEATURE_FAMILY.values():
        if fam_name not in seen_fams:
            seen_fams.add(fam_name)
            fam_handles.append(Patch(facecolor=fam_c, edgecolor=PALETTE["navy"], label=fam_name))
    ax_imp.legend(handles=fam_handles, loc="lower right", fontsize=7.2, framealpha=0.95)

    ab_short_map = {
        "all_14": "All 14\n(39b, 50fps)",
        "freq_only": "Freq Only\n(3 feat)",
        "freq_plus_rocof": "Freq+ROCOF\n(7 feat)",
        "no_spatial_angle": "No Spatial\n(9 feat)",
        "spatial_angle_only": "Spatial\n(5 feat)",
        "partial_13bus_pmu": "13/39 PMU\n(33% buses)",
        "fps_25_vs_50": "25 fps\nStream",
    }
    ab_labels = [ab_short_map.get(str(a), str(a)) for a in df_ab["ablation"]]
    xa = np.arange(len(df_ab))
    wa = 0.36

    id_mae_ab = df_ab["id_mae"].to_numpy(dtype=float) * 1000.0
    ood_mae_ab = df_ab["ood_mae"].to_numpy(dtype=float) * 1000.0
    b_idm = ax_ab_mae.bar(xa - wa / 2, id_mae_ab, width=wa, color=PALETTE["blue"], edgecolor=PALETTE["navy"], lw=0.65, label="ID Test MAE (mHz)")
    b_oom = ax_ab_mae.bar(xa + wa / 2, ood_mae_ab, width=wa, color=PALETTE["orange"], edgecolor=PALETTE["navy"], lw=0.65, label="OOD Test MAE (mHz)")
    for bar in list(b_idm) + list(b_oom):
        hv = bar.get_height()
        if np.isfinite(hv):
            ax_ab_mae.text(bar.get_x() + bar.get_width() / 2.0, hv + 0.8, f"{hv:.1f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    ax_ab_mae.set_xticks(xa)
    ax_ab_mae.set_xticklabels(ab_labels, fontsize=7.2, fontweight="bold")
    ax_ab_mae.set_ylim(0.0, float(np.nanmax(np.concatenate([id_mae_ab, ood_mae_ab]))) * 1.22)
    ax_ab_mae.set_title("(b) Feature-Group, 13-Bus PMU & 25-fps Ablations: Nadir MAE", fontweight="bold")
    ax_ab_mae.set_ylabel("MAE (mHz)")
    ax_ab_mae.grid(True, axis="y")
    ax_ab_mae.legend(loc="upper left", fontsize=7.2, framealpha=0.95)

    id_f1_ab = df_ab["id_f1"].to_numpy(dtype=float) * 100.0
    ood_f1_ab = df_ab["ood_f1"].to_numpy(dtype=float) * 1000.0 / 10.0
    b_idf = ax_ab_f1.bar(xa - wa / 2, id_f1_ab, width=wa, color=PALETTE["emerald"], edgecolor=PALETTE["navy"], lw=0.65, label="ID Test F1 (%)")
    b_oof = ax_ab_f1.bar(xa + wa / 2, ood_f1_ab, width=wa, color=PALETTE["teal"], edgecolor=PALETTE["navy"], lw=0.65, label="OOD Test F1 (%)")
    for bar in list(b_idf) + list(b_oof):
        hv = bar.get_height()
        if np.isfinite(hv):
            ax_ab_f1.text(bar.get_x() + bar.get_width() / 2.0, hv + 0.25, f"{hv:.1f}", ha="center", va="bottom", fontsize=6.4, fontweight="bold")
    ax_ab_f1.set_xticks(xa)
    ax_ab_f1.set_xticklabels(ab_labels, fontsize=7.2, fontweight="bold")
    lo_f1_ab = max(80.0, float(np.nanmin(np.concatenate([id_f1_ab, ood_f1_ab]))) - 3.5)
    ax_ab_f1.set_ylim(lo_f1_ab, 102.2)
    ax_ab_f1.set_title("(c) Feature-Group, 13-Bus PMU & 25-fps Ablations: Security F1", fontweight="bold")
    ax_ab_f1.set_ylabel("Security F1 (%)")
    ax_ab_f1.grid(True, axis="y")
    ax_ab_f1.legend(loc="lower right", fontsize=7.2, framealpha=0.95)

    return save_dual(fig, out_dir, "fig12_feature_importance_and_ablation", dpi=dpi)


# ---------------------------------------------------------------------------
# Figure 13: 6-Stage Hardware Latency Profile & Scenario Learning Curve
# ---------------------------------------------------------------------------
def plot_fig13_latency_and_learning_curve(
    out_dir: Path,
    tables_dir: Path,
    dpi: int = 240,
) -> Tuple[Path, Path]:
    df_prof = pd.read_csv(tables_dir / "latency_profile.csv")
    stages_path = tables_dir / "latency_stages.csv"
    if stages_path.exists():
        df_stg = pd.read_csv(stages_path)
    else:
        r0 = df_prof.iloc[0]
        df_stg = pd.DataFrame([
            {"Stage": "1. Causal Window Slicing", "Median (ms)": r0["t_slice_ms"], "P95 (ms)": r0["t_slice_ms"], "P99 (ms)": r0["t_slice_ms"]},
            {"Stage": "2. 14-Feature Extraction", "Median (ms)": r0["t_feat_ms"], "P95 (ms)": r0["t_feat_ms"], "P99 (ms)": r0["t_feat_ms"]},
            {"Stage": "3. Nadir Regression Inference", "Median (ms)": r0["t_reg_ms"], "P95 (ms)": r0["t_reg_ms"], "P99 (ms)": r0["t_reg_ms"]},
            {"Stage": "4. Security Classification Inference", "Median (ms)": r0["t_clf_ms"], "P95 (ms)": r0["t_clf_ms"], "P99 (ms)": r0["t_clf_ms"]},
            {"Stage": "5. Probability Calibration & Decision", "Median (ms)": r0["t_cal_ms"], "P95 (ms)": r0["t_cal_ms"], "P99 (ms)": r0["t_cal_ms"]},
            {"Stage": "6. Total End-to-End Online Prediction", "Median (ms)": r0["t_total_ms"], "P95 (ms)": r0["t_total_p95_ms"], "P99 (ms)": r0["t_total_p99_ms"]},
            {"Stage": "Reference: 10s Time-Domain ODE Simulation", "Median (ms)": r0["sim_time_ms"], "P95 (ms)": r0["sim_time_p95_ms"], "P99 (ms)": r0["sim_time_p95_ms"]},
        ])

    df_lc = pd.read_csv(tables_dir / "learning_curve.csv")
    speedup = float(df_prof["speedup_factor"].iloc[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.8, 5.2), gridspec_kw={"width_ratios": [1.14, 1.0]})

    stage_short = [
        "1. Window\nSlicing",
        "2. 14-Feat.\nExtraction",
        "3. Reg.\nInference",
        "4. Clf.\nInference",
        "5. Prob.\nCalibration",
        "6. Total\nOnline E2E",
        "Ref: 10-s\nODE Sim",
    ][: len(df_stg)]

    xs = np.arange(len(df_stg))
    ws = 0.26
    med_ms = df_stg["Median (ms)"].to_numpy(dtype=float)
    p95_ms = df_stg["P95 (ms)"].to_numpy(dtype=float)
    p99_ms = df_stg["P99 (ms)"].to_numpy(dtype=float)

    ax1.bar(xs - ws, med_ms, width=ws, color=PALETTE["blue"], edgecolor=PALETTE["navy"], lw=0.65, label="Median (ms)")
    ax1.bar(xs, p95_ms, width=ws, color="#38BDF8", edgecolor=PALETTE["navy"], lw=0.65, label="P95 Tail (ms)")
    ax1.bar(xs + ws, p99_ms, width=ws, color=PALETTE["amber"], edgecolor=PALETTE["navy"], lw=0.65, label="P99 Tail (ms)")

    ax1.set_yscale("log")
    ax1.axhline(10.0, color=PALETTE["red"], lw=1.5, ls="--", label="Real-Time Screening Budget (10.0 ms)")

    for idx_s, (m_v, p99_v) in enumerate(zip(med_ms, p99_ms)):
        top_v = max(m_v, p99_v)
        lbl_str = f"{m_v:.2f} ms" if m_v >= 0.1 else f"{m_v:.3f} ms"
        if m_v >= 20.0:
            lbl_str = f"{m_v:.1f} ms"
        ax1.text(xs[idx_s], top_v * 1.24, lbl_str, ha="center", va="bottom", fontsize=7.2, fontweight="bold", color=PALETTE["navy"])

    ax1.set_xticks(xs)
    ax1.set_xticklabels(stage_short, fontsize=7.8, fontweight="bold")
    ax1.set_ylim(max(0.012, float(np.min(med_ms)) * 0.45), float(np.max(p99_ms)) * 3.0)
    ax1.set_title(f"(a) 6-Stage Hardware Latency vs. ODE Sim ({speedup:.1f}× Speedup)", fontweight="bold")
    ax1.set_ylabel("Execution Time per Scenario (ms, log scale)")
    ax1.grid(True, axis="y", which="both")
    ax1.legend(loc="upper left", fontsize=7.4, framealpha=0.95, ncol=2)

    frac_pct = df_lc["train_fraction"].to_numpy(dtype=float) * 100.0
    n_tr_arr = df_lc["n_train_samples"].to_numpy(dtype=int)
    mae_m = df_lc["mae_mean"].to_numpy(dtype=float) * 1000.0
    mae_s = df_lc["mae_std"].to_numpy(dtype=float) * 1000.0
    f1_m = df_lc["f1_mean"].to_numpy(dtype=float) * 100.0
    f1_s = df_lc["f1_std"].to_numpy(dtype=float) * 100.0

    ax2_r = ax2.twinx()
    ax2.fill_between(frac_pct, np.maximum(0.0, mae_m - mae_s), mae_m + mae_s, color="#BFDBFE", alpha=0.55)
    l_mae = ax2.plot(frac_pct, mae_m, "o-", color=PALETTE["blue"], lw=2.3, markersize=6.2, label="Held-Out Test MAE ± 1σ (mHz) [Left]")

    ax2_r.fill_between(frac_pct, np.maximum(0.0, f1_m - f1_s), np.minimum(100.0, f1_m + f1_s), color="#A7F3D0", alpha=0.45)
    l_f1 = ax2_r.plot(frac_pct, f1_m, "s--", color=PALETTE["emerald"], lw=2.1, markersize=6.0, label="Held-Out Security F1 ± 1σ (%) [Right]")

    for fx, mv in zip(frac_pct, mae_m):
        ax2.text(fx, mv + 0.9, f"{mv:.1f}", ha="center", va="bottom", fontsize=7.2, fontweight="bold", color=PALETTE["blue"])

    ax2.set_xticks(frac_pct)
    ax2.set_xticklabels([f"{int(round(f))}%\n(N={n})" for f, n in zip(frac_pct, n_tr_arr)], fontsize=8.0, fontweight="bold")
    ax2.set_ylim(max(0.0, float(np.min(mae_m - mae_s)) - 3.5), float(np.max(mae_m + mae_s)) * 1.20)
    ax2_r.set_ylim(max(80.0, float(np.min(f1_m - f1_s)) - 3.0), 101.2)
    ax2.set_title("(b) Scenario-Level Learning Curve (20%–100% Train)", fontweight="bold")
    ax2.set_xlabel("Training Partition Fraction (Disjoint ID Scenarios)")
    ax2.set_ylabel("Nadir MAE (mHz)", color=PALETTE["blue"], fontweight="bold")
    ax2_r.set_ylabel("Security F1-Score (%)", color=PALETTE["emerald"], fontweight="bold")
    ax2.grid(True)
    lns = l_mae + l_f1
    ax2.legend(lns, [ln.get_label() for ln in lns], loc="center right", fontsize=7.3, framealpha=0.95)

    fig.tight_layout(pad=1.15)
    return save_dual(fig, out_dir, "fig13_latency_and_learning_curve", dpi=dpi)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Generate all 13 publication figures (PNG + PDF)")
    parser.add_argument("--output-dir", type=str, default="results/figures", help="Output directory for figures")
    parser.add_argument("--raw-dir", "--data-dir", dest="raw_dir", type=str, default="data/raw", help="Observable PMU .npz directory")
    parser.add_argument("--raw-sim-dir", type=str, default="data/raw_sim", help="Internal ODE simulator .npz directory")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Processed feature directory")
    parser.add_argument("--features", type=str, default="data/processed/features.npz", help="Primary features.npz path")
    parser.add_argument("--models-dir", type=str, default="models", help="Trained models directory")
    parser.add_argument("--tables-dir", type=str, default="results/tables", help="Benchmark CSV tables directory")
    parser.add_argument("--dpi", type=int, default=240, help="Raster PNG export DPI")
    return parser.parse_args(args)


def main(args=None) -> int:
    parsed = parse_args(args)
    out_dir = Path(parsed.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for stale in out_dir.glob("fig[1-9]_*.p*"):
        stale.unlink(missing_ok=True)

    raw_dir = Path(parsed.raw_dir)
    raw_sim_dir = Path(parsed.raw_sim_dir)
    proc_dir = Path(parsed.processed_dir)
    feat_path = Path(parsed.features)
    models_dir = Path(parsed.models_dir)
    tables_dir = Path(parsed.tables_dir)
    dpi = int(parsed.dpi)

    exported = [
        plot_fig01_system_and_architecture(out_dir, feat_path, models_dir, tables_dir, dpi=dpi),
        plot_fig02_generation_loss_response(out_dir, raw_dir, raw_sim_dir, dpi=dpi),
        plot_fig03_causal_window_illustration(out_dir, raw_dir, raw_sim_dir, proc_dir, dpi=dpi),
        plot_fig04_pmu_and_feature_distributions(out_dir, proc_dir, tables_dir, dpi=dpi),
        plot_fig05_nadir_parity_and_residuals(out_dir, feat_path, models_dir, dpi=dpi),
        plot_fig06_regression_model_comparison(out_dir, tables_dir, dpi=dpi),
        plot_fig07_classification_and_safety(out_dir, tables_dir, dpi=dpi),
        plot_fig08_calibration_reliability(out_dir, feat_path, models_dir, tables_dir, dpi=dpi),
        plot_fig09_window_sensitivity(out_dir, tables_dir, dpi=dpi),
        plot_fig10_ood_generalization(out_dir, tables_dir, dpi=dpi),
        plot_fig11_pmu_noise_robustness(out_dir, tables_dir, dpi=dpi),
        plot_fig12_feature_importance_and_ablation(out_dir, tables_dir, dpi=dpi),
        plot_fig13_latency_and_learning_curve(out_dir, tables_dir, dpi=dpi),
    ]

    print(f"Exported {len(exported)} PNG + {len(exported)} PDF figures ({2 * len(exported)} files) to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
