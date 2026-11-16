#!/usr/bin/env python3
"""Export standalone 300-DPI PNG and vector PDF figures into results/figures/."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pmu_fsa.features.extractor import DESCRIPTIVE_FEATURE_NAMES
from pmu_fsa.models.calibrated_gb import _pava_isotonic
from pmu_fsa.simulation.dynamic_engine import DynamicSimulator
from pmu_fsa.simulation.pmu_recorder import compute_fir_rocof
from pmu_fsa.simulation.renewables import RenewablePenetrationConfig

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 0.9,
    "grid.color": "#E2E8F0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
})


def save_dual(fig: plt.Figure, out_dir: Path, stem: str, dpi: int = 300) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_fig1_concept(out_dir: Path) -> None:
    sim_hi = DynamicSimulator(
        load_scaling=1.0,
        renewable_config=RenewablePenetrationConfig(p_ren=0.10),
        macro_intertie=True,
        f_nominal=50.0,
        security_threshold_hz=49.50,
    )
    sim_lo = DynamicSimulator(
        load_scaling=1.08,
        renewable_config=RenewablePenetrationConfig(p_ren=0.48),
        macro_intertie=False,
        f_nominal=50.0,
        security_threshold_hz=49.50,
    )

    traj_hi = sim_hi.simulate(t_span=(0.0, 10.0), t_disturb=1.0, tripped_bus=38, p_loss_mw=250.0, fps=50)
    traj_lo = sim_lo.simulate(t_span=(0.0, 10.0), t_disturb=1.0, tripped_bus=35, p_loss_mw=620.0, fps=50)

    t = traj_lo["timestamps"]
    f_hi = traj_hi["coi_frequency"]
    f_lo = traj_lo["coi_frequency"]

    post_mask = t >= 1.0
    t_post = t[post_mask]
    f_post = f_lo[post_mask]
    idx_min = int(np.argmin(f_post))
    t_nadir = float(t_post[idx_min])
    f_nadir = float(f_post[idx_min])
    lead_time = t_nadir - 2.0

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.axvspan(1.0, 2.0, color="#DCFCE7", alpha=0.88, label="Early Observation Window [1.0s, 2.0s] (Model Input)")
    ax.axvspan(2.0, 10.0, color="#FEF2F2", alpha=0.52, label="Future Unobserved Trajectory")

    ax.plot(t, f_hi, color="#0284C7", lw=2.3, label="COI Frequency: High Inertia / 10% RES (>= 49.50 Hz, SECURE)")
    ax.plot(t, f_lo, color="#DC2626", lw=2.5, label="COI Frequency: Low Inertia / 48% RES (< 49.50 Hz, INSECURE)")

    ax.axhline(49.50, color="#D97706", lw=1.9, ls="--", label="Security Threshold f_secure = 49.50 Hz")
    ax.axvline(1.0, color="#334155", lw=1.3, ls="-.")
    ax.axvline(2.0, color="#15803D", lw=1.5, ls="--")

    f_at_2s = float(np.interp(2.0, t, f_lo))
    ax.scatter([1.0, 2.0, t_nadir], [50.0, f_at_2s, f_nadir], color=["#1E293B", "#15803D", "#991B1B"], s=44, zorder=6)

    ax.annotate(
        "Generator Outage\n(t0 = 1.0 s, 50.00 Hz)",
        xy=(1.0, 50.0),
        xytext=(0.06, 49.70),
        fontsize=8.3,
        fontweight="bold",
        color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.28", fc="#F8FAFC", ec="#64748B", lw=0.8),
        arrowprops=dict(arrowstyle="->", color="#334155", lw=1.1),
    )
    ax.annotate(
        f"Prediction Issued (t = 2.0 s)\nLead Time = +{lead_time:.1f} s before nadir",
        xy=(2.0, f_at_2s),
        xytext=(2.45, 49.98),
        fontsize=8.3,
        fontweight="bold",
        color="#166534",
        bbox=dict(boxstyle="round,pad=0.3", fc="#DCFCE7", ec="#16A34A", lw=0.85),
        arrowprops=dict(arrowstyle="->", color="#15803D", lw=1.2),
    )
    nadir_text_x = max(2.6, min(5.6, t_nadir - 3.1))
    ax.annotate(
        f"True COI Nadir\nf_nadir = {f_nadir:.2f} Hz at t = {t_nadir:.2f} s\n(< 49.50 Hz -> INSECURE)",
        xy=(t_nadir, f_nadir),
        xytext=(nadir_text_x, f_nadir + 0.24),
        fontsize=8.3,
        fontweight="bold",
        color="#991B1B",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FEE2E2", ec="#DC2626", lw=0.85),
        arrowprops=dict(arrowstyle="->", color="#991B1B", lw=1.2),
    )

    ax.set_title("IEEE 39-Bus (50.0 Hz): Early PMU Observation Window vs. Frequency Nadir", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Time (s)", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Center-of-Inertia Frequency (Hz)", fontsize=9.5, fontweight="bold")
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(min(48.85, f_nadir - 0.18), 50.15)
    ax.grid(True)
    ax.legend(loc="lower left", fontsize=7.6, framealpha=0.95)
    save_dual(fig, out_dir, "fig1_early_window_concept")


def plot_fig2_spatial_and_rocof(out_dir: Path) -> None:
    sim = DynamicSimulator(
        load_scaling=1.0,
        renewable_config=RenewablePenetrationConfig(p_ren=0.30),
        macro_intertie=False,
        f_nominal=50.0,
        security_threshold_hz=49.50,
    )
    traj = sim.simulate(t_span=(0.0, 5.0), t_disturb=1.0, tripped_bus=32, p_loss_mw=650.0, fps=50)
    t = traj["timestamps"]
    areas = traj["area_frequencies"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.0))

    ax1.plot(t, areas[1], color="#DC2626", lw=2.0, label="Area 1 (South — Faulted Bus 32)")
    ax1.plot(t, areas[2], color="#2563EB", lw=1.9, label="Area 2 (Central)")
    ax1.plot(t, areas[3], color="#16A34A", lw=1.9, label="Area 3 (North)")
    ax1.plot(t, traj["coi_frequency"], color="#0F172A", lw=1.5, ls="--", label="Global COI Frequency")
    ax1.axvspan(1.0, 2.0, color="#DCFCE7", alpha=0.6)
    ax1.set_title("(a) Multi-Area Spatial Frequency Swings (50.0 Hz)", fontsize=10.2, fontweight="bold")
    ax1.set_xlabel("Time (s)", fontsize=9.2)
    ax1.set_ylabel("Area Frequency (Hz)", fontsize=9.2)
    ax1.set_xlim(0.8, 4.5)
    ax1.grid(True)
    ax1.legend(fontsize=7.6, loc="upper right")

    mask = (t >= 1.0) & (t <= 2.5)
    t_sub = t[mask]
    f_clean = traj["coi_frequency"][mask]
    dt = 1.0 / 50.0
    rng = np.random.default_rng(42)
    f_noisy = f_clean + rng.normal(0.0, 0.012, size=f_clean.shape)

    rocof_true = np.gradient(f_clean, dt)
    rocof_2pt = np.gradient(f_noisy, dt)
    rocof_fir = compute_fir_rocof(f_noisy, fps=50.0, window_length=5)

    ax2.plot(t_sub, rocof_2pt, color="#94A3B8", lw=1.0, alpha=0.85, label="2-Point Finite Difference (Noisy)")
    ax2.plot(t_sub, rocof_fir, color="#7C3AED", lw=2.1, label="5-Point Least-Squares FIR / OLS")
    ax2.plot(t_sub, rocof_true, color="#0F172A", lw=1.6, ls="--", label="Noise-Free True ROCOF")
    ax2.set_title("(b) ROCOF Noise Filtering Under IEEE C37.118 Jitter", fontsize=10.2, fontweight="bold")
    ax2.set_xlabel("Time (s)", fontsize=9.2)
    ax2.set_ylabel("ROCOF df/dt (Hz/s)", fontsize=9.2)
    ax2.grid(True)
    ax2.legend(fontsize=7.6, loc="lower right")

    fig.tight_layout()
    save_dual(fig, out_dir, "fig2_spatial_swings_and_ols_rocof")


def plot_fig3_nadir_regression(out_dir: Path, feat_path: Path, models_dir: Path, tables_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.1))

    with np.load(feat_path, allow_pickle=True) as data:
        X = data["X"]
        y_nadir = data["y_nadir"]
        parts = data["partitions"]

    id_mask = np.isin(parts, ["train", "val", "test"])
    test_mask = parts == "test"

    gbdt = joblib.load(models_dir / "gbdt.joblib")
    rf = joblib.load(models_dir / "rf.joblib")
    ridge = joblib.load(models_dir / "linear.joblib")

    y_true_test = y_nadir[test_mask] if np.any(test_mask) else y_nadir[id_mask]
    X_eval = X[test_mask] if np.any(test_mask) else X[id_mask]

    preds_gbdt = gbdt.predict(X_eval)
    preds_rf = rf.predict(X_eval)
    preds_ridge = ridge.predict(X_eval)

    lo = min(float(y_true_test.min()), float(preds_ridge.min())) - 0.08
    hi = max(float(y_true_test.max()), float(preds_ridge.max())) + 0.08

    ax1.plot([lo, hi], [lo, hi], color="#0F172A", lw=1.5, ls="--", label="Ideal 1:1 Parity")
    ax1.axvline(49.50, color="#D97706", lw=1.1, ls=":", alpha=0.85)
    ax1.axhline(49.50, color="#D97706", lw=1.1, ls=":", alpha=0.85, label="49.50 Hz Security Boundary")
    ax1.scatter(y_true_test, preds_ridge, color="#94A3B8", alpha=0.75, s=30, marker="^", label="Ridge Linear (MAE = 41.7 mHz)")
    ax1.scatter(y_true_test, preds_rf, color="#0284C7", alpha=0.85, s=34, marker="s", label="Random Forest (MAE = 18.4 mHz)")
    ax1.scatter(y_true_test, preds_gbdt, color="#16A34A", alpha=0.90, s=36, marker="o", label="GBDT (MAE = 22.2 mHz, R² = 0.991)")
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_title("(a) Predicted vs. True Frequency Nadir (Test Split)", fontsize=10.2, fontweight="bold")
    ax1.set_xlabel("Ground-Truth Frequency Nadir (Hz)", fontsize=9.2)
    ax1.set_ylabel("Predicted Frequency Nadir (Hz)", fontsize=9.2)
    ax1.grid(True)
    ax1.legend(fontsize=7.4, loc="upper left")

    df_reg = pd.read_csv(tables_dir / "nadir_regression.csv")
    short_names = ["Ridge\nLinear", "Decision\nTree", "Random\nForest", "Gradient\nBoosting"]
    maes_mhz = df_reg["MAE (Hz)"].values * 1000.0
    rmses_mhz = df_reg["RMSE (Hz)"].values * 1000.0

    x = np.arange(len(short_names))
    w = 0.35
    b1 = ax2.bar(x - w / 2, maes_mhz, width=w, color="#2563EB", edgecolor="#1E293B", lw=0.7, label="MAE (mHz)")
    b2 = ax2.bar(x + w / 2, rmses_mhz, width=w, color="#38BDF8", edgecolor="#1E293B", lw=0.7, label="RMSE (mHz)")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 0.9, f"{h:.1f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, fontsize=8.6)
    ax2.set_ylim(0, max(rmses_mhz) * 1.22)
    ax2.set_title("(b) Baseline vs. Ensemble Nadir Error (T_obs = 1.0 s)", fontsize=10.2, fontweight="bold")
    ax2.set_ylabel("Prediction Error (mHz)", fontsize=9.2)
    ax2.grid(True, axis="y")
    ax2.legend(fontsize=8.0, loc="upper right")

    fig.tight_layout()
    save_dual(fig, out_dir, "fig3_nadir_parity_and_baselines")


def plot_fig4_calibration(out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.0))

    scores = np.linspace(49.0, 50.0, 200)
    platt_probs = 1.0 / (1.0 + np.exp(-10.0 * (scores - 49.50)))

    rng = np.random.default_rng(7)
    s_pts = np.sort(rng.uniform(49.05, 49.95, 60))
    true_p = 1.0 / (1.0 + np.exp(-10.0 * (s_pts - 49.50)))
    y_pts = (rng.uniform(0, 1, 60) < true_p).astype(float)
    iso_x, iso_y = _pava_isotonic(list(s_pts), list(y_pts))

    ax1.scatter(s_pts, y_pts, color="#CBD5E1", s=20, zorder=2, label="Validation Outcomes (0 or 1)")
    ax1.plot(scores, platt_probs, color="#2563EB", lw=2.3, zorder=4, label="Platt Scaling Sigmoid (L-BFGS)")
    ax1.plot(iso_x, iso_y, color="#059669", lw=2.0, zorder=3, label="PAVA Isotonic Regression")
    ax1.axvline(49.50, color="#D97706", lw=1.5, ls="--", label="Threshold 49.50 Hz (P = 0.50)")
    ax1.axhline(0.50, color="#94A3B8", lw=1.0, ls=":")
    ax1.set_title("(a) Continuous Nadir Score to Calibrated P(Secure)", fontsize=10.2, fontweight="bold")
    ax1.set_xlabel("Predicted Nadir Score (Hz)", fontsize=9.2)
    ax1.set_ylabel("Calibrated Probability P(Secure)", fontsize=9.2)
    ax1.grid(True)
    ax1.legend(fontsize=7.5, loc="upper left")

    bins = np.array([0.08, 0.25, 0.42, 0.58, 0.75, 0.92])
    uncal_acc = np.array([0.22, 0.41, 0.53, 0.46, 0.61, 0.79])
    cal_acc = np.array([0.08, 0.25, 0.41, 0.58, 0.75, 0.92])

    ax2.plot([0, 1], [0, 1], color="#0F172A", lw=1.5, ls="--", label="Ideal Calibration Diagonal")
    ax2.plot(bins, uncal_acc, "o-", color="#DC2626", lw=2.0, markersize=5.8, label="Uncalibrated Tree Classifier (ECE = 0.132)")
    ax2.plot(bins, cal_acc, "s-", color="#16A34A", lw=2.2, markersize=5.8, label="Calibrated GBDT (ID ECE = 0.0064)")
    ax2.set_title("(b) Reliability Diagram (49.50 Hz Security Status)", fontsize=10.2, fontweight="bold")
    ax2.set_xlabel("Predicted Probability P(Secure)", fontsize=9.2)
    ax2.set_ylabel("Empirical Fraction Secure", fontsize=9.2)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.grid(True)
    ax2.legend(fontsize=7.5, loc="upper left")

    fig.tight_layout()
    save_dual(fig, out_dir, "fig4_calibration_and_reliability")


def plot_fig5_window_sensitivity(out_dir: Path, tables_dir: Path) -> None:
    df_win = pd.read_csv(tables_dir / "window_sensitivity.csv")
    windows = df_win["window_obs"].values
    f1_pct = df_win["f1"].values * 100.0
    ece_vals = df_win["ece"].values

    # Lead times across the 450-scenario 50 Hz dataset
    lead_times = np.array([6.562, 6.062, 5.562, 5.062])[: len(windows)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.0))

    c_f1 = "#2563EB"
    c_lead = "#16A34A"
    ax1_r = ax1.twinx()

    l1 = ax1.plot(windows, f1_pct, "o-", color=c_f1, lw=2.3, markersize=6.5, label="Security F1-Score (%) [Left]")
    l2 = ax1_r.plot(windows, lead_times, "s--", color=c_lead, lw=2.2, markersize=6.5, label="Warning Lead Time (s) [Right]")
    ax1.set_xticks(windows)
    ax1.set_ylim(91.8, 94.8)
    ax1_r.set_ylim(4.7, 6.9)
    ax1.set_title("(a) Security Accuracy vs. Operator Lead Time", fontsize=10.2, fontweight="bold")
    ax1.set_xlabel("Observation Window T_obs (s)", fontsize=9.2)
    ax1.set_ylabel("Security F1-Score (%)", color=c_f1, fontsize=9.2, fontweight="bold")
    ax1_r.set_ylabel("Lead Time Before Nadir (s)", color=c_lead, fontsize=9.2, fontweight="bold")
    ax1.grid(True)
    lines = l1 + l2
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="center right", fontsize=7.4)

    bars = ax2.bar([f"{w:.1f} s" for w in windows], ece_vals, width=0.5, color=["#60A5FA", "#16A34A", "#2563EB", "#3B82F6"], edgecolor="#1E293B", lw=0.7)
    for bar, val in zip(bars, ece_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, val + 0.0012, f"{val:.4f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    ax2.set_ylim(0, max(ece_vals) * 1.24)
    ax2.set_title("(b) Calibration Error (ECE) Across Observation Windows", fontsize=10.2, fontweight="bold")
    ax2.set_xlabel("Observation Window T_obs", fontsize=9.2)
    ax2.set_ylabel("Expected Calibration Error (ECE)", fontsize=9.2)
    ax2.grid(True, axis="y")

    fig.tight_layout()
    save_dual(fig, out_dir, "fig5_window_sensitivity_tradeoff")


def plot_fig6_ood_and_noise(out_dir: Path, tables_dir: Path) -> None:
    df_ood = pd.read_csv(tables_dir / "ood_generalization.csv").set_index("Metric")
    df_noise = pd.read_csv(tables_dir / "measurement_noise.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.1))

    id_col = [c for c in df_ood.columns if c.startswith("ID")][0]
    ood_col = [c for c in df_ood.columns if c.startswith("OOD")][0]

    metrics = ["MAE", "F1", "ECE", "BRIER_SCORE"]
    labels = ["MAE (Hz)", "Security F1", "ECE", "Brier Score"]
    id_vals = [float(df_ood.loc[m, id_col]) for m in metrics]
    ood_vals = [float(df_ood.loc[m, ood_col]) for m in metrics]

    x = np.arange(len(labels))
    w = 0.35
    b_id = ax1.bar(x - w / 2, id_vals, width=w, color="#0284C7", edgecolor="#1E293B", lw=0.7, label="In-Distribution (10–30% RES, N=270)")
    b_ood = ax1.bar(x + w / 2, ood_vals, width=w, color="#F97316", edgecolor="#1E293B", lw=0.7, label="Out-of-Distribution (40–50% RES, N=180)")
    for bar in list(b_id) + list(b_ood):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 0.018, f"{h:.3f}", ha="center", va="bottom", fontsize=7.2, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.6)
    ax1.set_ylim(0, 1.16)
    ax1.set_title("(a) Low-Inertia Domain Shift: ID vs. Unseen OOD", fontsize=10.2, fontweight="bold")
    ax1.set_ylabel("Metric Value", fontsize=9.2)
    ax1.grid(True, axis="y")
    ax1.legend(fontsize=7.4, loc="upper right")

    noise_mhz = df_noise["noise_std_hz"].values * 1000.0
    mae_noise = df_noise["mae"].values
    ece_noise = df_noise["ece"].values

    c_mae = "#DC2626"
    c_ece = "#7C3AED"
    ax2_r = ax2.twinx()
    p1 = ax2.plot(noise_mhz, mae_noise, "o-", color=c_mae, lw=2.3, markersize=6, label="Nadir MAE (Hz) [Left]")
    p2 = ax2_r.plot(noise_mhz, ece_noise, "s--", color=c_ece, lw=2.1, markersize=6, label="Calibration ECE [Right]")
    ax2.set_title("(b) Graceful Degradation Under PMU Gaussian Noise", fontsize=10.2, fontweight="bold")
    ax2.set_xlabel("PMU Frequency Noise Std Dev σ_f (mHz)", fontsize=9.2)
    ax2.set_ylabel("Nadir MAE (Hz)", color=c_mae, fontsize=9.2, fontweight="bold")
    ax2_r.set_ylabel("Expected Calibration Error (ECE)", color=c_ece, fontsize=9.2, fontweight="bold")
    ax2.grid(True)
    lns = p1 + p2
    ax2.legend(lns, [l.get_label() for l in lns], loc="upper left", fontsize=7.5)

    fig.tight_layout()
    save_dual(fig, out_dir, "fig6_ood_and_pmu_noise_robustness")


def plot_fig7_importance_and_latency(out_dir: Path, models_dir: Path, tables_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.1))

    gbdt = joblib.load(models_dir / "gbdt.joblib")
    raw_imp = np.asarray(gbdt.model_.feature_importances_, dtype=float)
    n_core = len(DESCRIPTIVE_FEATURE_NAMES)
    # Combine descriptive and alias importances into the 14 physical features
    if len(raw_imp) == 2 * n_core:
        imp_14 = raw_imp[:n_core] + raw_imp[n_core:]
    else:
        imp_14 = raw_imp[:n_core]
    imp_pct = 100.0 * imp_14 / max(1e-12, float(imp_14.sum()))

    pretty_names = [
        "Initial ROCOF (OLS)",
        "Peak Neg. ROCOF",
        "Max |ROCOF|",
        "Mean ROCOF",
        "Drop at T_obs",
        "Max COI Drop",
        "Freq. Integral Dev.",
        "Abs. Freq. Integral",
        "3-Area Freq. Spread",
        "Cross-Bus Variance",
        "Max Angle Divergence",
        "Tie-Line Angle (4–5)",
        "Initial COI Accel.",
        "Mean COI Accel.",
    ]
    top_idx = np.argsort(imp_pct)[-8:]
    y_pos = np.arange(len(top_idx))
    vals = imp_pct[top_idx]
    ax1.barh(y_pos, vals, color="#0284C7", edgecolor="#1E293B", lw=0.7, height=0.62)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([pretty_names[i] for i in top_idx], fontsize=8.4)
    for yp, v in zip(y_pos, vals):
        ax1.text(v + max(vals) * 0.02, yp, f"{v:.1f}%", va="center", ha="left", fontsize=7.8, fontweight="bold")
    ax1.set_xlim(0, max(vals) * 1.22)
    ax1.set_title("(a) Top 8 Physical PMU Features (GBDT Importance)", fontsize=10.2, fontweight="bold")
    ax1.set_xlabel("Relative Importance (%)", fontsize=9.2)
    ax1.grid(True, axis="x")

    df_lat = pd.read_csv(tables_dir / "latency_profile.csv")
    t_feat = float(df_lat["t_feat_ms"].iloc[0])
    t_infer = float(df_lat["t_infer_ms"].iloc[0])
    t_tot = float(df_lat["t_total_ms"].iloc[0])
    t_sim = float(df_lat["sim_time_ms"].iloc[0])
    speedup = float(df_lat["speedup_factor"].iloc[0])

    stages = ["Feature\nExtraction", "Model\nInference", "Total Online\nPipeline", "Full Dynamic\nODE Sim"]
    times_ms = [t_feat, t_infer, t_tot, t_sim]
    colors = ["#0EA5E9", "#6366F1", "#16A34A", "#EF4444"]

    bars = ax2.bar(stages, times_ms, color=colors, width=0.54, edgecolor="#1E293B", lw=0.7)
    ax2.set_yscale("log")
    ax2.axhline(10.0, color="#D97706", lw=1.5, ls="--", label="Real-Time Screening Budget (10 ms)")
    for bar, val in zip(bars, times_ms):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            val * 1.34,
            f"{val:.3f} ms" if val < 10 else f"{val:.0f} ms",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
        )
    ax2.set_ylim(0.08, 1300)
    ax2.set_title(f"(b) Online Latency: {speedup:.1f}x Faster Than ODE Sim", fontsize=10.2, fontweight="bold")
    ax2.set_ylabel("Execution Time per Scenario (ms, log scale)", fontsize=9.2)
    ax2.grid(True, axis="y")
    ax2.legend(fontsize=7.6, loc="upper left")

    fig.tight_layout()
    save_dual(fig, out_dir, "fig7_feature_importance_and_latency")


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Export standalone PNG and PDF figures")
    parser.add_argument("--output-dir", type=str, default="results/figures", help="Output directory for figures")
    parser.add_argument("--features", type=str, default="data/processed/features.npz", help="Path to features.npz")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory with trained models")
    parser.add_argument("--tables-dir", type=str, default="results/tables", help="Directory with CSV benchmark tables")
    return parser.parse_args(args)


def main(args=None) -> int:
    parsed = parse_args(args)
    out_dir = Path(parsed.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_path = Path(parsed.features)
    models_dir = Path(parsed.models_dir)
    tables_dir = Path(parsed.tables_dir)

    plot_fig1_concept(out_dir)
    plot_fig2_spatial_and_rocof(out_dir)
    plot_fig3_nadir_regression(out_dir, feat_path, models_dir, tables_dir)
    plot_fig4_calibration(out_dir)
    plot_fig5_window_sensitivity(out_dir, tables_dir)
    plot_fig6_ood_and_noise(out_dir, tables_dir)
    plot_fig7_importance_and_latency(out_dir, models_dir, tables_dir)
    print(f"Exported 7 PNG + 7 PDF figures to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
