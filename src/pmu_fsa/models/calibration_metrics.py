"""Probability calibration metrics and reliability diagram binning."""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np


def _prepare_prob_array(y: Any, name: str = "y_prob") -> np.ndarray:
    if y is None:
        raise ValueError(f"Cannot compute calibration metrics on None {name}")
    arr = np.asarray(y, dtype=np.float64)
    if arr.size == 0:
        raise ValueError(f"Cannot compute calibration metrics on empty {name}")
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        raise ValueError(f"Input {name} contains NaN or Inf values")
    if arr.ndim == 2 and arr.shape[1] == 2:
        arr = arr[:, 1]
    else:
        arr = arr.ravel()
    return arr


def _prepare_true_array(y: Any, name: str = "y_true", threshold: float = 49.50) -> np.ndarray:
    if y is None:
        raise ValueError(f"Cannot compute calibration metrics on None {name}")
    arr = np.asarray(y, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError(f"Cannot compute calibration metrics on empty {name}")
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        raise ValueError(f"Input {name} contains NaN or Inf values")
    if np.any(arr > 1.5):
        arr = np.where(arr >= float(threshold), 1.0, 0.0)
    else:
        arr = np.where(arr >= 0.5, 1.0, 0.0)
    return arr


def brier_score(y_true: Any, y_prob: Any, threshold: float = 49.50) -> float:
    """Compute mean squared probability error (Brier score)."""
    yt = _prepare_true_array(y_true, "y_true", threshold=threshold)
    yp = _prepare_prob_array(y_prob, "y_prob")
    if yt.size != yp.size:
        raise ValueError(f"Length mismatch: len(y_true)={yt.size}, len(y_prob)={yp.size}")

    return float(np.mean((yp - yt) ** 2))


def expected_calibration_error(
    y_true: Any,
    y_prob: Any,
    n_bins: int = 10,
    threshold: float = 49.50,
) -> float:
    """Compute Expected Calibration Error (ECE) across equal-width bins."""
    yt = _prepare_true_array(y_true, "y_true", threshold=threshold)
    yp = _prepare_prob_array(y_prob, "y_prob")
    if yt.size != yp.size:
        raise ValueError(f"Length mismatch: len(y_true)={yt.size}, len(y_prob)={yp.size}")
    if n_bins <= 0:
        raise ValueError(f"n_bins must be strictly positive, got {n_bins}")

    n = yt.size
    bin_step = 1.0 / n_bins
    ece = 0.0

    for m in range(n_bins):
        low = m * bin_step
        high = (m + 1) * bin_step
        if m == n_bins - 1:
            idx = np.where((yp >= low) & (yp <= high))[0]
        else:
            idx = np.where((yp >= low) & (yp < high))[0]

        if idx.size > 0:
            bin_acc = float(np.mean(yt[idx]))
            bin_conf = float(np.mean(yp[idx]))
            bin_weight = float(idx.size / n)
            ece += bin_weight * abs(bin_acc - bin_conf)

    return float(ece)


def maximum_calibration_error(
    y_true: Any,
    y_prob: Any,
    n_bins: int = 10,
    threshold: float = 49.50,
) -> float:
    """Compute Maximum Calibration Error (MCE) across non-empty bins."""
    yt = _prepare_true_array(y_true, "y_true", threshold=threshold)
    yp = _prepare_prob_array(y_prob, "y_prob")
    if yt.size != yp.size:
        raise ValueError(f"Length mismatch: len(y_true)={yt.size}, len(y_prob)={yp.size}")
    if n_bins <= 0:
        raise ValueError(f"n_bins must be strictly positive, got {n_bins}")

    bin_step = 1.0 / n_bins
    mce = 0.0

    for m in range(n_bins):
        low = m * bin_step
        high = (m + 1) * bin_step
        if m == n_bins - 1:
            idx = np.where((yp >= low) & (yp <= high))[0]
        else:
            idx = np.where((yp >= low) & (yp < high))[0]

        if idx.size > 0:
            bin_acc = float(np.mean(yt[idx]))
            bin_conf = float(np.mean(yp[idx]))
            diff = abs(bin_acc - bin_conf)
            if diff > mce:
                mce = diff

    return float(mce)


def reliability_diagram(
    y_true: Any,
    y_prob: Any,
    n_bins: int = 10,
    threshold: float = 49.50,
) -> Dict[str, Any]:
    """Bin predicted probabilities and empirical accuracies for reliability plots."""
    yt = _prepare_true_array(y_true, "y_true", threshold=threshold)
    yp = _prepare_prob_array(y_prob, "y_prob")
    if yt.size != yp.size:
        raise ValueError(f"Length mismatch: len(y_true)={yt.size}, len(y_prob)={yp.size}")
    if n_bins <= 0:
        raise ValueError(f"n_bins must be strictly positive, got {n_bins}")

    bin_step = 1.0 / n_bins
    bin_edges = [float(m * bin_step) for m in range(n_bins + 1)]
    bin_accs: List[float] = []
    bin_confs: List[float] = []
    bin_counts: List[int] = []

    for m in range(n_bins):
        low = bin_edges[m]
        high = bin_edges[m + 1]
        if m == n_bins - 1:
            idx = np.where((yp >= low) & (yp <= high))[0]
        else:
            idx = np.where((yp >= low) & (yp < high))[0]

        count = int(idx.size)
        bin_counts.append(count)
        if count > 0:
            bin_accs.append(float(np.mean(yt[idx])))
            bin_confs.append(float(np.mean(yp[idx])))
        else:
            bin_accs.append(0.0)
            bin_confs.append(float(0.5 * (low + high)))

    return {
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts": bin_counts,
        "bin_edges": bin_edges,
    }


def compute_calibration_metrics(
    y_true: Any,
    y_prob: Any,
    n_bins: int = 10,
    threshold: float = 49.50,
) -> Dict[str, Any]:
    """Compute Brier score, ECE, MCE, and reliability diagram bins (standardized n_bins=10)."""
    bs = brier_score(y_true, y_prob, threshold=threshold)
    ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins, threshold=threshold)
    mce = maximum_calibration_error(y_true, y_prob, n_bins=n_bins, threshold=threshold)
    diagram = reliability_diagram(y_true, y_prob, n_bins=n_bins, threshold=threshold)

    return {
        "brier_score": bs,
        "ece": ece,
        "mce": mce,
        "n_bins": int(n_bins),
        "bin_accuracies": diagram["bin_accuracies"],
        "bin_confidences": diagram["bin_confidences"],
        "bin_counts": diagram["bin_counts"],
        "bin_edges": diagram["bin_edges"],
    }
