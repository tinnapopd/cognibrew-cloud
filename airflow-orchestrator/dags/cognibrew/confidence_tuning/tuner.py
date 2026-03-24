import os
from dataclasses import dataclass

import numpy as np

from cognibrew.logger import Logger

logger = Logger().get_logger()

_SWEEP_MIN = float(os.getenv("THRESHOLD_SWEEP_MIN", "0.3"))
_SWEEP_MAX = float(os.getenv("THRESHOLD_SWEEP_MAX", "0.9"))
_SWEEP_STEP = float(os.getenv("THRESHOLD_SWEEP_STEP", "0.01"))

# When mean drift exceeds this, lower the threshold slightly to compensate
_DRIFT_ADJUSTMENT_FACTOR = 0.05


@dataclass
class SweepResult:
    """Result of threshold sweep optimisation."""

    optimal_threshold: float
    f1_score: float
    precision: float
    recall: float
    total_samples: int
    positive_samples: int
    negative_samples: int


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float, float]:
    """Compute precision, recall, F1 from boolean arrays."""
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def sweep_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    sweep_min: float | None = None,
    sweep_max: float | None = None,
    sweep_step: float | None = None,
) -> SweepResult:
    """Sweep similarity thresholds and find the one maximising F1.

    Args:
        scores: (N,) array of similarity scores ∈ [0,1].
        labels: (N,) boolean array, True = correct recognition.
        sweep_min/max/step: override default sweep range.

    Returns:
        SweepResult with the optimal threshold and its metrics.
    """
    lo = sweep_min or _SWEEP_MIN
    hi = sweep_max or _SWEEP_MAX
    step = sweep_step or _SWEEP_STEP

    thresholds = np.arange(lo, hi + step, step)

    best_f1 = -1.0
    best_threshold = lo
    best_precision = 0.0
    best_recall = 0.0

    for t in thresholds:
        predictions = scores >= t  # predict "correct" if score ≥ threshold
        precision, recall, f1 = _compute_metrics(labels, predictions)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)
            best_precision = precision
            best_recall = recall

    total = len(scores)
    positive = int(np.sum(labels))

    logger.info(
        "Threshold sweep: best=%.4f F1=%.4f P=%.4f R=%.4f (N=%d pos=%d neg=%d)",
        best_threshold,
        best_f1,
        best_precision,
        best_recall,
        total,
        positive,
        total - positive,
    )

    return SweepResult(
        optimal_threshold=best_threshold,
        f1_score=best_f1,
        precision=best_precision,
        recall=best_recall,
        total_samples=total,
        positive_samples=positive,
        negative_samples=total - positive,
    )


def adjust_for_drift(
    threshold: float,
    mean_drift: float,
    *,
    adjustment_factor: float = _DRIFT_ADJUSTMENT_FACTOR,
) -> tuple[float, bool]:
    """Optionally lower the threshold when drift is detected.

    When face vectors drift (e.g. aging, lighting changes), the similarity
    scores naturally decrease.  Lowering the threshold prevents false-negative
    fallback triggers.

    Returns:
        (adjusted_threshold, was_adjusted)
    """
    if mean_drift > 0.1:
        new_threshold = max(0.2, threshold - adjustment_factor)
        logger.info(
            "Drift adjustment: %.4f → %.4f (mean_drift=%.4f)",
            threshold,
            new_threshold,
            mean_drift,
        )
        return new_threshold, True
    return threshold, False
