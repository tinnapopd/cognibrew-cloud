import os
from dataclasses import dataclass

import numpy as np

from cognibrew.logger import Logger

logger = Logger().get_logger()

_DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.15"))
_IQR_FACTOR = float(os.getenv("OUTLIER_IQR_FACTOR", "1.5"))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return 1 - cos(a, b).  Range [0, 2]; 0 = identical."""
    dot = float(np.dot(a, b))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm == 0.0:
        return 1.0
    return 1.0 - dot / norm


def cosine_distances_to_reference(
    vectors: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Cosine distances of each row in *vectors* to *reference*."""
    ref_norm = reference / (np.linalg.norm(reference) + 1e-12)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    normed = vectors / norms
    sims = normed @ ref_norm
    return 1.0 - sims


@dataclass
class OutlierResult:
    """Result of IQR-based outlier filtering."""

    mask: np.ndarray  # bool[N] — True = keep
    q1: float
    q3: float
    iqr: float
    lower: float
    upper: float

    @property
    def kept(self) -> int:
        return int(self.mask.sum())

    @property
    def removed(self) -> int:
        return int((~self.mask).sum())


def filter_outliers_iqr(
    distances: np.ndarray, *, factor: float | None = None
) -> OutlierResult:
    """Flag outliers using Inter-Quartile Range on cosine distances.

    A vector whose distance to the reference falls outside
    [Q1 − factor*IQR, Q3 + factor*IQR] is considered an outlier.
    """
    factor = factor or _IQR_FACTOR
    if len(distances) < 4:
        # Too few points for IQR — keep everything
        return OutlierResult(
            mask=np.ones(len(distances), dtype=bool),
            q1=0.0,
            q3=0.0,
            iqr=0.0,
            lower=0.0,
            upper=float("inf"),
        )

    q1 = float(np.percentile(distances, 25))
    q3 = float(np.percentile(distances, 75))
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    mask = (distances >= lower) & (distances <= upper)

    logger.info(
        "IQR outlier filter: Q1=%.4f Q3=%.4f IQR=%.4f bounds=[%.4f, %.4f] kept=%d removed=%d",
        q1,
        q3,
        iqr,
        lower,
        upper,
        mask.sum(),
        (~mask).sum(),
    )
    return OutlierResult(
        mask=mask, q1=q1, q3=q3, iqr=iqr, lower=lower, upper=upper
    )


@dataclass
class DriftResult:
    """Result of drift detection for a user."""

    username: str
    mean_drift: float
    max_drift: float
    is_drifting: bool
    gallery_size: int


def detect_drift(
    username: str,
    new_vectors: np.ndarray,
    baseline: np.ndarray,
    gallery_size: int,
    *,
    threshold: float | None = None,
) -> DriftResult:
    """Detect concept drift by measuring cosine distance from baseline.

    Args:
        username: identity label.
        new_vectors: (N, 512) new embeddings.
        baseline: (512,) baseline vector.
        gallery_size: current gallery size for this user.
        threshold: drift threshold (default from env).

    Returns:
        DriftResult with mean/max drift and whether it exceeds threshold.
    """
    threshold = threshold or _DRIFT_THRESHOLD
    distances = cosine_distances_to_reference(new_vectors, baseline)
    mean_d = float(np.mean(distances))
    max_d = float(np.max(distances))
    is_drifting = mean_d > threshold

    logger.info(
        "Drift for %s: mean=%.4f max=%.4f threshold=%.4f drifting=%s",
        username,
        mean_d,
        max_d,
        threshold,
        is_drifting,
    )
    return DriftResult(
        username=username,
        mean_drift=mean_d,
        max_drift=max_d,
        is_drifting=is_drifting,
        gallery_size=gallery_size,
    )


@dataclass
class AnalysisSummary:
    """Summary of processing a batch of vectors for one user."""

    username: str
    total_input: int
    accepted: int  # after outlier filtering
    rejected: int  # outliers removed
    drift: DriftResult
    accepted_vectors: list[list[float]]  # filtered embeddings to store


def analyze_user_batch(
    username: str,
    embeddings: np.ndarray,
    baseline: np.ndarray | None,
    gallery_size: int,
) -> AnalysisSummary:
    """Full analysis pipeline for one user's batch.

    Steps:
        1. If baseline exists → compute drift, filter outliers.
        2. If no baseline → treat first vector as baseline.
        3. Outlier-filtered vectors are returned for gallery insertion.
    """
    n = len(embeddings)

    if baseline is None:
        # First-ever vector becomes the baseline
        logger.info(
            "No baseline for %s — using first vector as baseline", username
        )
        return AnalysisSummary(
            username=username,
            total_input=n,
            accepted=n,
            rejected=0,
            drift=DriftResult(
                username=username,
                mean_drift=0.0,
                max_drift=0.0,
                is_drifting=False,
                gallery_size=gallery_size,
            ),
            accepted_vectors=[emb.tolist() for emb in embeddings],
        )

    # Compute distances to baseline
    distances = cosine_distances_to_reference(embeddings, baseline)

    # Outlier filtering
    outlier_result = filter_outliers_iqr(distances)

    # Keep only non-outlier vectors
    kept_embeddings = embeddings[outlier_result.mask]
    kept_distances = distances[outlier_result.mask]

    # Drift detection on kept vectors
    if len(kept_distances) > 0:
        mean_d = float(np.mean(kept_distances))
        max_d = float(np.max(kept_distances))
    else:
        mean_d = 0.0
        max_d = 0.0

    drift = DriftResult(
        username=username,
        mean_drift=mean_d,
        max_drift=max_d,
        is_drifting=mean_d > _DRIFT_THRESHOLD,
        gallery_size=gallery_size + outlier_result.kept,
    )

    return AnalysisSummary(
        username=username,
        total_input=n,
        accepted=outlier_result.kept,
        rejected=outlier_result.removed,
        drift=drift,
        accepted_vectors=[emb.tolist() for emb in kept_embeddings],
    )
