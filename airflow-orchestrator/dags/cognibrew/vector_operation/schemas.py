from enum import Enum

from pydantic import BaseModel, Field


class AnchorType(str, Enum):
    BASELINE = "baseline"
    SECONDARY = "secondary"
    TEMPORAL = "temporal"


class VectorInput(BaseModel):
    """A single embedding with metadata."""

    username: str
    embedding: list[float] = Field(..., min_length=512, max_length=512)
    is_correct: bool = True
    is_fallback: bool = False
    anchor_type: AnchorType = AnchorType.TEMPORAL


class AnalyzeRequest(BaseModel):
    """Batch of vectors for drift detection + outlier filtering."""

    vectors: list[VectorInput] = Field(..., min_length=1)


class GalleryExpandRequest(BaseModel):
    """Fallback-verified vectors to add as secondary anchors."""

    vectors: list[VectorInput] = Field(..., min_length=1)


class VectorInfo(BaseModel):
    """Stored vector metadata from gallery."""

    point_id: str
    username: str
    anchor_type: AnchorType
    timestamp: str
    embedding: list[float] | None = None


class GalleryResponse(BaseModel):
    """Per-user vector gallery."""

    username: str
    total_vectors: int
    baseline_count: int
    secondary_count: int
    temporal_count: int
    vectors: list[VectorInfo]


class DriftSignal(BaseModel):
    """Per-user drift telemetry."""

    username: str
    mean_drift: float = Field(
        ..., description="Mean cosine distance from baseline"
    )
    max_drift: float = Field(
        ..., description="Maximum cosine distance from baseline"
    )
    gallery_size: int
    is_drifting: bool


class DriftResponse(BaseModel):
    """Aggregated drift signals for all users."""

    signals: list[DriftSignal]
    global_mean_drift: float


class AnalyzeResult(BaseModel):
    """Result of batch vector analysis."""

    accepted: int = Field(
        ..., description="Vectors kept after outlier filtering"
    )
    rejected: int = Field(..., description="Outliers removed")
    drift_detected_users: list[str] = Field(
        default_factory=list,
        description="Users whose drift exceeded threshold",
    )
    details: list[DriftSignal] = Field(default_factory=list)
