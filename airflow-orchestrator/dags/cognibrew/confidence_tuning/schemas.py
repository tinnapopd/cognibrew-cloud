from pydantic import BaseModel, Field


class FeedbackRecord(BaseModel):
    """A single recognition result with correctness feedback."""

    username: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    is_correct: bool


class TuneRequest(BaseModel):
    """Request to run threshold tuning on feedback data."""

    feedback: list[FeedbackRecord] = Field(..., min_length=1)


class ThresholdResult(BaseModel):
    """Optimal threshold after F1-sweep."""

    optimal_threshold: float
    f1_score: float
    precision: float
    recall: float
    total_samples: int
    positive_samples: int
    negative_samples: int


class TuneResponse(BaseModel):
    """Result of a threshold tuning run."""

    result: ThresholdResult
    drift_adjusted: bool = Field(
        False,
        description="True if threshold was adjusted based on drift signals",
    )
    previous_threshold: float | None = None


class ThresholdInfo(BaseModel):
    """Current threshold for a user or global."""

    threshold: float
    version: int
    username: str | None = None
    f1_score: float | None = None
    updated_at: str | None = None


class ConfidenceHistory(BaseModel):
    """Historical threshold entry stored in PostgreSQL."""

    id: int | None = None
    username: str | None = None
    threshold: float
    f1_score: float
    precision: float
    recall: float
    version: int
    created_at: str | None = None
