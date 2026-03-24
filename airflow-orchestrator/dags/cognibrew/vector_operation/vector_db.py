import os
import uuid
from datetime import datetime, timezone

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from cognibrew.logger import Logger

logger = Logger().get_logger()

_HOST = os.getenv("QDRANT_HOST", "qdrant")
_PORT = int(os.getenv("QDRANT_PORT", "6334"))
_COLLECTION = os.getenv("QDRANT_COLLECTION", "face_embeddings")
_DIM = int(os.getenv("EMBEDDING_DIM", "512"))


def client() -> QdrantClient:
    """Return a Qdrant gRPC client."""
    return QdrantClient(host=_HOST, port=_PORT, prefer_grpc=True)


def ensure_collection() -> None:
    """Create the face-gallery collection if it does not exist."""
    c = client()
    collections = [col.name for col in c.get_collections().collections]
    if _COLLECTION not in collections:
        logger.info(
            "Creating Qdrant collection: %s (%d-dim, cosine)",
            _COLLECTION,
            _DIM,
        )
        c.create_collection(
            collection_name=_COLLECTION,
            vectors_config=VectorParams(size=_DIM, distance=Distance.COSINE),
        )


def _user_filter(username: str) -> Filter:
    return Filter(
        must=[FieldCondition(key="username", match=MatchValue(value=username))]
    )


def upsert_vectors(
    vectors: list[tuple[str, list[float], str, bool, bool]],
) -> int:
    """Insert vectors into the gallery.

    Args:
        vectors: list of (username, embedding, anchor_type, is_correct, is_fallback)

    Returns:
        Number of points upserted.
    """
    points = []
    now = datetime.now(timezone.utc).isoformat()
    for username, embedding, anchor_type, is_correct, is_fallback in vectors:
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "username": username,
                    "anchor_type": anchor_type,
                    "is_correct": is_correct,
                    "is_fallback": is_fallback,
                    "timestamp": now,
                },
            )
        )
    if points:
        client().upsert(collection_name=_COLLECTION, points=points)
    return len(points)


def get_user_vectors(
    username: str, *, with_vectors: bool = True, limit: int = 1000
) -> list[dict]:
    """Retrieve all gallery vectors for a user.

    Returns list of dicts: {id, username, anchor_type, timestamp, embedding?}.
    """
    results = client().scroll(
        collection_name=_COLLECTION,
        scroll_filter=_user_filter(username),
        with_vectors=with_vectors,
        limit=limit,
    )[0]
    out = []
    for pt in results:
        if pt.payload is None:
            continue

        rec = {
            "point_id": str(pt.id),
            "username": pt.payload["username"],
            "anchor_type": pt.payload["anchor_type"],
            "timestamp": pt.payload.get("timestamp", ""),
        }
        if with_vectors and isinstance(pt.vector, list):
            rec["embedding"] = pt.vector

        out.append(rec)

    return out


def get_user_baseline(username: str) -> np.ndarray | None:
    """Return the baseline vector for a user (or None)."""
    filt = Filter(
        must=[
            FieldCondition(key="username", match=MatchValue(value=username)),
            FieldCondition(
                key="anchor_type", match=MatchValue(value="baseline")
            ),
        ]
    )
    results = client().scroll(
        collection_name=_COLLECTION,
        scroll_filter=filt,
        with_vectors=True,
        limit=1,
    )[0]
    if not results:
        return None
    vec = results[0].vector
    if not isinstance(vec, list):
        return None

    return np.array(vec, dtype=np.float32)


def get_all_usernames() -> list[str]:
    """Return distinct usernames in the gallery (sample-based)."""
    results = client().scroll(
        collection_name=_COLLECTION,
        with_vectors=False,
        limit=10_000,
    )[0]
    return list(
        {pt.payload["username"] for pt in results if pt.payload is not None}
    )
