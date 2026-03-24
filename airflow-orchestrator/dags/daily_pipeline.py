import asyncio
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
import os
import urllib.request

from cognibrew.logger import Logger
from cognibrew.confidence_tuning import tuner, database
from cognibrew.confidence_tuning import mlflow_logger as ct_mlflow
from cognibrew.vector_operation import vector_engine, vector_db
from cognibrew.vector_operation import mlflow_logger as vo_mlflow

logger = Logger().get_logger()

default_args = {
    "owner": "cognibrew",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _read_batch_callable(**context: Any) -> str:
    """Determine the S3 key for today's batch.

    Returns the S3 prefix `/{date}/` for downstream tasks.
    """
    ds: str = context["ds"]  # YYYY-MM-DD
    return ds


def process_vectors_callable(**context: Any):
    """Analyze incoming batch: drift detection + outlier filtering."""
    # In production, read vectors from S3 using context["ti"].xcom_pull(task_ids="read_batch")
    # Stub: empty list of vectors
    vectors = []

    vector_db.ensure_collection()

    by_user: dict[str, list[tuple[list[float], bool, bool]]] = defaultdict(
        list
    )
    for v in vectors:
        by_user[v["username"]].append(
            (
                v["embedding"],
                v.get("is_correct", True),
                v.get("is_fallback", False),
            )
        )

    total_accepted = 0
    total_rejected = 0
    drift_users: list[str] = []

    for username, records in by_user.items():
        embeddings = np.array([r[0] for r in records], dtype=np.float32)
        baseline = vector_db.get_user_baseline(username)
        existing = vector_db.get_user_vectors(username, with_vectors=False)
        gallery_size = len(existing)

        summary = vector_engine.analyze_user_batch(
            username, embeddings, baseline, gallery_size
        )

        if summary.accepted_vectors:
            to_upsert = []
            for i, emb in enumerate(summary.accepted_vectors):
                is_correct = records[i][1] if i < len(records) else True
                is_fallback = records[i][2] if i < len(records) else False
                anchor = (
                    "baseline" if baseline is None and i == 0 else "temporal"
                )
                to_upsert.append(
                    (username, emb, anchor, is_correct, is_fallback)
                )

            vector_db.upsert_vectors(to_upsert)

        total_accepted += summary.accepted
        total_rejected += summary.rejected

        if summary.drift.is_drifting:
            drift_users.append(username)

        vo_mlflow.log_drift_metrics(
            username,
            summary.drift.mean_drift,
            summary.drift.max_drift,
            summary.drift.is_drifting,
            summary.drift.gallery_size,
        )

    vo_mlflow.log_outlier_metrics(
        total_input=total_accepted + total_rejected,
        accepted=total_accepted,
        rejected=total_rejected,
    )

    return {
        "accepted": total_accepted,
        "rejected": total_rejected,
        "drift_users": drift_users,
    }


def tune_confidence_callable(**context: Any):
    """Run threshold sweep on feedback data."""

    async def async_tune():
        await database.init_db()
        # In production, read feedback from DB or S3.
        feedback = []
        if not feedback:
            logger.info("No feedback to tune.")
            return

        scores = np.array(
            [f["similarity_score"] for f in feedback], dtype=np.float32
        )
        labels = np.array([f["is_correct"] for f in feedback], dtype=bool)

        result = tuner.sweep_threshold(scores, labels)

        # Get global drift to optionally adjust
        usernames = vector_db.get_all_usernames()
        signals = []
        for username in usernames:
            baseline = vector_db.get_user_baseline(username)
            all_vecs = vector_db.get_user_vectors(username, with_vectors=True)
            if baseline is not None and len(all_vecs) >= 2:
                embeddings = np.array(
                    [v["embedding"] for v in all_vecs], dtype=np.float32
                )
                drift = vector_engine.detect_drift(
                    username, embeddings, baseline, len(all_vecs)
                )
                signals.append(drift.mean_drift)

        global_drift = float(np.mean(signals)) if signals else 0.0
        adjusted, was_adjusted = tuner.adjust_for_drift(
            result.optimal_threshold, global_drift
        )

        if was_adjusted:
            result = tuner.SweepResult(
                optimal_threshold=adjusted,
                f1_score=result.f1_score,
                precision=result.precision,
                recall=result.recall,
                total_samples=result.total_samples,
                positive_samples=result.positive_samples,
                negative_samples=result.negative_samples,
            )

        # Tune per-user
        feedback_usernames = {f["username"] for f in feedback}
        for uname in feedback_usernames:
            user_scores = np.array(
                [
                    f["similarity_score"]
                    for f in feedback
                    if f["username"] == uname
                ],
                dtype=np.float32,
            )
            user_labels = np.array(
                [f["is_correct"] for f in feedback if f["username"] == uname],
                dtype=bool,
            )
            if len(user_scores) >= 5:
                user_result = tuner.sweep_threshold(user_scores, user_labels)
                await database.save_confidence(
                    threshold=user_result.optimal_threshold,
                    f1_score=user_result.f1_score,
                    precision=user_result.precision,
                    recall=user_result.recall,
                    username=uname,
                )

        await database.save_confidence(
            threshold=result.optimal_threshold,
            f1_score=result.f1_score,
            precision=result.precision,
            recall=result.recall,
        )

        ct_mlflow.log_threshold_experiment(
            threshold=result.optimal_threshold,
            f1_score=result.f1_score,
            precision=result.precision,
            recall=result.recall,
            total_samples=result.total_samples,
            drift_adjusted=was_adjusted,
        )

    asyncio.run(async_tune())


def gallery_expansion_callable(**context: Any):
    """Fallback-verified vectors to add as secondary anchors."""
    # Stub: read from S3
    vectors = []
    to_upsert = []
    by_user = defaultdict(int)

    for v in vectors:
        to_upsert.append(
            (
                v["username"],
                v["embedding"],
                "secondary",
                v.get("is_correct", True),
                v.get("is_fallback", True),
            )
        )
        by_user[v["username"]] += 1

    vector_db.upsert_vectors(to_upsert)

    for username, added in by_user.items():
        existing = vector_db.get_user_vectors(username, with_vectors=False)
        vo_mlflow.log_gallery_expansion(username, added, len(existing))


def sync_edge_healthcheck_callable(**context: Any):
    """Confirm that edge-sync is alive and serving pull bundles.

    Edge devices use the pull pattern — they call
    GET /api/v1/sync/bundle on their own schedule.
    This task verifies the sync service is reachable so operators
    know the pipeline completed successfully.
    """
    edge_sync_url = os.getenv("EDGE_SYNC_URL", "http://edge-sync:8000")
    url = f"{edge_sync_url}/api/v1/sync/status"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            if resp.status != 200:
                raise RuntimeError(
                    f"Edge Sync health-check failed: HTTP {resp.status}"
                )
            logger.info(
                "Edge Sync is healthy and ready to serve pull bundles."
            )
    except Exception as exc:
        # Non-fatal: log warning but don't block the pipeline.
        # Edge devices can still pull bundles; this is just an observability check.
        logger.warning(
            "Edge Sync health-check could not reach service: %s", exc
        )


# ── DAG ───────────────────────────────────────────────────────

with DAG(
    dag_id="cognibrew_daily_pipeline",
    default_args=default_args,
    description="Daily: read batch → analyse vectors → tune confidence → expand gallery → confirm edge-sync ready",
    schedule="0 2 * * *",  # 02:00 UTC daily
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["cognibrew", "mlops"],
) as dag:
    # 1. Read raw batch data key
    read_batch = PythonOperator(
        task_id="read_batch",
        python_callable=_read_batch_callable,
    )

    # 2. Trigger vector analysis (drift + outlier + gallery expansion)
    process_vectors = PythonOperator(
        task_id="process_vectors",
        python_callable=process_vectors_callable,
    )

    # 3. Trigger confidence tuning
    tune_confidence = PythonOperator(
        task_id="tune_confidence",
        python_callable=tune_confidence_callable,
    )

    # 4. Gallery expansion
    gallery_expansion = PythonOperator(
        task_id="gallery_expansion",
        python_callable=gallery_expansion_callable,
    )

    # 5. Confirm edge-sync is healthy and ready to serve pull bundles.
    #    Edge devices pull GET /api/v1/sync/bundle on their own schedule.
    #    No push is involved — this is purely an observability health-check.
    sync_edge_healthcheck = PythonOperator(
        task_id="sync_edge_healthcheck",
        python_callable=sync_edge_healthcheck_callable,
    )

    # ── Task Dependencies ─────────────────────────────────────
    (
        read_batch
        >> process_vectors
        >> tune_confidence
        >> gallery_expansion
        >> sync_edge_healthcheck
    )
