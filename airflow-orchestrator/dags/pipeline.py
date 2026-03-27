"""Task flow
-------------------------------------------------------------------------------
1. read_batch: determine the date key for today's data.
2. process_vectors: download today's batch of vectors from S3 and POST to
   vector-operation API to update user baselines.
3. get_thresholds: query vector-operation API for updated similarity thresholds
   based on today's batch.
4. get_vectors: query vector-operation API for vectors to update edge sync.
5. edge_sync_update: POST updated thresholds and vectors to edge-sync API.
"""

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from typing import Any, Dict, List

import boto3
from botocore.client import Config

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from logger import Logger

logger = Logger().get_logger()

# Service URLs
VECTOR_OP_URL = os.getenv(
    "VECTOR_OP_URL", "http://vector-operation:8000/api/v1"
)
EDGE_SYNC_URL = os.getenv("EDGE_SYNC_URL", "http://edge-sync:8000/api/v1")

# S3 / RustFS
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://rustfs:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "rustfsadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "rustfsadmin")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "cognibrew-raw")


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


# Helpers
def _post(url: str, payload: dict, *, timeout: int = 60) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        logger.error("POST %s --> HTTP %s: %s", url, exc.code, body)
        raise


def _get(url: str, *, timeout: int = 30) -> dict:
    """GET *url* and return the parsed JSON response."""
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        logger.error("GET %s → HTTP %s: %s", url, exc.code, body)
        raise


def _read_batch_callable(**context: Any) -> List[Dict[str, Any]]:
    """List all device JSON files for today from S3 and return flattened vectors.

    Pushes the list of VectorRecord dicts to XCom so downstream tasks
    (process_vectors, tune_confidence, gallery_expansion) can reuse them
    without re-downloading from S3.
    """
    logical_date = context.get("logical_date") or datetime.now()
    ds = logical_date.strftime("%Y-%m-%d")
    s3 = _s3_client()

    # List all objects under the date prefix
    paginator = s3.get_paginator("list_objects_v2")
    keys = [
        obj["Key"]
        for page in paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=f"{ds}/")
        for obj in page.get("Contents", [])
        if obj["Key"].endswith(".json")
    ]

    if not keys:
        logger.warning("No batch files found in S3 for date %s", ds)
        return []

    vectors = []
    for key in keys:
        obj = s3.get_object(Bucket=S3_BUCKET_NAME, Key=key)
        payload = json.loads(obj["Body"].read())
        device_id = payload.get("device_id", "unknown")
        for v in payload.get("vectors", []):
            vectors.append(
                {
                    "device_id": device_id,
                    "username": v["username"],
                    "embedding": v["embedding"],
                    "is_correct": v.get("is_correct", True),
                    "is_fallback": not v.get("is_correct", True),
                }
            )

    logger.info(
        "read_batch: %d vectors from %d files for %s",
        len(vectors),
        len(keys),
        ds,
    )
    return vectors


def process_vectors_callable(**context: Any) -> None:
    """
    Download today's vectors from S3 (via XCom) and POST to
    vector-operation API.
    """
    vectors = context["ti"].xcom_pull(task_ids="read_batch") or []

    if not vectors:
        logger.info("process_vectors: no vectors to process — skipping.")
        return None

    # Group vectors by device_id
    group_by_device = {}
    for vector in vectors:
        device_id = vector["device_id"]
        username = vector["username"]

        # Create device_id if not exists
        if device_id not in group_by_device:
            group_by_device[device_id] = {}

        # Create username if not exists
        if username not in group_by_device[device_id]:
            group_by_device[device_id][username] = []

        # Append vector to the user's list
        group_by_device[device_id][username].append(
            {
                "embedding": vector["embedding"],
                "is_correct": vector["is_correct"],
            }
        )

    for device_id, user_vectors in group_by_device.items():
        logger.info(
            "process_vectors: %d vectors from device %s",
            len(user_vectors),
            device_id,
        )
        for username, user_embeddings in user_vectors.items():
            _post(
                f"{VECTOR_OP_URL}/vectors/update/user-baseline",
                {
                    "device_id": device_id,
                    "username": username,
                    "vectors": user_embeddings,
                },
            )

    logger.info(
        "process_vectors: completed processing vectors for %d device(s).",
        len(group_by_device),
    )


def get_thresholds_callable(**context: Any) -> Dict[str, Dict[str, Any]]:
    vectors = context["ti"].xcom_pull(task_ids="read_batch") or []
    if not vectors:
        logger.info("get_thresholds: no vectors to analyze — skipping.")
        return {}

    device_ids = {v["device_id"] for v in vectors}
    logger.info(
        "get_thresholds: analyzing threshold for %d device(s): %s",
        len(device_ids),
        device_ids,
    )

    devices_threshold = {}
    skipped = []
    for device_id in device_ids:
        logger.info(
            "get_thresholds: fetching threshold for device %s", device_id
        )
        threshold_url = f"{VECTOR_OP_URL}/vectors/threshold/{device_id}"
        try:
            result = _get(threshold_url)
        except Exception as exc:
            logger.warning(
                "get_thresholds: could not fetch threshold for device %s: %s",
                device_id,
                exc,
            )
            skipped.append(device_id)
            continue

        devices_threshold[device_id] = {
            "optimal_threshold": result["optimal_threshold"],
            "sample_count": result["sample_count"],
        }

    if skipped:
        logger.warning(
            "get_thresholds: %d/%d device(s) skipped due to errors: %s",
            len(skipped),
            len(device_ids),
            skipped,
        )

    logger.info(
        "get_thresholds: fetched threshold for %d/%d device(s).",
        len(devices_threshold),
        len(device_ids),
    )

    return devices_threshold


def get_vectors_callable(**context: Any) -> Dict[str, Dict[str, Any]]:
    vectors = context["ti"].xcom_pull(task_ids="read_batch") or []
    if not vectors:
        logger.info("get_vectors: no vectors to fetch — skipping.")
        return {}

    device_ids = {v["device_id"] for v in vectors}
    logger.info(
        "get_vectors: fetching vectors for %d device(s): %s",
        len(device_ids),
        device_ids,
    )

    devices_vectors = {}
    skipped = []
    for device_id in device_ids:
        logger.info("get_vectors: fetching vectors for device %s", device_id)
        vectors_url = f"{VECTOR_OP_URL}/vectors/{device_id}"
        try:
            result = _get(vectors_url)
            devices_vectors[device_id] = result
        except Exception as exc:
            logger.warning(
                "get_vectors: could not fetch vectors for device %s: %s",
                device_id,
                exc,
            )
            skipped.append(device_id)
            continue

    if skipped:
        logger.warning(
            "get_vectors: %d/%d device(s) skipped due to errors: %s",
            len(skipped),
            len(device_ids),
            skipped,
        )

    logger.info(
        "get_vectors: fetched vectors for %d/%d device(s).",
        len(devices_vectors),
        len(device_ids),
    )

    return devices_vectors


def edge_sync_update_callable(**context: Any) -> None:
    devices_threshold = (
        context["ti"].xcom_pull(task_ids="get_thresholds") or {}
    )
    devices_vectors = context["ti"].xcom_pull(task_ids="get_vectors") or {}
    if not devices_threshold or not devices_vectors:
        logger.info(
            "edge_sync_update: missing data for thresholds or vectors — skipping."
        )
        return

    for device_id, device_info in devices_vectors.items():
        threshold_info = devices_threshold.get(device_id)
        if not threshold_info:
            logger.warning(
                "edge_sync_update: no threshold info for device %s — skipping",
                device_id,
            )
            continue

        users = device_info.get("users", [])
        for user in users:
            username = user.get("username")
            if not username:
                logger.warning(
                    "edge_sync_update: missing username for device %s — skipping user",
                    device_id,
                )
                continue

            vectors = user.get("vectors", [])
            if not vectors:
                logger.warning(
                    "edge_sync_update: no vectors for user %s on device %s — skipping user",
                    username,
                    device_id,
                )
                continue

            for vector in vectors:
                payload = {
                    "device_id": device_id,
                    "threshold": threshold_info["optimal_threshold"],
                    "username": username,
                    "embedding": vector,
                }
                try:
                    _post(f"{EDGE_SYNC_URL}/sync/update", payload)
                except Exception as exc:
                    logger.warning(
                        "edge_sync_update: failed to update edge sync for "
                        "device %s, user %s: %s",
                        device_id,
                        username,
                        exc,
                    )
                    continue

    logger.info("edge_sync_update: completed updates to edge sync service.")


# DAG
default_args = {
    "owner": "cognibrew",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="cognibrew_pipeline",
    default_args=default_args,
    description=(
        "Read batch --> process vectors (vector-operation API) "
        "--> get thresholds (vector-operation API) "
        "--> get vectors (vector-operation API) "
        "--> update edge-sync"
    ),
    schedule="0 0 * * *",  # 00:00 UTC daily
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["cognibrew"],
) as dag:
    # 1. Read raw batch data key
    read_batch = PythonOperator(
        task_id="read_batch",
        python_callable=_read_batch_callable,
    )

    # 2. Process vectors via the vector-operation API
    process_vectors = PythonOperator(
        task_id="process_vectors",
        python_callable=process_vectors_callable,
    )

    # 3. Get similarity threshold via the vector-operation API
    get_thresholds = PythonOperator(
        task_id="get_thresholds",
        python_callable=get_thresholds_callable,
    )

    # 4. Get vectors for edge sync update
    get_vectors = PythonOperator(
        task_id="get_vectors",
        python_callable=get_vectors_callable,
    )

    # 5. Update edge sync with new thresholds and vectors
    edge_sync_update = PythonOperator(
        task_id="edge_sync_update",
        python_callable=edge_sync_update_callable,
    )

    # Task dependencies
    (
        read_batch
        >> process_vectors
        >> [get_thresholds, get_vectors]
        >> edge_sync_update
    )  # type: ignore
