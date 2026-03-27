"""Task flow
-------------------------------------------------------------------------------
1. read_enrollments: scan S3 `enrollments/{username}/*.json` and collect
   enrollment records.
2. process_baseline: POST each enrollment record to the vector-operation API
   to register the user's baseline embedding.
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


def read_enrollments_callable(**context: Any) -> List[Dict[str, Any]]:
    """Scan S3 enrollments/{username}/*.json and return enrollment records.

    Each JSON file contains:
        {
            "username": str,
            "embedding": list[float],
            "device_id": str,
        }
    """
    s3 = _s3_client()
    prefix = "enrollments/"

    paginator = s3.get_paginator("list_objects_v2")
    keys = [
        obj["Key"]
        for page in paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=prefix)
        for obj in page.get("Contents", [])
        if obj["Key"].endswith(".json")
    ]

    if not keys:
        logger.warning(
            "read_enrollments: no enrollment files found under s3://%s/%s",
            S3_BUCKET_NAME,
            prefix,
        )
        return []

    enrollments: List[Dict[str, Any]] = []
    for key in keys:
        try:
            obj = s3.get_object(Bucket=S3_BUCKET_NAME, Key=key)
            record = json.loads(obj["Body"].read())
            enrollments.append(
                {
                    "username": record["username"],
                    "embedding": record["embedding"],
                    "device_id": record["device_id"],
                }
            )
        except Exception as exc:
            logger.warning("read_enrollments: failed to read %s: %s", key, exc)
            continue

    logger.info(
        "read_enrollments: collected %d enrollment(s) from %d file(s)",
        len(enrollments),
        len(keys),
    )
    return enrollments


def process_baseline_callable(**context: Any) -> None:
    enrollments = context["ti"].xcom_pull(task_ids="read_enrollments") or []
    if not enrollments:
        logger.info("process_baseline: no enrollments to process — skipping.")
        return None

    succeeded = 0
    failed = 0
    for record in enrollments:
        payload = {
            "device_id": record["device_id"],
            "username": record["username"],
            "vectors": [
                {
                    "embedding": record["embedding"],
                    "is_correct": True,
                }
            ],
        }
        try:
            _post(
                f"{VECTOR_OP_URL}/vectors/update/user-baseline",
                payload,
            )
            succeeded += 1
        except Exception as exc:
            logger.warning(
                "process_baseline: failed for user %s (device %s): %s",
                record["username"],
                record["device_id"],
                exc,
            )
            failed += 1
            continue

    logger.info(
        "process_baseline: completed — %d succeeded, %d failed out of %d",
        succeeded,
        failed,
        len(enrollments),
    )


# DAG
default_args = {
    "owner": "cognibrew",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="cognibrew_baseline_pipeline",
    default_args=default_args,
    description=(
        "Read enrollment JSONs from S3 enrollments/{username}/ "
        "--> process baseline via vector-operation API"
    ),
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["cognibrew"],
) as dag:
    # 1. Read enrollment records from S3
    read_enrollments = PythonOperator(
        task_id="read_enrollments",
        python_callable=read_enrollments_callable,
    )

    # 2. Process baseline via vector-operation API
    process_baseline = PythonOperator(
        task_id="process_baseline",
        python_callable=process_baseline_callable,
    )

    # Task dependencies
    read_enrollments >> process_baseline
