import os

import mlflow

from cognibrew.logger import Logger

logger = Logger().get_logger()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "vector-evolution"))


def log_drift_metrics(
    username: str,
    mean_drift: float,
    max_drift: float,
    is_drifting: bool,
    gallery_size: int,
) -> None:
    """Log per-user drift metrics to MLflow."""
    try:
        with mlflow.start_run(run_name=f"drift-{username}", nested=True):
            mlflow.log_param("username", username)
            mlflow.log_metric("mean_drift", mean_drift)
            mlflow.log_metric("max_drift", max_drift)
            mlflow.log_metric("is_drifting", int(is_drifting))
            mlflow.log_metric("gallery_size", gallery_size)
    except Exception:
        logger.warning("Failed to log drift metrics to MLflow", exc_info=True)


def log_outlier_metrics(
    total_input: int,
    accepted: int,
    rejected: int,
) -> None:
    """Log outlier filtering summary to MLflow."""
    try:
        with mlflow.start_run(run_name="outlier-filter", nested=True):
            mlflow.log_metric("total_input", total_input)
            mlflow.log_metric("accepted", accepted)
            mlflow.log_metric("rejected", rejected)
            if total_input > 0:
                mlflow.log_metric("rejection_rate", rejected / total_input)
    except Exception:
        logger.warning(
            "Failed to log outlier metrics to MLflow", exc_info=True
        )


def log_gallery_expansion(
    username: str,
    vectors_added: int,
    new_gallery_size: int,
) -> None:
    """Log gallery expansion event."""
    try:
        with mlflow.start_run(
            run_name=f"gallery-expand-{username}", nested=True
        ):
            mlflow.log_param("username", username)
            mlflow.log_metric("vectors_added", vectors_added)
            mlflow.log_metric("new_gallery_size", new_gallery_size)
    except Exception:
        logger.warning(
            "Failed to log gallery expansion to MLflow", exc_info=True
        )
