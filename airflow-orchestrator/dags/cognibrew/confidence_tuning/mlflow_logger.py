import os

import mlflow

from cognibrew.logger import Logger


logger = Logger().get_logger()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "confidence-tuning"))


def log_threshold_experiment(
    threshold: float,
    f1_score: float,
    precision: float,
    recall: float,
    total_samples: int,
    drift_adjusted: bool,
    username: str | None = None,
) -> None:
    """Log a threshold tuning experiment run."""
    try:
        run_name = f"tune-{username}" if username else "tune-global"
        with mlflow.start_run(run_name=run_name, nested=True):
            if username:
                mlflow.log_param("username", username)
            mlflow.log_param("drift_adjusted", drift_adjusted)
            mlflow.log_metric("threshold", threshold)
            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("total_samples", total_samples)
    except Exception:
        logger.warning(
            "Failed to log threshold experiment to MLflow", exc_info=True
        )
