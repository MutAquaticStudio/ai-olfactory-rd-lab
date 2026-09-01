"""Local MLflow adapter for reproducible training manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple


def _numeric_metrics(payload: object, prefix: str = "") -> Iterable[Tuple[str, float]]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _numeric_metrics(value, next_prefix)
    elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
        yield prefix, float(payload)


def log_manifest_to_mlflow(
    manifest: Dict[str, object],
    manifest_path: Path,
    tracking_root: Path,
) -> str:
    """Log immutable metadata to local SQLite/artifact storage and return run ID."""
    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError("Training tracking requires requirements-training.txt") from error
    root = Path(tracking_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    database = root / "mlflow.sqlite3"
    artifacts = root / "mlruns"
    artifacts.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{database}")
    experiment_name = str(manifest.get("model_family", manifest.get("architecture", "scent-model")))
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = (
        mlflow.create_experiment(experiment_name, artifact_location=artifacts.as_uri())
        if experiment is None
        else experiment.experiment_id
    )
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=str(manifest.get("run_id", "training-run")),
    ) as run:
        parameters = {
            key: value
            for key, value in manifest.items()
            if key in {"run_id", "model_version", "dataset_version", "split_hash", "seed", "git_commit", "status"}
            and value is not None
        }
        if parameters:
            mlflow.log_params({key: str(value) for key, value in parameters.items()})
        metrics = {
            key[:250]: value
            for key, value in _numeric_metrics(manifest.get("metrics", manifest.get("benchmark", {})))
            if value == value
        }
        if metrics:
            mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(manifest_path), artifact_path="manifests")
        return run.info.run_id
