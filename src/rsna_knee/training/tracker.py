"""MLOps experiment tracker — Phase 6.

Provides a unified ExperimentTracker that simultaneously logs to:
- MLflow (always enabled by default)
- Weights & Biases (optional)
- TensorBoard (optional)

Usage
-----
    tracker = ExperimentTracker(cfg)
    tracker.start_run("fold-0")
    tracker.log_params({"lr": 3e-4, "epochs": 12})
    tracker.log_metrics({"val/auc": 0.91}, step=1)
    tracker.log_artifact("results/model.pt")
    tracker.end_run()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ExperimentTracker:
    """Unified tracker for MLflow + W&B + TensorBoard."""

    def __init__(
        self,
        *,
        experiment_name: str = "rsna-knee",
        mlflow_uri: str = "mlruns",
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        tensorboard_log_dir: str = "logs/tensorboard",
        wandb_project: str = "rsna-knee-abnormality",
        wandb_entity: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self._tags = tags or {}
        self._use_wandb = use_wandb
        self._use_tb = use_tensorboard
        self._tb_log_dir = tensorboard_log_dir
        self._wandb_project = wandb_project
        self._wandb_entity = wandb_entity

        # MLflow
        try:
            import mlflow  # noqa: PLC0415

            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment(experiment_name)
            self._mlflow = mlflow
            logger.info("MLflow configured", tracking_uri=mlflow_uri, experiment=experiment_name)
        except ImportError:
            self._mlflow = None
            logger.warning("mlflow not installed; MLflow tracking disabled")

        # TensorBoard
        self._tb_writer: Any = None

    def start_run(
        self,
        run_name: str = "",
        *,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Start a new tracking run."""
        if self._mlflow is not None:
            self._mlflow.start_run(run_name=run_name, tags=self._tags)
            if params:
                self._mlflow.log_params(params)

        if self._use_wandb:
            try:
                import wandb  # noqa: PLC0415

                wandb.init(
                    project=self._wandb_project,
                    entity=self._wandb_entity,
                    name=run_name,
                    config=params or {},
                    tags=list(self._tags.values()),
                    reinit=True,
                )
                self._wandb = wandb
            except ImportError:
                logger.warning("wandb not installed; W&B tracking disabled")
                self._use_wandb = False

        if self._use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415

                log_dir = Path(self._tb_log_dir) / run_name
                self._tb_writer = SummaryWriter(log_dir=str(log_dir))
                logger.debug("TensorBoard writer created", log_dir=str(log_dir))
            except ImportError:
                logger.warning("tensorboard not installed; TensorBoard disabled")
                self._use_tb = False

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters."""
        if self._mlflow is not None:
            self._mlflow.log_params(params)
        if self._use_wandb and hasattr(self, "_wandb"):
            self._wandb.config.update(params)

    def log_metrics(self, metrics: dict[str, float], *, step: int | None = None) -> None:
        """Log scalar metrics at an optional step."""
        if self._mlflow is not None:
            self._mlflow.log_metrics(metrics, step=step)
        if self._use_wandb and hasattr(self, "_wandb"):
            self._wandb.log(metrics, step=step)
        if self._tb_writer is not None:
            for key, value in metrics.items():
                self._tb_writer.add_scalar(key, value, global_step=step)

    def log_artifact(self, artifact_path: str | Path) -> None:
        """Log a file or directory artifact."""
        path = Path(artifact_path)
        if not path.exists():
            logger.warning("Artifact path does not exist", path=str(path))
            return
        if self._mlflow is not None:
            if path.is_dir():
                self._mlflow.log_artifacts(str(path))
            else:
                self._mlflow.log_artifact(str(path))
        if self._use_wandb and hasattr(self, "_wandb"):
            self._wandb.save(str(path))

    def log_figure(self, figure: Any, name: str, *, step: int | None = None) -> None:  # noqa: ANN401
        """Log a matplotlib figure."""
        if self._mlflow is not None:
            import tempfile  # noqa: PLC0415

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                figure.savefig(tmp.name, dpi=120, bbox_inches="tight")
                self._mlflow.log_artifact(tmp.name, artifact_path="figures")
        if self._use_wandb and hasattr(self, "_wandb"):
            self._wandb.log({name: self._wandb.Image(figure)}, step=step)
        if self._tb_writer is not None:
            self._tb_writer.add_figure(name, figure, global_step=step)

    def register_model(
        self,
        model_path: str | Path,
        *,
        model_name: str = "rsna-knee-model",
        tags: dict[str, str] | None = None,
    ) -> None:
        """Register model in MLflow model registry."""
        if self._mlflow is None:
            logger.warning("MLflow not available; model registration skipped")
            return
        try:
            run = self._mlflow.active_run()
            if run is None:
                logger.warning("No active MLflow run; model registration skipped")
                return
            run_id = run.info.run_id
            model_uri = f"runs:/{run_id}/model"
            self._mlflow.register_model(model_uri=model_uri, name=model_name, tags=tags)
            logger.info("Model registered", name=model_name, run_id=run_id)
        except Exception as exc:  # noqa: BLE001
            logger.error("Model registration failed", error=str(exc))

    def end_run(self) -> None:
        """Close all tracking backends."""
        if self._mlflow is not None:
            self._mlflow.end_run()
        if self._use_wandb and hasattr(self, "_wandb"):
            self._wandb.finish()
        if self._tb_writer is not None:
            self._tb_writer.close()

    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, *args: Any) -> None:  # noqa: ANN401
        self.end_run()
