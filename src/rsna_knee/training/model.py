"""Modular model backbone factory.

Supports: ResNet, EfficientNet, DenseNet, ConvNeXt, Vision Transformers (ViT).
All models are loaded via timm with configurable frozen/fine-tuned backbone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import timm
import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..constants import N_TARGETS
from ..utils.logging import get_logger

logger = get_logger(__name__)


# ── Model config ──────────────────────────────────────────────────────────────


@dataclass
class ModelConfig:
    """Typed configuration for model construction."""

    name: str
    timm_name: str
    embedding_dim: int
    image_size: int
    freeze_backbone: bool = False
    dropout: float = 0.2
    head_hidden_dim: int = 256
    num_classes: int = N_TARGETS
    pretrained: bool = True

    @classmethod
    def from_omegaconf(cls, cfg: DictConfig) -> "ModelConfig":
        return cls(
            name=cfg.name,
            timm_name=cfg.timm_name,
            embedding_dim=cfg.embedding_dim,
            image_size=cfg.image_size,
            freeze_backbone=cfg.get("freeze_backbone", False),
            dropout=cfg.get("dropout", 0.2),
            head_hidden_dim=cfg.get("head_hidden_dim", 256),
            num_classes=cfg.get("num_classes", N_TARGETS),
            pretrained=cfg.get("pretrained", True),
        )


# ── Classification head ───────────────────────────────────────────────────────


class MultilabelHead(nn.Module):
    """Two-layer MLP classification head for multilabel outputs."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.net(x)


# ── Main model ────────────────────────────────────────────────────────────────


class KneeAbnormalityModel(nn.Module):
    """Knee MRI multilabel classifier.

    Architecture
    ------------
    - timm backbone (any supported architecture)
    - Global average pool (already built into most timm models)
    - Two-layer MLP head with LayerNorm, GELU, and Dropout
    - Raw logits output (apply sigmoid externally for probabilities)
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Build backbone without classification head
        self.backbone = timm.create_model(
            config.timm_name,
            pretrained=config.pretrained,
            num_classes=0,  # Remove classifier
            global_pool="avg",
        )

        if config.freeze_backbone:
            self._freeze_backbone()

        # Verify embedding dimension
        dummy = torch.zeros(1, 3, config.image_size, config.image_size)
        with torch.no_grad():
            out = self.backbone(dummy)
        actual_dim = out.shape[-1]
        if actual_dim != config.embedding_dim:
            logger.warning(
                "Embedding dim mismatch; using actual",
                expected=config.embedding_dim,
                actual=actual_dim,
                model=config.timm_name,
            )

        self.head = MultilabelHead(
            in_features=actual_dim,
            hidden_dim=config.head_hidden_dim,
            num_classes=config.num_classes,
            dropout=config.dropout,
        )

        logger.info(
            "Model created",
            backbone=config.timm_name,
            embedding_dim=actual_dim,
            freeze_backbone=config.freeze_backbone,
            params_total=self._count_params(),
            params_trainable=self._count_params(trainable_only=True),
        )

    def _freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.debug("Backbone frozen")

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone for fine-tuning (e.g., second training phase)."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen")

    def _count_params(self, *, trainable_only: bool = False) -> int:
        params = (p for p in self.parameters())
        if trainable_only:
            params = (p for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits of shape (B, num_classes)."""
        features = self.backbone(x)
        return self.head(features)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities (inference convenience)."""
        with torch.inference_mode():
            return torch.sigmoid(self(x))

    def get_feature_maps(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (feature_maps, logits) for GradCAM; backbone must expose features.

        Works for CNN backbones that expose intermediate feature maps via hooks.
        """
        # Forward through backbone internals to get spatial features
        feature_maps: list[torch.Tensor] = []

        def hook_fn(module: nn.Module, inp: Any, out: torch.Tensor) -> None:  # noqa: ANN401
            feature_maps.append(out)

        # Attach hook to last conv/attention layer
        target_layer = self._get_target_layer()
        handle = target_layer.register_forward_hook(hook_fn)
        try:
            logits = self(x)
        finally:
            handle.remove()

        if not feature_maps:
            raise RuntimeError("No feature maps captured; check target layer selection.")
        return feature_maps[0], logits

    def _get_target_layer(self) -> nn.Module:
        """Return the last feature-producing layer for GradCAM targeting."""
        backbone = self.backbone

        # Try common attribute names used by timm models
        for attr in ("layer4", "blocks", "features", "stages"):
            layer = getattr(backbone, attr, None)
            if layer is not None:
                if isinstance(layer, nn.Sequential):
                    return layer[-1]
                if isinstance(layer, nn.ModuleList):
                    return layer[-1]

        # Fallback: last named module that is a conv or attention block
        last_module = None
        for _, module in backbone.named_modules():
            if isinstance(module, (nn.Conv2d, nn.MultiheadAttention)):
                last_module = module
        if last_module is not None:
            return last_module

        raise RuntimeError(
            f"Cannot auto-detect target layer for {type(backbone).__name__}. "
            "Please specify explicitly."
        )

    def checkpoint_dict(self) -> dict[str, Any]:
        """Return a versioned checkpoint dictionary."""
        from ..constants import TARGETS, TARGET_TO_INDEX  # noqa: PLC0415

        return {
            "model_state_dict": self.state_dict(),
            "config": {
                "name": self.config.name,
                "timm_name": self.config.timm_name,
                "embedding_dim": self.config.embedding_dim,
                "image_size": self.config.image_size,
                "freeze_backbone": self.config.freeze_backbone,
                "dropout": self.config.dropout,
                "head_hidden_dim": self.config.head_hidden_dim,
                "num_classes": self.config.num_classes,
            },
            "targets": list(TARGETS),
            "target_to_index": TARGET_TO_INDEX,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, Any]) -> "KneeAbnormalityModel":
        """Load model from a checkpoint dictionary."""
        config = ModelConfig(**checkpoint["config"], pretrained=False)
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model


# ── Factory ───────────────────────────────────────────────────────────────────

_SUPPORTED_ARCHITECTURES = {
    "vit_small_dinov2": "vit_small_patch14_dinov2.lvd142m",
    "resnet50": "resnet50.a1_in1k",
    "resnet34": "resnet34.a1_in1k",
    "efficientnet_b3": "efficientnet_b3.ra2_in1k",
    "efficientnet_b4": "efficientnet_b4.ra2_in1k",
    "densenet121": "densenet121.ra_in1k",
    "convnext_small": "convnext_small.in12k_ft_in1k",
    "convnext_tiny": "convnext_tiny.in12k_ft_in1k",
}


def list_supported_architectures() -> list[str]:
    """Return all supported architecture names."""
    return sorted(_SUPPORTED_ARCHITECTURES)


def build_model(config: ModelConfig) -> KneeAbnormalityModel:
    """Construct and return a model from a ModelConfig."""
    return KneeAbnormalityModel(config)
