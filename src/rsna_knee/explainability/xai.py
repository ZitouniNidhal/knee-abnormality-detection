"""Explainable AI module — Phase 5.

Implements:
- GradCAM (Selvaraju et al., 2017)
- GradCAM++ (Chattopadhyay et al., 2018)
- Integrated Gradients (Sundararajan et al., 2017)
- SHAP DeepExplainer (shapley values)
- Per-patient explanation reports
- Automatic output storage
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..constants import TARGETS
from ..utils.logging import get_logger

logger = get_logger(__name__)


# ── GradCAM ───────────────────────────────────────────────────────────────────


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017).

    Works with any model that has a convolutional backbone. For ViT models,
    attention rollout should be used instead; a RuntimeError is raised if no
    conv features are found.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._gradients: torch.Tensor | None = None
        self._activations: torch.Tensor | None = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(
            module: nn.Module, inp: Any, out: torch.Tensor  # noqa: ANN401
        ) -> None:
            self._activations = out.detach()

        def backward_hook(
            module: nn.Module,
            grad_in: Any,  # noqa: ANN401
            grad_out: tuple[torch.Tensor, ...],
        ) -> None:
            self._gradients = grad_out[0].detach()

        self._fwd_handle = self.target_layer.register_forward_hook(forward_hook)
        self._bwd_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: int,
    ) -> np.ndarray:
        """Compute GradCAM heatmap for *class_idx*.

        Parameters
        ----------
        input_tensor: (1, C, H, W) preprocessed image tensor
        class_idx: target class index (0–11)

        Returns
        -------
        np.ndarray of shape (H, W) with values in [0, 1]
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)
        self.model.zero_grad()

        output = self.model(input_tensor)  # (1, num_classes)
        score = output[0, class_idx]
        score.backward()

        if self._gradients is None or self._activations is None:
            raise RuntimeError("Hooks did not capture gradients/activations.")

        # Pool gradients across spatial dimensions
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = torch.relu(cam).squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam.astype(np.float32)

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *args: Any) -> None:  # noqa: ANN401
        self.remove_hooks()


class GradCAMPlusPlus(GradCAM):
    """GradCAM++ (Chattopadhyay et al., 2018) — improved localization."""

    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: int,
    ) -> np.ndarray:
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)
        self.model.zero_grad()

        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()

        if self._gradients is None or self._activations is None:
            raise RuntimeError("Hooks did not capture gradients/activations.")

        grads = self._gradients  # (1, C, H, W)
        acts = self._activations  # (1, C, H, W)

        grads_sq = grads**2
        grads_cu = grads**3
        acts_sum = acts.sum(dim=(2, 3), keepdim=True)

        alpha_num = grads_sq
        alpha_denom = 2 * grads_sq + acts_sum * grads_cu + 1e-8
        alpha = alpha_num / alpha_denom

        relu_grad = torch.relu(score.exp() * grads)
        weights = (alpha * relu_grad).mean(dim=(2, 3), keepdim=True)

        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam.astype(np.float32)


# ── Integrated Gradients ──────────────────────────────────────────────────────


class IntegratedGradients:
    """Integrated Gradients attribution (Sundararajan et al., 2017)."""

    def __init__(self, model: nn.Module, n_steps: int = 50) -> None:
        self.model = model
        self.n_steps = n_steps

    def attribute(
        self,
        input_tensor: torch.Tensor,
        class_idx: int,
        baseline: torch.Tensor | None = None,
    ) -> np.ndarray:
        """Compute IG attributions.

        Returns np.ndarray of shape (C, H, W) — absolute values summed → (H, W)
        for visualization.
        """
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)

        # Interpolated inputs: n_steps from baseline to input
        alphas = torch.linspace(0, 1, self.n_steps, device=input_tensor.device)
        interpolated = baseline + alphas.view(-1, 1, 1, 1) * (input_tensor - baseline)
        interpolated.requires_grad_(True)

        self.model.eval()
        output = self.model(interpolated)  # (n_steps, num_classes)
        score = output[:, class_idx].sum()
        score.backward()

        if interpolated.grad is None:
            raise RuntimeError("No gradients computed for integrated gradients.")

        grads = interpolated.grad  # (n_steps, C, H, W)
        avg_grads = grads.mean(dim=0)  # (C, H, W)
        ig = (input_tensor.squeeze(0) - baseline.squeeze(0)) * avg_grads
        ig_np = ig.detach().cpu().numpy()  # (C, H, W)

        # Sum absolute value across channels for visualization
        saliency = np.abs(ig_np).sum(axis=0)
        saliency_max = saliency.max()
        if saliency_max > 1e-8:
            saliency = saliency / saliency_max
        return saliency.astype(np.float32)


# ── SHAP ──────────────────────────────────────────────────────────────────────


class SHAPExplainer:
    """SHAP DeepExplainer wrapper for PyTorch models."""

    def __init__(
        self,
        model: nn.Module,
        background_samples: torch.Tensor,
    ) -> None:
        try:
            import shap  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "shap is required for SHAPExplainer. "
                "Install with: pip install shap"
            ) from exc

        self.model = model
        import shap  # noqa: PLC0415

        self._explainer = shap.DeepExplainer(model, background_samples)

    def explain(self, inputs: torch.Tensor, class_idx: int) -> np.ndarray:
        """Return per-pixel SHAP values summed across channels (H, W)."""
        shap_values = self._explainer.shap_values(inputs)
        # shap_values: list[array(B, C, H, W)] per class
        sv = np.array(shap_values[class_idx])  # (B, C, H, W)
        saliency = np.abs(sv).sum(axis=1)  # (B, H, W)
        return saliency.astype(np.float32)


# ── Visualization helpers ─────────────────────────────────────────────────────


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """Overlay a grayscale heatmap on an RGB image.

    Parameters
    ----------
    image: (H, W, 3) uint8 RGB image
    heatmap: (H, W) float32 in [0, 1]
    alpha: transparency of overlay
    colormap: matplotlib colormap name

    Returns
    -------
    np.ndarray (H, W, 3) uint8
    """
    import cv2  # noqa: PLC0415

    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap_color + (1 - alpha) * image).clip(0, 255).astype(np.uint8)
    return overlay


def save_explanation_figure(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    *,
    target_name: str,
    method: str,
    study_id: str,
    output_dir: str | Path,
) -> Path:
    """Save a side-by-side (original | heatmap) explanation figure."""
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        logger.warning("matplotlib not installed; skipping figure save")
        return Path(output_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_image, cmap="gray")
    axes[0].set_title("Original MRI Slice")
    axes[0].axis("off")

    overlay = overlay_heatmap(
        np.stack([original_image] * 3, axis=-1) if original_image.ndim == 2 else original_image,
        heatmap,
    )
    axes[1].imshow(overlay)
    axes[1].set_title(f"{method} — {target_name}")
    axes[1].axis("off")

    safe_id = study_id.replace("/", "_")[:40]
    safe_target = target_name.replace(" ", "_").replace("'", "")
    out_path = output_dir / f"{safe_id}_{method}_{safe_target}.png"
    fig.suptitle(f"Study: {study_id} | Finding: {target_name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── Explanation report ────────────────────────────────────────────────────────


def generate_explanation_report(
    study_id: str,
    explanations: dict[str, dict[str, str]],
    output_path: str | Path,
) -> Path:
    """Write a per-patient explanation report in Markdown.

    Parameters
    ----------
    study_id: Patient study identifier
    explanations: {target → {method → figure_path}}
    output_path: Where to write the .md file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Explainability Report — {study_id}",
        "\n> Auto-generated by the RSNA Knee Abnormality Detection pipeline.",
        "\n---\n",
    ]
    for target, methods in explanations.items():
        lines.append(f"## {target}\n")
        for method, fig_path in methods.items():
            lines.append(f"### {method}")
            lines.append(f"![{method} heatmap]({fig_path})\n")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


# ── Storage helper ────────────────────────────────────────────────────────────


class ExplainabilityStorage:
    """Manages automatic storage of XAI outputs with an index JSON."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._index: list[dict[str, Any]] = []
        self._index_path = self.base_dir / "xai_index.json"

    def store(
        self,
        figure_path: Path,
        study_id: str,
        target: str,
        method: str,
        score: float,
    ) -> None:
        self._index.append({
            "study_id": study_id,
            "target": target,
            "method": method,
            "score": score,
            "figure": str(figure_path),
        })
        self._index_path.write_text(json.dumps(self._index, indent=2), encoding="utf-8")

    def get_index(self) -> list[dict[str, Any]]:
        return list(self._index)
