"""Loss functions for multilabel knee MRI classification.

Implements:
- WeightedBCELoss: gold vs pseudo-label sample weighting
- FocalLoss: hard-example mining (Lin et al., 2017)
- AsymmetricLoss: asymmetric focusing for medical imbalance (Ridnik et al., 2021)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class WeightedBCELoss(nn.Module):
    """Binary cross-entropy with per-sample weights.

    Used to up-weight gold labels (radiologist-confirmed) over pseudo-labels
    (report-extracted with NLP heuristics).
    """

    def __init__(
        self,
        gold_label_weight: float = 8.0,
        pseudo_label_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.gold_weight = gold_label_weight
        self.pseudo_weight = pseudo_label_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute weighted BCE.

        Parameters
        ----------
        logits: (B, C) raw logits
        targets: (B, C) binary labels in [0, 1]
        sample_weights: (B,) per-sample weight; 1.0 = pseudo, gold_weight = gold
        """
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # (B, C)
        if sample_weights is not None:
            loss = loss * sample_weights.unsqueeze(1)
        return loss.mean()


class FocalLoss(nn.Module):
    """Focal loss for class imbalance (Lin et al., 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce
        if sample_weights is not None:
            loss = loss * sample_weights.unsqueeze(1)
        return loss.mean()


class AsymmetricLoss(nn.Module):
    """Asymmetric loss for extreme class imbalance (Ridnik et al., 2021).

    Different focusing factors for positives (gamma_pos) and negatives (gamma_neg).
    Probability margin *m* shifts negative probabilities down to reduce false-positive noise.
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1 - xs_pos

        # Probability margin
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.gamma_pos > 0:
                pt0 = xs_pos * targets
                los_pos = los_pos * (1 - pt0) ** self.gamma_pos
            if self.gamma_neg > 0:
                pt1 = xs_neg * (1 - targets)
                los_neg = los_neg * (1 - pt1) ** self.gamma_neg

        loss = -(los_pos + los_neg)
        if sample_weights is not None:
            loss = loss * sample_weights.unsqueeze(1)
        return loss.mean()


def build_loss(name: str, **kwargs: float) -> nn.Module:
    """Factory for loss functions by name."""
    losses = {
        "weighted_bce": WeightedBCELoss,
        "focal": FocalLoss,
        "asymmetric": AsymmetricLoss,
    }
    if name not in losses:
        raise ValueError(f"Unknown loss '{name}'. Supported: {sorted(losses)}")
    return losses[name](**kwargs)
