"""Single dispatch point for `build_model(name, **cfg)`."""

from __future__ import annotations

from torch import nn

from mlops_project.models.baseline import StatsLogisticRegression
from mlops_project.models.simple_cnn import SimpleCNN
from mlops_project.models.transfer import ResNet50Transfer
from mlops_project.models.unet_classifier import UNetClassifier

MODEL_NAMES = ("baseline", "simple_cnn", "unet_classifier", "resnet50_transfer")


def build_model(name: str, **kwargs) -> nn.Module:
    """Construct one of the four supported architectures.

    Raises:
        ValueError: if `name` is not in MODEL_NAMES.
    """
    if name == "baseline":
        return StatsLogisticRegression()
    if name == "simple_cnn":
        return SimpleCNN(**kwargs)
    if name == "unet_classifier":
        return UNetClassifier(**kwargs)
    if name == "resnet50_transfer":
        return ResNet50Transfer(**kwargs)
    raise ValueError(f"Unknown model: {name!r}. Valid: {MODEL_NAMES}")


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
