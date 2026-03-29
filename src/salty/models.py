"""Model factory for creating architectures from config."""

import torch.nn as nn

from salty.resnet import ResNet, BasicBlock, BottleNeck
from salty.wide_resnet import WideResNet


RESNET_CONFIGS = {
    "resnet18": (BasicBlock, [2, 2, 2, 2]),
    "resnet34": (BasicBlock, [3, 4, 6, 3]),
    "resnet50": (BottleNeck, [3, 4, 6, 3]),
    "resnet101": (BottleNeck, [3, 4, 23, 3]),
    "resnet152": (BottleNeck, [3, 8, 36, 3]),
}

WIDERESNET_CONFIGS = {
    "wrn-28-10": (28, 10),
    "wrn-28-2": (28, 2),
    "wrn-16-8": (16, 8),
    "wrn-40-2": (40, 2),
}


def get_model(name: str, num_classes: int = 100, dropout: float = 0.0) -> nn.Module:
    """Create a model by name from config.

    Args:
        name: Model name (resnet18/34/50/101/152, wrn-28-10, wrn-28-2, etc.)
        num_classes: Number of output classes
        dropout: Dropout rate (only applies to WideResNet variants)

    Returns:
        Model instance
    """
    name = name.lower()

    if name in RESNET_CONFIGS:
        block, layers = RESNET_CONFIGS[name]
        return ResNet(block, layers, num_classes=num_classes)

    if name in WIDERESNET_CONFIGS:
        depth, widen_factor = WIDERESNET_CONFIGS[name]
        return WideResNet(depth=depth, num_classes=num_classes, widen_factor=widen_factor, dropRate=dropout)

    all_models = list(RESNET_CONFIGS.keys()) + list(WIDERESNET_CONFIGS.keys())
    raise ValueError(f"Unknown model: {name}. Available: {all_models}")


def get_model_from_config(cfg: dict) -> nn.Module:
    """Create a model from an experiment config dict.

    Reads model.name, data.num_classes, and model.dropout from the config.
    """
    model_cfg = cfg["model"]
    return get_model(
        name=model_cfg["name"],
        num_classes=cfg["data"]["num_classes"],
        dropout=model_cfg.get("dropout", 0.0),
    )


# Backward compatibility
def get_resnet50_model(num_classes: int = 100) -> ResNet:
    return get_model("resnet50", num_classes)
