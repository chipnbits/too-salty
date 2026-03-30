"""Shared evaluation utilities: model evaluation, checkpoint loading, key helpers."""

import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from salty.models import get_model, get_model_from_config

# Standard filename pattern for finetuned models: epoch_{N}_model_{M}.pt (legacy flat files)
fine_tuned_pattern = re.compile(r"epoch_(\d+)_model_(\d+)\.pt")

# Directory pattern for finetuned models: *-epoch_{N}_model_{M} (new layout)
fine_tuned_dir_pattern = re.compile(r".*-epoch_(\d+)_model_(\d+)$")


def canonical_key(key_a, key_b):
    """Create a canonical (sorted) key for a pair of model keys."""
    return ";".join(sorted([str(key_a), str(key_b)]))


def evaluate_model(model, dataloader, device):
    """Evaluate a model on a dataloader, returning (accuracy, avg_loss)."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            predicted = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)

    accuracy = 100 * correct / total
    average_loss = total_loss / total
    return accuracy, average_loss


def load_model_from_checkpoint(path, device, model_name="resnet50", num_classes=100):
    """Load a model from a checkpoint file. Infers model type from checkpoint config if available."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", None)
    if cfg is not None:
        model = get_model_from_config(cfg)
    else:
        model = get_model(model_name, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    start_epoch = checkpoint.get("epoch", 0)
    return model, start_epoch


def extract_key_from_filename(filename: str, pattern: re.Pattern) -> Optional[str]:
    """Extract a unique key {epoch}_{model_id} from a saved model filename."""
    match = pattern.match(filename)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    return None
