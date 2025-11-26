import math
import os
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import torch
import torch.functional as F
import yaml

import wandb
from salty.datasets import CIFAR100_MEAN, CIFAR100_STD


def normalize_cifar100(tensor):
    """
    Normalize a tensor image with CIFAR-100 mean and standard deviation.

    Args:
        tensor: Tensor image of size (C, H, W) to be normalized

    Returns:
        Normalized tensor image
    """
    normalize = F.normalize(
        tensor,
        mean=torch.tensor(CIFAR100_MEAN, device=tensor.device).view(-1, 1, 1),
        std=torch.tensor(CIFAR100_STD, device=tensor.device).view(-1, 1, 1),
    )
    return normalize


def denormalize_cifar100(tensor):
    """
    Denormalize a tensor image with CIFAR-100 mean and standard deviation.

    Args:
        tensor: Tensor image of size (C, H, W) to be denormalized

    Returns:
        Denormalized tensor image
    """
    denormalize = tensor * torch.tensor(CIFAR100_STD, device=tensor.device).view(-1, 1, 1) + torch.tensor(
        CIFAR100_MEAN, device=tensor.device
    ).view(-1, 1, 1)
    return denormalize


def show_batch_with_labels(images, labels, class_names, pad=4):
    """
    Display a batch of CIFAR images in the closest square grid,
    with white padding and class labels under each tile.
    """
    B, C, H, W = images.shape

    # ---- compute grid size ----
    n = math.ceil(math.sqrt(B))
    num_cells = n * n

    # ---- pad batch if needed ----
    if num_cells > B:
        pad_count = num_cells - B
        pad_imgs = torch.ones((pad_count, C, H, W), device=images.device)  # white padding
        pad_labels = torch.full((pad_count,), -1, device=labels.device)  # no label
        images = torch.cat([images, pad_imgs], 0)
        labels = torch.cat([labels, pad_labels], 0)

    # ---- add white border around each image ----
    # pad=(left,right,top,bottom)
    images = torch.nn.functional.pad(images, (pad, pad, pad, pad), value=1.0)
    Hp, Wp = H + 2 * pad, W + 2 * pad

    # ---- einops grid: (n*n, C, Hp, Wp) -> (n*Hp, n*Wp, C) ----
    img_grid = einops.rearrange(
        images,
        "(n1 n2) c hp wp -> (n1 hp) (n2 wp) c",
        n1=n,
        n2=n,
    )

    # ---- plot grid first ----
    plt.figure(figsize=(n * 2.4, n * 2.4))
    plt.imshow(img_grid.cpu().numpy())
    plt.axis("off")

    # ---- draw labels under each tile ----
    for idx in range(num_cells):
        if labels[idx] == -1:
            continue

        row = idx // n
        col = idx % n

        x = col * Wp + Wp / 2
        y = row * Hp + Hp - 2  # bottom of the tile

        label_str = class_names[labels[idx].item()]

        plt.text(
            x,
            y,
            label_str,
            ha="center",
            va="bottom",
            fontsize=12,
            color="black",
        )

    plt.tight_layout()
    plt.show()


class RunningAverage:
    """A simple class that maintains the running average of a quantity."""

    def __init__(self, val=0.0):
        self.steps = 0
        self.total = val

    def update(self, val: float) -> None:
        self.total += val
        self.steps += 1

    @property
    def average(self) -> float:
        return self.total / self.steps if self.steps > 0 else ZeroDivisionError("No steps to compute average.")

    @property
    def value(self) -> float:
        return self.total


def load_config(config_path):
    path = Path(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    cfg,
    best_val_acc=None,
    global_step=None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "config": cfg,
        "best_val_acc": best_val_acc,
        "global_step": global_step,
        "wandb_run_id": wandb.run.id if wandb.run is not None else None,
    }
    torch.save(state, path)
    print(f"Saved checkpoint: {path}")


def load_checkpoint(path, model: torch.nn.Module, optimizer=None, scheduler=None, scaler=None):
    print(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    if scaler is not None and checkpoint.get("scaler_state") is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_val_acc = checkpoint.get("best_val_acc", None)
    loaded_cfg = checkpoint.get("config", None)
    global_step = checkpoint.get("global_step", 0)
    wandb_run_id = checkpoint.get("wandb_run_id", None)

    return start_epoch, best_val_acc, loaded_cfg, global_step, wandb_run_id


def load_checkpoint_config(path):
    checkpoint = torch.load(path, map_location="cpu")
    loaded_cfg = checkpoint.get("config", None)
    return loaded_cfg
