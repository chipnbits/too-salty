"""Train a baseline model from a config file.

Usage:
    python scripts/02_train_baseline.py --config configs/resnet50_baseline.yaml
    python scripts/02_train_baseline.py --config configs/wrn2810.yaml
"""

import argparse
import os
from copy import deepcopy

import torch.nn as nn
from dotenv import load_dotenv

import wandb
from salty.datasets import get_data_loaders
from salty.models import get_model_from_config
from salty.training import (
    build_optimizer,
    build_scheduler,
    get_device,
    train_loop,
)
from salty.utils import (
    load_checkpoint,
    load_config,
    set_seed,
)

load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR", "./models")
DATA_DIR = os.getenv("DATA_DIR", "./data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--config", type=str, default="configs/resnet50_baseline.yaml", help="Path to the config file")
    parser.add_argument("--device", type=str, default=None, help="Device to use for training")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume training from")
    parser.add_argument("--seed", type=int, default=None, help="Override seed from config")
    parser.add_argument("--run-name", type=str, default=None, help="Override run name from config")
    args = parser.parse_args()
    config = load_config(args.config)

    # Seed: CLI overrides config, config defaults to 42
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    set_seed(seed)

    # Override run name if provided
    if args.run_name is not None:
        config["run_name"] = args.run_name

    # Handle resume from checkpoint
    if args.resume is not None:
        config.setdefault("resume", {})
        config["resume"]["checkpoint_path"] = args.resume

    # Build model from config
    model = get_model_from_config(config)

    # Build optimizer & scheduler
    optimizer = build_optimizer(config["optimizer"], model)
    scheduler = build_scheduler(config.get("scheduler", {}), optimizer)

    # Get data loaders from config
    train_loader, val_loader, test_loader = get_data_loaders(config, DATA_DIR)

    device = get_device(args.device)
    model = model.to(device)

    train_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Resume logic
    start_epoch = 0
    best_val_acc = None
    start_global_step = 0

    resume_cfg = config.get("resume", {})
    resume_path = resume_cfg.get("checkpoint_path", None)

    if resume_path is not None:
        start_epoch, best_val_acc, loaded_cfg, start_global_step, wandb_run_id = load_checkpoint(
            resume_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        print(
            f"Resumed from epoch {start_epoch}, "
            f"best_val_acc={best_val_acc}, "
            f"start_step={start_global_step}"
        )

    # wandb init
    wandb_config = deepcopy(config)
    wandb_config["seed"] = seed

    wandb.init(
        project=config["project_name"],
        name=config.get("run_name", None),
        config=wandb_config,
    )

    save_dir = os.path.join(MODEL_DIR, config.get("project_name"), config.get("run_name"))

    train_loop(
        model,
        train_loader,
        val_loader,
        train_loss_fn,
        optimizer,
        scheduler,
        device,
        config,
        save_dir=save_dir,
        start_epoch=start_epoch,
        best_val_acc=best_val_acc,
        global_step=start_global_step,
    )
