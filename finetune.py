"""Finetune multiple models from a common checkpoint with varied optimizer hyperparameters.

This script loads a trained checkpoint and continues training multiple models with
randomized optimizer hyperparameters (learning rate, momentum, weight decay).
Each model diverges from the common starting point with different training dynamics.
"""

import argparse
import os
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv

import wandb
from salty.datasets import get_cifar100_loaders
from salty.models import get_resnet50_model
from salty.utils import load_checkpoint, load_checkpoint_config, load_config, save_checkpoint
from train import (
    build_optimizer,
    build_scheduler,
    get_device,
    train_loop,
)

load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR", "./models")
DATA_DIR = os.getenv("DATA_DIR", "./data")


def randomize_optimizer_hyperparams(
    optimizer, lr_scale_range=1.0, momentum_scale_range=1.0, wd_scale_range=1.0, seed=None
):
    """
    Randomize optimizer hyperparameters by scaling them with uniform random multipliers.

    Args:
        optimizer: PyTorch optimizer instance
        lr_scale_range: Maximum scale factor for learning rate (uniform in [1-range, 1+range])
        momentum_scale_range: Maximum scale factor for momentum (uniform in [1-range, 1+range])
        wd_scale_range: Maximum scale factor for weight decay (uniform in [1-range, 1+range])
        seed: Random seed for reproducibility

    Returns:
        Dictionary with original and new hyperparameters
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    def log_uniform_scale(scale_range):
        log_scale = rng.uniform(0.2 * scale_range, scale_range)
        random_sign = rng.choice([-1, 1])
        log_scale *= random_sign
        return 2**log_scale

    lr_new = optimizer.param_groups[0]["lr"] * log_uniform_scale(lr_scale_range)
    momentum_new = optimizer.param_groups[0].get("momentum", 0.9) * log_uniform_scale(momentum_scale_range)
    wd_new = optimizer.param_groups[0].get("weight_decay", 0.0) * log_uniform_scale(wd_scale_range)

    original_params = {}
    new_params = {}

    for group_idx, param_group in enumerate(optimizer.param_groups):
        # Store original hyperparameters
        original_params[group_idx] = {
            "lr": param_group["lr"],
            "momentum": param_group.get("momentum", None),
            "weight_decay": param_group.get("weight_decay", None),
        }

        # Update hyperparameters
        param_group["lr"] = lr_new
        if "momentum" in param_group:
            param_group["momentum"] = momentum_new
        if "weight_decay" in param_group:
            param_group["weight_decay"] = wd_new

        # Store new hyperparameters
        new_params[group_idx] = {
            "lr": param_group["lr"],
            "momentum": param_group.get("momentum", None),
            "weight_decay": param_group.get("weight_decay", None),
        }

    return {
        "original": original_params,
        "new": new_params,
        "scales": {
            "lr_scale": lr_new / original_params[0]["lr"],
            "momentum_scale": (
                (momentum_new / original_params[0]["momentum"]) if original_params[0]["momentum"] is not None else None
            ),
            "wd_scale": (
                (wd_new / original_params[0]["weight_decay"])
                if original_params[0]["weight_decay"] is not None
                else None
            ),
        },
    }


def finetune_from_checkpoint(
    checkpoint_path,
    num_models,
    lr_scale_range=1.0,
    momentum_scale_range=1.0,
    wd_scale_range=1.0,
    device=None,
    base_seed=42,
    model_save_idx=0,
):
    """
    Finetune multiple models from a common checkpoint with varied optimizer hyperparameters.

    Args:
        checkpoint_path: Path to checkpoint file to load
        num_models: Number of models to train from the checkpoint
        additional_epochs: Number of additional epochs to train each model
        lr_scale_range: Max random log scale range for learning rate (default: 1.0 = 100%)
        momentum_scale_range: Max random log scale range for momentum (default: 1.0 = 100%)
        wd_scale_range: Max random log scale range for weight decay (default: 1.0 = 100%)
        device: Device to train on (cuda/cpu/mps), auto-detected if None
        base_seed: Base random seed for reproducibility
    """

    # Strip name from the .pt checkpoint file for saving
    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    # Load checkpoint to get config
    print(f"Loading base checkpoint from: {checkpoint_path}")
    base_config = load_checkpoint_config(checkpoint_path)

    device = get_device(device)

    # Data loaders (same as original training)
    train_loader, val_loader, test_loader = get_cifar100_loaders(
        batch_size=base_config["training"]["batch_size"],
        num_workers=base_config["training"]["num_workers"],
        data_dir=DATA_DIR,
        val_ratio=base_config["data"].get("validation_split", 0.0),
        seed=base_config["data"]["split_seed"],
        augment=base_config["data"].get("augment", True),
    )

    # Train each model variant
    for model_idx in range(num_models):
        print(f"\n{'='*80}")
        print(f"Training model {model_idx + 1}/{num_models}")
        print(f"{'='*80}\n")

        model_seed = base_seed + model_idx
        torch.manual_seed(model_seed)
        np.random.seed(model_seed)
        random.seed(model_seed)

        # Build model
        model_cfg = base_config["model"]
        if model_cfg["name"] == "resnet50":
            model = get_resnet50_model(num_classes=base_config["data"]["num_classes"])
        else:
            raise NotImplementedError(f"Model {model_cfg['name']} not implemented.")
        model = model.to(device)

        # Build optimizer & scheduler
        optimizer = build_optimizer(base_config["optimizer"], model)
        scheduler = build_scheduler(base_config.get("scheduler", {}), optimizer)

        # Load checkpoint for THIS model & optimizer
        start_epoch, best_val_acc, loaded_cfg, global_step, wandb_run_id = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, scaler=None
        )

        # Overwrite best to ensure saved model is from finetuning
        best_val_acc = 0.0  # reset best

        # Now randomize hyperparams
        hp_changes = randomize_optimizer_hyperparams(
            optimizer,
            lr_scale_range=lr_scale_range,
            momentum_scale_range=momentum_scale_range,
            wd_scale_range=wd_scale_range,
            seed=model_seed,
        )

        print(f"Optimizer hyperparameter randomization:")
        print(f"  LR scale:     {hp_changes['scales']['lr_scale']:.4f}x")
        print(f"  Momentum scale: {hp_changes['scales']['momentum_scale']:.4f}x")
        print(f"  Weight decay scale: {hp_changes['scales']['wd_scale']:.4f}x")
        print(f"  New LR:       {hp_changes['new'][0]['lr']:.6f}")
        print(f"  New momentum: {hp_changes['new'][0]['momentum']:.6f}")
        print(f"  New weight decay: {hp_changes['new'][0]['weight_decay']:.6f}")
        print()

        # Create modified config for this model
        # Make a termination estimate based on original epochs + additional
        additional_epochs = base_config["training"].get("epochs", 300) - start_epoch + 10
        finetune_config = deepcopy(base_config)
        finetune_config["training"]["epochs"] = start_epoch + additional_epochs
        finetune_config["project_name"] = base_config["project_name"]
        finetune_config["run_name"] = f"{checkpoint_name}_model_{model_idx+1+model_save_idx}"
        finetune_config["logging"]["checkpoint_interval"] = 1000  # Only save best and last

        # Initialize wandb for this run
        wandb.init(
            project=finetune_config["project_name"],
            name=finetune_config["run_name"],
            config={
                **finetune_config,
                "finetune_from_checkpoint": checkpoint_path,
                "finetune_start_epoch": start_epoch,
                "finetune_model_idx": model_idx + model_save_idx,
                "finetune_seed": model_seed,
                "hp_scales": hp_changes["scales"],
                "hp_original": hp_changes["original"][0],
                "hp_new": hp_changes["new"][0],
            },
        )

        # Loss function
        train_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Train this model
        train_loop(
            model,
            train_loader,
            val_loader,
            train_loss_fn,
            optimizer,
            scheduler,
            device,
            finetune_config,
            start_epoch=start_epoch,
            best_val_acc=best_val_acc,
            global_step=global_step,
        )

        # Finish wandb run
        wandb.finish()

        print(f"\nCompleted model {model_idx + 1}/{num_models}")

    print(f"\n{'='*80}")
    print(f"Finetuning complete! Trained {num_models} models.")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finetune multiple models from a checkpoint with varied optimizer hyperparameters"
    )
    parser.add_argument("--checkpoint", type=str, required=False, help="Path to the checkpoint file to finetune from")
    parser.add_argument(
        "--num-models", type=int, default=2, help="Number of models to train from the checkpoint (default: 2)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of additional epochs to train each model (default: 50)"
    )
    parser.add_argument(
        "--lr-scale",
        type=float,
        default=1.0,
        help="Max random log scale range for learning rate, (default: 1.0 = 100%%)",
    )
    parser.add_argument(
        "--momentum-scale",
        type=float,
        default=1.0,
        help="Max random log scale range for momentum (default: 1.0 = 100%%)",
    )
    parser.add_argument(
        "--wd-scale", type=float, default=1.0, help="Max random log scale range for weight decay (default: 1.0 = 100%%)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for training (cuda/cpu/mps), auto-detected if not specified",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    # Validate checkpoint exists

    # finetune_from_checkpoint(
    #     checkpoint_path=args.checkpoint,
    #     num_models=args.num_models,
    #     additional_epochs=args.epochs,
    #     lr_scale_range=args.lr_scale,
    #     momentum_scale_range=args.momentum_scale,
    #     wd_scale_range=args.wd_scale,
    #     device=args.device,
    #     base_seed=args.seed,
    # )

    # Parse all checkpoints
    baseline_dir = "/data/sghyselincks/too-salty/models/cifar100-resnet50/baseline-resnet50"
    checkpoint_files = [f for f in os.listdir(baseline_dir) if f.endswith(".pt")]

    # Order them by epoch number if possible _110.pt
    def extract_epoch_num(filename):
        try:
            parts = filename.split("_")
            epoch_part = parts[-1]  # e.g., "110.pt"
            epoch_str = epoch_part.split(".")[0]  # e.g., "110"
            return int(epoch_str)
        except (IndexError, ValueError):
            return -1  # Default if parsing fails

    # Largest epoch first
    checkpoint_files.sort(key=extract_epoch_num, reverse=True)

    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(baseline_dir, checkpoint_file)
        finetune_from_checkpoint(
            checkpoint_path=checkpoint_path,
            num_models=args.num_models,
            lr_scale_range=1.0,
            momentum_scale_range=0.4,
            wd_scale_range=0.4,
            device=args.device,
            base_seed=args.seed,
            model_save_idx=2,
        )

    print(f"Found {len(checkpoint_files)} checkpoint files in {baseline_dir}")
