"""Finetune model variants from baseline checkpoints with scaled optimizer hyperparameters.

Each variant branches from a checkpoint and trains to completion with perturbed
learning rate, momentum, and weight decay. Scales are defined in a variants config
(e.g., configs/resnet50_finetune.yaml) or can be generated randomly.

Usage:
    # Single model from single checkpoint:
    python scripts/03_finetune_branches.py --checkpoint models/.../epoch_50.pt --model-idx 1

    # All models from single checkpoint:
    python scripts/03_finetune_branches.py --checkpoint models/.../epoch_50.pt

    # All models from all checkpoints in a directory:
    python scripts/03_finetune_branches.py --checkpoint-dir models/.../baseline-resnet50/

    # Generate random scales instead of using config:
    python scripts/03_finetune_branches.py --checkpoint ... --random-scales --num-models 4
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
    generate_random_scales,
    get_device,
    scale_optimizer,
    train_loop,
)
from salty.utils import load_checkpoint, load_checkpoint_config, load_config, set_seed

load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR", "./models")
DATA_DIR = os.getenv("DATA_DIR", "./data")


def finetune_single_model(
    checkpoint_path,
    model_idx,
    lr_scale,
    momentum_scale,
    wd_scale,
    device=None,
    seed=42,
):
    """Finetune a single model variant from a checkpoint with explicit scales.

    Args:
        checkpoint_path: Path to baseline checkpoint
        model_idx: Variant index (used for naming and seed offset)
        lr_scale: Multiplicative scale for learning rate
        momentum_scale: Multiplicative scale for momentum
        wd_scale: Multiplicative scale for weight decay
        device: Device to train on, auto-detected if None
        seed: Random seed for this variant
    """
    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    base_config = load_checkpoint_config(checkpoint_path)
    device = get_device(device)

    model_seed = seed + model_idx
    set_seed(model_seed)

    # Build model, optimizer, scheduler from config
    model = get_model_from_config(base_config).to(device)
    optimizer = build_optimizer(base_config["optimizer"], model)
    scheduler = build_scheduler(base_config.get("scheduler", {}), optimizer)

    # Load checkpoint weights and optimizer state
    start_epoch, _, _, global_step, _ = load_checkpoint(
        checkpoint_path, model, optimizer, scheduler, scaler=None
    )

    # Apply scales to optimizer hyperparameters
    hp_info = scale_optimizer(optimizer, lr_scale, momentum_scale, wd_scale)

    print(f"Model {model_idx} scales: LR={lr_scale:.4f}x, Momentum={momentum_scale:.4f}x, WD={wd_scale:.4f}x")
    print(f"  LR: {hp_info['original'][0]['lr']:.6f} -> {hp_info['scaled'][0]['lr']:.6f}")
    print(f"  Momentum: {hp_info['original'][0]['momentum']:.6f} -> {hp_info['scaled'][0]['momentum']:.6f}")
    print(f"  WD: {hp_info['original'][0]['weight_decay']:.6f} -> {hp_info['scaled'][0]['weight_decay']:.6f}")

    # Config for this finetune run
    additional_epochs = base_config["training"].get("epochs", 300) - start_epoch + 10
    ft_config = deepcopy(base_config)
    ft_config["training"]["epochs"] = start_epoch + additional_epochs
    ft_config["run_name"] = f"{checkpoint_name}_model_{model_idx}"
    ft_config["logging"]["checkpoint_interval"] = 1000  # Only save best and last

    # Data loaders
    train_loader, val_loader, _ = get_data_loaders(base_config, DATA_DIR)

    # wandb
    wandb.init(
        project=ft_config["project_name"],
        name=ft_config["run_name"],
        config={
            **ft_config,
            "finetune_from_checkpoint": checkpoint_path,
            "finetune_start_epoch": start_epoch,
            "finetune_model_idx": model_idx,
            "finetune_seed": model_seed,
            "hp_scales": {"lr": lr_scale, "momentum": momentum_scale, "wd": wd_scale},
            "hp_original": hp_info["original"][0],
            "hp_scaled": hp_info["scaled"][0],
        },
    )

    save_dir = os.path.join(MODEL_DIR, ft_config["project_name"], ft_config["run_name"])

    train_loop(
        model,
        train_loader,
        val_loader,
        nn.CrossEntropyLoss(label_smoothing=0.1),
        optimizer,
        scheduler,
        device,
        ft_config,
        save_dir=save_dir,
        start_epoch=start_epoch,
        best_val_acc=0.0,
        global_step=global_step,
    )

    wandb.finish()
    print(f"Completed model {model_idx}")


def run_variants(checkpoint_path, variants, device=None, seed=42, model_indices=None):
    """Run finetune for selected variants from a single checkpoint.

    Args:
        checkpoint_path: Path to baseline checkpoint
        variants: Dict mapping model_idx -> {lr_scale, momentum_scale, wd_scale}
        device: Device to train on
        seed: Base random seed
        model_indices: If set, only run these variant indices. Otherwise run all.
    """
    indices = model_indices if model_indices is not None else sorted(variants.keys())

    for idx in indices:
        v = variants[idx]
        print(f"\n{'='*80}")
        print(f"Variant {idx} from {os.path.basename(checkpoint_path)}")
        print(f"{'='*80}\n")

        finetune_single_model(
            checkpoint_path=checkpoint_path,
            model_idx=idx,
            lr_scale=v["lr_scale"],
            momentum_scale=v["momentum_scale"],
            wd_scale=v["wd_scale"],
            device=device,
            seed=seed,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finetune model variants from checkpoints with scaled optimizer hyperparameters"
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a single checkpoint to finetune from")
    parser.add_argument("--checkpoint-dir", type=str, help="Directory of checkpoints (processes all .pt epoch files)")
    parser.add_argument(
        "--variants-config", type=str, default="configs/resnet50_finetune.yaml",
        help="YAML file defining variant scales (default: configs/resnet50_finetune.yaml)",
    )
    parser.add_argument("--model-idx", type=int, default=None, help="Run only this variant index")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu/mps), auto-detected if omitted")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")

    # Alternative: generate random scales instead of using config
    parser.add_argument("--random-scales", action="store_true", help="Generate random scales instead of using config")
    parser.add_argument("--num-models", type=int, default=4, help="Number of random variants to generate (default: 4)")
    parser.add_argument("--lr-scale-range", type=float, default=1.0, help="Log2 scale range for LR (default: 1.0)")
    parser.add_argument("--momentum-scale-range", type=float, default=0.4, help="Log2 scale range for momentum (default: 0.4)")
    parser.add_argument("--wd-scale-range", type=float, default=0.4, help="Log2 scale range for weight decay (default: 0.4)")

    args = parser.parse_args()

    # Load or generate variants
    if args.random_scales:
        scale_list = generate_random_scales(
            args.num_models,
            lr_scale_range=args.lr_scale_range,
            momentum_scale_range=args.momentum_scale_range,
            wd_scale_range=args.wd_scale_range,
            seed=args.seed,
        )
        variants = {i + 1: s for i, s in enumerate(scale_list)}
        print(f"Generated {len(variants)} random scale variants:")
        for idx, v in variants.items():
            print(f"  Model {idx}: LR={v['lr_scale']:.4f}, Mom={v['momentum_scale']:.4f}, WD={v['wd_scale']:.4f}")
    else:
        variants_cfg = load_config(args.variants_config)
        variants = {int(k): v for k, v in variants_cfg["variants"].items()}

    # Select which model indices to run
    model_indices = [args.model_idx] if args.model_idx is not None else None

    # Collect checkpoints
    if args.checkpoint_dir is not None:
        checkpoint_files = sorted(
            [f for f in os.listdir(args.checkpoint_dir) if f.endswith(".pt") and "epoch" in f],
            key=lambda f: int(f.split("epoch_")[-1].split(".")[0]) if "epoch_" in f else -1,
        )

        for ckpt_file in checkpoint_files:
            ckpt_path = os.path.join(args.checkpoint_dir, ckpt_file)
            run_variants(ckpt_path, variants, device=args.device, seed=args.seed, model_indices=model_indices)

        print(f"\nProcessed {len(checkpoint_files)} checkpoints from {args.checkpoint_dir}")

    elif args.checkpoint is not None:
        run_variants(args.checkpoint, variants, device=args.device, seed=args.seed, model_indices=model_indices)

    else:
        parser.error("Either --checkpoint or --checkpoint-dir must be provided.")
