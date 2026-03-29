"""Shared training utilities: loops, optimizer/scheduler builders, device selection."""

import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

import wandb


def train_loop(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    scheduler,
    device,
    cfg,
    save_dir,
    start_epoch=0,
    best_val_acc=None,
    scaler=None,
    global_step=0,
):
    """Standard training loop for baseline model training (no EMA/SWA)."""
    from salty.utils import save_checkpoint

    num_epochs = cfg["training"]["epochs"]
    val_interval = cfg["logging"]["val_interval"]
    ckpt_interval = cfg["logging"]["checkpoint_interval"]

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        # train for one epoch
        train_loss, train_acc, global_step = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            epoch,
            cfg,
            scaler=scaler,
            global_step=global_step,
        )

        # validate every val_interval epochs
        if (epoch + 1) % val_interval == 0:
            val_loss, val_acc = validate(
                model,
                val_loader,
                loss_fn,
                device,
                epoch,
            )
        else:
            val_loss, val_acc = None, None

        # Step the scheduler each epoch
        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch} completed in {epoch_time:.1f}s")

        # Epoch-level logging to wandb
        if wandb.run is not None:
            wandb.log(
                {
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                    "global_step": global_step,
                    "timing/epoch_seconds": epoch_time,
                },
                step=global_step,
            )

        # Save last checkpoint
        last_model_path = os.path.join(save_dir, "last_model.pt")
        save_checkpoint(
            last_model_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            cfg,
            best_val_acc=best_val_acc,
            global_step=global_step,
        )

        # Epoch checkpoint every N epochs
        if (epoch + 1) % ckpt_interval == 0:
            ckpt_path = os.path.join(save_dir, f"{cfg['run_name']}-epoch_{epoch+1}.pt")
            save_checkpoint(
                ckpt_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                cfg,
                best_val_acc=best_val_acc,
                global_step=global_step,
            )

        # Save best model
        if best_val_acc is None or val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(save_dir, "best_model.pt")
            save_checkpoint(
                best_model_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                cfg,
                best_val_acc=best_val_acc,
                global_step=global_step,
            )
            print(f"New best model saved with val_acc: {best_val_acc:.2f}%")


def train_one_epoch(
    model,
    train_loader,
    loss_fn,
    optimizer,
    device,
    epoch,
    cfg,
    scaler=None,
    global_step=0,
    ema_model=None,
):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    log_interval = cfg["logging"]["log_interval"]
    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False)

    for batch_idx, (img_batch, label_batch) in enumerate(pbar):
        img_batch = img_batch.to(device, non_blocking=True)
        label_batch = label_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast():
                outputs = model(img_batch)
                loss = loss_fn(outputs, label_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(img_batch)
            loss = loss_fn(outputs, label_batch)
            loss.backward()
            optimizer.step()

        # Update EMA model after each optimizer step
        if ema_model is not None:
            ema_model.update_parameters(model)

        _, preds = outputs.max(1)
        running_loss += loss.item() * img_batch.size(0)
        running_correct += preds.eq(label_batch).sum().item()
        running_total += img_batch.size(0)

        avg_loss = running_loss / running_total
        avg_acc = 100.0 * running_correct / running_total

        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc:.2f}%"})

        # Step-level logging to wandb
        global_step += 1
        if wandb.run is not None and (batch_idx % log_interval == 0):
            wandb.log(
                {
                    "train/loss": avg_loss,
                    "train/acc": avg_acc,
                    "train/step": global_step,
                    "train/epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                },
                step=global_step,
            )

    epoch_loss = running_loss / running_total
    epoch_acc = 100.0 * running_correct / running_total

    return epoch_loss, epoch_acc, global_step


@torch.no_grad()
def validate(model, val_loader, loss_fn, device, epoch):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    pbar = tqdm(val_loader, desc=f"Val   Epoch {epoch}", leave=False)

    for batch_idx, (img_batch, label_batch) in enumerate(pbar):
        img_batch = img_batch.to(device, non_blocking=True)
        label_batch = label_batch.to(device, non_blocking=True)

        outputs = model(img_batch)
        loss = loss_fn(outputs, label_batch)

        _, preds = outputs.max(1)
        running_loss += loss.item() * img_batch.size(0)
        running_correct += preds.eq(label_batch).sum().item()
        running_total += img_batch.size(0)

        avg_loss = running_loss / running_total
        avg_acc = 100.0 * running_correct / running_total

        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc:.2f}%"})

    epoch_loss = running_loss / running_total
    epoch_acc = 100.0 * running_correct / running_total

    return epoch_loss, epoch_acc


def build_optimizer(opt_cfg, model):
    """Build optimizer from config."""
    opt_name = opt_cfg["name"].lower()

    if opt_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=opt_cfg.get("lr", 0.1),
            momentum=opt_cfg.get("momentum", 0.9),
            weight_decay=opt_cfg.get("weight_decay", 5e-4),
            nesterov=opt_cfg.get("nesterov", True),
        )
    elif opt_name == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=opt_cfg.get("lr", 1e-3),
            weight_decay=opt_cfg.get("weight_decay", 0.0),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")
    return optimizer


def build_scheduler(sch_cfg, optimizer):
    """Build learning rate scheduler from config."""
    name = sch_cfg["name"].lower()

    if name == "multisteplr":
        scheduler = MultiStepLR(
            optimizer,
            milestones=sch_cfg.get("milestones", [100, 150]),
            gamma=sch_cfg.get("gamma"),
        )
    elif name == "steplr":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sch_cfg.get("step_size", 30),
            gamma=sch_cfg.get("gamma"),
        )
    elif name == "cosineannealinglr":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sch_cfg.get("T_max", 280),
            eta_min=sch_cfg.get("eta_min", 0),
        )
    elif name == "none" or name is None:
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {name}")
    return scheduler


def get_device(config_device=None):
    if config_device is not None:
        device = config_device
    else:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Using device: {device}")
    return device


def scale_optimizer(optimizer, lr_scale=1.0, momentum_scale=1.0, wd_scale=1.0):
    """Apply multiplicative scales to optimizer hyperparameters.

    Args:
        optimizer: PyTorch optimizer instance
        lr_scale: Multiplicative factor for learning rate
        momentum_scale: Multiplicative factor for momentum
        wd_scale: Multiplicative factor for weight decay

    Returns:
        Dict with original and scaled hyperparameters
    """
    original = {}
    scaled = {}

    for group_idx, param_group in enumerate(optimizer.param_groups):
        original[group_idx] = {
            "lr": param_group["lr"],
            "momentum": param_group.get("momentum", None),
            "weight_decay": param_group.get("weight_decay", None),
        }

        param_group["lr"] *= lr_scale
        if "momentum" in param_group:
            param_group["momentum"] *= momentum_scale
        if "weight_decay" in param_group:
            param_group["weight_decay"] *= wd_scale

        scaled[group_idx] = {
            "lr": param_group["lr"],
            "momentum": param_group.get("momentum", None),
            "weight_decay": param_group.get("weight_decay", None),
        }

    return {"original": original, "scaled": scaled}


def generate_random_scales(num_variants, lr_scale_range=1.0, momentum_scale_range=0.4,
                           wd_scale_range=0.4, seed=42):
    """Generate random optimizer scales for multiple model variants.

    Each variant gets an independent seed (base seed + variant index) so that
    scales are reproducible per-variant regardless of how many variants are
    generated. Uses log-uniform sampling: log2(scale) is drawn uniformly from
    [0.2*range, range] with a random sign, so scale and 1/scale are equally
    likely.

    Args:
        num_variants: Number of variant scale sets to generate
        lr_scale_range: Max log2 scale magnitude for learning rate (default: 1.0)
        momentum_scale_range: Max log2 scale magnitude for momentum (default: 0.4)
        wd_scale_range: Max log2 scale magnitude for weight decay (default: 0.4)
        seed: Base random seed; variant i uses seed + i

    Returns:
        List of dicts with lr_scale, momentum_scale, wd_scale
    """
    variants = []

    for i in range(num_variants):
        rng = np.random.RandomState(seed + i)

        def _sample(scale_range):
            log_scale = rng.uniform(0.2 * scale_range, scale_range)
            sign = rng.choice([-1, 1])
            return 2 ** (log_scale * sign)

        variants.append({
            "lr_scale": _sample(lr_scale_range),
            "momentum_scale": _sample(momentum_scale_range),
            "wd_scale": _sample(wd_scale_range),
        })

    return variants
