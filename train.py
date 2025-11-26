import argparse
import os
import random
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from dotenv import load_dotenv
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

import wandb
from salty.datasets import get_cifar100_class_names, get_cifar100_loaders
from salty.models import get_resnet50_model
from salty.utils import (
    denormalize_cifar100,
    load_checkpoint,
    load_config,
    normalize_cifar100,
    save_checkpoint,
    show_batch_with_labels,
)

load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR", "./models")
DATA_DIR = os.getenv("DATA_DIR", "./data")


def train_loop(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    scheduler,
    device,
    cfg,
    start_epoch=0,
    best_val_acc=None,
    scaler=None,
    global_step=0,
):
    num_epochs = cfg["training"]["epochs"]
    val_interval = cfg["logging"]["val_interval"]
    ckpt_interval = cfg["logging"]["checkpoint_interval"]
    save_dir = os.path.join(MODEL_DIR, cfg.get("project_name"), cfg.get("run_name"))

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(start_epoch, num_epochs):
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
    """Build optimizer from config"""

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
    """Build learning rate scheduler from config"""
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on CIFAR-100")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to the config file")
    parser.add_argument("--device", type=str, default=None, help="Device to use for training")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume training from")
    args = parser.parse_args()
    config = load_config(args.config)

    # Handle resume from checkpoint
    if args.resume is not None:
        config.setdefault("resume", {})
        config["resume"]["checkpoint_path"] = args.resume

    # Build model
    model_cfg = config["model"]
    if model_cfg["name"] == "resnet50":
        model = get_resnet50_model(num_classes=config["data"]["num_classes"])
    else:
        raise NotImplementedError(f"Model {model_cfg['name']} not implemented.")

    # Build optimizer
    optimizer = build_optimizer(config["optimizer"], model)

    # Build scheduler
    scheduler = build_scheduler(config.get("scheduler", {}), optimizer)

    # Get data loaders and class names
    train_loader, val_loader, test_loader = get_cifar100_loaders(
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        data_dir=DATA_DIR,
        val_ratio=config["data"].get("validation_split"),
        seed=config["data"]["split_seed"],
        augment=config["data"].get("augment", True),
    )

    device = get_device(args.device)  # Get device override or auto-detect from system
    model = model.to(device)

    train_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ------------------------
    # Resume logic
    # ------------------------
    start_epoch = 0
    best_val_acc = None
    start_global_step = 0
    wandb_run_id = None

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
            f"start_step={start_global_step}, "
            f"wandb_run_id={wandb_run_id}"
        )

    # ------------------------
    # wandb init
    # ------------------------
    wandb.init(
        project=config["project_name"],
        name=config.get("run_name", None),
        config=deepcopy(config),
    )

    # ------------------------
    # Training loop
    # ------------------------

    train_loop(
        model,
        train_loader,
        val_loader,
        train_loss_fn,
        optimizer,
        scheduler,
        device,
        config,
        start_epoch=start_epoch,
        best_val_acc=best_val_acc,
        global_step=start_global_step,
    )

    # images, labels = next(iter(train_loader))
    # print(f"Sample batch images shape: {images.shape}")
    # print(f"Sample batch labels shape: {labels.shape}")

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = model.to(device)

    # images = images.to(device)
    # labels = labels.to(device)
    # # Run a forward pass with one batch
    # outputs = model(images)
    # print(f"Model outputs shape: {outputs.shape}")

    # images_denorm = denormalize_cifar100(images)
    # show_batch_with_labels(images_denorm, labels, label_names)
