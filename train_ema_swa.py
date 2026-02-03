"""
Batch training script for running multiple experiments with EMA and SWA.
Each run trains a baseline model while tracking EMA and SWA variants,
saving the best validation checkpoint for each.
"""

import argparse
import gc
import os
import random

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from dotenv import load_dotenv
from torch.optim.swa_utils import AveragedModel, SWALR, get_ema_multi_avg_fn, update_bn
import torch.multiprocessing as mp

import wandb
from salty.datasets import get_cifar100_loaders
from salty.models import get_resnet50_model
from salty.utils import load_config, save_checkpoint
from train import build_optimizer, build_scheduler, validate, train_one_epoch

load_dotenv()
MODEL_DIR = os.getenv("MODEL_DIR", "./models")
DATA_DIR = os.getenv("DATA_DIR", "./data")


def run_single_experiment(run_index, config, args):
    """Run a single training experiment with baseline, EMA, and SWA tracking."""

    seed = config["data"]["split_seed"] + run_index
    run_id = f"run_{run_index:02d}_seed_{seed}"
    save_dir = os.path.join(MODEL_DIR, config["project_name"], "ema_swa_experiments", run_id)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Starting {run_id} ({run_index + 1}/{args.runs})")
    print(f"{'='*60}")

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Data loaders
    train_loader, val_loader, test_loader = get_cifar100_loaders(
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        data_dir=DATA_DIR,
        val_ratio=config["data"].get("validation_split", 0.05),
        seed=seed,
        augment=config["data"].get("augment", True),
    )

    # Build model, optimizer, scheduler
    model = get_resnet50_model(num_classes=config["data"]["num_classes"]).to(args.device)
    optimizer = build_optimizer(config["optimizer"], model)
    scheduler = build_scheduler(config.get("scheduler", {}), optimizer)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    # EMA model - updates every step
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay))
    print(f"EMA enabled with decay={args.ema_decay}")

    # SWA model - starts late in training
    num_epochs = config["training"]["epochs"]
    swa_start = args.swa_start if args.swa_start is not None else int(0.80 * num_epochs)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr, anneal_strategy="linear", anneal_epochs=5)
    print(f"SWA enabled, starting at epoch {swa_start} with lr={args.swa_lr}")

    # Initialize wandb
    wandb_config = deepcopy(config)
    wandb_config["seed"] = seed
    wandb_config["ema"] = {"enabled": True, "decay": args.ema_decay}
    wandb_config["swa"] = {"enabled": True, "start": swa_start, "lr": args.swa_lr}

    wandb.init(
        project=config["project_name"],
        name=run_id,
        group="ema_swa_batch",
        config=wandb_config,
        reinit=True,
    )

    # Training state
    best_baseline_acc = 0.0
    best_ema_acc = 0.0
    best_swa_acc = 0.0
    global_step = 0
    val_interval = config["logging"]["val_interval"]

    # Training loop
    for epoch in range(num_epochs):
        in_swa_phase = epoch >= swa_start

        # Train one epoch (EMA updates happen inside train_one_epoch)
        train_loss, train_acc, global_step = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            args.device,
            epoch,
            config,
            scaler=None,
            global_step=global_step,
            ema_model=ema_model,
        )

        # Update SWA model at end of epoch (only in SWA phase)
        if in_swa_phase:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            if scheduler is not None:
                scheduler.step()

        # Validation
        if (epoch + 1) % val_interval == 0:
            # Validate baseline model
            val_loss, val_acc = validate(model, val_loader, loss_fn, args.device, epoch)

            # Log to wandb
            wandb.log({
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
            }, step=global_step)

            # Save best baseline
            if val_acc > best_baseline_acc:
                best_baseline_acc = val_acc
                torch.save({
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "config": config,
                }, os.path.join(save_dir, "baseline_best.pt"))
                print(f"  New best baseline: {val_acc:.2f}%")

            # Validate EMA model periodically
            if (epoch + 1) % (val_interval * 5) == 0 or epoch == num_epochs - 1:
                ema_val_loss, ema_val_acc = validate(ema_model, val_loader, loss_fn, args.device, f"{epoch}_ema")
                wandb.log({"val/ema_acc": ema_val_acc, "val/ema_loss": ema_val_loss}, step=global_step)

                if ema_val_acc > best_ema_acc:
                    best_ema_acc = ema_val_acc
                    # Save intermediate EMA checkpoint
                    torch.save({
                        "model_state": ema_model.module.state_dict(),
                        "epoch": epoch,
                        "val_acc": ema_val_acc,
                        "config": config,
                    }, os.path.join(save_dir, "ema_best.pt"))
                    print(f"  New best EMA: {ema_val_acc:.2f}%")

    # Finalize: Update BN stats for EMA and SWA
    print("\nFinalizing EMA model (updating BN stats)...")
    update_bn(train_loader, ema_model, device=args.device)

    print("Finalizing SWA model (updating BN stats)...")
    update_bn(train_loader, swa_model, device=args.device)

    # Final validation of EMA
    ema_final_loss, ema_final_acc = validate(ema_model, val_loader, loss_fn, args.device, "EMA_final")
    print(f"EMA Final Accuracy: {ema_final_acc:.2f}%")

    # Final validation of SWA
    swa_final_loss, swa_final_acc = validate(swa_model, val_loader, loss_fn, args.device, "SWA_final")
    print(f"SWA Final Accuracy: {swa_final_acc:.2f}%")

    # Log final metrics
    wandb.log({
        "final/ema_acc": ema_final_acc,
        "final/ema_loss": ema_final_loss,
        "final/swa_acc": swa_final_acc,
        "final/swa_loss": swa_final_loss,
        "final/baseline_best_acc": best_baseline_acc,
    })

    # Save final models
    torch.save({
        "model_state": ema_model.module.state_dict(),
        "val_acc": ema_final_acc,
        "config": config,
    }, os.path.join(save_dir, "ema_final.pt"))

    torch.save({
        "model_state": swa_model.module.state_dict(),
        "val_acc": swa_final_acc,
        "config": config,
    }, os.path.join(save_dir, "swa_final.pt"))

    # Save last baseline model
    torch.save({
        "model_state": model.state_dict(),
        "epoch": num_epochs - 1,
        "config": config,
    }, os.path.join(save_dir, "baseline_last.pt"))

    print(f"\nRun {run_id} complete!")
    print(f"  Best Baseline: {best_baseline_acc:.2f}%")
    print(f"  Best EMA: {best_ema_acc:.2f}%")
    print(f"  Final EMA: {ema_final_acc:.2f}%")
    print(f"  Final SWA: {swa_final_acc:.2f}%")
    print(f"  Models saved to: {save_dir}")

    wandb.finish()

    # Cleanup
    del model, ema_model, swa_model, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "run_id": run_id,
        "seed": seed,
        "best_baseline_acc": best_baseline_acc,
        "best_ema_acc": best_ema_acc,
        "final_ema_acc": ema_final_acc,
        "final_swa_acc": swa_final_acc,
    }

def gpu_worker(gpu_id, queue, config, base_args):
    """
    Worker process that pulls tasks from the queue and runs them on a specific GPU.
    """
    print(f"Worker started on GPU {gpu_id}")
    
    # Process-specific device setting
    worker_args = deepcopy(base_args)
    worker_args.device = f"cuda:{gpu_id}"
    
    while True:
        try:
            # Try to get a run index from the queue (non-blocking)
            run_index = queue.get_nowait()
        except Exception:
            # Queue is empty, worker is done
            break
            
        try:
            # Run the experiment
            run_single_experiment(run_index, config, worker_args)
        except Exception as e:
            print(f"!!! Error in Run {run_index} on GPU {gpu_id}: {e}")
            import traceback
            traceback.print_exc()

    print(f"Worker on GPU {gpu_id} finished all tasks.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch training with EMA and SWA")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--runs", type=int, default=20, help="Total number of runs")
    # Use parallel dispatch for device
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--swa-start", type=int, default=None)
    parser.add_argument("--swa-lr", type=float, default=0.05)
    parser.add_argument("--start-run", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs found! Falling back to single CPU process.")
        num_gpus = 0 # Handle CPU case separately if needed, or just set device='cpu'
    else:
        print(f"Detected {num_gpus} GPUs. Starting parallel pool...")

    #use 'spawn' context which is required for CUDA multiprocessing
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    
    # Fill the queue with run indices
    for i in range(args.start_run, args.runs):
        queue.put(i)

    # Launch Workers (One per GPU)
    processes = []
    
    if num_gpus > 0:
        for gpu_id in range(num_gpus):
            p = ctx.Process(
                target=gpu_worker, 
                args=(gpu_id, queue, config, args)
            )
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
    else:
        worker_args = deepcopy(args)
        worker_args.device = "cpu"
        while not queue.empty():
            run_index = queue.get()
            run_single_experiment(run_index, config, worker_args)
