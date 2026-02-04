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
    """Run a single training experiment with branched Baseline/EMA and SWA paths."""

    seed = config["data"]["split_seed"] + run_index
    run_id = f"run_{run_index:02d}_seed_{seed}"
    save_dir = os.path.join(MODEL_DIR, config["project_name"], "ema_swa_experiments", run_id)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Starting {run_id} ({run_index + 1}/{args.runs})")
    print(f"{'='*60}")

    # Set random seed
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

    # ---------------------------------------------------------
    # 1. SETUP SHARED INITIAL STATE
    # ---------------------------------------------------------
    model = get_resnet50_model(num_classes=config["data"]["num_classes"]).to(args.device)
    optimizer = build_optimizer(config["optimizer"], model)
    scheduler = build_scheduler(config.get("scheduler", {}), optimizer)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay))
    
    # SWA settings
    num_epochs = config["training"]["epochs"]
    swa_start = args.swa_start if args.swa_start is not None else int(0.75 * num_epochs)
    
    # WandB Init
    wandb_config = deepcopy(config)
    wandb_config["seed"] = seed
    wandb_config["ema"] = {"enabled": True, "decay": args.ema_decay}
    wandb_config["swa"] = {"enabled": True, "start": swa_start, "lr": args.swa_lr}

    wandb.init(
        project=config["project_name"],
        name=run_id,
        group="ema_swa_branching",
        config=wandb_config,
        reinit=True,
    )

    # State tracking
    best_baseline_acc = 0.0
    best_ema_acc = 0.0
    global_step = 0
    val_interval = config["logging"]["val_interval"]
    branch_point_path = os.path.join(save_dir, "ckpt_branch_point.pt")

    # ---------------------------------------------------------
    # 2. PHASE 1: MAIN TRUNK (Baseline + EMA)
    # ---------------------------------------------------------
    print(f"Phase 1: Training Main Trunk (Epochs 0-{num_epochs})")
    
    for epoch in range(num_epochs):
        
        # --- BRANCHING LOGIC ---
        # We save the state right before the SWA start epoch.
        if epoch == swa_start:
            print(f" >> Reached SWA fork point (Epoch {epoch}). Saving branch checkpoint...")
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
            }, branch_point_path)
        # -----------------------

        # Train one epoch (Baseline updates weights, EMA follows)
        train_loss, train_acc, global_step = train_one_epoch(
            model, train_loader, loss_fn, optimizer, args.device,
            epoch, config, scaler=None, global_step=global_step,
            ema_model=ema_model 
        )

        if scheduler is not None:
            scheduler.step()

        # Validation (Baseline)
        if (epoch + 1) % val_interval == 0:
            val_loss, val_acc = validate(model, val_loader, loss_fn, args.device, epoch)
            
            wandb.log({
                "train/loss": train_loss, "train/acc": train_acc,
                "val/loss": val_loss, "val/acc": val_acc,
                "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
            }, step=global_step)

            if val_acc > best_baseline_acc:
                best_baseline_acc = val_acc
                torch.save({
                    "model_state": model.state_dict(),
                    "val_acc": val_acc,
                    "config": config,
                }, os.path.join(save_dir, "baseline_best.pt"))

            # Validation (EMA) - Check periodically
            if (epoch + 1) % (val_interval * 5) == 0 or epoch == num_epochs - 1:
                ema_loss, ema_acc = validate(ema_model, val_loader, loss_fn, args.device, f"{epoch}_ema")
                wandb.log({"val/ema_acc": ema_acc, "val/ema_loss": ema_loss}, step=global_step)
                
                if ema_acc > best_ema_acc:
                    best_ema_acc = ema_acc
                    torch.save({
                        "model_state": ema_model.module.state_dict(),
                        "val_acc": ema_acc,
                        "config": config,
                    }, os.path.join(save_dir, "ema_best.pt"))

    # Finalize EMA (Main Trunk Complete)
    print("\nFinalizing EMA model (updating BN stats)...")
    update_bn(train_loader, ema_model, device=args.device)
    ema_final_loss, ema_final_acc = validate(ema_model, val_loader, loss_fn, args.device, "EMA_final")
    
    # Save Final Baseline
    torch.save({
        "model_state": model.state_dict(),
        "epoch": num_epochs - 1,
        "config": config,
    }, os.path.join(save_dir, "baseline_last.pt"))

    torch.save({
        "model_state": ema_model.module.state_dict(),
        "val_acc": ema_final_acc,
    }, os.path.join(save_dir, "ema_final.pt"))

    # ---------------------------------------------------------
    # 3. PHASE 2: SWA BRANCH
    # ---------------------------------------------------------
    print(f"\nPhase 2: SWA Branch (Retraining from Epoch {swa_start}-{num_epochs})")
    
    # RESET: Load the model/optimizer state from the branch point
    if os.path.exists(branch_point_path):
        checkpoint = torch.load(branch_point_path, map_location=args.device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        print(f" >> Loaded branch point from {branch_point_path}")
    else:
        print(" !! Warning: No branch point found. SWA will run on the fully trained model (Suboptimal).")

    # Initialize SWA specific components
    swa_model = AveragedModel(model)
    # Important: Create a FRESH scheduler for SWA, ignoring the old scheduler state
    swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr, anneal_strategy="linear", anneal_epochs=5)
    
    print(f"SWA Strategy: Starting at epoch {swa_start}, LR={args.swa_lr}")

    # SWA Loop
    # Note: We do not update global_step here to keep wandb charts clean, 
    # or we log to a separate section.
    for epoch in range(swa_start, num_epochs):
        
        # Train one epoch (Model weights diverge here from Baseline)
        # We pass ema_model=None because EMA is already done in Phase 1
        train_loss, train_acc, _ = train_one_epoch(
            model, train_loader, loss_fn, optimizer, args.device,
            epoch, config, scaler=None, global_step=0, 
            ema_model=None 
        )
        
        # Update SWA parameters and Scheduler
        swa_model.update_parameters(model)
        swa_scheduler.step()
        
        wandb.log({
            "swa_branch/train_loss": train_loss,
            "swa_branch/train_acc": train_acc,
            "swa_branch/lr": optimizer.param_groups[0]["lr"],
            "epoch": epoch
        })

    # Finalize SWA
    print("Finalizing SWA model (updating BN stats)...")
    update_bn(train_loader, swa_model, device=args.device)
    swa_final_loss, swa_final_acc = validate(swa_model, val_loader, loss_fn, args.device, "SWA_final")
    print(f"SWA Final Accuracy: {swa_final_acc:.2f}%")

    # ---------------------------------------------------------
    # 4. WRAP UP & CLEANUP
    # ---------------------------------------------------------
    
    # Log final comparison
    wandb.log({
        "final/ema_acc": ema_final_acc,
        "final/swa_acc": swa_final_acc,
        "final/baseline_best_acc": best_baseline_acc,
    })

    # Save Models
    torch.save({
        "model_state": swa_model.module.state_dict(),
        "val_acc": swa_final_acc,
    }, os.path.join(save_dir, "swa_final.pt"))

    # Cleanup temp checkpoint
    if os.path.exists(branch_point_path):
        os.remove(branch_point_path)

    print(f"\nRun {run_id} complete!")
    print(f"  Best Baseline: {best_baseline_acc:.2f}%")
    print(f"  Final EMA:     {ema_final_acc:.2f}%")
    print(f"  Final SWA:     {swa_final_acc:.2f}%")
    
    wandb.finish()

    # Memory cleanup
    del model, ema_model, swa_model, optimizer, scheduler, swa_scheduler
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "run_id": run_id,
        "seed": seed,
        "best_baseline_acc": best_baseline_acc,
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
    parser.add_argument("--swa-start", type=int, default=220)
    parser.add_argument("--swa-lr", type=float, default=0.03)
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
