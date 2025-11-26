"""
An evaluation script for pretrained models on both CIFAR-100 test and CIFAR-100C
"""

import os
import re

import pandas as pd
import torch
from dotenv import load_dotenv
from tqdm import tqdm

from salty.datasets import (
    get_cifar100_loaders,
    get_cifar100c_loaders_by_corruption,
)
from salty.models import get_resnet50_model
from salty.utils import (
    load_checkpoint,
)

load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR", "./models")
SOUP_DIR = os.getenv("SOUP_DIR", "./models/soup_models")
soup_pattern = re.compile(r"epoch_(\d+)_model_(\d+)\.pt")


def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def load_soup_model(path, device):
    model = get_resnet50_model(num_classes=100)
    start_epoch, best_val_acc, loaded_cfg, global_step, wandb_run_id = load_checkpoint(path, model)
    model.to(device)
    return model, start_epoch


def evaluate_all_soup_models(
    device: str = "cuda",
    batch_size: int = 128,
    severities=(1, 2, 3, 4, 5),
    max_models: int | None = None,
):
    # --- CIFAR-100 test loader ---
    _, _, test_loader = get_cifar100_loaders(batch_size=batch_size)

    results = []
    num_models = 0  # how many models we've actually evaluated

    for filename in sorted(os.listdir(SOUP_DIR)):
        match = soup_pattern.match(filename)
        if not match:
            continue

        epoch, model_n = map(int, match.groups())
        model_path = os.path.join(SOUP_DIR, filename)

        print(f"\n=== Loading {filename} (epoch={epoch}, model={model_n}) ===")
        model, best_epoch = load_soup_model(model_path, device)

        # Evaluate once on clean test set (independent of severity)
        acc_clean = evaluate_model(model, test_loader, device)

        # Now loop over all severities for CIFAR-100C
        for severity in severities:
            print(f"\n  -> Evaluating CIFAR-100C, severity={severity}")

            corruption_loaders = get_cifar100c_loaders_by_corruption(
                batch_size=batch_size,
                severity=severity,
            )

            corruption_accs = {}
            for corr_name, loader in corruption_loaders.items():
                acc_corr = evaluate_model(model, loader, device)
                corruption_accs[corr_name] = acc_corr

            row = {
                "branch_epoch": epoch,
                "termination_epoch": best_epoch,
                "model_id": model_n,
                "model_path": filename,
                "severity": severity,
                "clean_accuracy": acc_clean,
            }
            # add per-corruption accuracies
            row.update({f"corr_{k}": v for k, v in corruption_accs.items()})

            results.append(row)

        num_models += 1
        if max_models is not None and num_models >= max_models:
            print(f"\nReached max_models={max_models}, stopping early.")
            break

    df = pd.DataFrame(results)
    return df


def reduce_to_severity_columns(df):
    """
    Collapse the long-form df (multiple rows per model) into
    a single wide row per model:

    clean_accuracy, severity_1, ..., severity_5
    """
    corr_cols = [c for c in df.columns if c.startswith("corr_")]

    # Step 1 — Compute mean accuracy across all corruption types per severity
    df["severity_mean"] = df[corr_cols].mean(axis=1)

    # Step 2 — Pivot to get one column per severity
    wide = df.pivot_table(
        index=["model_id", "branch_epoch", "termination_epoch", "model_path", "clean_accuracy"],
        columns="severity",
        values="severity_mean",
    ).reset_index()

    # Rename severity columns → severity_1, severity_2, ...
    wide.columns = ["model_id", "branch_epoch", "termination_epoch", "model_path", "clean_accuracy"] + [
        f"severity_{int(c)}" for c in wide.columns[5:]
    ]

    return wide


if __name__ == "__main__":
    severities = [1, 2, 3, 4, 5]
    df = evaluate_all_soup_models(device="cuda:2", batch_size=1024, severities=severities, max_models=None)
    wide = reduce_to_severity_columns(df)

    # Paths
    parquet_path = os.path.join(SOUP_DIR, "soup_model_results.parquet")
    csv_path = os.path.join(SOUP_DIR, "soup_model_results.csv")

    # Best format
    wide.to_parquet(parquet_path, index=False)
    print(f"Saved Parquet: {parquet_path}")

    # Optional human-readable version
    wide.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    print("\n=== Final Results ===")
    print(wide)
