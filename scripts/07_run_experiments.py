"""
Run various experiments on finetuned models stored in SOUP_DIR:
- Experiment 2: Permutation ablation
- Experiment 3: Predicting soupability (similarity metrics)
- Experiment 5: Deviation metrics from shared ancestor

Usage:
    python scripts/07_run_experiments.py
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from salty.datasets import get_cifar100_loaders
from salty.evaluation import canonical_key, evaluate_model
from salty.model_manager import ModelManager
from salty.models import get_model
from salty.permute_model import permute_models
from salty.similarity_metrics_from_logits import cka_similarity, logit_mse_kl
from salty.similarity_metrics_from_models import cosine_similarity, l2_distance
from salty.utils import load_checkpoint, soup_models

MODEL_DIR = os.getenv("MODEL_DIR", "./models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOUP_DIR = os.getenv("SOUP_DIR", "./models/soup_models")


# Experiment 2 - Permutation Ablation
def run_permutation_ablation_experiment(
    mm: ModelManager,
    model_pairs: List[Tuple[int, int]],
    output_csv: str = "permutation_ablation_results.csv",
) -> None:
    """
    Clean acc and loss of all pairs of models after being permuted.
    """
    csv_file = Path(output_csv)
    COLUMNS = ["key_a", "key_b", "clean_accuracy_permuted", "clean_loss_permuted"]

    existing_canonical_keys = set()
    if not csv_file.exists():
        pd.DataFrame(columns=COLUMNS).to_csv(csv_file, index=False)
        print(f"Created new results file: {output_csv}")
    else:
        try:
            df_existing = pd.read_csv(csv_file)
            existing_canonical_keys = set(
                canonical_key(row["key_a"], row["key_b"]) for _, row in df_existing.iterrows()
            )
            print(f"Resuming experiment. Found {len(existing_canonical_keys)} previously computed pairs.")

        except pd.errors.EmptyDataError:
            print("Existing file is empty. Starting fresh.")
        except Exception as e:
            print(f"Error reading existing CSV: {e}. Starting fresh.")

    # Filter out model pairs that have already been computed
    pairs_to_process = [
        pair
        for pair in model_pairs
        if canonical_key(mm.get_key(pair[0]), mm.get_key(pair[1])) not in existing_canonical_keys
    ]
    print(f"Total pairs to process: {len(pairs_to_process)}")

    dataloader_cifar100_test = get_cifar100_loaders(batch_size=128, num_workers=4)[2]

    for idx_a, idx_b in pairs_to_process:
        model_a = mm[idx_a]
        model_b = mm[idx_b]
        key_a = model_a.key
        key_b = model_b.key
        print(f"Processing pair: {key_a} + {key_b}")

        # Permute and Soup
        permuted_model_b = permute_models(model_a, model_b)
        model_soup = soup_models(model_a, permuted_model_b, alpha=0.5)

        acc, loss = evaluate_model(
            model=model_soup,
            dataloader=dataloader_cifar100_test,
            device=mm.device,
        )

        if key_a > key_b:
            key_a, key_b = key_b, key_a

        new_row = {
            "key_a": key_a,
            "key_b": key_b,
            "clean_accuracy_permuted": acc,
            "clean_loss_permuted": loss,
        }
        df_new = pd.DataFrame([new_row])
        df_new.to_csv(csv_file, mode="a", header=False, index=False)

        # Clean up models and free GPU memory
        del model_a, model_b, permuted_model_b, model_soup
        if mm.device.type == "cuda":
            torch.cuda.empty_cache()

        print(f"  Saved (Acc: {acc:.2f}, Loss: {loss:.4f})")

    print("\nExperiment run completed.")
    df_final = pd.read_csv(csv_file)
    df_final.to_parquet(csv_file.with_suffix(".parquet"), engine="fastparquet")
    print(f"Final results saved to {csv_file} and {csv_file.with_suffix('.parquet')}")


# Experiment 3 - Predicting Soupability


def record_logits_features(mm: ModelManager, dataloader: DataLoader) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    """
    Record logits and penultimate activations for all models in the ModelManager on the given dataloader by key.
    """

    logits = {}
    penultimate_activations = {}
    for model in mm:
        model_logits = []
        model_penultimate = []
        for inputs, _ in dataloader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            with torch.no_grad():
                penultimate = model.forward_features(inputs)
                outputs = model.fc(penultimate)
            model_logits.append(outputs.cpu())
            model_penultimate.append(penultimate.cpu())
        logits[model.key] = torch.cat(model_logits, dim=0)
        penultimate_activations[model.key] = torch.cat(model_penultimate, dim=0)
    return logits, penultimate_activations


def pickle_logits_features(
    logits: Dict[str, Tensor], penultimate_activations: Dict[str, Tensor], filename: str
) -> None:
    with open(filename, "wb") as f:
        pickle.dump({"logits": logits, "penultimate_activations": penultimate_activations}, f)
    print(f"Saved logits and penultimate activations to {filename}")


def load_pickled_logits_features(filename: str) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    with open(filename, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded logits and penultimate activations from {filename}")
    return data["logits"], data["penultimate_activations"]


def record_similarity_metrics(
    mm: ModelManager,
    logits: Dict[str, Tensor],
    penultimate_activations: Dict[str, Tensor],
    csv_output: str = "similarity_metrics.csv",
) -> None:
    rows = []
    for i in range(len(mm)):
        for j in range(i + 1, len(mm)):
            model_a = mm[i]
            model_b = mm[j]
            l2 = l2_distance(model_a, model_b)["l2"]
            cosine = cosine_similarity(model_a, model_b)
            logits_model_a = logits[model_a.key]
            logits_model_b = logits[model_b.key]
            features_model_a = penultimate_activations[model_a.key]
            features_model_b = penultimate_activations[model_b.key]
            cka_logits = cka_similarity(logits_model_a, logits_model_b)
            mse_kl = logit_mse_kl(logits_model_a, logits_model_b)
            mse_logits = mse_kl["mse"]
            kl_logits = mse_kl["kl"]
            cka_features = cka_similarity(features_model_a, features_model_b)

            # Canonical order for keys
            key_a, key_b = model_a.key, model_b.key
            if key_a > key_b:
                key_a, key_b = key_b, key_a

            row = {
                "key_a": key_a,
                "key_b": key_b,
                "l2_distance": l2,
                "cosine_similarity": cosine,
                "cka_logits": cka_logits,
                "mse_logits": mse_logits,
                "kl_logits": kl_logits,
                "cka_features": cka_features,
            }
            print(f"Recorded similarity for pair: {key_a} + {key_b}")
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_output, index=False)
    parquet_output = csv_output.replace(".csv", ".parquet")
    df.to_parquet(parquet_output, index=False)


# Experiment 5 - Comparison of angle metric from the last shared checkpoint
def record_deviation_metrics(
    mm: ModelManager,
    csv_output: str = "angle_change_metrics.csv",
) -> None:
    rows = []
    for i in range(len(mm)):
        for j in range(i + 1, len(mm)):
            model_a = mm[i]
            model_b = mm[j]

            model_ancestor_epoch = min(model_a.epoch, model_b.epoch)
            path = os.path.join(
                MODEL_DIR, f"cifar100-resnet50/baseline-resnet50/baseline-resnet50-epoch_{model_ancestor_epoch}.pt"
            )

            model_ancestor = get_model(mm.model_name, num_classes=mm.num_classes, dropout=mm.dropout)
            load_checkpoint(path, model_ancestor)
            model_ancestor.eval()
            model_ancestor.to(mm.device)

            # Will give 2x the difference of model_a from the ancestor
            model_a_diff = soup_models(model_ancestor, model_a, alpha=-1.0)
            model_b_diff = soup_models(model_ancestor, model_b, alpha=-1.0)

            # rescale by 0.5 to get the actual difference
            l2 = l2_distance(model_a, model_b)["l2"] / 2.0
            cosine = cosine_similarity(model_a, model_b)

            # Canonical order for keys
            key_a, key_b = model_a.key, model_b.key
            if key_a > key_b:
                key_a, key_b = key_b, key_a

            row = {
                "key_a": key_a,
                "key_b": key_b,
                "l2_distance": l2,
                "cosine_similarity": cosine,
            }
            print(f"Recorded similarity for pair: {key_a} + {key_b}")
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_output, index=False)
    parquet_output = csv_output.replace(".csv", ".parquet")
    df.to_parquet(parquet_output, index=False)


if __name__ == "__main__":
    mm = ModelManager(models_dir=SOUP_DIR, device=DEVICE)
    # Example usage of get_index_pairs
    index_pairs = mm.get_index_pairs(
        model_ids=[1, 2, 3, 4],
        epochs=list(range(10, 301, 10)),
        unique=True,
    )
    print(f"Total unique model pairs: {len(index_pairs)}")

    experiments_to_run = [2,3,5]  # Specify which experiments to run

    # Experiment 2 - Permutation Ablation
    if 2 in experiments_to_run:
        run_permutation_ablation_experiment(mm, index_pairs, output_csv="analysis/permutation_ablation_results.csv")

    # Experiment 3 - Predicting Soupability
    if 3 in experiments_to_run:
        save_logits_features_path = "logits_penultimate_activations.pkl"
        if not os.path.exists(save_logits_features_path):
            _, _, test_loader = get_cifar100_loaders()
            logits, penultimate_activations = record_logits_features(mm, test_loader)
            pickle_logits_features(logits, penultimate_activations, save_logits_features_path)
        else:
            logits, penultimate_activations = load_pickled_logits_features(save_logits_features_path)

        record_similarity_metrics(
            mm,
            logits,
            penultimate_activations,
            csv_output="analysis/similarity_metrics.csv",
        )

    if 5 in experiments_to_run:
        record_deviation_metrics(
            mm,
            csv_output="analysis/angle_change_metrics.csv",
        )
