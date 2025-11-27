"""
An evaluation script for pretrained models on both CIFAR-100 test and CIFAR-100C
"""

import itertools
import os
import random
import re
import warnings
from multiprocessing import Process
from typing import Any, Dict, List, Optional, Tuple

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
    soup_models,
)

load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR", "./models")
SOUP_DIR = os.getenv("SOUP_DIR", "./models/soup_models")
fine_tuned_pattern = re.compile(r"epoch_(\d+)_model_(\d+)\.pt")


def canonical_key(key_a, key_b):
    return ";".join(sorted([str(key_a), str(key_b)]))


def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            predicted = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)

    accuracy = 100 * correct / total
    average_loss = total_loss / total
    return accuracy, average_loss


def load_resnet50_model(path, device):
    model = get_resnet50_model(num_classes=100)
    start_epoch, best_val_acc, loaded_cfg, global_step, wandb_run_id = load_checkpoint(path, model)
    model.to(device)
    return model, start_epoch


def extract_key_from_filename(filename: str, pattern: re.Pattern) -> Optional[str]:
    """
    Extracts a unique key {epoch}_{model_id} from the saved model filename.
    """
    match = pattern.match(filename)
    if match:
        # Assuming group(1) is the epoch and group(2) is the model ID
        return f"{match.group(1)}_{match.group(2)}"
    return None


def get_model_index_pairs(
    model_filenames: List[str],
    model_filename_pattern: re.Pattern,
    model_ids: List[int] = [1, 2, 3, 4],
    epochs=list(range(10, 301, 10)),
) -> List[Tuple[int, int]]:
    """
    1. Parses available models and maps their unique key to their index.
    2. Generates all possible key pairings within specified epoch and model ID ranges.
    3. Filters for available pairings from filenames and returns their indices
    """

    # --- Step 1: Parse and Map Available Models (Key -> Index) ---
    # The input model_filenames should already be sorted.
    available_key_to_index: Dict[str, int] = {}

    for index, filename in enumerate(model_filenames):
        key = extract_key_from_filename(filename, model_filename_pattern)
        if key:
            # Store the unique key (e.g., "50_4") mapped to its sorted list index (e.g., 5)
            available_key_to_index[key] = index

    # --- Step 2: Generate All Theoretical Key Pairings ---
    # List of all theoretical keys (e.g., "10_1", "10_2", ..., "300_4")
    all_possible_keys = [f"{e}_{m}" for e, m in itertools.product(epochs, model_ids)]

    # All N choose 2 unique pairings from the theoretical keys
    all_possible_key_pairs = list(itertools.combinations(all_possible_keys, 2))

    # --- Step 3: Filter Available Pairs and Convert to Indices ---
    final_index_pairs: List[Tuple[int, int]] = []

    for key_a, key_b in all_possible_key_pairs:
        # Check if BOTH keys exist in the available files map
        if key_a in available_key_to_index and key_b in available_key_to_index:
            idx_a = available_key_to_index[key_a]
            idx_b = available_key_to_index[key_b]
            # Add the index pairing to the final list
            final_index_pairs.append((idx_a, idx_b))

    return final_index_pairs


def evaluate_all_soup_models(
    device: str = "cuda",
    batch_size: int = 128,
    severities=(
        1,
        3,
    ),
    max_models: int | None = None,
):
    # --- CIFAR-100 test loader ---
    _, _, test_loader = get_cifar100_loaders(batch_size=batch_size)

    results = []
    num_models = 0  # how many models we've actually evaluated

    for filename in sorted(os.listdir(SOUP_DIR)):
        match = fine_tuned_pattern.match(filename)
        if not match:
            continue

        epoch, model_n = map(int, match.groups())
        model_path = os.path.join(SOUP_DIR, filename)

        print(f"\n=== Loading {filename} (epoch={epoch}, model={model_n}) ===")
        model, best_epoch = load_resnet50_model(model_path, device)

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


def soup_rand_models(
    device: str = "cuda",
    batch_size: int = 128,
    severities=(3,),
    seed: None | int = None,
    result_path: str = os.path.join(SOUP_DIR, "soup_rand_results.csv"),
):
    # --- CIFAR-100 test loader ---
    _, _, test_loader = get_cifar100_loaders(batch_size=batch_size)

    existing_results = pd.DataFrame()
    computed_keys = set()

    # ---- Results Storage ----
    if os.path.exists(result_path):
        try:
            existing_results = pd.read_csv(result_path)
            # The unique identifier is the sorted combination of the two keys (A and B)
            # We must ensure the keys are sorted (e.g., "50_1;100_3") regardless of which
            # model was A or B in the original computation.
            existing_results["unique_key"] = existing_results.apply(
                lambda row: canonical_key(row["key_a"], row["key_b"]), axis=1
            )
            computed_keys = set(existing_results["unique_key"])
            print(f"Resuming experiment. Found {len(computed_keys)} previously computed pairs.")
        except pd.errors.EmptyDataError:
            print("Existing result file is empty. Starting fresh.")
        except Exception as e:
            print(f"Error loading existing results: {e}. Starting fresh.")
            existing_results = pd.DataFrame()
            computed_keys = set()

    # Filter by regex for model filenames
    model_filenames = [f for f in os.listdir(SOUP_DIR) if fine_tuned_pattern.match(f)]
    model_filenames = sorted(model_filenames)

    # Get pairings of available models
    index_pairs = get_model_index_pairs(
        model_filenames,
        fine_tuned_pattern,
        model_ids=[1, 2, 3, 4],
        epochs=list(range(10, 301, 10)),
    )

    # Rand shuffle the pairs for order of computation
    if seed is not None:
        random.seed(seed)
    random.shuffle(index_pairs)

    # --- 3. Iteratively Compute and Save ---
    new_results_to_save = []

    print(f"Total model pairs to potentially process: {len(index_pairs)}")

    for idx_a, idx_b in index_pairs:
        # --- RESUME CHECK: Create the unique ID ---
        key_a = extract_key_from_filename(model_filenames[idx_a], fine_tuned_pattern)
        key_b = extract_key_from_filename(model_filenames[idx_b], fine_tuned_pattern)

        if key_a is None or key_b is None:
            warnings.warn(
                f"Warning: Could not extract keys for files {model_filenames[idx_a]} or {model_filenames[idx_b]}. Skipping."
            )
            continue

        # The canonical key is sorted to ensure a match regardless of A/B assignment
        canonical_key_val = canonical_key(key_a, key_b)

        if canonical_key_val in computed_keys:
            print(f"   Skipping pair ({key_a}, {key_b}). Already computed.")
            continue

        # --- Load and soup models ---
        model_path_a = os.path.join(SOUP_DIR, model_filenames[idx_a])
        model_path_b = os.path.join(SOUP_DIR, model_filenames[idx_b])
        match_a = fine_tuned_pattern.match(model_filenames[idx_a])
        match_b = fine_tuned_pattern.match(model_filenames[idx_b])
        epoch_a, model_n_a = map(int, match_a.groups())
        epoch_b, model_n_b = map(int, match_b.groups())

        print(f"\n=== Computing NEW souped model: {key_a} + {key_b} ===")
        model_a, best_epoch_a = load_resnet50_model(model_path_a, device)
        model_b, best_epoch_b = load_resnet50_model(model_path_b, device)
        souped_model = soup_models(model_a, model_b, alpha=0.5)
        souped_model.to(device)

        row = {
            "key_a": key_a,  # Saved for the unique key check on resume
            "key_b": key_b,  # Saved for the unique key check on resume
            "branch_epoch_a": epoch_a,
            "branch_epoch_b": epoch_b,
            "model_id_a": model_n_a,
            "model_id_b": model_n_b,
            "termination_epoch_a": best_epoch_a,
            "termination_epoch_b": best_epoch_b,
        }

        # --- Evaluation

        # Evaluate once on clean test set
        acc_clean, loss_clean = evaluate_model(souped_model, test_loader, device)
        print(f"\n  -> Clean accuracy of souped model: {acc_clean:.2f}%")
        print(f"\n  -> Clean loss of souped model: {loss_clean:.4f}")
        row.update(
            {
                "clean_accuracy": acc_clean,
                "clean_loss": loss_clean,
            }
        )

        # (Then your CIFAR-100C severity loop, if you want)
        for severity in severities:
            print(f"\n  -> Evaluating CIFAR-100C, severity={severity}")
            corruption_loaders = get_cifar100c_loaders_by_corruption(
                batch_size=batch_size,
                severity=severity,
            )

            corruption_accs = {}
            corruption_losses = {}
            for corr_name, loader in corruption_loaders.items():
                acc_corr, loss_corr = evaluate_model(souped_model, loader, device)
                corruption_accs[corr_name] = acc_corr
                corruption_losses[corr_name] = loss_corr

            # Pool results into a single acc and loss over all corruptions
            mean_acc = sum(corruption_accs.values()) / len(corruption_accs)
            mean_loss = sum(corruption_losses.values()) / len(corruption_losses)

            row[f"acc_corr_{severity}"] = mean_acc
            row[f"loss_corr_{severity}"] = mean_loss

        # --- Incremental Save ---
        new_results_to_save.append(row)

        # Save after every successful computation (or batch of them)
        # We append new results to the file.
        df_new = pd.DataFrame([row])

        if os.path.exists(result_path):
            df_new.to_csv(result_path, mode="a", header=False, index=False)
        else:
            df_new.to_csv(result_path, mode="w", header=True, index=False)

        # Add the just-computed key to the set to prevent re-computation in this session
        computed_keys.add(canonical_key_val)
        print(f"Result saved for {key_a} + {key_b}. Total computed: {len(computed_keys)}")

    # --- 4. Final Dataframe Assembly ---
    # Load all results one last time to get the final complete dataframe
    final_df = pd.read_csv(result_path)
    return final_df


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


def gpu_worker_process(
    device_id: int,
    pairs_subset: List[Tuple[int, int]],
    model_filenames: List[str],
    fine_tuned_pattern: re.Pattern,
    severities: Tuple[int, ...],
    batch_size: int,
    # Worker-specific result path
    worker_result_path: str,
    acc_threshold: float = 5.0,  # minimum clean accuracy to record corruption results
):

    device = f"cuda:{device_id}"

    # --- Setup and Dataloader Initialization (MUST be done inside the worker) ---
    # Re-initialize dataloaders for the worker process
    _, _, test_loader = get_cifar100_loaders(batch_size=batch_size)

    # --- Resume Logic for Worker's Subset ---
    computed_keys = set()
    if os.path.exists(worker_result_path):
        try:
            worker_df = pd.read_csv(worker_result_path)
            worker_df["unique_key"] = worker_df.apply(lambda row: canonical_key(row["key_a"], row["key_b"]), axis=1)
            computed_keys = set(worker_df["unique_key"])
            print(f"[Worker {device_id}] Resuming. Found {len(computed_keys)} computed pairs in its subset.")
        except pd.errors.EmptyDataError:
            pass  # File exists but is empty
        except Exception as e:
            print(f"[Worker {device_id}] Error loading result file: {e}. Starting fresh.")

    # --- Main Loop ---
    for idx_a, idx_b in pairs_subset:
        key_a = extract_key_from_filename(model_filenames[idx_a], fine_tuned_pattern)
        key_b = extract_key_from_filename(model_filenames[idx_b], fine_tuned_pattern)
        if key_a is None or key_b is None:
            continue

        canonical_key_val = canonical_key(key_a, key_b)

        if canonical_key_val in computed_keys:
            continue

        # --- Computation and Evaluation (Your core logic) ---
        model_path_a = os.path.join(os.path.dirname(worker_result_path), model_filenames[idx_a])
        model_path_b = os.path.join(os.path.dirname(worker_result_path), model_filenames[idx_b])

        match_a = fine_tuned_pattern.match(model_filenames[idx_a])
        match_b = fine_tuned_pattern.match(model_filenames[idx_b])
        assert match_a is not None and match_b is not None

        epoch_a, model_n_a = map(int, match_a.groups())
        epoch_b, model_n_b = map(int, match_b.groups())

        print(f"\n[Worker {device_id}] Computing NEW souped model: {key_a} + {key_b}")
        model_a, best_epoch_a = load_resnet50_model(model_path_a, device)
        model_b, best_epoch_b = load_resnet50_model(model_path_b, device)
        souped_model = soup_models(model_a, model_b, alpha=0.5)
        souped_model.to(device)

        row = {
            "key_a": key_a,
            "key_b": key_b,
            "branch_epoch_a": epoch_a,
            "branch_epoch_b": epoch_b,
            "model_id_a": model_n_a,
            "model_id_b": model_n_b,
        }

        # Clean Evaluation
        acc_clean, loss_clean = evaluate_model(souped_model, test_loader, device)
        row.update({"clean_accuracy": acc_clean, "clean_loss": loss_clean})

        # Corrupted Evaluation
        if acc_clean >= acc_threshold:

            corruption_losses = {}
            for severity in severities:
                corruption_loaders = get_cifar100c_loaders_by_corruption(batch_size=batch_size, severity=severity)
                corruption_accs = {}
                for corr_name, loader in corruption_loaders.items():
                    acc_corr, loss_corr = evaluate_model(souped_model, loader, device)
                    corruption_accs[corr_name] = acc_corr
                    corruption_losses[corr_name] = loss_corr

                mean_acc = sum(corruption_accs.values()) / len(corruption_accs)
                mean_loss = sum(corruption_losses.values()) / len(corruption_losses)
                row[f"acc_corr_{severity}"] = mean_acc
                row[f"loss_corr_{severity}"] = mean_loss
        else:
            for severity in severities:
                row[f"acc_corr_{severity}"] = None
                row[f"loss_corr_{severity}"] = None

        # --- Incremental Save to Worker's File ---
        df_new = pd.DataFrame([row])
        write_header = not os.path.exists(worker_result_path)
        df_new.to_csv(worker_result_path, mode="a", header=write_header, index=False)
        computed_keys.add(canonical_key_val)
        print(f"[Worker {device_id}] ✅ Saved {key_a} + {key_b}")


def soup_rand_models_parallel(
    device: str = "cuda",
    batch_size: int = 128,
    severities=(3,),
    seed: None | int = None,
    result_path: str = os.path.join(SOUP_DIR, "soup_rand_results.csv"),
    num_gpus: int = 3,  # Explicitly use 3 GPUs
):
    if device != "cuda" and num_gpus > 1:
        warnings.warn("Parallel run requested but device is not 'cuda'. Running sequentially.")
        num_gpus = 1

    # --- Initial Setup and Pairing ---
    model_filenames = [f for f in os.listdir(SOUP_DIR) if fine_tuned_pattern.match(f)]
    model_filenames = sorted(model_filenames)

    index_pairs = get_model_index_pairs(
        model_filenames,
        fine_tuned_pattern,
        model_ids=[1, 2, 3, 4],
        epochs=list(range(10, 301, 10)),
    )

    if seed is not None:
        random.seed(seed)
    random.shuffle(index_pairs)

    print(f"Total model pairs to process: {len(index_pairs)}")

    # --- Split Pairs and Launch Workers ---

    chunk_size = len(index_pairs) // num_gpus
    chunks = []
    for i in range(num_gpus):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_gpus - 1 else len(index_pairs)
        chunks.append(index_pairs[start:end])

    processes = []

    print(f"Starting {num_gpus} GPU workers...")
    temp_files = []

    for i in range(num_gpus):
        if not chunks[i]:
            continue

        worker_result_path = os.path.join(SOUP_DIR, f"soup_temp_worker_{i}_results.csv")
        temp_files.append(worker_result_path)

        p = Process(
            target=gpu_worker_process,
            args=(
                i,  # device_id (0, 1, 2)
                chunks[i],
                model_filenames,
                fine_tuned_pattern,
                severities,
                batch_size,
                worker_result_path,
            ),
        )
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    print("All GPU workers finished. Combining results...")

    # --- Combine Worker Results into Final File ---
    all_results = []

    # Check if the final result file exists to determine header writing
    final_header_needed = not os.path.exists(result_path)

    for temp_file in temp_files:
        if os.path.exists(temp_file):
            try:
                df = pd.read_csv(temp_file)

                # Append to the final file
                df.to_csv(result_path, mode="a", header=final_header_needed, index=False)
                final_header_needed = False  # Header only needed for the very first write

                # Clean up the temporary worker file
                os.remove(temp_file)
            except pd.errors.EmptyDataError:
                os.remove(temp_file)  # Clean up empty file
            except Exception as e:
                print(f"Warning: Could not read or combine temporary file {temp_file}. Error: {e}")

    # Load all results one last time to get the final complete dataframe
    try:
        final_df = pd.read_csv(result_path)
    except pd.errors.EmptyDataError:
        print("Final result file is empty.")
        final_df = pd.DataFrame()

    return final_df


if __name__ == "__main__":
    severities = [
        3,
    ]
    result_naming = "rand_soups_seed_42"
    result_path = os.path.join(SOUP_DIR, f"{result_naming}.csv")
    # df = soup_rand_models(device="cuda:2", batch_size=1024, severities=severities,  seed=42, result_path=result_path)
    df = soup_rand_models_parallel(
        device="cuda",
        batch_size=1024,
        severities=severities,
        seed=42,
        result_path=result_path,
        num_gpus=3,
    )

    # Paths
    parquet_path = os.path.join(SOUP_DIR, f"{result_naming}.parquet")
    csv_path = os.path.join(SOUP_DIR, f"{result_naming}.csv")

    # Best format
    df.to_parquet(parquet_path, index=False)
    print(f"Saved Parquet: {parquet_path}")

    # Optional human-readable version
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    print("\n=== Final Results ===")
    print(df)
