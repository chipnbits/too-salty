"""
Evaluate pairwise model soups on CIFAR-100 test and CIFAR-100-C.

Uses ModelManager for model discovery, loading, and souping. Results are saved
incrementally so the experiment can resume if interrupted.

Usage:
    python scripts/05_evaluate_soups.py --variants-dir models/cifar100-resnet50
    python scripts/05_evaluate_soups.py --variants-dir models/soup_models  # legacy flat layout
"""

import argparse
import itertools
import os
import random
from typing import List, Tuple

import pandas as pd
from dotenv import load_dotenv

from salty.datasets import get_cifar100_loaders, get_cifar100c_loaders_by_corruption
from salty.evaluation import canonical_key
from salty.model_manager import ModelManager

load_dotenv()

VARIANTS_DIR = os.getenv("VARIANTS_DIR", "./models/cifar100-resnet50")


def _load_computed_keys(result_path: str) -> set:
    """Load previously computed canonical keys from an incremental CSV."""
    if not os.path.exists(result_path):
        return set()
    try:
        df = pd.read_csv(result_path)
        return set(
            df.apply(lambda row: canonical_key(row["key_a"], row["key_b"]), axis=1)
        )
    except (pd.errors.EmptyDataError, Exception):
        return set()


def _save_row(row: dict, result_path: str) -> None:
    """Append a single result row to the CSV (create header if needed)."""
    df = pd.DataFrame([row])
    write_header = not os.path.exists(result_path)
    df.to_csv(result_path, mode="a", header=write_header, index=False)


def evaluate_all_soups(
    mm: ModelManager,
    result_path: str,
    batch_size: int = 1024,
    severities: List[int] = [3],
    seed: int = 42,
    acc_threshold: float = 5.0,
    num_workers: int = 1,
    worker_id: int = 0,
) -> pd.DataFrame:
    """Evaluate all pairwise model soups managed by *mm*.

    Results are written incrementally to *result_path* so the run can resume
    after interruption.  When *num_workers* > 1, pairs are deterministically
    split across workers so multiple GPUs can evaluate in parallel, each
    writing to its own shard CSV.
    """
    _, _, test_loader = get_cifar100_loaders(batch_size=batch_size)

    computed_keys = _load_computed_keys(result_path)
    if computed_keys:
        print(f"Resuming. Found {len(computed_keys)} previously computed pairs.")

    # Canonical pairs only: (ka, kb) with ka <= kb (includes self-pairings)
    all_keys = sorted(mm.keys)
    all_pairs = list(itertools.combinations_with_replacement(all_keys, 2))

    random.seed(seed)
    random.shuffle(all_pairs)

    # Split pairs across workers
    if num_workers > 1:
        all_pairs = [p for i, p in enumerate(all_pairs) if i % num_workers == worker_id]

    print(f"Discovered {len(mm)} models, {len(all_pairs)} pairs for worker {worker_id}/{num_workers}")

    for key_a, key_b in all_pairs:
        canon = canonical_key(key_a, key_b)
        if canon in computed_keys:
            continue

        # Soup and evaluate
        souped = mm.soup(key_a, key_b, alpha=0.5)
        acc_clean, loss_clean = mm.evaluate(souped, test_loader)

        epoch_a, mid_a = mm._entries[mm.get_index(key_a)]["epoch"], mm._entries[mm.get_index(key_a)]["model_id"]
        epoch_b, mid_b = mm._entries[mm.get_index(key_b)]["epoch"], mm._entries[mm.get_index(key_b)]["model_id"]

        row = {
            "key_a": key_a,
            "key_b": key_b,
            "branch_epoch_a": epoch_a,
            "branch_epoch_b": epoch_b,
            "model_id_a": mid_a,
            "model_id_b": mid_b,
            "clean_accuracy": acc_clean,
            "clean_loss": loss_clean,
        }

        # Corrupted evaluation (skip if clean acc is too low)
        if acc_clean >= acc_threshold:
            for severity in severities:
                corruption_loaders = get_cifar100c_loaders_by_corruption(
                    batch_size=batch_size, severity=severity,
                )
                accs, losses = {}, {}
                for name, loader in corruption_loaders.items():
                    a, l = mm.evaluate(souped, loader)
                    accs[name] = a
                    losses[name] = l

                row[f"acc_corr_{severity}"] = sum(accs.values()) / len(accs)
                row[f"loss_corr_{severity}"] = sum(losses.values()) / len(losses)
        else:
            for severity in severities:
                row[f"acc_corr_{severity}"] = None
                row[f"loss_corr_{severity}"] = None

        _save_row(row, result_path)
        computed_keys.add(canon)
        print(f"[{len(computed_keys)}] {key_a} + {key_b}: clean_acc={acc_clean:.2f}%")

    return pd.read_csv(result_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pairwise model soups")
    parser.add_argument("--variants-dir", type=str, default=VARIANTS_DIR,
                        help="Directory containing variant model subdirectories")
    parser.add_argument("--output-dir", type=str, default="./analysis",
                        help="Directory to save results")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--severities", type=int, nargs="+", default=[3])
    parser.add_argument("--acc-threshold", type=float, default=5.0,
                        help="Skip corruption eval if clean acc below this (default: 5.0)")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Total number of parallel workers (SLURM array size)")
    parser.add_argument("--worker-id", type=int, default=0,
                        help="This worker's ID (0-indexed, use SLURM_ARRAY_TASK_ID)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    mm = ModelManager(args.variants_dir, device="cuda")

    result_name = f"rand_soups_seed_{args.seed}"
    if args.num_workers > 1:
        result_path = os.path.join(args.output_dir, f"{result_name}_worker_{args.worker_id}.csv")
    else:
        result_path = os.path.join(args.output_dir, f"{result_name}.csv")

    df = evaluate_all_soups(
        mm,
        result_path=result_path,
        batch_size=args.batch_size,
        severities=args.severities,
        seed=args.seed,
        acc_threshold=args.acc_threshold,
        num_workers=args.num_workers,
        worker_id=args.worker_id,
    )

    print(f"Saved: {result_path}")
    print(f"Total pairs evaluated by this worker: {len(df)}")

    # If single worker, also save parquet
    if args.num_workers <= 1:
        parquet_path = os.path.join(args.output_dir, f"{result_name}.parquet")
        df.to_parquet(parquet_path, index=False)
        print(f"Saved parquet: {parquet_path}")
