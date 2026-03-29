"""
Evaluate accuracy over the ternary simplex (3-model convex combinations).

For each triplet of models, generates a grid of weights (w1, w2, w3) summing
to 1, blends the model parameters, and evaluates on CIFAR-100 test set.

Usage:
    python scripts/ternary_simplex_eval.py --triplet "240_3,170_2,250_4" --resolution 20
    python scripts/ternary_simplex_eval.py --all --resolution 20
"""

import argparse
import copy
import os
import time

import pandas as pd
import torch

from salty.datasets import get_cifar100_loaders
from salty.evaluation import evaluate_model
from salty.model_manager import ModelManager
from salty.utils import ternary_soup_state_dict

VARIANTS_DIR = os.getenv("VARIANTS_DIR", "./models/cifar100-resnet50")

TRIPLETS = [
    ("240_3", "170_2", "250_4"),
    ("250_4", "230_4", "240_3"),
    ("240_3", "190_2", "250_4"),
    ("240_3", "210_2", "250_4"),
    ("240_3", "250_4", "200_2"),
    ("270_4", "220_2", "250_4"),
    ("260_4", "180_4", "280_1"),
    ("240_3", "250_4", "160_4"),
]


def generate_simplex_grid(resolution):
    """Generate all (w1, w2, w3) with wi = k/resolution, sum = 1."""
    points = []
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            k = resolution - i - j
            points.append((i / resolution, j / resolution, k / resolution))
    return points


def evaluate_triplet(triplet_keys, mm, test_loader, device, resolution=10):
    """
    Evaluate all simplex grid points for a triplet of models.

    Returns a DataFrame with columns: w1, w2, w3, accuracy, loss, key_a, key_b, key_c
    """
    key_a, key_b, key_c = triplet_keys

    # Load models and extract state_dicts once
    model_a = mm.load(key_a)
    model_b = mm.load(key_b)
    model_c = mm.load(key_c)

    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    sd_c = model_c.state_dict()

    # Single base model to reuse
    base_model = copy.deepcopy(model_a)
    base_model.to(device)
    base_model.eval()

    # Free originals
    del model_a, model_b, model_c

    grid = generate_simplex_grid(resolution)
    n_points = len(grid)
    print(f"Evaluating triplet ({key_a}, {key_b}, {key_c}): {n_points} grid points")

    rows = []
    for idx, (w1, w2, w3) in enumerate(grid):
        blended_sd = ternary_soup_state_dict([sd_a, sd_b, sd_c], [w1, w2, w3])
        base_model.load_state_dict(blended_sd)

        acc, loss = evaluate_model(base_model, test_loader, device)
        rows.append({
            "w1": w1, "w2": w2, "w3": w3,
            "accuracy": acc, "loss": loss,
            "key_a": key_a, "key_b": key_b, "key_c": key_c,
        })

        if (idx + 1) % 10 == 0 or idx == n_points - 1:
            print(f"  [{idx+1}/{n_points}] w=({w1:.2f},{w2:.2f},{w3:.2f}) acc={acc:.2f}%")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Ternary simplex soup evaluation")
    parser.add_argument("--triplet", type=str, default=None,
                        help="Comma-separated triplet keys, e.g. '240_3,170_2,250_4'")
    parser.add_argument("--all", action="store_true",
                        help="Run all predefined triplets")
    parser.add_argument("--resolution", type=int, default=20,
                        help="Grid resolution (number of ticks per side)")
    parser.add_argument("--variants-dir", type=str, default=VARIANTS_DIR,
                        help="Path to model checkpoints directory")
    parser.add_argument("--output-dir", type=str, default="analysis/ternary",
                        help="Output directory for parquet files")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    if not args.triplet and not args.all:
        parser.error("Specify --triplet or --all")

    triplets = []
    if args.all:
        triplets = TRIPLETS
    else:
        keys = args.triplet.split(",")
        if len(keys) != 3:
            parser.error("--triplet must be exactly 3 comma-separated keys")
        triplets = [tuple(keys)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mm = ModelManager(args.variants_dir, device=device)
    _, _, test_loader = get_cifar100_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for triplet in triplets:
        key_a, key_b, key_c = triplet
        out_path = os.path.join(
            args.output_dir, f"ternary_{key_a}_{key_b}_{key_c}.parquet"
        )

        if os.path.exists(out_path):
            print(f"Skipping {triplet} — already computed at {out_path}")
            continue

        t0 = time.time()
        df = evaluate_triplet(triplet, mm, test_loader, device, args.resolution)
        elapsed = time.time() - t0

        df.to_parquet(out_path, index=False)
        print(f"Saved {out_path} ({len(df)} rows, {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
