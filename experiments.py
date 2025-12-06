"""
This file runs provides functions to run various experiments for the project, based off the finetuned models stored in SOUP_DIR with
"""

import itertools
import os
import pickle
from pathlib import Path
from typing import Dict, List, Pattern, Tuple

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from evaluate import canonical_key, evaluate_model, extract_key_from_filename, fine_tuned_pattern
from salty.datasets import get_cifar100_loaders
from salty.models import get_resnet50_model
from salty.permute_model import permute_models
from salty.similarity_metrics_from_logits import cka_similarity, logit_mse_kl
from salty.similarity_metrics_from_models import cosine_similarity, l2_distance
from salty.utils import load_checkpoint, soup_models

MODEL_DIR = os.getenv("MODEL_DIR", "./models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOUP_DIR = os.getenv("SOUP_DIR", "./models/soup_models")


class ModelManager:
    """Class to manage loading and storing models for experiments.

    This class functions as an iterable and indexable collection of models.
    Models are loaded lazily from disk when accessed by index.

    Example:
        manager = ModelManager()

        # Indexable - loads model on access
        model = manager[2]

        # Iterable - loads models as you iterate
        for model in manager:
            print(model.key, model.epoch, model.model_id)

        # Get key/index mappings
        key = manager.get_key(2)  # e.g., "50_4"
        index = manager.get_index("50_4")  # e.g., 2

        # Get index pairs for specific epochs and model_ids
        pairs = manager.get_index_pairs(
            model_ids=[1, 2, 3, 4],
            epochs=[10, 20, 30]
        )
    """

    def __init__(self, model_dir: str = SOUP_DIR, pattern: Pattern = fine_tuned_pattern, device: torch.device = DEVICE):
        self.soup_dir = model_dir
        self.pattern = pattern
        self.device = device
        self._model_paths: List[Path] = []
        self._model_keys: List[str] = []
        self._init_models()

    def _init_models(self) -> None:
        """Initialize model paths and keys from the soup directory."""
        model_filenames = [f for f in os.listdir(self.soup_dir) if self.pattern.match(f)]
        model_filenames = sorted(model_filenames)

        self._model_paths = [Path(self.soup_dir) / f for f in model_filenames]
        self._model_keys = [extract_key_from_filename(f, self.pattern) for f in model_filenames]

    def __len__(self) -> int:
        """Return the number of models."""
        return len(self._model_paths)

    def __getitem__(self, index: int) -> torch.nn.Module:
        """Load and return model at the given index.

        Args:
            index: Index of the model to load

        Returns:
            Loaded model with .key, .epoch, and .model_id attributes set
        """
        if index < 0 or index >= len(self._model_paths):
            raise IndexError(f"Model index {index} out of range [0, {len(self._model_paths)})")

        path = self._model_paths[index]
        key = self._model_keys[index]
        model = self._load_model(path)
        model.key = key

        # Parse and attach epoch and model_id from key
        epoch, model_id = self._parse_key(key)
        model.epoch = epoch
        model.model_id = model_id

        return model

    def __iter__(self):
        """Return an iterator over all models."""
        self._iter_index = 0
        return self

    def __next__(self) -> torch.nn.Module:
        """Return the next model in iteration."""
        if self._iter_index >= len(self._model_paths):
            raise StopIteration
        model = self[self._iter_index]
        self._iter_index += 1
        return model

    def _load_model(self, path: Path) -> torch.nn.Module:
        """Load a single model from path.

        Args:
            path: Path to model checkpoint

        Returns:
            Loaded model
        """
        model = get_resnet50_model(num_classes=100)
        load_checkpoint(path, model)
        model.eval()
        model.to(self.device)
        return model

    @staticmethod
    def _parse_key(key: str) -> Tuple[int | None, int | None]:
        """Parse epoch and model_id from key.

        Args:
            key: Canonical key string (e.g., "50_4")

        Returns:
            Tuple of (epoch, model_id), or (None, None) if parsing fails
        """
        if not key:
            return None, None
        parts = key.split("_")
        if len(parts) == 2:
            try:
                return int(parts[0]), int(parts[1])
            except ValueError:
                return None, None
        return None, None

    @property
    def model_paths(self) -> List[Path]:
        """Get list of model paths."""
        return self._model_paths

    @property
    def model_keys(self) -> List[str]:
        """Get list of model keys."""
        return self._model_keys

    def get_key(self, index: int) -> str:
        """Get the canonical key for a model at the given index.

        Args:
            index: Index of the model

        Returns:
            Canonical key string (e.g., "50_4")
        """
        if index < 0 or index >= len(self._model_keys):
            raise IndexError(f"Model index {index} out of range [0, {len(self._model_keys)})")
        return self._model_keys[index]

    def get_epoch(self, index: int) -> int | None:
        """Get the epoch for a model at the given index.

        Args:
            index: Index of the model

        Returns:
            Epoch number, or None if not available
        """
        key = self.get_key(index)
        epoch, _ = self._parse_key(key)
        return epoch

    def get_model_id(self, index: int) -> int | None:
        """Get the model_id for a model at the given index.

        Args:
            index: Index of the model

        Returns:
            Model ID, or None if not available
        """
        key = self.get_key(index)
        _, model_id = self._parse_key(key)
        return model_id

    def get_index(self, key: str) -> int:
        """Get the index of a model with the given canonical key.

        Args:
            key: Canonical key string (e.g., "50_4")

        Returns:
            Index of the model, or -1 if not found
        """
        try:
            return self._model_keys.index(key)
        except ValueError:
            return -1

    def get_index_pairs(
        self,
        model_ids: List[int] = [1, 2, 3, 4],
        epochs: List[int] = list(range(10, 301, 10)),
        unique: bool = True,
    ) -> List[Tuple[int, int]]:
        """Get all valid index pairs for models matching the specified epochs and model_ids.

        This generates all possible pairings (including self-pairings) of models that
        match the given epoch and model_id filters.

        Args:
            model_ids: List of model IDs to include (default: [1, 2, 3, 4])
            epochs: List of epochs to include (default: range(10, 301, 10))
            unique: If True, returns only unique pairings (A, B) where canonical_key(A, B)
                    is generated only once, using combinations_with_replacement.
                    If False, returns all permutations (A, B and B, A).

        Returns:
            List of (index_a, index_b) tuples for all valid model pairs
        """
        # Build a map from key to index for available models
        available_key_to_index: Dict[str, int] = {}
        for index, key in enumerate(self._model_keys):
            if key:
                available_key_to_index[key] = index

        # Generate all theoretical keys from epochs and model_ids
        all_possible_keys = [f"{e}_{m}" for e, m in itertools.product(epochs, model_ids)]
        available_keys = sorted([k for k in all_possible_keys if k in available_key_to_index])

        if unique:
            key_pairs_iterator = itertools.combinations_with_replacement(available_keys, 2)
        else:
            # Use product to get all permutations (A, B and B, A)
            key_pairs_iterator = itertools.product(available_keys, available_keys)

        index_pairs: List[Tuple[int, int]] = []
        for key_a, key_b in key_pairs_iterator:
            idx_a = available_key_to_index[key_a]
            idx_b = available_key_to_index[key_b]
            index_pairs.append((idx_a, idx_b))

        return index_pairs


# Experiment 1 - # Shared Epochs Already done! (messy versuion in evaluate.py file)


# Experiment 2 - Permutation Ablation
def run_permutation_ablation_experiment(
    mm: ModelManager,  # Use Any or the actual type ModelManager
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

    # Filter out model pairs that have already been computed (checked by canonical key)
    pairs_to_process = [
        pair
        for pair in model_pairs
        if canonical_key(mm.get_key(pair[0]), mm.get_key(pair[1])) not in existing_canonical_keys
    ]
    print(f"Total pairs to process: {len(pairs_to_process)}")

    dataloader_cifar100_test = get_cifar100_loaders(batch_size=128, num_workers=4)[2]  # Test loader

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
            device=mm.device,  # Use mm.device for consistency
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

        print(f"  ✅ Saved (Acc: {acc:.2f}, Loss: {loss:.4f})")

    print("\nExperiment run completed.")
    # Save final CSV and parquet file
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

            # Cannonical order for keys
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


# Experiment 4 - Transitivity Prediction - Also already done !

if __name__ == "__main__":
    mm = ModelManager(model_dir=SOUP_DIR, pattern=fine_tuned_pattern, device=DEVICE)
    # Example usage of get_index_pairs
    index_pairs = mm.get_index_pairs(
        model_ids=[1, 2, 3, 4],
        epochs=list(range(10, 301, 10)),
        unique=True,
    )
    print(f"Total unique model pairs: {len(index_pairs)}")

    experiments_to_run = [2]  # Specify which experiments to run

    # Experiment 2 - Permutation Ablation
    if 2 in experiments_to_run:
        run_permutation_ablation_experiment(mm, index_pairs, output_csv="permutation_ablation_results.csv")

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
            csv_output="similarity_metrics.csv",
        )
    # compare to rows in
