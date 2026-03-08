"""Unified ModelManager for discovering, loading, souping, and evaluating models.

Supports two filesystem layouts:
  - Directory layout: ``*-epoch_{N}_model_{M}/best_model.pt``
  - Flat file layout: ``epoch_{N}_model_{M}.pt`` (legacy)

Example usage::

    from salty.model_manager import ModelManager

    mm = ModelManager("./models/cifar100-resnet50", device="cuda")

    # Iterate over available model entries (no loading)
    for entry in mm.entries:
        print(entry["key"], entry["epoch"], entry["model_id"])

    # Load a model by index or key
    model = mm[0]
    model = mm.load("50_1")

    # Soup two models
    souped = mm.soup("50_1", "50_2", alpha=0.5)

    # Evaluate
    acc, loss = mm.evaluate(souped, test_loader)

    # Get pairwise index pairs for experiments
    pairs = mm.get_index_pairs(model_ids=[1,2,3,4], epochs=range(50,310,10))
"""

import itertools
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from salty.evaluation import (
    evaluate_model,
    fine_tuned_dir_pattern,
    fine_tuned_pattern,
    load_model_from_checkpoint,
)
from salty.utils import soup_models


class ModelManager:
    """Lazy-loading collection of finetuned model checkpoints.

    Models are discovered on construction but only loaded from disk when
    accessed via ``__getitem__``, ``load()``, or ``soup()``.
    """

    def __init__(
        self,
        models_dir: str,
        device: Union[str, torch.device] = "cpu",
        model_name: str = "resnet50",
        num_classes: int = 100,
        dropout: float = 0.0,
    ):
        self.models_dir = models_dir
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout = dropout

        self._entries: List[Dict] = []
        self._key_to_index: Dict[str, int] = {}
        self._discover_models()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover_models(self) -> None:
        """Scan *models_dir* for checkpoints in either directory or flat-file layout."""
        entries = []

        for entry_name in os.listdir(self.models_dir):
            # Directory layout: *-epoch_{N}_model_{M}/best_model.pt
            dir_match = fine_tuned_dir_pattern.match(entry_name)
            if dir_match:
                best_path = os.path.join(self.models_dir, entry_name, "best_model.pt")
                if os.path.exists(best_path):
                    epoch, model_id = map(int, dir_match.groups())
                    entries.append({
                        "key": f"{epoch}_{model_id}",
                        "path": best_path,
                        "epoch": epoch,
                        "model_id": model_id,
                    })
                continue

            # Flat file layout: epoch_{N}_model_{M}.pt
            file_match = fine_tuned_pattern.match(entry_name)
            if file_match:
                filepath = os.path.join(self.models_dir, entry_name)
                epoch, model_id = map(int, file_match.groups())
                entries.append({
                    "key": f"{epoch}_{model_id}",
                    "path": filepath,
                    "epoch": epoch,
                    "model_id": model_id,
                })

        entries.sort(key=lambda e: (e["epoch"], e["model_id"]))
        self._entries = entries
        self._key_to_index = {e["key"]: i for i, e in enumerate(entries)}

    # ------------------------------------------------------------------
    # Collection interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> nn.Module:
        """Load and return the model at *index*, with metadata attributes attached."""
        if index < 0 or index >= len(self._entries):
            raise IndexError(f"Model index {index} out of range [0, {len(self._entries)})")
        entry = self._entries[index]
        model = self._load_model(entry["path"])
        model.key = entry["key"]
        model.epoch = entry["epoch"]
        model.model_id = entry["model_id"]
        return model

    def __iter__(self):
        self._iter_index = 0
        return self

    def __next__(self) -> nn.Module:
        if self._iter_index >= len(self._entries):
            raise StopIteration
        model = self[self._iter_index]
        self._iter_index += 1
        return model

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load_model(self, path: str) -> nn.Module:
        """Load a single checkpoint, infer architecture from config if available."""
        model, _ = load_model_from_checkpoint(
            path, self.device, model_name=self.model_name, num_classes=self.num_classes
        )
        return model

    def load(self, key_or_index: Union[str, int]) -> nn.Module:
        """Load a model by its string key (e.g. ``"50_1"``) or integer index."""
        if isinstance(key_or_index, str):
            index = self.get_index(key_or_index)
            if index < 0:
                raise KeyError(f"No model found with key '{key_or_index}'")
            return self[index]
        return self[key_or_index]

    # ------------------------------------------------------------------
    # Souping
    # ------------------------------------------------------------------

    def soup(
        self,
        key_a: Union[str, int],
        key_b: Union[str, int],
        alpha: float = 0.5,
    ) -> nn.Module:
        """Load two models, soup them, and return the result on *self.device*."""
        model_a = self.load(key_a)
        model_b = self.load(key_b)
        souped = soup_models(model_a, model_b, alpha=alpha)
        souped.to(self.device)
        return souped

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model_or_key: Union[nn.Module, str, int],
        dataloader,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tuple[float, float]:
        """Evaluate a model (or load one by key/index), returning ``(accuracy, loss)``."""
        if isinstance(model_or_key, (str, int)):
            model = self.load(model_or_key)
        else:
            model = model_or_key
        dev = device or self.device
        return evaluate_model(model, dataloader, dev)

    # ------------------------------------------------------------------
    # Metadata accessors
    # ------------------------------------------------------------------

    @property
    def entries(self) -> List[Dict]:
        """List of ``{"key", "path", "epoch", "model_id"}`` dicts (no loading)."""
        return list(self._entries)

    @property
    def keys(self) -> List[str]:
        """All available model keys, sorted by (epoch, model_id)."""
        return [e["key"] for e in self._entries]

    @property
    def model_paths(self) -> List[Path]:
        """All model checkpoint paths."""
        return [Path(e["path"]) for e in self._entries]

    @property
    def model_keys(self) -> List[str]:
        """Alias for *keys* — backward compatibility with old ModelManager."""
        return self.keys

    def get_key(self, index: int) -> str:
        if index < 0 or index >= len(self._entries):
            raise IndexError(f"Model index {index} out of range")
        return self._entries[index]["key"]

    def get_index(self, key: str) -> int:
        return self._key_to_index.get(key, -1)

    def get_epoch(self, index: int) -> Optional[int]:
        return self._entries[index]["epoch"]

    def get_model_id(self, index: int) -> Optional[int]:
        return self._entries[index]["model_id"]

    # ------------------------------------------------------------------
    # Pairing utilities
    # ------------------------------------------------------------------

    def get_index_pairs(
        self,
        model_ids: List[int] = [1, 2, 3, 4],
        epochs: List[int] = list(range(10, 301, 10)),
        unique: bool = True,
    ) -> List[Tuple[int, int]]:
        """Return index pairs for all available models matching *epochs* and *model_ids*.

        If *unique* is True, returns combinations_with_replacement (no duplicates).
        """
        available_keys = sorted(
            k for k in self._key_to_index
            if any(k == f"{e}_{m}" for e, m in itertools.product(epochs, model_ids))
        )

        if unique:
            key_pairs = itertools.combinations_with_replacement(available_keys, 2)
        else:
            key_pairs = itertools.product(available_keys, available_keys)

        return [
            (self._key_to_index[ka], self._key_to_index[kb])
            for ka, kb in key_pairs
        ]

    def get_key_pairs(
        self,
        model_ids: List[int] = [1, 2, 3, 4],
        epochs: List[int] = list(range(10, 301, 10)),
        unique: bool = True,
    ) -> List[Tuple[str, str]]:
        """Like ``get_index_pairs`` but returns key-string tuples."""
        index_pairs = self.get_index_pairs(model_ids, epochs, unique)
        return [(self.get_key(a), self.get_key(b)) for a, b in index_pairs]
