"""
This file runs provides functions to run various experiments for the project.
"""

from typing import Dict, Tuple
from torch import Tensor
import torch
from torch.utils.data import DataLoader
import pandas as pd

from salty.similarity_metrics_from_logits import cka_similarity, logit_mse_kl
from salty.similarity_metrics_from_models import cosine_similarity, l2_distance
from salty.permute_model import permute_models
from salty.datasets import get_cifar100_loaders

# Experiment 1 - # Shared Epochs Already done!

# Experiment 2 - Permutation Ablation


def run_permutation_ablation_experiment(N=1000):
    for i in range(N):
        model_a, model_b = sample_model_pair()
        permuted_model_b = permute_models(model_a, model_b)
        # Run souping experiment with model_a and permuted_model_b


# Experiment 3 - Predicting Soupability


def record_logits_features(dataloader: DataLoader) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    models = []
    logits = {}
    penultimate_activations = {}
    for model in models:
        model_logits = []
        model_penultimate = []
        for inputs, _ in dataloader:
            with torch.no_grad():
                outputs, penultimate = model(inputs)
            model_logits.append(outputs)
            model_penultimate.append(penultimate)
        logits[model.key] = torch.cat(model_logits, dim=0)
        penultimate_activations[model.key] = torch.cat(model_penultimate, dim=0)
    return logits, penultimate_activations


def record_similarity_metrics(logits: Dict[str, Tensor], penultimate_activations: Dict[str, Tensor]) -> None:
    models = []
    rows = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model_a = models[i]
            model_b = models[j]
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
            row = {
                "model_a": model_a.name,
                "model_b": model_b.name,
                "l2_distance": l2,
                "cosine_similarity": cosine,
                "cka_logits": cka_logits,
                "mse_logits": mse_logits,
                "kl_logits": kl_logits,
                "cka_features": cka_features,
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("similarity_metrics.csv", index=False)


def run_soupability_prediction_experiment():
    _, _, test_loader = get_cifar100_loaders()
    logits, features = record_logits_features(test_loader)
    record_similarity_metrics(logits, features)


# Experiment 4 - Transitivity Prediction - Also already done !
