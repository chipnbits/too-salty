"""Similarity metrics between two resnet models.

Provided functions
- ``l2_distance(model_a, model_b, ...)``: returns SSE, L2, RMSE
- ``cosine_similarity(...)``: returns cosine similarity between two models
- ``logit_mse_kl(...)``: returns MSE and KL divergence between two models' logits on random inputs
- ``probe_set_similarities(...)``: returns logit MSE/KL and CKA similarities
- ``compare_resnet50_pair(...)``: helper that compares two ResNet-50
  instances via `get_resnet50_model` (useful for quick checks).
"""

from typing import Dict, Iterable, Optional, Tuple, Generator

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .models import get_resnet50_model
from .resnet import ResNet


def _default_excluded_suffixes() -> Tuple[str, ...]:
    return ("running_mean", "running_var", "num_batches_tracked")


def _iter_common_tensors(
    model_a: ResNet,
    model_b: ResNet,
    excluded_suffixes: Optional[Iterable[str]] = None,
) -> Generator[Tuple[str, torch.Tensor, torch.Tensor], None, None]:
    """Yield (key, tensor_a_cpu, tensor_b_cpu) for comparable state_dict entries.

    This centralises the filtering logic so other metrics can reuse it.
    """
    if excluded_suffixes is None:
        excluded_suffixes = _default_excluded_suffixes()

    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()

    for key, ta in sd_a.items():
        if any(key.endswith(suf) for suf in excluded_suffixes):
            continue
        tb = sd_b.get(key)
        if tb is None:
            continue
        if ta.shape != tb.shape:
            continue

        yield key, ta.detach().float().cpu(), tb.detach().float().cpu()


def l2_distance(
    model_a: ResNet,
    model_b: ResNet,
    excluded_suffixes: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """Compute L2 statistics between two models' state_dict entries.

    Uses the shared iterator to avoid duplicating filtering logic.
    """
    sse = 0.0
    total_elems = 0

    for _, ta_f, tb_f in _iter_common_tensors(model_a, model_b, excluded_suffixes):
        diff = ta_f - tb_f
        sse += float((diff * diff).sum().item())
        total_elems += diff.numel()

    l2 = float(sse**0.5)
    rmse = float((sse / total_elems) ** 0.5) if total_elems > 0 else 0.0

    return {"sse": sse, "l2": l2, "rmse": rmse, "n_elements": total_elems}


def cosine_similarity(
    model_a: ResNet,
    model_b: ResNet,
    excluded_suffixes: Optional[Iterable[str]] = None,
) -> float:
    """Compute cosine similarity between two models' state_dict entries.

    Uses the same filtering as ``l2_distance`` via the shared iterator.
    """
    dot_product = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for _, ta_f, tb_f in _iter_common_tensors(model_a, model_b, excluded_suffixes):
        dot_product += float((ta_f * tb_f).sum().item())
        norm_a += float((ta_f * ta_f).sum().item())
        norm_b += float((tb_f * tb_f).sum().item())

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / ((norm_a**0.5) * (norm_b**0.5))


def _linear_cka(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Linear CKA between two representation matrices.

    x: [n, d1], y: [n, d2]
    """
    if x.shape[0] < 2:
        return x.new_tensor(0.0)

    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    x_ty = x.t() @ y

    num = (x_ty * x_ty).sum()

    x_tx = x.t() @ x
    y_ty = y.t() @ y
    denom = (x_tx * x_tx).sum().sqrt() * (y_ty * y_ty).sum().sqrt()

    if denom == 0:
        return x.new_tensor(0.0)

    return num / denom


def probe_set_similarities(
    model_a: ResNet,
    model_b: ResNet,
    probe_set: DataLoader,
    *,
    compute_logit_mse_kl: bool = True,
    compute_cka_logits: bool = True,
    compute_cka_features: bool = True,
) -> Dict[str, float]:
    """Compute probe-set similarities between two models.

    Computes, depending on flags:
        - 'mse': mean squared error between logits
        - 'kl': symmetric KL divergence between softmax outputs
        - 'cka_logits': linear CKA on logits
        - 'cka_features': linear CKA on feature representations
        - 'n_samples': total number of probe samples used
    """
    device = next(model_a.parameters()).device

    mse_sum = 0.0
    kl_sum = 0.0
    n_total = 0

    logits_a_all: list[torch.Tensor] = []
    logits_b_all: list[torch.Tensor] = []
    feats_a_all: list[torch.Tensor] = []
    feats_b_all: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in probe_set:
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch

            x = x.to(device, non_blocking=True)

            # Single conv pass per model: features then logits
            feats_a = model_a.forward_features(x)
            feats_b = model_b.forward_features(x)

            logits_a = model_a.fc(feats_a)
            logits_b = model_b.fc(feats_b)

            if logits_a.shape != logits_b.shape:
                raise ValueError(f"Logit shapes differ: {logits_a.shape} vs {logits_b.shape}")

            batch_size = logits_a.shape[0]
            n_total += batch_size

            # ----- Logit MSE / KL -----
            if compute_logit_mse_kl:
                diff = logits_a - logits_b
                mse_sum += float((diff.pow(2).sum(dim=1)).sum().item())

                log_p = F.log_softmax(logits_a, dim=1)
                log_q = F.log_softmax(logits_b, dim=1)
                p = log_p.exp()
                q = log_q.exp()

                kl_pq = (p * (log_p - log_q)).sum(dim=1)
                kl_qp = (q * (log_q - log_p)).sum(dim=1)
                skl = 0.5 * (kl_pq + kl_qp)
                kl_sum += float(skl.sum().item())

            # ----- CKA on logits -----
            if compute_cka_logits:
                logits_a_all.append(logits_a.detach().cpu())
                logits_b_all.append(logits_b.detach().cpu())

            # ----- CKA on penultimate features -----
            if compute_cka_features:
                feats_a_all.append(feats_a.detach().cpu())
                feats_b_all.append(feats_b.detach().cpu())

    result: Dict[str, float] = {"n_samples": float(n_total)}

    if n_total == 0:
        if compute_logit_mse_kl:
            result["mse"] = 0.0
            result["kl"] = 0.0
        if compute_cka_logits:
            result["cka_logits"] = 0.0
        if compute_cka_features:
            result["cka_features"] = 0.0
        return result

    if compute_logit_mse_kl:
        result["mse"] = mse_sum / n_total
        result["kl"] = kl_sum / n_total

    if compute_cka_logits:
        logits_a_full = torch.cat(logits_a_all, dim=0)
        logits_b_full = torch.cat(logits_b_all, dim=0)
        result["cka_logits"] = float(_linear_cka(logits_a_full, logits_b_full).item())

    if compute_cka_features:
        feats_a_full = torch.cat(feats_a_all, dim=0)
        feats_b_full = torch.cat(feats_b_all, dim=0)
        result["cka_features"] = float(_linear_cka(feats_a_full, feats_b_full).item())

    return result


def compare_resnet50_pair(
    a: ResNet,
    b: ResNet,
    permute: bool = False,
    probe_set: Optional[DataLoader] = None,
    *,
    compute_logit_mse_kl: bool = True,
    compute_cka_logits: bool = True,
    compute_cka_features: bool = True,
) -> Dict[str, float]:
    """Compare two ResNet-50 models across all metrics."""
    if permute:
        raise NotImplementedError("Permutation not implemented yet.")

    stats: Dict[str, float] = {}
    stats["l2_distance"] = l2_distance(a, b)["l2"]
    stats["cosine_similarity"] = cosine_similarity(a, b)

    if probe_set is not None and (compute_logit_mse_kl or compute_cka_logits or compute_cka_features):
        probe_stats = probe_set_similarities(
            a,
            b,
            probe_set,
            compute_logit_mse_kl=compute_logit_mse_kl,
            compute_cka_logits=compute_cka_logits,
            compute_cka_features=compute_cka_features,
        )

        if compute_logit_mse_kl:
            stats["logit_mse"] = probe_stats.get("mse", 0.0)
            stats["logit_kl"] = probe_stats.get("kl", 0.0)

        if compute_cka_logits:
            stats["cka_logits"] = probe_stats.get("cka_logits", 0.0)

        if compute_cka_features:
            stats["cka_features"] = probe_stats.get("cka_features", 0.0)

    return stats
