from typing import Dict, Optional, Tuple, Iterable
import itertools

import torch
import torch.nn.functional as F


def _linear_cka(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r"""Linear CKA between two representation matrices.

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


def logit_mse_kl(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
) -> Dict[str, float]:
    r"""Compute MSE and symmetric KL between two logit matrices.

    logits_*: [n, c]
    Returns:
        {
            'n_samples': n,
            'mse': mean squared error per sample,
            'kl': symmetric KL per sample,
        }
    """
    if logits_a.shape != logits_b.shape:
        raise ValueError(f"Logit shapes differ: {logits_a.shape} vs {logits_b.shape}")

    if logits_a.numel() == 0:
        return {"n_samples": 0.0, "mse": 0.0, "kl": 0.0}

    # ensure float
    logits_a = logits_a.float()
    logits_b = logits_b.float()

    n = logits_a.shape[0]

    diff = logits_a - logits_b
    # sum over classes, then average over samples
    mse = float(diff.pow(2).sum(dim=1).mean().item())

    log_p = F.log_softmax(logits_a, dim=1)
    log_q = F.log_softmax(logits_b, dim=1)
    p = log_p.exp()
    q = log_q.exp()

    kl_pq = (p * (log_p - log_q)).sum(dim=1)
    kl_qp = (q * (log_q - log_p)).sum(dim=1)
    skl = 0.5 * (kl_pq + kl_qp)

    kl_mean = float(skl.mean().item())

    return {"n_samples": float(n), "mse": mse, "kl": kl_mean}


def cka_similarity(
    rep_a: torch.Tensor,
    rep_b: torch.Tensor,
) -> float:
    r"""Convenience wrapper: linear CKA between two representation matrices.

    rep_*: [n, d]
    """
    if rep_a.shape[0] != rep_b.shape[0]:
        raise ValueError(f"Number of samples differ: {rep_a.shape[0]} vs {rep_b.shape[0]}")
    # cast to float and move to cpu for safety
    rep_a = rep_a.detach().float().cpu()
    rep_b = rep_b.detach().float().cpu()
    return float(_linear_cka(rep_a, rep_b).item())
