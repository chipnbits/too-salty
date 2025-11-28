import math

import torch
from torch.utils.data import DataLoader, TensorDataset

import pytest

from salty.resnet import resnet18, resnet50
from salty.similarity_metrics import (
    l2_distance,
    cosine_similarity,
    probe_set_similarities,
    compare_resnet50_pair,
)


def _make_probe_loader(
    n_samples: int = 64,
    batch_size: int = 16,
    image_size: int = 32,
) -> DataLoader:
    torch.manual_seed(0)
    x = torch.randn(n_samples, 3, image_size, image_size)
    # no labels needed; similarity code only uses inputs
    dataset: TensorDataset = TensorDataset(x)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_l2_and_cosine_identical_models():
    torch.manual_seed(0)
    m1 = resnet18()
    m2 = resnet18()
    m2.load_state_dict(m1.state_dict())

    dist = l2_distance(m1, m2)
    cos = cosine_similarity(m1, m2)

    assert dist["l2"] == pytest.approx(0.0, abs=1e-6)
    assert dist["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert cos == pytest.approx(1.0, abs=1e-6)


def test_l2_and_cosine_different_models():
    torch.manual_seed(0)
    m1 = resnet18()
    torch.manual_seed(1)
    m2 = resnet18()

    dist = l2_distance(m1, m2)
    cos = cosine_similarity(m1, m2)

    assert dist["l2"] > 0.0
    # should not be almost perfectly aligned
    assert cos < 0.999


def test_probe_set_similarities_identical_models():
    torch.manual_seed(0)
    m1 = resnet18()
    m2 = resnet18()
    m2.load_state_dict(m1.state_dict())

    loader = _make_probe_loader()

    sims = probe_set_similarities(
        m1,
        m2,
        loader,
        compute_logit_mse_kl=True,
        compute_cka_logits=True,
        compute_cka_features=True,
    )

    n_samples = len(loader.dataset)  # type: ignore
    assert sims["n_samples"] == pytest.approx(float(n_samples))

    # identical models, same inputs: logits identical
    assert sims["mse"] == pytest.approx(0.0, abs=1e-6)
    assert sims["kl"] == pytest.approx(0.0, abs=1e-6)

    # CKA on identical representations should be very close to 1
    assert sims["cka_logits"] == pytest.approx(1.0, rel=1e-4, abs=1e-4)
    assert sims["cka_features"] == pytest.approx(1.0, rel=1e-4, abs=1e-4)


def test_probe_set_similarities_different_models_reasonable_values():
    torch.manual_seed(0)
    m1 = resnet18()
    torch.manual_seed(1)
    m2 = resnet18()

    loader = _make_probe_loader()

    sims = probe_set_similarities(
        m1,
        m2,
        loader,
        compute_logit_mse_kl=True,
        compute_cka_logits=True,
        compute_cka_features=True,
    )

    assert sims["n_samples"] == pytest.approx(float(len(loader.dataset)))  # type: ignore

    # different random weights: distances strictly positive
    assert sims["mse"] > 0.0
    assert sims["kl"] > 0.0

    # CKA similarities should be in [0, 1]
    assert 0.0 <= sims["cka_logits"] <= 1.0
    assert 0.0 <= sims["cka_features"] <= 1.0


def test_probe_set_similarities_flags_disable_metrics():
    torch.manual_seed(0)
    m1 = resnet18()
    m2 = resnet18()
    loader = _make_probe_loader()

    sims = probe_set_similarities(
        m1,
        m2,
        loader,
        compute_logit_mse_kl=False,
        compute_cka_logits=False,
        compute_cka_features=False,
    )

    # only n_samples should be present
    assert "n_samples" in sims
    assert len(sims) == 1


def test_probe_set_similarities_empty_probe_set():
    torch.manual_seed(0)
    m1 = resnet18()
    m2 = resnet18()

    # empty dataset
    x = torch.empty(0, 3, 32, 32)
    loader = DataLoader(TensorDataset(x), batch_size=16)

    sims = probe_set_similarities(
        m1,
        m2,
        loader,
        compute_logit_mse_kl=True,
        compute_cka_logits=True,
        compute_cka_features=True,
    )

    assert sims["n_samples"] == 0.0
    assert sims["mse"] == 0.0
    assert sims["kl"] == 0.0
    assert sims["cka_logits"] == 0.0
    assert sims["cka_features"] == 0.0


def test_compare_resnet50_pair_identical_models():
    torch.manual_seed(0)
    a = resnet50()
    b = resnet50()
    b.load_state_dict(a.state_dict())

    loader = _make_probe_loader()

    stats = compare_resnet50_pair(
        a,
        b,
        probe_set=loader,
        compute_logit_mse_kl=True,
        compute_cka_logits=True,
        compute_cka_features=True,
    )

    # state-dict identical
    assert stats["l2_distance"] == pytest.approx(0.0, abs=1e-6)
    assert stats["cosine_similarity"] == pytest.approx(1.0, abs=1e-6)

    # logits identical
    assert stats["logit_mse"] == pytest.approx(0.0, abs=1e-6)
    assert stats["logit_kl"] == pytest.approx(0.0, abs=1e-6)

    # CKA ~ 1 for identical reps
    assert stats["cka_logits"] == pytest.approx(1.0, rel=1e-4, abs=1e-4)
    assert stats["cka_features"] == pytest.approx(1.0, rel=1e-4, abs=1e-4)


def test_compare_resnet50_pair_flags_control_outputs():
    torch.manual_seed(0)
    a = resnet50()
    b = resnet50()
    loader = _make_probe_loader()

    stats = compare_resnet50_pair(
        a,
        b,
        probe_set=loader,
        compute_logit_mse_kl=False,
        compute_cka_logits=True,
        compute_cka_features=False,
    )

    # always present
    assert "l2_distance" in stats
    assert "cosine_similarity" in stats

    # disabled
    assert "logit_mse" not in stats
    assert "logit_kl" not in stats

    # enabled
    assert "cka_logits" in stats
    assert "cka_features" not in stats
