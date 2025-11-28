"""Tests for permute_models, which aligns two ResNet models by permuting channels."""

import copy

import pytest
import torch
import torch.nn as nn

pytest.importorskip("rebasin")

from salty.resnet import ResNet, resnet18, resnet50
from salty.permute_model import permute_models


def _models_equal(m1: nn.Module, m2: nn.Module, atol: float = 1e-6) -> bool:
    sd1 = m1.state_dict()
    sd2 = m2.state_dict()
    if sd1.keys() != sd2.keys():
        return False
    for k in sd1.keys():
        if not torch.allclose(sd1[k], sd2[k], atol=atol):
            return False
    return True


def _outputs_equal(m1: nn.Module, m2: nn.Module, x: torch.Tensor, atol: float = 1e-6) -> bool:
    m1.eval()
    m2.eval()
    with torch.no_grad():
        y1 = m1(x)
        y2 = m2(x)
    return torch.allclose(y1, y2, atol=atol)


@pytest.mark.parametrize("constructor", [resnet18, resnet50])
def test_permutes_returns_new_model_without_modifying_original(constructor) -> None:
    torch.manual_seed(0)
    a = constructor()
    b = constructor()

    # Save original copy of b for later comparison
    b_original = copy.deepcopy(b)

    b_perm = permute_models(a, b)

    # Returned model should be a different object
    assert b_perm is not b

    # Original b should be unchanged
    assert _models_equal(b, b_original)

    # Architecture (parameter keys) should be preserved
    assert list(b_perm.state_dict().keys()) == list(b.state_dict().keys())


@pytest.mark.parametrize("constructor", [resnet18, resnet50])
def test_identical_models_yield_equivalent_model(constructor) -> None:
    torch.manual_seed(0)

    # Start from a single model, clone weights so they are exactly identical
    a = constructor()
    b = copy.deepcopy(a)

    # Sanity: outputs identical before permutation
    x = torch.randn(4, 3, 32, 32)
    assert _outputs_equal(a, b, x)

    b_perm = permute_models(a, b)

    # After permutation, b_perm should still be functionally identical to a
    assert _outputs_equal(a, b_perm, x)

    # b itself should still match the original a (permute_models must not touch it)
    assert _outputs_equal(a, b, x)


def test_type_and_device_are_preserved() -> None:
    torch.manual_seed(0)

    a = resnet50().to("cpu")
    b = resnet50().to("cpu")

    b_perm = permute_models(a, b)

    # Same class
    assert isinstance(b_perm, ResNet)

    # Same device as original b
    devices_b = {p.device for p in b.parameters()}
    devices_b_perm = {p.device for p in b_perm.parameters()}
    assert devices_b == devices_b_perm == {torch.device("cpu")}


def test_raises_on_mismatched_architectures() -> None:
    torch.manual_seed(0)

    a = resnet18()
    b = resnet50()

    with pytest.raises(TypeError):
        _ = permute_models(a, b)
