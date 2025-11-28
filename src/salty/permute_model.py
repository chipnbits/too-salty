"""Permutation alignment of two ResNet models so that the neurones are as closely aligned as possible."""

import copy
from typing import Tuple

import torch
import torch.nn as nn
from rebasin import PermutationCoordinateDescent

from .resnet import ResNet


def _get_device(model: nn.Module) -> torch.device:
    params = list(model.parameters())
    if not params:
        return torch.device("cpu")
    return params[0].device


def _make_dummy_input(model: nn.Module) -> torch.Tensor:
    """Create a dummy input tensor that is compatible with the model."""
    in_channels = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            in_channels = module.in_channels
            break

    if in_channels is None:
        raise ValueError("Could not infer input channels from model (no Conv2d found).")

    # Spatial size 32x32 works for CIFAR-style ResNets and is safe for typical configurations.
    # If your model expects a different resolution, you can change this.
    return torch.randn(1, in_channels, 32, 32)


def _prepare_models(
    a: ResNet,
    b: ResNet,
) -> Tuple[ResNet, ResNet, torch.Tensor]:
    """Return deep-copied models on a common device and a dummy input for tracing."""
    if type(a) is not type(b):
        raise TypeError("permute_models expects models of the same class and architecture.")

    device_a = _get_device(a)
    device_b = _get_device(b)

    # Use device of model a as the working device
    work_device = device_a

    a_work = copy.deepcopy(a).to(work_device)
    b_work = copy.deepcopy(b).to(work_device)

    dummy_input = _make_dummy_input(a_work).to(work_device)

    # We do not move the originals; only the copies are modified
    return a_work, b_work, dummy_input.to(work_device)


def permute_models(a: ResNet, b: ResNet) -> ResNet:
    """
    Return a copy of model b whose neurons (channels) have been permuted
    to align as closely as possible with model a, using Git Re-Basin style
    weight-matching (PermutationCoordinateDescent).

    The original models a and b are left unchanged.

    Both models must:
      - Have identical architecture (same class, same layer configuration).
      - Be compatible with the same input shape.

    Parameters
    ----------
    a : ResNet
        Reference model that stays in its original channel ordering.
    b : ResNet
        Model to be permuted to match a.

    Returns
    -------
    ResNet
        A deep copy of b, with its parameters permuted in place.
    """

    # Require same class
    if type(a) is not type(b):
        raise TypeError("permute_models expects models of the same class.")

    # Require identical parameter shapes
    params_a = list(a.parameters())
    params_b = list(b.parameters())
    if len(params_a) != len(params_b) or any(pa.shape != pb.shape for pa, pb in zip(params_a, params_b)):
        raise TypeError("permute_models expects models with identical parameter shapes.")

    # Prepare deep-copied working models on a common device
    a_work, b_work, dummy_input = _prepare_models(a, b)

    # Run permutation coordinate descent to align b_work with a_work
    pcd = PermutationCoordinateDescent(a_work, b_work, dummy_input)
    pcd.rebasin()  # mutates b_work in place

    # Move the permuted copy back to the original device of b
    target_device = _get_device(b)
    b_permuted = b_work.to(target_device)

    return b_permuted
