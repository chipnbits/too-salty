""" Model architectures available for training CIFAR-100"""

from salty.resnet import resnet50


def get_resnet50_model(num_classes: int = 100):
    """Returns a ResNet-50 model adjusted for CIFAR-100."""
    return resnet50()


def get_vit_model(num_classes: int = 100):
    """Returns a Vision Transformer model adjusted for CIFAR-100."""
    return NotImplementedError("ViT model is not implemented yet.")
