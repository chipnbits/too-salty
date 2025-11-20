import torch

from salty.datasets import get_cifar100_class_names, get_cifar100_loaders
from salty.models import get_resnet50_model
from salty.utils import denormalize_cifar100, normalize_cifar100, show_batch_with_labels

# TODO: add training loop, see https://github.com/weiaicunzai/pytorch-cifar100/blob/master/train.py for reference


if __name__ == "__main__":
    model = get_resnet50_model()
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")
    train_loader, test_loader = get_cifar100_loaders()
    label_names = get_cifar100_class_names()
    print(f"Class names: {label_names}")

    images, labels = next(iter(train_loader))
    print(f"Sample batch images shape: {images.shape}")
    print(f"Sample batch labels shape: {labels.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    images = images.to(device)
    labels = labels.to(device)
    # Run a forward pass with one batch
    outputs = model(images)
    print(f"Model outputs shape: {outputs.shape}")

    images_denorm = denormalize_cifar100(images)
    show_batch_with_labels(images_denorm, labels, label_names)
