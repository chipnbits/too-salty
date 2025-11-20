"""Dataset loaders for CIFAR-100 and CIFAR-100-C with corruptions."""

import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# Load environment variables
load_dotenv()

# CIFAR-100 normalization constants
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# CIFAR-100-C download URL
CIFAR100C_URL = "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar"

# CIFAR-100-C corruption types
CORRUPTION_TYPES = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
    "speckle_noise",
    "gaussian_blur",
    "spatter",
    "saturate",
]


def get_cifar100_loaders(
    batch_size: int = 128,
    num_workers: int = 4,
    data_dir: Optional[str] = None,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-100 train and test DataLoaders with standard normalization.
    Assumes datasets have been pre-downloaded using build_dataset.py.

    Args:
        batch_size: Batch size for both train and test loaders
        num_workers: Number of worker processes for data loading
        data_dir: Directory containing CIFAR-100 data (default: from DATA_DIR env var or './data')
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        Tuple of (train_loader, test_loader)
    """
    if data_dir is None:
        data_dir = os.getenv("DATA_DIR", "./data")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])

    train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=False, transform=transform)

    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=False, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, test_loader


class CIFAR100CDataset(Dataset):
    """
    CIFAR-100-C dataset with corruption support.

    CIFAR-100-C should be downloaded from:
    https://zenodo.org/record/3555552/files/CIFAR-100-C.tar
    or via the `build_dataset.py` script.

    Expected directory structure:
        data_dir/
            CIFAR-100-C/
                labels.npy
                gaussian_noise.npy
                shot_noise.npy
                ...
    """

    def __init__(
        self, data_dir: str, corruption_type: str, severity: int = 5, transform: Optional[transforms.Compose] = None
    ):
        """
        Initialize CIFAR-100-C dataset for a specific corruption.

        Args:
            data_dir: Root directory containing CIFAR-100-C folder
            corruption_type: Type of corruption (see CORRUPTION_TYPES)
            severity: Corruption severity level (1-5, where 5 is most severe)
            transform: Optional transforms to apply
        """
        if corruption_type not in CORRUPTION_TYPES:
            raise ValueError(f"Invalid corruption type '{corruption_type}'. " f"Must be one of: {CORRUPTION_TYPES}")

        if not 1 <= severity <= 5:
            raise ValueError(f"Severity must be between 1 and 5, got {severity}")

        cifar_c_dir = Path(data_dir) / "CIFAR-100-C"

        # Load corruption data
        corruption_path = cifar_c_dir / f"{corruption_type}.npy"
        if not corruption_path.exists():
            raise FileNotFoundError(
                f"Corruption file not found: {corruption_path}\n"
                f"Please download CIFAR-100-C from: "
                f"{CIFAR100C_URL}"
                f" or run the build_dataset.py script."
            )

        # Load labels
        labels_path = cifar_c_dir / "labels.npy"
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        # Each .npy file contains all 5 severity levels (50000 images)
        # Images are organized as: [severity_1, severity_2, ..., severity_5]
        # Each severity level has 10000 images
        all_data = np.load(corruption_path)
        all_labels = np.load(labels_path)

        # Extract data for the specified severity level
        start_idx = (severity - 1) * 10000
        end_idx = severity * 10000

        self.data = all_data[start_idx:end_idx]
        self.targets = all_labels[start_idx:end_idx]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, target = self.data[idx], int(self.targets[idx])

        # Convert to PIL Image for transform compatibility
        from PIL import Image

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def get_cifar100c_loader(
    batch_size: int = 128,
    num_workers: int = 4,
    data_dir: Optional[str] = None,
    corruption_type: Optional[str] = None,
    severity: int = 5,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Get CIFAR-100-C test DataLoader with corruptions.

    If corruption_type is None, returns a DataLoader that iterates through
    all corruption types. Otherwise, returns a DataLoader for the specific
    corruption type.

    Args:
        batch_size: Batch size for the loader
        num_workers: Number of worker processes for data loading
        data_dir: Directory containing CIFAR-100-C data (default: from DATA_DIR env var or './data')
        corruption_type: Specific corruption type, or None for all types
        severity: Corruption severity level (1-5)
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        DataLoader for CIFAR-100-C test data
    """
    if data_dir is None:
        data_dir = os.getenv("DATA_DIR", "./data")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])

    if corruption_type is not None:
        # Single corruption type
        dataset = CIFAR100CDataset(
            data_dir=data_dir, corruption_type=corruption_type, severity=severity, transform=transform
        )
    else:
        # All corruption types - concatenate them
        datasets_list = []
        for corr_type in CORRUPTION_TYPES:
            try:
                ds = CIFAR100CDataset(
                    data_dir=data_dir, corruption_type=corr_type, severity=severity, transform=transform
                )
                datasets_list.append(ds)
            except FileNotFoundError as e:
                print(f"Warning: Skipping {corr_type}: {e}")
                continue

        if not datasets_list:
            raise RuntimeError(
                "No corruption datasets found. Please download CIFAR-100-C from: "
                "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar"
            )

        # Concatenate all corruption datasets
        from torch.utils.data import ConcatDataset

        dataset = ConcatDataset(datasets_list)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return loader


def get_cifar100c_loaders_by_corruption(
    batch_size: int = 128,
    num_workers: int = 4,
    data_dir: Optional[str] = None,
    severity: int = 5,
    pin_memory: bool = True,
) -> dict[str, DataLoader]:
    """
    Get a dictionary of CIFAR-100-C DataLoaders, one for each corruption type.

    This is useful when you want to evaluate performance on each corruption
    type separately.

    Args:
        batch_size: Batch size for all loaders
        num_workers: Number of worker processes for data loading
        data_dir: Directory containing CIFAR-100-C data (default: from DATA_DIR env var or './data')
        severity: Corruption severity level (1-5)
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        Dictionary mapping corruption type names to DataLoaders
    """
    if data_dir is None:
        data_dir = os.getenv("DATA_DIR", "./data")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])

    loaders = {}
    for corr_type in CORRUPTION_TYPES:
        try:
            dataset = CIFAR100CDataset(
                data_dir=data_dir, corruption_type=corr_type, severity=severity, transform=transform
            )
            loaders[corr_type] = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
            )
        except FileNotFoundError as e:
            print(f"Warning: Skipping {corr_type}: {e}")
            continue

    if not loaders:
        raise RuntimeError(
            "No corruption datasets found. Please download CIFAR-100-C from: "
            "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar"
        )

    return loaders
