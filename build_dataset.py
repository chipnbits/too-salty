"""Download CIFAR-100 and CIFAR-100-C datasets.

The CIFAR-100 dataset comes in train and test sets with 50,000 and 10,000 images
(32x32 color images with 100 classes).

CIFAR-100-C contains corrupted versions of the test set with 19 corruption types
at 5 severity levels each.
"""

import argparse
import os
import tarfile
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from torchvision import datasets

from salty.datasets import CIFAR100C_URL

# Load environment variables
load_dotenv()

parser = argparse.ArgumentParser(description="Download CIFAR-100 and CIFAR-100-C datasets")
parser.add_argument(
    "--skip-cifar100", action="store_true", help="Skip downloading CIFAR-100 (only download CIFAR-100-C)"
)
parser.add_argument(
    "--skip-cifar100c", action="store_true", help="Skip downloading CIFAR-100-C (only download CIFAR-100)"
)


def _download_file(url: str, dest: Path) -> None:
    """Download a file from URL to dest (simple wrapper)."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _progress(block_num: int, block_size: int, total_size: int):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded / total_size * 100
            print(f"\rDownloading {dest.name}: {percent:5.1f}%", end="")
        else:
            print(f"\rDownloading {dest.name}: {downloaded} bytes", end="")

    print(f"Downloading CIFAR-100-C from {url} to {dest} ...")
    urllib.request.urlretrieve(url, dest, _progress)
    print("\nDownload complete.")


def _extract_tar(archive_path: Path, extract_to: Path) -> None:
    """Extract a .tar archive to a directory."""
    print(f"Extracting {archive_path} to {extract_to} ...")
    with tarfile.open(archive_path, "r") as tar:
        tar.extractall(path=extract_to)
    print("Extraction complete.")


def ensure_cifar100c_downloaded(data_dir: str) -> Path:
    """
    Ensure CIFAR-100-C is downloaded and extracted under data_dir.

    Returns:
        Path to the CIFAR-100-C directory.
    """
    root = Path(data_dir)
    cifar_c_dir = root / "CIFAR-100-C"
    labels_path = cifar_c_dir / "labels.npy"

    # If labels exist, assume dataset is ready
    if labels_path.exists():
        return cifar_c_dir

    # If directory exists but labels don't, try to re-extract
    tar_path = root / "CIFAR-100-C.tar"

    if not tar_path.exists():
        # Need to download
        try:
            _download_file(CIFAR100C_URL, tar_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download CIFAR-100-C from {CIFAR100C_URL}: {e}") from e

    # Extract archive
    try:
        _extract_tar(tar_path, root)
    except Exception as e:
        raise RuntimeError(f"Failed to extract CIFAR-100-C archive: {e}") from e

    # Optionally, remove archive to save disk space
    try:
        tar_path.unlink()
    except OSError:
        pass

    if not labels_path.exists():
        raise RuntimeError(f"After extraction, CIFAR-100-C labels file not found at {labels_path}.")

    return cifar_c_dir


if __name__ == "__main__":
    args = parser.parse_args()

    # Determine data directory (priority: env var > default)
    data_dir = os.getenv("DATA_DIR", "./data")
    # Convert to absolute path
    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using data directory: {data_dir}")
    print("-" * 60)

    # Download CIFAR-100
    if not args.skip_cifar100:
        print("\n[1/2] Downloading CIFAR-100...")
        print("This will download the train and test sets (~170 MB)")

        # Download train set
        datasets.CIFAR100(root=str(data_dir), train=True, download=True)

        # Download test set
        datasets.CIFAR100(root=str(data_dir), train=False, download=True)

        print("✓ CIFAR-100 download complete!")
    else:
        print("\n[1/2] Skipping CIFAR-100 download")

    # Download CIFAR-100-C
    if not args.skip_cifar100c:
        print("\n[2/2] Downloading CIFAR-100-C...")
        print("This will download all corruption types (~2.7 GB)")

        ensure_cifar100c_downloaded(str(data_dir))

        print("✓ CIFAR-100-C download complete!")
    else:
        print("\n[2/2] Skipping CIFAR-100-C download")

    print("\n" + "=" * 60)
    print("Dataset setup complete!")
    print(f"Data directory: {data_dir}")
    print("=" * 60)
