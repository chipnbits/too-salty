import os
import re
import shutil
import tarfile
import zipfile

ROOT_DIR = "/data/sghyselincks/too-salty/models/cifar100-resnet50"
SOUP_DIR = "/data/sghyselincks/too-salty/models/soup_models"
pattern = re.compile(r"baseline-resnet50-epoch_(\d+)_model_(\d+)")


def walk_folders(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        print(f"Directory: {dirpath}")
        for dirname in dirnames:
            print(f"  Subdirectory: {dirname}")
        for filename in filenames:
            print(f"  File: {filename}")


def consolidate_soup_models(root_dir, soup_dir):
    for dirname in os.listdir(root_dir):
        match = pattern.match(dirname)
        if match:
            epoch = match.group(1)
            model_num = match.group(2)
            source_path = os.path.join(root_dir, dirname, "best_model.pt")
            target_filename = f"epoch_{epoch}_model_{model_num}.pt"
            target_path = os.path.join(soup_dir, target_filename)
            os.makedirs(soup_dir, exist_ok=True)

            if os.path.exists(source_path):
                shutil.copy2(source_path, target_path)
                print(f"Copied {source_path} -> {target_path}")
            else:
                print(f"Source file does not exist: {source_path}")


def compress_soup_models_zip(soup_dir, archive_name="soup_models.zip"):
    archive_path = os.path.join(soup_dir, archive_name)

    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for filename in os.listdir(soup_dir):
            if filename.endswith(".pt"):
                zipf.write(os.path.join(soup_dir, filename), arcname=filename)
                print(f"Added {filename} to ZIP")

    print(f"\nCreated ZIP archive: {archive_path}")
    return archive_path


def compress_soup_models_targz(soup_dir, archive_name="soup_models.tar.gz"):
    archive_path = os.path.join(soup_dir, archive_name)

    with tarfile.open(archive_path, "w:gz") as tar:
        for filename in os.listdir(soup_dir):
            if filename.endswith(".pt"):
                tar.add(os.path.join(soup_dir, filename), arcname=filename)
                print(f"Added {filename} to tar.gz")

    print(f"\nCreated tar.gz archive: {archive_path}")
    return archive_path


def cleanup_soup_models(soup_dir):
    removed = 0
    for filename in os.listdir(soup_dir):
        if filename.endswith(".pt"):
            path = os.path.join(soup_dir, filename)
            os.remove(path)
            removed += 1
            print(f"Deleted: {filename}")
    print(f"\nDeleted {removed} temp files.")


if __name__ == "__main__":
    consolidate_soup_models(ROOT_DIR, SOUP_DIR)
    # archive_path = compress_soup_models_targz(SOUP_DIR)
    # cleanup_soup_models(SOUP_DIR)
    # print(f"\nDone. Final archive: {archive_path}")
