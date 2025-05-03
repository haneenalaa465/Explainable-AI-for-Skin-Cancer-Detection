"""
Dataset preparation and loading utilities.
"""

import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import v2
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from XAI.preprocessing.C_LAHE import CLAHE
from XAI.preprocessing.enhance_clarity import EnhanceClarityCV
from XAI.preprocessing.hair_removal import HairRemoval
from XAI.preprocessing.contrast_stretch import ContrastStretch
from skimage import io

from XAI.config import (
    INTERIM_DATA_DIR,
    RAW_DATA_DIR,
    HAM10000_METADATA,
    HAM10000_IMAGES_PART1,
    HAM10000_IMAGES_PART2,
    CLASS_NAMES,
    RANDOM_SEED,
    BATCH_SIZE,
    MODEL_INPUT_SIZE,
)


def download_and_extract_ham10000():
    """
    Download and extract the HAM10000 dataset.
    This function assumes you have kaggle API set up.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    # Download dataset using kaggle API
    print("Downloading HAM10000 dataset...")
    os.system(f"kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p {RAW_DATA_DIR}")

    # Extract the dataset
    print("Extracting dataset...")
    zip_path = RAW_DATA_DIR / "skin-cancer-mnist-ham10000.zip"
    if zip_path.exists():
        import zipfile

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
        print("Extraction complete!")
    else:
        print(f"Error: ZIP file not found at {zip_path}")


def organize_data():
    """
    Organize the HAM10000 dataset by:
    1. Creating a directory structure by class
    2. Copying images to their respective class directories
    """
    # Load metadata
    metadata_path = HAM10000_METADATA
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    print("Merging Directories")

    shutil.copytree(HAM10000_IMAGES_PART2, HAM10000_IMAGES_PART1, dirs_exist_ok=True)
    metadata = pd.read_csv(metadata_path)

    # Create class directories
    organized_dir = INTERIM_DATA_DIR / "organized_by_class"
    for class_name in CLASS_NAMES.keys():
        os.makedirs(organized_dir / class_name, exist_ok=True)

    # Copy images to their respective class directories
    image_dirs = [HAM10000_IMAGES_PART1, HAM10000_IMAGES_PART2]

    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
        img_id = row['image_id']
        dx = row['dx']  # Diagnosis/class

        # Find the image
        found = False
        for img_dir in image_dirs:
            img_path = img_dir / f"{img_id}.jpg"
            if img_path.exists():
                found = True
                # Copy to class directory
                dst_path = organized_dir / dx / f"{img_id}.jpg"
                shutil.copy(img_path, dst_path)
                break

        if not found:
            print(f"Warning: Image {img_id} not found")

    print("Data organization complete!")


class HAM10000Dataset(Dataset):
    """HAM10000 dataset class for PyTorch."""

    def __init__(self, df, img_dir, transform=None, is_binary=False):
        """
        Args:
            df (pandas.DataFrame): DataFrame with image info
            img_dir (Path): Directory with images
            transform: Transformations to apply to images
        """
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_binary = is_binary

        # Create a mapping from class names to indices
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(CLASS_NAMES.keys())}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["image_id"]
        dx = row["dx"]  # Diagnosis/class

        # Find the image path
        img_path = HAM10000_IMAGES_PART1 / f"{img_id}.jpg"

        # Load image and apply transformations
        image = io.imread(img_path)
        # Get class index
        label = self.class_to_idx[dx]

        if self.is_binary:
            if label == 4 or label == 1:
                label = 1
            else:
                label = 0

        if self.transform:

            image = self.transform(image)
        # print(image.shape)
        return image, label


def get_transforms(stage="train"):
    """
    Get transforms for different stages (train, val, test).

    Args:
        stage (str): One of "train", "val", or "test"

    Returns:
        albumentations.Compose: Composition of transforms
    """
    if stage == "train":
        return v2.Compose(
            [
                HairRemoval(),
                CLAHE(),
                EnhanceClarityCV(),
                ContrastStretch(),
                v2.ToImage(),  # If using tensor transforms afterwards
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomRotation(25),
                # v2.ColorJitter(0.2, 0.2),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:  # val or test
        return v2.Compose(
            [
                HairRemoval(),
                # CLAHE(),
                # EnhanceClarityCV(),
                CLAHE(),
                EnhanceClarityCV(),
                ContrastStretch(),
                v2.ToImage(),  # If using tensor transforms afterwards
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


def prepare_data(metadata_path=None, balanced=True, is_binary=False):
    """
    Prepare train, validation, and test datasets.

    Args:
        metadata_path (Path): Path to metadata CSV file
        balanced (bool): Whether to balance classes

    Returns:
        tuple: train_loader, val_loader, test_loader
    """
    if metadata_path is None:
        metadata_path = HAM10000_METADATA

    # Load metadata
    metadata = pd.read_csv(metadata_path)

    # Get class distributions
    class_counts = metadata["dx"].value_counts()
    print("Class distribution in the dataset:")
    for class_name, count in class_counts.items():
        print(f"{CLASS_NAMES[class_name]}: {count} images")

    # Balance classes if required
    if balanced:
        # Find the minimum class size
        min_class_count = class_counts.min()

        # Downsample or augment to balance
        balanced_dfs = []
        for class_name in CLASS_NAMES.keys():
            class_df = metadata[metadata["dx"] == class_name]

            if len(class_df) <= min_class_count:
                # For small classes, include all samples
                balanced_dfs.append(class_df)
            else:
                # For larger classes, sample without replacement
                sampled_df = class_df.sample(min_class_count, random_state=RANDOM_SEED)
                balanced_dfs.append(sampled_df)

        # Combine balanced dataframes
        metadata = pd.concat(balanced_dfs, ignore_index=True)

    # Split data into train, validation, and test sets
    train_df, temp_df = train_test_split(
        metadata, test_size=0.3, random_state=RANDOM_SEED, stratify=metadata["dx"]
    )

    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_df["dx"]
    )

    print(f"Train set: {len(train_df)} images")
    print(f"Validation set: {len(val_df)} images")
    print(f"Test set: {len(test_df)} images")

    # Create datasets
    train_dataset = HAM10000Dataset(
        train_df,
        HAM10000_IMAGES_PART1,
        transform=get_transforms("train"),
        is_binary=is_binary,
    )

    val_dataset = HAM10000Dataset(
        val_df, HAM10000_IMAGES_PART1, transform=get_transforms("val"), is_binary=is_binary
    )

    test_dataset = HAM10000Dataset(
        test_df,
        HAM10000_IMAGES_PART1,
        transform=get_transforms("test"),
        is_binary=is_binary,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # If run as a script, download and organize the dataset
    download_and_extract_ham10000()
    organize_data()
    prepare_data()
