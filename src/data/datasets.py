"""Dataset loaders for Flowers102 and CUB-200-2011."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import Flowers102
from datasets import load_dataset

from .transforms import CLIPTransform


class CUBDatasetWrapper(torch.utils.data.Dataset):
    """Wrapper to apply transforms to HuggingFace CUB dataset."""

    def __init__(self, hf_dataset, transform):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_flowers102_dataset(root_dir, transform, download=True):
    """Load Flowers102 dataset splits.

    Args:
        root_dir: Root directory for dataset
        transform: Transform to apply to images
        download: Whether to download if not present

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, class_names, num_classes)
    """
    train_dts = Flowers102(
        root=root_dir, split="train", transform=transform, download=download
    )
    val_dts = Flowers102(
        root=root_dir, split="val", transform=transform, download=download
    )
    test_dts = Flowers102(
        root=root_dir, split="test", transform=transform, download=download
    )

    # Load class names
    cat_to_name_path = Path("cat_to_name.json")
    with open(cat_to_name_path, "r") as f:
        cat_to_name = json.load(f)

    # Convert to 0-indexed list (Flowers102 uses 1-indexed labels)
    class_names = [cat_to_name[str(i + 1)] for i in range(102)]
    num_classes = 102

    return train_dts, val_dts, test_dts, class_names, num_classes


def get_cub200_dataset(transform, val_split=0.1, seed=42):
    """Load CUB-200-2011 dataset splits.

    Args:
        transform: Transform to apply to images
        val_split: Fraction of train data to use for validation
        seed: Random seed for train/val split

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, class_names, num_classes)
    """
    birds_200 = load_dataset("bentrevett/caltech-ucsd-birds-200-2011")

    # Split train into train and validation
    cub_train_val = birds_200["train"]
    train_size = int((1 - val_split) * len(cub_train_val))
    val_size = len(cub_train_val) - train_size
    cub_train_split, cub_val_split = random_split(
        cub_train_val,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_dts = CUBDatasetWrapper(cub_train_split, transform)
    val_dts = CUBDatasetWrapper(cub_val_split, transform)
    test_dts = CUBDatasetWrapper(birds_200["test"], transform)

    class_names = sorted(set(birds_200["train"].features["label"].names))
    num_classes = 200

    return train_dts, val_dts, test_dts, class_names, num_classes


def get_dataset(dataset_name, clip_processor, root_dir="../data", **kwargs):
    """Factory function to get dataset by name.

    Args:
        dataset_name: Name of dataset ("flowers102" or "cub200")
        clip_processor: CLIP processor for transforms
        root_dir: Root directory for datasets
        **kwargs: Additional dataset-specific arguments

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, class_names, num_classes)
    """
    transform = CLIPTransform(clip_processor)

    if dataset_name == "flowers102":
        return get_flowers102_dataset(root_dir, transform, download=kwargs.get("download", True))
    elif dataset_name == "cub200":
        return get_cub200_dataset(transform, val_split=kwargs.get("val_split", 0.1))
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_dataloaders(train_dts, val_dts, test_dts, batch_size, num_workers=4):
    """Create DataLoaders for train/val/test splits.

    Args:
        train_dts: Training dataset
        val_dts: Validation dataset
        test_dts: Test dataset
        batch_size: Batch size for all loaders
        num_workers: Number of worker processes

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dts, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dts, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dts, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader
