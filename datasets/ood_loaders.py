"""Standalone OOD dataset loader utilities for cross-dataset evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset
from torchvision import datasets, transforms

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def cifar_norm_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )


def imagenet_norm_transform(input_size: int = 32) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(36),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def imagenet_cifar_norm_transform(input_size: int = 32) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(36),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )


def build_cifar10_id_loader(
    data_root: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=cifar_norm_transform(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_cifar100_ood_loader(
    data_root: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = datasets.CIFAR100(
        root=data_root,
        train=False,
        download=True,
        transform=cifar_norm_transform(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


class HFImageNetStream(IterableDataset):
    """Stream ImageNet-1k validation samples from Hugging Face datasets."""

    def __init__(self, transform: transforms.Compose, max_samples: int = 5000):
        self.transform = transform
        self.max_samples = int(max_samples)

    def __iter__(self):
        try:
            from datasets import load_dataset
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Hugging Face fallback requested but `datasets` is not installed. "
                "Install with: pip install -r requirements_ood.txt"
            ) from exc

        stream = load_dataset("imagenet-1k", split="validation", streaming=True)
        emitted = 0
        for item in stream:
            image = item.get("image")
            if image is None:
                continue
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            if not isinstance(image, Image.Image):
                continue
            image = image.convert("RGB")
            tensor = self.transform(image)
            yield tensor, -1
            emitted += 1
            if emitted >= self.max_samples:
                break


def build_imagenet_ood_loaders(
    imagenet_path: str | None,
    batch_size: int,
    num_workers: int,
    hf_max_samples: int = 5000,
) -> List[Tuple[str, DataLoader]]:
    """Build ImageNet OOD loaders for both normalization modes.

    Returns a list of (dataset_key, loader) pairs where dataset_key includes
    source and normalization mode for explicit downstream reporting.
    """
    local_path = Path(imagenet_path).expanduser() if imagenet_path else None
    loaders: List[Tuple[str, DataLoader]] = []

    norm_modes = {
        "imagenet_norm": imagenet_norm_transform(),
        "cifar_norm": imagenet_cifar_norm_transform(),
    }

    if local_path is not None and local_path.exists() and local_path.is_dir():
        for norm_name, tfm in norm_modes.items():
            dataset = datasets.ImageFolder(root=str(local_path), transform=tfm)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
            loaders.append((f"imagenet_local_{norm_name}", loader))
        return loaders

    hf_available = True
    try:
        import datasets as _hf  # noqa: F401
    except Exception:
        hf_available = False

    if hf_available:
        for norm_name, tfm in norm_modes.items():
            dataset = HFImageNetStream(transform=tfm, max_samples=hf_max_samples)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
            loaders.append((f"imagenet_hf_{norm_name}", loader))
        return loaders

    raise RuntimeError(
        "ImageNet OOD requested but neither a valid --imagenet-path directory "
        "nor Hugging Face `datasets` fallback is available. "
        "Provide --imagenet-path <local_imagenet_val_dir> or install fallback deps "
        "with: pip install -r requirements_ood.txt"
    )
