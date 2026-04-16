"""Configuration module for CIFAR-10 energy-based classification."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


DEFAULT_ID_CLASSES = tuple(range(8))
DEFAULT_OOD_CLASSES = (8, 9)


def _validate_class_indices(indices: Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(v) for v in indices)
    if not values:
        raise ValueError("Class index list cannot be empty.")
    if len(set(values)) != len(values):
        raise ValueError("Class index list contains duplicates.")
    if any(v < 0 or v >= len(CIFAR10_CLASSES) for v in values):
        raise ValueError(f"Class indices must be in [0, {len(CIFAR10_CLASSES) - 1}].")
    return values


def parse_class_indices(text: str) -> tuple[int, ...]:
    """Parse comma-separated class indices (example: '0,1,2,3')."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("Expected at least one class index.")
    return _validate_class_indices(int(p) for p in parts)


def class_names_from_indices(indices: Iterable[int]) -> list[str]:
    """Return CIFAR-10 class names for the given class indices."""
    return [CIFAR10_CLASSES[i] for i in _validate_class_indices(indices)]


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters and file-system paths."""

    data_root: str = "./data"
    checkpoints_dir: str = "./checkpoints"
    results_dir: str = "./results"

    model_name: str = "resnet18"
    id_classes: tuple[int, ...] = field(default_factory=lambda: DEFAULT_ID_CLASSES)
    ood_classes: tuple[int, ...] = field(default_factory=lambda: DEFAULT_OOD_CLASSES)
    num_classes: int = len(DEFAULT_ID_CLASSES)

    epochs: int = 200
    batch_size: int = 128
    num_workers: int = 4

    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    label_smoothing: float = 0.1
    entropy_lambda: float = 0.01

    temperature: float = 1.0
    tau: float = 3.0
    odin_epsilon: float = 0.002

    seed: int = 42
    val_split: float = 0.1

    def __post_init__(self) -> None:
        id_classes = _validate_class_indices(self.id_classes)
        ood_classes = _validate_class_indices(self.ood_classes)

        if set(id_classes) & set(ood_classes):
            raise ValueError("ID and OOD class sets must be disjoint.")
        if len(id_classes) + len(ood_classes) != len(CIFAR10_CLASSES):
            raise ValueError("ID and OOD class sets must cover all CIFAR-10 classes.")
        if self.num_classes != len(id_classes):
            raise ValueError("num_classes must match the number of ID classes.")


DEFAULT_CONFIG = TrainConfig()


def ensure_dirs(cfg: TrainConfig) -> None:
    """Create checkpoint and results directories if they do not exist."""
    Path(cfg.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)
