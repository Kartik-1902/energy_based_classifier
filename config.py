"""Configuration module for CIFAR-10 energy-based classification."""

from dataclasses import dataclass
from pathlib import Path


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


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters and file-system paths."""

    data_root: str = "./data"
    checkpoints_dir: str = "./checkpoints"
    results_dir: str = "./results"

    model_name: str = "resnet18"
    num_classes: int = 10

    epochs: int = 200
    batch_size: int = 128
    num_workers: int = 4

    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4

    temperature: float = 1.0
    tau: float = 3.0

    seed: int = 42
    val_split: float = 0.1


DEFAULT_CONFIG = TrainConfig()


def ensure_dirs(cfg: TrainConfig) -> None:
    """Create checkpoint and results directories if they do not exist."""
    Path(cfg.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)
