"""Train CIFAR-10 classifier and generate post-training energy profiles."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tqdm import tqdm
from torchvision import datasets, transforms

from config import DEFAULT_CONFIG, TrainConfig, class_names_from_indices, parse_class_indices, ensure_dirs
from model import build_model
from profile_energy import compute_and_save_profiles


def set_seed(seed: int) -> None:
    """Set reproducibility seed across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CIFARClassSubset(Dataset):
    """Filter CIFAR dataset to selected classes and remap labels to [0, num_id_classes)."""

    def __init__(self, dataset: datasets.CIFAR10, selected_classes: tuple[int, ...]):
        self.dataset = dataset
        self.selected_classes = tuple(int(c) for c in selected_classes)
        self.class_to_local = {c: i for i, c in enumerate(self.selected_classes)}
        self.indices = [
            idx
            for idx, label in enumerate(dataset.targets)
            if int(label) in self.class_to_local
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, label = self.dataset[self.indices[index]]
        return image, self.class_to_local[int(label)]


def build_dataloaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    """Create CIFAR-10 train and validation dataloaders."""
    train_tfm = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    eval_tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    full_train = datasets.CIFAR10(root=cfg.data_root, train=True, download=True, transform=train_tfm)
    full_train_eval = datasets.CIFAR10(root=cfg.data_root, train=True, download=False, transform=eval_tfm)

    filtered_train = CIFARClassSubset(full_train, cfg.id_classes)
    filtered_eval = CIFARClassSubset(full_train_eval, cfg.id_classes)

    val_size = int(len(filtered_train) * cfg.val_split)
    train_size = len(filtered_train) - val_size
    train_subset_idx, val_subset_idx = random_split(
        range(len(filtered_train)),
        lengths=[train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    train_subset = Subset(filtered_train, list(train_subset_idx))
    val_subset = Subset(filtered_eval, list(val_subset_idx))

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    epoch_idx: int,
    total_epochs: int,
    use_progress_bar: bool,
) -> tuple[float, float]:
    """Run one train or evaluation epoch and return loss/accuracy."""
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    correct = 0
    total = 0

    phase = "train" if is_train else "val"
    iterator = tqdm(
        loader,
        desc=f"Epoch {epoch_idx:03d}/{total_epochs:03d} [{phase}]",
        leave=False,
        dynamic_ncols=True,
        disable=not use_progress_bar,
    )

    for images, labels in iterator:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if use_progress_bar:
            iterator.set_postfix(
                {
                    "loss": f"{(running_loss / max(total, 1)):.4f}",
                    "acc": f"{(correct / max(total, 1)):.4f}",
                }
            )

    return running_loss / total, correct / total


def save_checkpoint(path: Path, state: dict) -> None:
    """Save a training checkpoint atomically at the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def append_train_log(log_path: Path, row: dict) -> None:
    """Append one epoch summary row to CSV training logs."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "lr",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "best_val_acc",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train CIFAR-10 model with energy profiling")
    parser.add_argument("--model-name", type=str, default=DEFAULT_CONFIG.model_name, choices=["resnet18", "wideresnet28-2", "wideresnet", "wrn28-2"])
    parser.add_argument("--data-root", type=str, default=DEFAULT_CONFIG.data_root)
    parser.add_argument("--checkpoints-dir", type=str, default=DEFAULT_CONFIG.checkpoints_dir)
    parser.add_argument("--results-dir", type=str, default=DEFAULT_CONFIG.results_dir)
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_CONFIG.num_workers)
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG.lr)
    parser.add_argument("--momentum", type=float, default=DEFAULT_CONFIG.momentum)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG.weight_decay)
    parser.add_argument("--temperature", type=float, default=DEFAULT_CONFIG.temperature)
    parser.add_argument("--tau", type=float, default=DEFAULT_CONFIG.tau)
    parser.add_argument(
        "--id-classes",
        type=str,
        default=",".join(str(i) for i in DEFAULT_CONFIG.id_classes),
        help="Comma-separated CIFAR-10 class indices used as in-distribution classes",
    )
    parser.add_argument(
        "--ood-classes",
        type=str,
        default=",".join(str(i) for i in DEFAULT_CONFIG.ood_classes),
        help="Comma-separated CIFAR-10 class indices held out for OOD evaluation",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    parser.add_argument("--val-split", type=float, default=DEFAULT_CONFIG.val_split)
    parser.add_argument("--save-every", type=int, default=10, help="Save periodic checkpoints every N epochs")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints/latest.pth if it exists")
    parser.add_argument("--no-progress-bar", action="store_true", help="Disable tqdm batch progress bars")
    return parser.parse_args()


def main() -> None:
    """Entry point for model training and post-training energy profiling."""
    args = parse_args()
    id_classes = parse_class_indices(args.id_classes)
    ood_classes = parse_class_indices(args.ood_classes)
    cfg = TrainConfig(
        data_root=args.data_root,
        checkpoints_dir=args.checkpoints_dir,
        results_dir=args.results_dir,
        model_name=args.model_name,
        id_classes=id_classes,
        ood_classes=ood_classes,
        num_classes=len(id_classes),
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        tau=args.tau,
        seed=args.seed,
        val_split=args.val_split,
    )

    ensure_dirs(cfg)
    set_seed(cfg.seed)

    print(f"ID classes: {cfg.id_classes} -> {class_names_from_indices(cfg.id_classes)}")
    print(f"OOD classes: {cfg.ood_classes} -> {class_names_from_indices(cfg.ood_classes)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(model_name=cfg.model_name, num_classes=cfg.num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_val_acc = 0.0
    start_epoch = 1
    checkpoints_dir = Path(cfg.checkpoints_dir)
    best_ckpt_path = checkpoints_dir / "best_model.pth"
    latest_ckpt_path = checkpoints_dir / "latest.pth"
    log_path = Path(cfg.results_dir) / "train_log.csv"

    if args.resume and latest_ckpt_path.exists():
        latest = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(latest["model_state_dict"])
        optimizer.load_state_dict(latest["optimizer_state_dict"])
        scheduler.load_state_dict(latest["scheduler_state_dict"])
        start_epoch = int(latest["epoch"]) + 1
        best_val_acc = float(latest.get("best_val_acc", 0.0))
        print(f"Resumed from {latest_ckpt_path} at epoch {start_epoch}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch_idx=epoch,
            total_epochs=cfg.epochs,
            use_progress_bar=not args.no_progress_bar,
        )
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
            None,
            device,
            epoch_idx=epoch,
            total_epochs=cfg.epochs,
            use_progress_bar=not args.no_progress_bar,
        )
        scheduler.step()
        current_lr = float(optimizer.param_groups[0]["lr"])

        state = {
            "epoch": epoch,
            "model_name": cfg.model_name,
            "num_classes": cfg.num_classes,
            "id_classes": list(cfg.id_classes),
            "ood_classes": list(cfg.ood_classes),
            "id_class_names": class_names_from_indices(cfg.id_classes),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "temperature": cfg.temperature,
            "tau": cfg.tau,
            "args": vars(args),
        }

        save_checkpoint(latest_ckpt_path, state)

        if args.save_every > 0 and (epoch % args.save_every == 0):
            periodic_path = checkpoints_dir / f"epoch_{epoch:03d}.pth"
            save_checkpoint(periodic_path, state)
            print(f"Saved periodic checkpoint to {periodic_path}")

        print(
            f"Epoch [{epoch:03d}/{cfg.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={current_lr:.6f}"
        )

        append_train_log(
            log_path,
            {
                "epoch": epoch,
                "lr": f"{current_lr:.8f}",
                "train_loss": f"{train_loss:.6f}",
                "train_acc": f"{train_acc:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_acc": f"{val_acc:.6f}",
                "best_val_acc": f"{best_val_acc:.6f}",
            },
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            state["best_val_acc"] = best_val_acc
            save_checkpoint(best_ckpt_path, state)
            print(f"Saved new best checkpoint to {best_ckpt_path}")

    print(f"Best validation accuracy: {best_val_acc:.4f}")

    profiles_path = str(Path(cfg.checkpoints_dir) / "energy_profiles.pkl")
    plot_path = str(Path(cfg.results_dir) / "energy_distributions.png")

    compute_and_save_profiles(
        checkpoint_path=str(best_ckpt_path),
        output_path=profiles_path,
        model_name=cfg.model_name,
        data_root=cfg.data_root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        num_classes=cfg.num_classes,
        id_classes=cfg.id_classes,
        plot_path=plot_path,
    )


if __name__ == "__main__":
    main()
