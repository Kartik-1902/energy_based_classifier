"""Compute and persist per-class energy profiles for a trained model."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch
from torchvision import datasets, transforms

from config import CIFAR10_CLASSES, DEFAULT_CONFIG
from energy import class_energy, compute_energy_profiles
from model import build_model
from utils import plot_energy_distributions


def build_profile_loader(data_root: str, batch_size: int, num_workers: int) -> torch.utils.data.DataLoader:
    """Return dataloader over all CIFAR-10 training samples (no augmentation)."""
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=tfm)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def compute_and_save_profiles(
    checkpoint_path: str,
    output_path: str,
    model_name: str = DEFAULT_CONFIG.model_name,
    data_root: str = DEFAULT_CONFIG.data_root,
    batch_size: int = DEFAULT_CONFIG.batch_size,
    num_workers: int = DEFAULT_CONFIG.num_workers,
    num_classes: int = DEFAULT_CONFIG.num_classes,
    plot_path: str | None = None,
) -> dict:
    """Load model checkpoint, compute per-class energy profiles, and save to disk."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_name = checkpoint.get("model_name", model_name)
    model = build_model(model_name=model_name, num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = build_profile_loader(data_root=data_root, batch_size=batch_size, num_workers=num_workers)
    profiles = compute_energy_profiles(model=model, dataloader=loader, num_classes=num_classes, device=device)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(profiles, f)

    print("\nPer-class energy profile summary")
    print("Class         |      mu |   sigma |     min |     max | count")
    print("-" * 66)
    for k in range(num_classes):
        p = profiles[k]
        print(
            f"{CIFAR10_CLASSES[k]:<13}| {p['mu']:>7.4f} | {p['sigma']:>7.4f} | "
            f"{p['min']:>7.4f} | {p['max']:>7.4f} | {p['count']:>5d}"
        )

    if plot_path:
        per_class_energies = [[] for _ in range(num_classes)]
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                for k in range(num_classes):
                    mask = labels == k
                    if mask.any():
                        per_class_energies[k].extend(class_energy(logits[mask], k).cpu().numpy().tolist())

        plot_energy_distributions(per_class_energies, CIFAR10_CLASSES, plot_path)
        print(f"Saved energy distribution plot to {plot_path}")

    print(f"Saved energy profiles to {output_path}\n")
    return profiles


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for profile generation."""
    parser = argparse.ArgumentParser(description="Compute per-class energy profiles for CIFAR-10")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pth")
    parser.add_argument("--output", type=str, default="./checkpoints/energy_profiles.pkl")
    parser.add_argument("--model-name", type=str, default=DEFAULT_CONFIG.model_name)
    parser.add_argument("--data-root", type=str, default=DEFAULT_CONFIG.data_root)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_CONFIG.num_workers)
    parser.add_argument("--plot-path", type=str, default="./results/energy_distributions.png")
    return parser.parse_args()


def main() -> None:
    """Entry point for profile generation."""
    args = parse_args()
    compute_and_save_profiles(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        model_name=args.model_name,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        plot_path=args.plot_path,
    )


if __name__ == "__main__":
    main()
