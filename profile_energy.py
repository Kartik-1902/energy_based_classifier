"""Compute and persist per-class energy profiles for a trained model."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from config import DEFAULT_CONFIG, class_names_from_indices, parse_class_indices
from energy import class_energy, compute_energy_profiles
from model import build_model
from utils import plot_energy_distributions


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


def build_profile_loader(
    data_root: str,
    batch_size: int,
    num_workers: int,
    id_classes: tuple[int, ...],
) -> torch.utils.data.DataLoader:
    """Return dataloader over ID CIFAR-10 training samples (no augmentation)."""
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=tfm)
    filtered_dataset = CIFARClassSubset(dataset, id_classes)
    return torch.utils.data.DataLoader(
        filtered_dataset,
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
    id_classes: tuple[int, ...] = DEFAULT_CONFIG.id_classes,
    temperature: float = DEFAULT_CONFIG.temperature,
    plot_path: str | None = None,
) -> dict:
    """Load model checkpoint, compute per-class energy profiles, and save to disk."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_name = checkpoint.get("model_name", model_name)
    num_classes = int(checkpoint.get("num_classes", num_classes))
    id_classes = tuple(checkpoint.get("id_classes", list(id_classes)))
    class_names = checkpoint.get("id_class_names", class_names_from_indices(id_classes))

    # Keep explicit CLI temperature for profile sweeps instead of forcing checkpoint temperature.
    temperature = float(temperature)

    model = build_model(model_name=model_name, num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = build_profile_loader(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        id_classes=id_classes,
    )
    profiles = compute_energy_profiles(
        model=model,
        dataloader=loader,
        num_classes=num_classes,
        temperature=temperature,
        device=device,
    )

    profiles_payload = {
        "profiles": profiles,
        "num_classes": num_classes,
        "id_classes": list(id_classes),
        "id_class_names": list(class_names),
        "temperature": temperature,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(profiles_payload, f)

    print("\nPer-class energy profile summary")
    print("Class         |      mu |   sigma |     min |     max | count")
    print("-" * 66)
    for k in range(num_classes):
        p = profiles[k]
        print(
            f"{class_names[k]:<13}| {p['mu']:>7.4f} | {p['sigma']:>7.4f} | "
            f"{p['min']:>7.4f} | {p['max']:>7.4f} | {p['count']:>5d}"
        )

    if plot_path:
        per_class_energies = [[] for _ in range(num_classes)]
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images) / max(temperature, 1e-6)
                for k in range(num_classes):
                    mask = labels == k
                    if mask.any():
                        per_class_energies[k].extend(class_energy(logits[mask], k).cpu().numpy().tolist())

        plot_energy_distributions(per_class_energies, class_names, plot_path)
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
    parser.add_argument("--temperature", type=float, default=DEFAULT_CONFIG.temperature)
    parser.add_argument(
        "--id-classes",
        type=str,
        default=",".join(str(i) for i in DEFAULT_CONFIG.id_classes),
        help="Comma-separated CIFAR-10 class indices used as in-distribution classes",
    )
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
        id_classes=parse_class_indices(args.id_classes),
        temperature=args.temperature,
        plot_path=args.plot_path,
    )


if __name__ == "__main__":
    main()
