"""Evaluate CIFAR-10 model with softmax and energy-based decision rules."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from config import CIFAR10_CLASSES, DEFAULT_CONFIG
from energy import energy_predict, marginal_energy
from model import build_model
from utils import compute_ece, plot_calibration, plot_ood_separation


def build_eval_transform() -> transforms.Compose:
    """Build deterministic normalization transform for evaluation."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )


def load_model_and_profiles(
    checkpoint_path: str,
    profiles_path: str,
    model_name: str,
    num_classes: int,
    device: torch.device,
):
    """Load trained model checkpoint and per-class energy profile dictionary."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", model_name)

    model = build_model(model_name=model_name, num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with open(profiles_path, "rb") as f:
        profiles = pickle.load(f)

    return model, profiles


def compute_fpr_at_95_tpr(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute FPR at 95% TPR using ROC curve interpolation by threshold sweep."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    mask = tpr >= 0.95
    if np.any(mask):
        return float(np.min(fpr[mask]))
    return 1.0


def collect_logits_labels(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect logits and labels from a dataloader."""
    logits_all = []
    labels_all = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images).cpu().numpy()
            logits_all.append(logits)
            labels_all.append(labels.numpy())

    return np.concatenate(logits_all, axis=0), np.concatenate(labels_all, axis=0)


def optimize_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit temperature by minimizing negative log-likelihood on calibration logits."""

    logits_t = torch.from_numpy(logits)
    labels_t = torch.from_numpy(labels)

    def objective(temp: float) -> float:
        t = max(temp, 1e-3)
        loss = F.cross_entropy(logits_t / t, labels_t)
        return float(loss.item())

    result = minimize_scalar(objective, bounds=(0.5, 5.0), method="bounded")
    return float(result.x)


def evaluate_id_metrics(
    logits_np: np.ndarray,
    labels_np: np.ndarray,
    profiles: Dict[int, Dict[str, float]],
    tau: float,
    temperature: float,
) -> Dict[str, float | np.ndarray]:
    """Evaluate softmax and energy classifiers on in-distribution CIFAR-10."""
    logits = torch.from_numpy(logits_np)

    probs = torch.softmax(logits / temperature, dim=1)
    softmax_conf, softmax_preds = probs.max(dim=1)

    energy_preds, is_ood, z_scores = energy_predict(logits, profiles, tau=tau)
    min_z, _ = z_scores.min(dim=1)
    energy_conf = 1.0 - torch.clamp(min_z / max(tau, 1e-6), min=0.0, max=1.0)

    labels_t = torch.from_numpy(labels_np)

    softmax_acc = float((softmax_preds == labels_t).float().mean().item())
    energy_acc = float((energy_preds == labels_t).float().mean().item())
    rejection_rate = float(is_ood.float().mean().item())

    softmax_ece = compute_ece(
        softmax_conf.numpy(),
        softmax_preds.numpy(),
        labels_np,
        n_bins=10,
    )
    energy_ece = compute_ece(
        energy_conf.numpy(),
        energy_preds.numpy(),
        labels_np,
        n_bins=10,
    )

    per_class_acc = {}
    for class_idx, class_name in enumerate(CIFAR10_CLASSES):
        mask = labels_np == class_idx
        if np.any(mask):
            per_class_acc[class_name] = float((energy_preds.numpy()[mask] == labels_np[mask]).mean())
        else:
            per_class_acc[class_name] = float("nan")

    return {
        "softmax_acc": softmax_acc,
        "energy_acc": energy_acc,
        "softmax_ece": softmax_ece,
        "energy_ece": energy_ece,
        "rejection_rate": rejection_rate,
        "softmax_preds": softmax_preds.numpy(),
        "softmax_conf": softmax_conf.numpy(),
        "energy_preds": energy_preds.numpy(),
        "energy_conf": energy_conf.numpy(),
        "per_class_acc": per_class_acc,
    }


def evaluate_ood(
    model: torch.nn.Module,
    id_loader: DataLoader,
    ood_loader: DataLoader,
    device: torch.device,
    temperature: float,
) -> Dict[str, float | np.ndarray]:
    """Evaluate OOD detection using marginal energy as score."""

    def collect_marginal_energies(loader: DataLoader) -> np.ndarray:
        vals = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                logits = model(images)
                e = marginal_energy(logits, temperature=temperature)
                vals.append(e.cpu().numpy())
        return np.concatenate(vals, axis=0)

    id_scores = collect_marginal_energies(id_loader)
    ood_scores = collect_marginal_energies(ood_loader)

    y_true = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    y_scores = np.concatenate([id_scores, ood_scores])

    auroc = float(roc_auc_score(y_true, y_scores))
    fpr95 = float(compute_fpr_at_95_tpr(y_true, y_scores))

    return {
        "auroc": auroc,
        "fpr95": fpr95,
        "id_scores": id_scores,
        "ood_scores": ood_scores,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line options for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate energy-based CIFAR-10 classifier")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pth")
    parser.add_argument("--profiles", type=str, default="./checkpoints/energy_profiles.pkl")
    parser.add_argument("--model-name", type=str, default=DEFAULT_CONFIG.model_name)
    parser.add_argument("--data-root", type=str, default=DEFAULT_CONFIG.data_root)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_CONFIG.num_workers)
    parser.add_argument("--tau", type=float, default=DEFAULT_CONFIG.tau)
    parser.add_argument("--temperature", type=float, default=DEFAULT_CONFIG.temperature)
    parser.add_argument("--ood-dataset", type=str, default="cifar100", choices=["cifar100", "svhn"])
    parser.add_argument("--calibrate-temperature", action="store_true")
    parser.add_argument("--results-dir", type=str, default=DEFAULT_CONFIG.results_dir)
    return parser.parse_args()


def main() -> None:
    """Entry point for full evaluation on ID and OOD datasets."""
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    model, profiles = load_model_and_profiles(
        checkpoint_path=args.checkpoint,
        profiles_path=args.profiles,
        model_name=args.model_name,
        num_classes=len(CIFAR10_CLASSES),
        device=device,
    )

    tfm = build_eval_transform()
    cifar10_test = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=tfm)
    id_loader = DataLoader(cifar10_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if args.ood_dataset == "cifar100":
        ood_dataset = datasets.CIFAR100(root=args.data_root, train=False, download=True, transform=tfm)
    else:
        ood_dataset = datasets.SVHN(root=args.data_root, split="test", download=True, transform=tfm)
    ood_loader = DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    logits_np, labels_np = collect_logits_labels(model, id_loader, device)
    temperature = args.temperature

    if args.calibrate_temperature:
        calib_size = int(0.2 * len(cifar10_test))
        eval_size = len(cifar10_test) - calib_size
        calib_subset, eval_subset = random_split(
            cifar10_test,
            lengths=[calib_size, eval_size],
            generator=torch.Generator().manual_seed(42),
        )

        calib_loader = DataLoader(calib_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        eval_loader = DataLoader(eval_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        calib_logits, calib_labels = collect_logits_labels(model, calib_loader, device)
        temperature = optimize_temperature(calib_logits, calib_labels)

        logits_np, labels_np = collect_logits_labels(model, eval_loader, device)
        id_loader_for_ood = eval_loader
        print(f"Optimized temperature (NLL): {temperature:.4f}")
    else:
        id_loader_for_ood = id_loader

    id_metrics = evaluate_id_metrics(
        logits_np=logits_np,
        labels_np=labels_np,
        profiles=profiles,
        tau=args.tau,
        temperature=temperature,
    )

    ood_metrics = evaluate_ood(
        model=model,
        id_loader=id_loader_for_ood,
        ood_loader=ood_loader,
        device=device,
        temperature=temperature,
    )

    plot_calibration(
        id_metrics["softmax_conf"],
        id_metrics["softmax_preds"],
        labels_np,
        save_path=str(Path(args.results_dir) / "reliability_softmax.png"),
        title="Reliability Diagram - Softmax",
    )
    plot_calibration(
        id_metrics["energy_conf"],
        id_metrics["energy_preds"],
        labels_np,
        save_path=str(Path(args.results_dir) / "reliability_energy.png"),
        title="Reliability Diagram - Energy",
    )
    plot_ood_separation(
        ood_metrics["id_scores"],
        ood_metrics["ood_scores"],
        save_path=str(Path(args.results_dir) / "ood_energy_separation.png"),
    )

    print("\nPer-class accuracy (Energy predictor)")
    for cls_name, acc in id_metrics["per_class_acc"].items():
        print(f"  {cls_name:<11}: {acc:.4f}")

    print("\nComparison Table")
    print("Method          | Accuracy | ECE    | AUROC (OOD)")
    print("-" * 50)
    print(
        f"Softmax baseline| {id_metrics['softmax_acc']:.4f}   | {id_metrics['softmax_ece']:.4f} | {ood_metrics['auroc']:.4f}"
    )
    print(
        f"Energy (ours)   | {id_metrics['energy_acc']:.4f}   | {id_metrics['energy_ece']:.4f} | {ood_metrics['auroc']:.4f}"
    )

    print("\nAdditional metrics")
    print(f"OOD dataset: {args.ood_dataset}")
    print(f"FPR@95TPR: {ood_metrics['fpr95']:.4f}")
    print(f"Energy rejection rate on ID: {id_metrics['rejection_rate']:.4f}")
    print(f"Using temperature: {temperature:.4f}")
    print(f"Saved plots to {args.results_dir}")


if __name__ == "__main__":
    main()
