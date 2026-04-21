"""Standalone additive cross-dataset OOD evaluation script."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import pickle
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from energy import marginal_energy
from model import build_model


def _load_ood_loader_module():
    module_path = Path(__file__).resolve().parent / "datasets" / "ood_loaders.py"
    spec = importlib.util.spec_from_file_location("local_ood_loaders", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load OOD loader module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone cross-dataset OOD evaluator")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--profiles-path", type=str, required=True)
    parser.add_argument("--imagenet-path", type=str, default=None)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--tau", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default="results/cross_dataset")
    parser.add_argument("--hf-max-samples", type=int, default=5000)
    parser.add_argument("--skip-imagenet", action="store_true")
    parser.add_argument("--skip-cifar100", action="store_true")
    return parser.parse_args()


def load_model_and_metadata(model_path: str, profiles_path: str, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)
    model_name = checkpoint.get("model_name", "resnet18")
    num_classes = int(checkpoint.get("num_classes", 10))

    model = build_model(model_name=model_name, num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with open(profiles_path, "rb") as f:
        profiles_blob = pickle.load(f)

    profiles_meta: Dict[str, object] = {}
    if isinstance(profiles_blob, dict):
        if "profiles" in profiles_blob:
            profiles_meta["num_classes"] = profiles_blob.get("num_classes", num_classes)
            profiles_meta["id_classes"] = profiles_blob.get("id_classes", checkpoint.get("id_classes", []))
        else:
            profiles_meta["num_classes"] = len(profiles_blob)
            profiles_meta["id_classes"] = checkpoint.get("id_classes", [])

    metadata = {
        "model_name": model_name,
        "num_classes": num_classes,
        "id_classes": checkpoint.get("id_classes", []),
        "ood_classes": checkpoint.get("ood_classes", []),
        "profiles_meta": profiles_meta,
    }
    return model, metadata


def collect_energy_scores(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    temperature: float,
) -> np.ndarray:
    scores: List[np.ndarray] = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            logits = model(images)
            energy = marginal_energy(logits, temperature=temperature)
            scores.append(energy.cpu().numpy())
    return np.concatenate(scores, axis=0)


def fpr_at_95_tpr(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    mask = tpr >= 0.95
    if np.any(mask):
        return float(np.min(fpr[mask]))
    return 1.0


def summarize_stats(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def evaluate_one_dataset(
    dataset_key: str,
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    tau: float,
):
    id_stats = summarize_stats(id_scores)
    ood_stats = summarize_stats(ood_scores)

    threshold = id_stats["mean"] + tau * max(id_stats["std"], 1e-8)

    y_true = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    y_score = np.concatenate([id_scores, ood_scores])

    auroc = float(roc_auc_score(y_true, y_score))
    aupr = float(average_precision_score(y_true, y_score))
    fpr95 = float(fpr_at_95_tpr(y_true, y_score))

    y_pred_ood = (y_score > threshold).astype(np.int64)
    det_acc = float(np.mean(y_pred_ood == y_true))

    separation = float((ood_stats["mean"] - id_stats["mean"]) / max(id_stats["std"], 1e-8))

    fpr, tpr, _ = roc_curve(y_true, y_score)

    row = {
        "dataset": dataset_key,
        "auroc": auroc,
        "aupr": aupr,
        "fpr95": fpr95,
        "detection_accuracy": det_acc,
        "threshold": float(threshold),
        "id_mean": id_stats["mean"],
        "id_std": id_stats["std"],
        "id_min": id_stats["min"],
        "id_max": id_stats["max"],
        "ood_mean": ood_stats["mean"],
        "ood_std": ood_stats["std"],
        "ood_min": ood_stats["min"],
        "ood_max": ood_stats["max"],
        "separation_score": separation,
        "roc_fpr": fpr,
        "roc_tpr": tpr,
        "id_scores": id_scores,
        "ood_scores": ood_scores,
    }
    return row


def save_summary_csv(rows: List[Dict[str, object]], output_dir: Path) -> Path:
    out_path = output_dir / "summary.csv"
    fieldnames = [
        "dataset",
        "auroc",
        "aupr",
        "fpr95",
        "detection_accuracy",
        "threshold",
        "id_mean",
        "id_std",
        "id_min",
        "id_max",
        "ood_mean",
        "ood_std",
        "ood_min",
        "ood_max",
        "separation_score",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})
    return out_path


def save_energy_histograms(rows: List[Dict[str, object]], output_dir: Path) -> Path:
    out_path = output_dir / "energy_histograms.png"
    fig, axes = plt.subplots(len(rows), 1, figsize=(9, max(4, 3 * len(rows))), squeeze=False)

    for i, row in enumerate(rows):
        ax = axes[i][0]
        id_scores = np.asarray(row["id_scores"])
        ood_scores = np.asarray(row["ood_scores"])
        ax.hist(id_scores, bins=50, alpha=0.6, label="ID (CIFAR-10 test)", density=True)
        ax.hist(ood_scores, bins=50, alpha=0.6, label=f"OOD ({row['dataset']})", density=True)
        ax.axvline(float(row["threshold"]), color="black", linestyle="--", linewidth=1.2, label="Threshold")
        ax.set_title(f"Energy Histogram: {row['dataset']}")
        ax.set_xlabel("Marginal energy")
        ax.set_ylabel("Density")
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_roc_curves(rows: List[Dict[str, object]], output_dir: Path) -> Path:
    out_path = output_dir / "roc_curves.png"
    plt.figure(figsize=(8, 6))
    for row in rows:
        plt.plot(row["roc_fpr"], row["roc_tpr"], label=f"{row['dataset']} (AUROC={row['auroc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("OOD ROC Curves (marginal energy)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def save_text_report(rows: List[Dict[str, object]], output_dir: Path, args: argparse.Namespace, metadata: Dict[str, object]) -> Path:
    out_path = output_dir / "results_report.txt"
    lines = []
    lines.append("Cross-Dataset OOD Evaluation Report")
    lines.append("=")
    lines.append(f"Model path: {args.model_path}")
    lines.append(f"Profiles path: {args.profiles_path}")
    lines.append(f"Temperature: {args.temperature}")
    lines.append(f"Tau: {args.tau}")
    lines.append(
        "Detection threshold rule: threshold = mean(ID energy) + tau * std(ID energy), "
        "predict OOD when E(x) > threshold"
    )
    lines.append(f"Model metadata: {metadata}")
    lines.append("")

    for row in rows:
        lines.append(f"Dataset: {row['dataset']}")
        lines.append(f"  AUROC: {row['auroc']:.6f}")
        lines.append(f"  AUPR: {row['aupr']:.6f}")
        lines.append(f"  FPR@95TPR: {row['fpr95']:.6f}")
        lines.append(f"  Detection Accuracy: {row['detection_accuracy']:.6f}")
        lines.append(f"  Threshold: {row['threshold']:.6f}")
        lines.append(
            "  ID stats(mean/std/min/max): "
            f"{row['id_mean']:.6f} / {row['id_std']:.6f} / {row['id_min']:.6f} / {row['id_max']:.6f}"
        )
        lines.append(
            "  OOD stats(mean/std/min/max): "
            f"{row['ood_mean']:.6f} / {row['ood_std']:.6f} / {row['ood_min']:.6f} / {row['ood_max']:.6f}"
        )
        lines.append(f"  Separation score: {row['separation_score']:.6f}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def print_results_table(rows: List[Dict[str, object]]) -> None:
    header = (
        f"{'Dataset':<28} {'AUROC':>8} {'AUPR':>8} {'FPR95':>8} {'DetAcc':>8} "
        f"{'Thr':>10} {'Sep':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['dataset']:<28} "
            f"{row['auroc']:>8.4f} "
            f"{row['aupr']:>8.4f} "
            f"{row['fpr95']:>8.4f} "
            f"{row['detection_accuracy']:>8.4f} "
            f"{row['threshold']:>10.4f} "
            f"{row['separation_score']:>8.4f}"
        )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_model_and_metadata(args.model_path, args.profiles_path, device)

    loaders_mod = _load_ood_loader_module()

    id_loader = loaders_mod.build_cifar10_id_loader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    id_scores = collect_energy_scores(model, id_loader, device, temperature=args.temperature)

    rows: List[Dict[str, object]] = []

    if not args.skip_cifar100:
        cifar100_loader = loaders_mod.build_cifar100_ood_loader(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        cifar100_scores = collect_energy_scores(model, cifar100_loader, device, temperature=args.temperature)
        rows.append(evaluate_one_dataset("cifar100", id_scores, cifar100_scores, tau=args.tau))

    if not args.skip_imagenet:
        imagenet_loaders = loaders_mod.build_imagenet_ood_loaders(
            imagenet_path=args.imagenet_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            hf_max_samples=args.hf_max_samples,
        )
        for dataset_key, loader in imagenet_loaders:
            ood_scores = collect_energy_scores(model, loader, device, temperature=args.temperature)
            rows.append(evaluate_one_dataset(dataset_key, id_scores, ood_scores, tau=args.tau))

    if not rows:
        raise RuntimeError("No OOD dataset selected. Disable fewer skip flags.")

    summary_path = save_summary_csv(rows, output_dir)
    hist_path = save_energy_histograms(rows, output_dir)
    roc_path = save_roc_curves(rows, output_dir)
    report_path = save_text_report(rows, output_dir, args, metadata)

    print_results_table(rows)
    print(f"Saved summary: {summary_path}")
    print(f"Saved histograms: {hist_path}")
    print(f"Saved ROC curves: {roc_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
