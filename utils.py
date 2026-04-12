"""Utility functions for metrics, calibration, and visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np


def compute_ece(confidences: np.ndarray, predictions: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) with fixed-width confidence bins."""
    confidences = np.asarray(confidences, dtype=np.float64)
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        left, right = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (confidences >= left) & (confidences <= right)
        else:
            in_bin = (confidences >= left) & (confidences < right)

        if np.any(in_bin):
            acc = np.mean(predictions[in_bin] == labels[in_bin])
            conf = np.mean(confidences[in_bin])
            ece += np.mean(in_bin) * abs(acc - conf)

    return float(ece)


def plot_energy_distributions(
    energies_per_class: Iterable[np.ndarray],
    class_names: Iterable[str],
    save_path: str,
) -> None:
    """Plot overlaid histograms of class-conditional energies for each class."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 7))
    for energies, name in zip(energies_per_class, class_names):
        plt.hist(np.asarray(energies), bins=40, alpha=0.35, density=True, label=name)
    plt.xlabel("Class-conditional energy E(x, y) = -logit_y")
    plt.ylabel("Density")
    plt.title("Per-class Energy Distributions")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_ood_separation(in_energies: np.ndarray, ood_energies: np.ndarray, save_path: str) -> None:
    """Plot score distributions for in-distribution vs OOD marginal energies."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    in_energies = np.asarray(in_energies)
    ood_energies = np.asarray(ood_energies)

    plt.figure(figsize=(10, 6))
    plt.hist(in_energies, bins=50, alpha=0.6, density=True, label="CIFAR-10 (ID)")
    plt.hist(ood_energies, bins=50, alpha=0.6, density=True, label="OOD")
    plt.xlabel("Marginal energy E(x)")
    plt.ylabel("Density")
    plt.title("OOD Separation via Marginal Energy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def reliability_curve(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute confidence and accuracy values for reliability diagrams."""
    confidences = np.asarray(confidences, dtype=np.float64)
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    accs = np.zeros(n_bins, dtype=np.float64)
    confs = np.zeros(n_bins, dtype=np.float64)

    for i in range(n_bins):
        left, right = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (confidences >= left) & (confidences <= right)
        else:
            in_bin = (confidences >= left) & (confidences < right)

        if np.any(in_bin):
            accs[i] = np.mean(predictions[in_bin] == labels[in_bin])
            confs[i] = np.mean(confidences[in_bin])
        else:
            accs[i] = np.nan
            confs[i] = np.nan

    return bin_centers, accs, confs


def plot_calibration(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    title: str = "Reliability Diagram",
    n_bins: int = 10,
) -> None:
    """Plot and save a reliability diagram (confidence vs accuracy)."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    bin_centers, accs, confs = reliability_curve(confidences, predictions, labels, n_bins=n_bins)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Perfect calibration")
    valid = ~np.isnan(accs)
    plt.plot(confs[valid], accs[valid], marker="o", linewidth=2, label="Model")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
