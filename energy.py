"""Energy computation utilities for energy-based classification and OOD rejection."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch


def marginal_energy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute JEM marginal energy: E(x) = -T * logsumexp(logits / T)."""
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


def class_energy(logits: torch.Tensor, class_idx: int) -> torch.Tensor:
    """Compute class-conditional energy: E(x, y) = -f(x)[y]."""
    return -logits[:, class_idx]


def compute_energy_profiles(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_classes: int = 10,
    device: torch.device | None = None,
) -> Dict[int, Dict[str, float]]:
    """Compute per-class mean/std profile for class-conditional energies over a dataset."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    per_class_energies: List[List[float]] = [[] for _ in range(num_classes)]

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)

            for k in range(num_classes):
                mask = labels == k
                if mask.any():
                    energies_k = class_energy(logits[mask], k)
                    per_class_energies[k].extend(energies_k.detach().cpu().numpy().tolist())

    profiles: Dict[int, Dict[str, float]] = {}
    eps = 1e-6
    for k in range(num_classes):
        values = np.array(per_class_energies[k], dtype=np.float64)
        if values.size == 0:
            raise ValueError(f"No samples found for class {k} while profiling energies.")
        sigma = float(values.std(ddof=0))
        profiles[k] = {
            "mu": float(values.mean()),
            "sigma": sigma if sigma > eps else eps,
            "min": float(values.min()),
            "max": float(values.max()),
            "count": int(values.size),
        }
    return profiles


def energy_predict(
    logits: torch.Tensor,
    profiles: Dict[int, Dict[str, float]],
    tau: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict class and OOD flag using z-score distance to per-class energy profiles."""
    num_classes = len(profiles)

    z_scores = []
    for k in range(num_classes):
        e_k = class_energy(logits, k)
        mu_k = profiles[k]["mu"]
        sigma_k = max(profiles[k]["sigma"], 1e-6)
        z_k = torch.abs((e_k - mu_k) / sigma_k)
        z_scores.append(z_k.unsqueeze(1))

    z_scores_tensor = torch.cat(z_scores, dim=1)
    min_z, preds = torch.min(z_scores_tensor, dim=1)
    is_ood = min_z > tau
    return preds, is_ood, z_scores_tensor
