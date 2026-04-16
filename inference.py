"""Run single-image inference with energy-profile-based OOD rejection."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config import DEFAULT_CONFIG, class_names_from_indices
from energy import energy_predict
from model import build_model


def build_inference_transform() -> transforms.Compose:
    """Create preprocessing transform identical to evaluation preprocessing."""
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )


def load_artifacts(
    checkpoint_path: str,
    profiles_path: str,
    model_name: str,
    num_classes: int,
    device: torch.device,
):
    """Load trained model checkpoint and serialized energy profiles."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", model_name)
    num_classes = int(checkpoint.get("num_classes", num_classes))
    id_classes = tuple(checkpoint.get("id_classes", list(DEFAULT_CONFIG.id_classes)))
    class_names = checkpoint.get("id_class_names", class_names_from_indices(id_classes))

    model = build_model(model_name=model_name, num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with open(profiles_path, "rb") as f:
        profiles_blob = pickle.load(f)

    profiles = profiles_blob
    if isinstance(profiles_blob, dict) and "profiles" in profiles_blob:
        profiles = profiles_blob["profiles"]
        id_classes = tuple(profiles_blob.get("id_classes", list(id_classes)))
        class_names = profiles_blob.get("id_class_names", class_names)

    return model, profiles, list(class_names)


def plot_z_scores(z_scores: torch.Tensor, class_names: list[str], output_path: str) -> None:
    """Visualize per-class z-scores for a single input image."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    values = z_scores.detach().cpu().numpy().reshape(-1)

    plt.figure(figsize=(10, 5))
    plt.bar(class_names, values)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Energy z-score")
    plt.title("Per-class Energy Z-scores")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for image inference."""
    parser = argparse.ArgumentParser(description="Single-image inference with energy-based rejection")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pth")
    parser.add_argument("--profiles", type=str, default="./checkpoints/energy_profiles.pkl")
    parser.add_argument("--model-name", type=str, default=DEFAULT_CONFIG.model_name)
    parser.add_argument("--temperature", type=float, default=DEFAULT_CONFIG.temperature)
    parser.add_argument("--tau", type=float, default=DEFAULT_CONFIG.tau)
    parser.add_argument("--plot", action="store_true", help="Save z-score bar chart")
    parser.add_argument("--plot-path", type=str, default="./results/inference_zscores.png")
    return parser.parse_args()


def main() -> None:
    """Entry point for single-image energy-based prediction."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, profiles, class_names = load_artifacts(
        checkpoint_path=args.checkpoint,
        profiles_path=args.profiles,
        model_name=args.model_name,
        num_classes=DEFAULT_CONFIG.num_classes,
        device=device,
    )

    image = Image.open(args.image).convert("RGB")
    tfm = build_inference_transform()
    x = tfm(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits / args.temperature, dim=1)
        softmax_conf, softmax_pred = probs.max(dim=1)
        pred, is_ood, z_scores = energy_predict(logits, profiles, tau=args.tau, temperature=args.temperature)

    min_z = z_scores[0, pred.item()].item()
    confidence = float(max(0.0, min(1.0, 1.0 - (min_z / max(args.tau, 1e-6)))))

    print(f"Image: {args.image}")
    print(f"Softmax baseline prediction: {class_names[softmax_pred.item()]} (p={softmax_conf.item():.4f})")
    print(f"Energy prediction: {class_names[pred.item()]}")
    print(f"Min energy z-score: {min_z:.4f}")
    print(f"Confidence (1 - min_z/tau): {confidence:.4f}")
    print(f"OOD rejected: {bool(is_ood.item())}")

    print("\nPer-class z-scores:")
    for k, cls_name in enumerate(class_names):
        print(f"  {cls_name:<11} -> {z_scores[0, k].item():.4f}")

    if args.plot:
        plot_z_scores(z_scores[0], class_names, args.plot_path)
        print(f"Saved z-score plot to {args.plot_path}")


if __name__ == "__main__":
    main()
