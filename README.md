# Energy-Based Classifier for CIFAR-10 (8 ID / 2 OOD)

This project implements an energy-based classifier inspired by EPOTTA and JEM.
It trains a CNN on selected in-distribution CIFAR-10 classes, computes per-class
energy profiles, and performs energy-based prediction plus OOD rejection.

Current default setup:

- In-distribution (ID): CIFAR-10 classes `0..7`
- Held-out OOD: CIFAR-10 classes `8,9`

## Implemented Math

Given model logits `f(x)` and temperature `T`:

- Marginal energy (JEM):

  \[
  E(x) = -T \cdot \log \sum_{k=1}^{C} \exp(f(x)_k / T)
  \]

- Class-conditional energy:

  \[
  E(x, y) = -f(x)_y
  \]

Per-class profile for class `k` is built from all ID-train samples with local label `k`:

- `mu_k = mean(E(x, k))`
- `sigma_k = std(E(x, k))`

Inference rule:

- `z_k = |E(x, k) - mu_k| / sigma_k`
- `pred = argmin_k z_k`
- reject as OOD if `min_k z_k > tau`

## Project Files

- `train.py`: train model and save best checkpoint by validation accuracy
- `profile_energy.py`: compute and save per-class energy profiles
- `inference.py`: classify one image and optionally reject as OOD
- `evaluate.py`: evaluate ID accuracy/calibration and OOD detection
- `model.py`: ResNet-18 (CIFAR stem) and WideResNet-28-2
- `energy.py`: energy formulas, profiling, and energy decision logic
- `config.py`: global hyperparameters and defaults
- `utils.py`: ECE and plotting utilities

## Setup

```bash
cd energy_based_classifier
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .\.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
```

## Train (ID Classes Only)

```bash
python train.py --model-name resnet18 --epochs 200 --batch-size 128 --id-classes 0,1,2,3,4,5,6,7 --ood-classes 8,9
```

Outputs:

- `checkpoints/best_model.pth`
- `checkpoints/energy_profiles.pkl`
- `results/energy_distributions.png`

Training details:

- CIFAR-10 augmentation: `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`
- Optimizer: SGD with momentum
- Scheduler: cosine annealing
- Loss: cross-entropy

Notes:

- The classifier output dimension is the number of ID classes.
- With `--id-classes 0,1,2,3,4,5,6,7`, model head has 8 outputs.

## Compute Profiles Separately

```bash
python profile_energy.py --checkpoint ./checkpoints/best_model.pth --output ./checkpoints/energy_profiles.pkl --id-classes 0,1,2,3,4,5,6,7
```

This prints class-wise summary: class, `mu`, `sigma`, `min`, `max`, `count`.

## Single-Image Inference

```bash
python inference.py path/to/image.png --checkpoint ./checkpoints/best_model.pth --profiles ./checkpoints/energy_profiles.pkl --tau 3.0 --plot
```

Replace `path/to/image.png` with a real image file path.

Example output format:

```text
Image: ./example.png
Softmax baseline prediction: truck (p=0.9132)
Energy prediction: truck
Min energy z-score: 1.2041
Confidence (1 - min_z/tau): 0.5986
OOD rejected: False
```

## Evaluation

```bash
python evaluate.py --checkpoint ./checkpoints/best_model.pth --profiles ./checkpoints/energy_profiles.pkl --id-classes 0,1,2,3,4,5,6,7 --ood-classes 8,9 --ood-dataset heldout-cifar10
```

By default, `--ood-dataset heldout-cifar10` evaluates OOD rejection using the held-out CIFAR-10 classes.

Alternative OOD datasets are still available:

```bash
python evaluate.py --checkpoint ./checkpoints/best_model.pth --profiles ./checkpoints/energy_profiles.pkl --ood-dataset cifar100
python evaluate.py --checkpoint ./checkpoints/best_model.pth --profiles ./checkpoints/energy_profiles.pkl --ood-dataset svhn
```

Optional temperature calibration:

```bash
python evaluate.py --calibrate-temperature
```

Reported metrics:

- Softmax accuracy and ECE
- Energy-based accuracy and ECE
- Per-class accuracy (energy predictor, ID classes)
- OOD AUROC and FPR@95TPR using marginal energy scores
- Reliability diagrams and OOD separation plot in `results/`

## Experiment Design

- This repository is configured for OOD rejection, not novel-class discovery.
- Held-out classes should be rejected as unknown, not assigned new class indices.
- If you change `--id-classes`, keep `--ood-classes` disjoint and covering the remaining CIFAR-10 classes.

## Notes

- Models output raw logits (no softmax in `forward`).
- `sigma_k` is clamped with a small epsilon for numerical stability.
- OOD score uses marginal energy where larger values indicate more OOD-like inputs.
