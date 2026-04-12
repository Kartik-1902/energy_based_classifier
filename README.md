# Energy-Based Classifier for CIFAR-10

This project implements an energy-based classifier inspired by EPOTTA and JEM.
It trains a CNN on CIFAR-10, computes per-class energy profiles, and performs
energy-based prediction plus OOD rejection.

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

Per-class profile for class `k` is built from all CIFAR-10 train samples with label `k`:

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
. .venv/Scripts/activate
pip install -r requirements.txt
```

## Train

```bash
python train.py --model-name resnet18 --epochs 200 --batch-size 128
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

## Compute Profiles Separately

```bash
python profile_energy.py --checkpoint ./checkpoints/best_model.pth --output ./checkpoints/energy_profiles.pkl
```

This prints class-wise summary: class, `mu`, `sigma`, `min`, `max`, `count`.

## Single-Image Inference

```bash
python inference.py path/to/image.png --checkpoint ./checkpoints/best_model.pth --profiles ./checkpoints/energy_profiles.pkl --tau 3.0 --plot
```

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
python evaluate.py --checkpoint ./checkpoints/best_model.pth --profiles ./checkpoints/energy_profiles.pkl --ood-dataset cifar100
```

Optional temperature calibration:

```bash
python evaluate.py --calibrate-temperature
```

Reported metrics:

- Softmax accuracy and ECE
- Energy-based accuracy and ECE
- Per-class accuracy (energy predictor)
- OOD AUROC and FPR@95TPR using marginal energy scores
- Reliability diagrams and OOD separation plot in `results/`

## Expected Targets (Typical)

- CIFAR-10 accuracy around 93-95% (ResNet-18)
- Energy-based accuracy close to softmax
- OOD AUROC versus CIFAR-100 often above 0.85
- ECE improved with calibration/energy-based confidence

## Notes

- Models output raw logits (no softmax in `forward`).
- `sigma_k` is clamped with a small epsilon for numerical stability.
- OOD score uses marginal energy where larger values indicate more OOD-like inputs.
