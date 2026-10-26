# Leaffiction 

Computer-vision pipeline for **leaf disease classification**, built as part of the 42 Paris curriculum.

The project covers the full workflow: dataset analysis, data augmentation to fix class imbalance, image transformations with PlantCV, and a PyTorch classifier based on **EfficientNet-B0** (transfer learning + fine-tuning).

---

## Features

| Stage | Script | What it does |
|---|---|---|
| 1. Analysis | `src/data/distribution.py` | Pie + bar charts of the number of images per class |
| 2. Augmentation | `src/data/augmentation.py` | 6 augmentations per image: Flip, Rotate, Skew, Shear, Crop, Distortion |
| 3. Transformation | `src/data/transformation.py` | PlantCV pipeline: Gaussian blur, mask, ROI, object analysis, pseudolandmarks, color histogram |
| 4. Training | `train.py` | Train/val split → class balancing → two-stage EfficientNet-B0 training → `learnings.zip` |
| 5. Prediction | `predict.py` | Predicts the disease for one image (with visualisation) or a whole directory |

---

## Project structure

```
Leaffiction/
├── train.py                     # training CLI
├── predict.py                   # inference CLI
├── setup.sh                     # environment setup (uv + Python 3.11)
├── requirements.txt
├── signature_dataset.txt        # SHA-1 of the dataset archive
├── signature_learnings.txt      # SHA-1 of learnings.zip
├── src/
│   ├── data/
│   │   ├── distribution.py      # class distribution charts
│   │   ├── augmentation.py      # data augmentation
│   │   ├── transformation.py    # PlantCV transformations (CLI)
│   │   └── transform_predict.py # leaf segmentation used by predict.py
│   ├── ml/
│   │   ├── model.py             # LeafDiseaseClassifier (EfficientNet-B0)
│   │   └── trainer.py           # training / validation loops
│   └── utils/
│       ├── dataset.py           # split, balancing, DataLoaders, class weights
│       ├── export.py            # packaging into learnings.zip
│       └── reorganize_by_plant.py
└── eval/
    ├── split.py                 # standalone train/val split (stdlib only)
    └── reorganize_by_plant.py
```

---

## Installation

```bash
git clone https://github.com/dzolotuk/Leaffiction.git
cd Leaffiction
./setup.sh
source venv/bin/activate
```

`setup.sh` installs [uv](https://github.com/astral-sh/uv) if needed, a portable Python 3.11, creates a `venv/` and installs the dependencies — no `sudo` required.

Main dependencies: PyTorch, torchvision, PlantCV, OpenCV, scikit-learn, matplotlib.

---

## Dataset

The project uses the 42 `leaves.zip` dataset (Apple and Grape leaves). Images are expected to be organised **one folder per class**:

```
Apple/
├── Apple_healthy/
├── Apple_Black_rot/
├── Apple_rust/
└── Apple_scab/
Grape/
├── Grape_healthy/
├── Grape_Black_rot/
├── Grape_Esca/
└── Grape_spot/
```
You can use similar dataset from here: https://www.kaggle.com/datasets/mhantor/apple-leaf-diseases

If the archive extracts into a flat `images/<Plant_Disease>/` folder, regroup it by plant type:

```bash
python src/utils/reorganize_by_plant.py leaves/images leaves_by_plant
```

---

## Usage

### 1. Class distribution

```bash
python src/data/distribution.py ./Apple
```

### 2. Augmentation

```bash
# One image: saves 6 augmented copies next to it and displays them
python src/data/augmentation.py "./Apple/Apple_healthy/image (1).JPG"

# A directory: balances every class up to the size of the largest one
python src/data/augmentation.py ./Apple
```

Generated files are named `<image>_Flip.JPG`, `<image>_Rotate.JPG`, `<image>_Skew.JPG`, etc.

### 3. Transformation

```bash
# One image: displays all transformations + RGB color histogram
python src/data/transformation.py "./Apple/Apple_healthy/image (1).JPG"

# A directory: saves the transformations to -dst
python src/data/transformation.py -src ./Apple/Apple_healthy -dst ./transformed

# Only one type of transformation
python src/data/transformation.py -src ./Apple/Apple_healthy -dst ./masks -type Mask
```

Available types: `Original`, `Gaussian`, `Mask`, `ROI`, `Analyze`, `Pseudolandmarks`.

### 4. Training

```bash
python train.py ./Apple/                # one model for apple diseases
python train.py ./Apple/ ./Grape/       # one model for all classes combined
```

| Option | Default | Description |
|---|---|---|
| `--val-ratio` | `0.2` | Share of images kept for validation |
| `--epochs` | `15` | Epochs for stage 1 (classification head only) |
| `--finetune-epochs` | `5` | Epochs for stage 2 (fine-tuning, `0` to skip) |
| `--lr` | `1e-3` | Learning rate, stage 1 |
| `--finetune-lr` | `1e-5` | Learning rate, stage 2 |
| `--batch-size` | `32` | Batch size |
| `--num-workers` | `4` | DataLoader workers |
| `--workdir` | `<dirs>_prepared` | Where the prepared train/val split is stored |
| `--output-zip` | `learnings.zip` | Output archive |
| `--seed` | `42` | Random seed |

At the end, a classification report and a confusion matrix are printed for the validation set.

### 5. Prediction

```bash
# One image: shows original vs transformed leaf + predicted class and confidence
python predict.py "./Apple/Apple_healthy/image (1).JPG"

# A directory: recursive predictions printed to the console
python predict.py ./test_images/

# Use a specific model
python predict.py image.JPG --model path/to/model.pth
```

`--model` accepts either `learnings.zip` (default) or a `.pth` checkpoint.

---

## How it works

### Data preparation

1. **Split first, augment second.** Raw images are split into `train/` and `val/` per class (seeded, 80/20 by default) *before* any augmentation, so augmented copies of a photo can never leak into the validation set.
2. **Balancing.** Minority classes in `train/` are augmented (6 new images per call) until they roughly match the largest class. The validation set stays untouched.
3. **Weighted loss.** Inverse-frequency class weights are passed to `CrossEntropyLoss` to compensate for any remaining imbalance.

### Model

- **Backbone:** EfficientNet-B0 pretrained on ImageNet (`torchvision`), with ImageNet preprocessing.
- **Head:** the final linear layer is replaced to output one logit per class.
- **Stage 1 – head training:** backbone frozen, only the classifier is trained (AdamW, `lr=1e-3`).
- **Stage 2 – fine-tuning:** the last 2 backbone blocks are unfrozen and trained with a much lower learning rate (`1e-5`).
- `ReduceLROnPlateau` on validation accuracy; the best checkpoint (by val accuracy) is kept.

### Output

`learnings.zip` contains:

```
learnings.zip
├── model.pth            # best weights + class mapping
├── class_to_idx.json    # class name → index
└── images/              # augmented training set, by class
```

---

## Signatures

The SHA-1 hashes of the dataset and of the trained model archive are stored in `signature_dataset.txt` and `signature_learnings.txt`. To verify:

```bash
sha1sum learnings.zip
```

---

## Tech stack

Python 3.11 · PyTorch · torchvision · PlantCV · OpenCV · scikit-learn · NumPy · matplotlib

## Authors

- [Dmytro Zolotukhin](https://github.com/dzolotuk) — 42 Paris
