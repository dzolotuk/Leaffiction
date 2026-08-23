"""
predict.py -- inference for LeafDiseaseClassifier.

Usage:
    python predict.py "./Apple/apple_healthy/image (1).JPG"
    python predict.py "./Apple"

By default it loads weights from learnings.zip (what train.py packages), but
you can point directly to a .pth via --model.

If a single image is provided, displays the original image next to the
transformed one (transform_leaf, using PlantCV), and prints the predicted
disease class with the model's confidence.

If a directory is provided, all images in its subdirectories are processed
recursively and their predictions are printed without displaying images.
"""

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.models import EfficientNet_B0_Weights

from src.ml.model import create_model
from src.data.transform_predict import transform_leaf

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def load_learnings(model_path: Path):
    """
    Accepts either learnings.zip (from train.py) or a .pth directly.
    Returns (state_dict, class_to_idx).
    """
    if model_path.suffix == ".zip":
        tmp_dir = Path(tempfile.mkdtemp())

        with zipfile.ZipFile(model_path) as zf:
            zf.extract("model.pth", tmp_dir)
            zf.extract("class_to_idx.json", tmp_dir)

        checkpoint = torch.load(tmp_dir / "model.pth", map_location="cpu")

        with open(tmp_dir / "class_to_idx.json", encoding="utf-8") as f:
            class_to_idx = json.load(f)
        return checkpoint["model_state"], class_to_idx
    
    checkpoint = torch.load(model_path, map_location="cpu")
    return checkpoint["model_state"], checkpoint["class_to_idx"]


def predict(image_path: Path, model, preprocess, idx_to_class):
    """
    Predicts the disease class for one image.
    """
    original = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(original).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax())

    predicted_class = idx_to_class[pred_idx]
    confidence = float(probs[pred_idx])

    transformed = transform_leaf(image_path)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(transformed)
    axes[1].set_title("Transformed")
    axes[1].axis("off")

    fig.suptitle(f"{predicted_class} ({confidence:.1%})")
    plt.show()

    print(f"Class: {predicted_class} (confidence: {confidence:.1%})")

    return predicted_class, confidence


def predict_directory(image_dir: Path, model, preprocess, idx_to_class):
    """
    Predicts the disease class for all images in a directory recursively.
    """
    image_paths = sorted(
    path
    for path in image_dir.rglob("*")
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print(f"No images found in: {image_dir}")
        return

    print(f"Found {len(image_paths)} images.")
    print()

    for image_path in image_paths:
        try:
            original = Image.open(image_path).convert("RGB")
            input_tensor = preprocess(original).unsqueeze(0)

            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)[0]
                pred_idx = int(probs.argmax())

            predicted_class = idx_to_class[pred_idx]
            confidence = float(probs[pred_idx])

            print(
                f"{image_path}: "
                f"{predicted_class} (confidence: {confidence:.1%})"
            )

        except Exception as e:
            print(f"{image_path}: error: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--model", default="learnings.zip",
    help="learnings.zip or a path to a .pth")
    args = parser.parse_args()

    image_path = Path(args.image_path)

    if not image_path.exists():
        raise ValueError(f"Path does not exist: {image_path}")

    state_dict, class_to_idx = load_learnings(Path(args.model))
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    model = create_model(num_classes)
    model.load_state_dict(state_dict)
    model.eval()

    preprocess = EfficientNet_B0_Weights.DEFAULT.transforms()

    if image_path.is_file():
        predict(
            image_path,
            model,
            preprocess,
            idx_to_class,
        )
    elif image_path.is_dir():
        predict_directory(
            image_path,
            model,
            preprocess,
            idx_to_class,
        )

if __name__ == "__main__":
    main()

