#!/usr/bin/env python3
"""
split.py -- splits raw images into train/val with a fixed seed.

Usage:
    ./split.py ./Apple/ --output prepared --val-ratio 0.2 --seed 42
    ./split.py ./Apple/ ./Grape/ --output prepared --seed 42

Standalone -- no imports from the rest of the project (stdlib only), so it
can be run and graded entirely on its own.
"""

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(class_dir: Path) -> list[Path]:
    return sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_dataset(raw_dirs: list[Path], train_dir: Path, val_dir: Path,
                   val_ratio: float, seed: int) -> None:
    """
    Splits RAW images into train_dir/<class>/ and val_dir/<class>/.
    Accepts multiple raw_dirs (e.g. one per plant type) and merges their
    classes into a single dataset.
    """
    rng = random.Random(seed)

    class_dirs = []
    seen_names = set()
    for raw_dir in raw_dirs:
        for class_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
            if class_dir.name in seen_names:
                raise ValueError(
                    f"Class '{class_dir.name}' appears in more than one raw_dir "
                    "-- class names must be unique across all directories passed in."
                )
            seen_names.add(class_dir.name)
            class_dirs.append(class_dir)

    if not class_dirs:
        raise ValueError(f"No class subdirectories found in {raw_dirs}")

    for class_dir in class_dirs:
        images = list_images(class_dir)
        rng.shuffle(images)

        n_val = max(1, round(len(images) * val_ratio))
        val_images, train_images = images[:n_val], images[n_val:]

        train_class_dir = train_dir / class_dir.name
        val_class_dir = val_dir / class_dir.name
        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)

        for img in train_images:
            shutil.copy2(img, train_class_dir / img.name)
        for img in val_images:
            shutil.copy2(img, val_class_dir / img.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dirs", nargs="+", help="one or more directories with raw photos by class")
    parser.add_argument("--output", default="prepared", help="where to write train/ and val/")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="fraction of data used for validation")
    parser.add_argument("--seed", type=int, default=42, help="random seed for the split")
    args = parser.parse_args()

    raw_dirs = [Path(d) for d in args.raw_dirs]
    output_dir = Path(args.output)
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"

    split_dataset(raw_dirs, train_dir, val_dir, args.val_ratio, args.seed)

    train_count = sum(1 for p in train_dir.rglob("*") if p.is_file())
    val_count = sum(1 for p in val_dir.rglob("*") if p.is_file())
    print(f"train: {train_count} images -> {train_dir}")
    print(f"val:   {val_count} images -> {val_dir}")


if __name__ == "__main__":
    main()