"""
dataset.py -- data preparation: splitting raw images into train/val,
calling the teammate's Augmentation.[extension] to balance train/, and
building DataLoaders for training.
"""

import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import EfficientNet_B0_Weights
from src.data.augmentation import augment_image

import warnings
warnings.filterwarnings("ignore")

MIN_VAL_IMAGES = 100  # subject requirement: at least 100 images in val
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
    classes into a single dataset -- useful for training one model across
    several plant types at once instead of one model per directory.

    The split happens BEFORE any augmentation -- val stays untouched and
    honest, as the subject requires (augmented copies of the same photo
    must never end up split across train and val).
    """
    rng = random.Random(seed)

    class_dirs = []
    seen_names = set()
    for raw_dir in raw_dirs:
        for class_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
            if class_dir.name in seen_names:
                raise ValueError(
                    f"Class '{class_dir.name}' appears in more than one dir "
                    "-- class names must be unique across all dirs passed in."
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


def _run_augmentation(image_path: Path) -> None:
    """
    Calls the teammate's Augmentation.[extension] on a single image. It has
    no CLI flags, no import guard, and everything at module level, so
    subprocess is the only way to use it (importing it would execute the
    whole script immediately). It also calls plt.show() unconditionally at
    the end -- MPLBACKEND=Agg makes that a no-op instead of blocking on a
    GUI window, since we call this in a loop with no one there to close it.
    """
    augment_image(str(image_path), show=False)


def balance_with_augmentation(train_dir: Path, seed: int = 42) -> None:
    """
    Balances train/ using the teammate's Augmentation.[extension]
    (Flip/Rotate/Skew/Shear/Crop/Distortion, subject Part 2). Each call on
    one image produces exactly 6 new images saved next to it
    ("<name>_Flip<ext>", "<name>_Rotate<ext>", ...), so classes are brought
    up to the target count in steps of 6 (slightly overshooting the target
    is fine -- exact parity isn't required).

    Each call is a whole separate Python process (cv2 + matplotlib import
    from scratch every time), so with a large deficit this can mean
    hundreds of subprocess launches. They're independent, so they run in a
    thread pool instead of one at a time -- threads are fine here since
    each one just blocks on subprocess.run, releasing the GIL while it
    waits. Progress is printed per class so this doesn't look stuck.
    """
    class_dirs = sorted(p for p in train_dir.iterdir() if p.is_dir())
    if not class_dirs:
        return

    rng = random.Random(seed)
    counts = {class_dir: list_images(class_dir) for class_dir in class_dirs}
    target = max(len(images) for images in counts.values())
    max_workers = min(32, (os.cpu_count() or 4) * 2)

    for class_dir, images in counts.items():
        if not images:
            continue

        deficit = target - len(images)
        if deficit <= 0:
            continue

        n_calls = -(-deficit // 6)  # ceil division: images needed / 6 per call
        pool = images.copy()
        rng.shuffle(pool)
        sources = [pool[i % len(pool)] for i in range(n_calls)]

        print(f"Balancing {class_dir.name}: {len(images)} -> {target}"
              "({n_calls} augmentation calls, {max_workers} workers)...")

        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_augmentation, src)
                       for src in sources]
            for future in as_completed(futures):
                future.result()
                done += 1
                if done % 20 == 0 or done == n_calls:
                    print(f"  {class_dir.name}: {done}/{n_calls} done")


def prepare_dataset(raw_dirs: list[Path], prepared_dir: Path,
                    val_ratio: float, seed: int) -> tuple[Path, Path]:
    train_dir = prepared_dir / "train"
    val_dir = prepared_dir / "val"

    if train_dir.exists() and val_dir.exists():
        return train_dir, val_dir  # already prepared earlier, don't redo it

    print("Splitting raw images into train/val...")
    split_dataset(raw_dirs, train_dir, val_dir, val_ratio, seed)
    print("Split done. Balancing train/ via augmentation...")
    balance_with_augmentation(train_dir, seed)
    print("Balancing done.")

    return train_dir, val_dir


def build_dataloaders(train_dir: Path, val_dir: Path,
                      batch_size: int, num_workers: int):
    preprocess = EfficientNet_B0_Weights.DEFAULT.transforms()

    train_dataset = ImageFolder(str(train_dir), transform=preprocess)
    val_dataset = ImageFolder(str(val_dir), transform=preprocess)

    if train_dataset.classes != val_dataset.classes:
        raise ValueError(
            "Classes in train/ and val/ don't match: "
            f"{train_dataset.classes} vs {val_dataset.classes}."
        )

    if len(val_dataset) < MIN_VAL_IMAGES:
        print(
            f"WARNING: val/ only has {len(val_dataset)} images, "
            f"but the subject requires at least {MIN_VAL_IMAGES}. "
        )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_dataset


def compute(dataset: ImageFolder, device: torch.device) -> torch.Tensor:
    counts = np.bincount([label for _, label in dataset.samples],
                         minlength=len(dataset.classes))
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)
