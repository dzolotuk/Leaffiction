"""
train.py -- CLI entry point for training the LeafDiseaseClassifier.

Usage:
    python train.py ./Apple/
    python train.py ./Apple/ ./Grape/

One or more directories with RAW images organized by class. Passing several
directories trains a single model across all of their classes combined
(e.g. one model for both apple and grape diseases at once) instead of one
model per directory. Class names must be unique across all directories
passed in -- that's already true here since they're prefixed per plant
type (Apple_healthy, Grape_healthy, ...).

    ./Apple/apple_healthy/
    ./Apple/apple_black_rot/
    ./Grape/grape_healthy/
    ./Grape/grape_black_rot/
    ...

This file only does CLI orchestration -- the actual work lives in:
  - src/utils/dataset.py -- split_dataset/prepare_dataset
                            /build_dataloaders/compute
  - src/ml/model.py      -- LeafDiseaseClassifier/create_model
                        /unfreeze_last_blocks
  - src/ml/trainer.py    -- train_epoch/validate/train_stage
                        /evaluate_with_predictions
  - src/utils/export.py  -- package_zip

Optional flags (--epochs, --batch-size, etc.) exist, but by default
`./train.py ./Apple/` works out of the box with sensible values.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from src.utils.dataset import prepare_dataset, build_dataloaders, compute
from src.ml.model import create_model, unfreeze_last_blocks
from src.ml.trainer import train_stage, evaluate_with_predictions
from src.utils.export import package_zip


import argparse
import random
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "raw_dirs", nargs="+",
        help="one or more directories with raw photos by class",
    )
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--output-zip", default="learnings.zip")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    raw_dirs = [Path(d) for d in args.raw_dirs]
    if args.workdir:
        prepared_dir = Path(args.workdir)
    else:
        combined_name = "_".join(d.name for d in raw_dirs)
        prepared_dir = raw_dirs[0].parent / f"{combined_name}_prepared"

    train_dir, val_dir = prepare_dataset(raw_dirs, prepared_dir,
                                         args.val_ratio, args.seed)

    train_loader, val_loader, train_dataset = build_dataloaders(
        train_dir, val_dir, args.batch_size, args.num_workers
    )
    class_to_idx = train_dataset.class_to_idx
    num_classes = len(class_to_idx)
    print(f"classes ({num_classes}): {list(class_to_idx.keys())}")

    model = create_model(num_classes).to(device)
    class_weights = compute(train_dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    checkpoint_path = prepared_dir / "model.pth"

    # --- stage 1: train only the head (backbone is frozen in create_model) ---
    optimizer = torch.optim.AdamW(model.model.classifier[1].parameters(),
                                  lr=args.lr)
    best_acc = train_stage(
        model, train_loader, val_loader, criterion, optimizer, device,
        args.epochs, best_acc=0.0, checkpoint_path=checkpoint_path,
        class_to_idx=class_to_idx, stage_name="head",
    )

    # --- stage 2: unfreeze the last backbone blocks and fine-tune---
    if args.finetune_epochs > 0:
        unfreeze_last_blocks(model, n_blocks=2)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.finetune_lr)
        best_acc = train_stage(
            model, train_loader, val_loader, criterion, optimizer, device,
            args.finetune_epochs, best_acc=best_acc,
            checkpoint_path=checkpoint_path,
            class_to_idx=class_to_idx, stage_name="finetune",
        )

    # --- final report on the best checkpoint ---
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    preds, labels = evaluate_with_predictions(model, val_loader, device)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    target_names = [idx_to_class[i] for i in range(num_classes)]

    print("\n=== classification report (val) ===")
    print(classification_report(labels, preds,
                                target_names=target_names, zero_division=0))
    print("=== confusion matrix (val) ===")
    print(confusion_matrix(labels, preds))
    print(f"\nbest val_acc: {best_acc:.2f}%")

    package_zip(Path(args.output_zip),
                checkpoint_path, train_dir, class_to_idx)


if __name__ == "__main__":
    main()
