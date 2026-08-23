"""
export.py -- packages the trained model and its training images into the
learnings.zip that the subject requires as train.py's output.
"""

import json
import zipfile
from pathlib import Path


def package_zip(output_zip: Path, checkpoint_path: Path,
                train_dir: Path, class_to_idx: dict) -> None:
    """Packages the learnings (weights + metadata)
    and augmented train images into a .zip."""
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(checkpoint_path, arcname="model.pth")
        zf.writestr("class_to_idx.json",
                    json.dumps(class_to_idx, ensure_ascii=False, indent=2))

        for image_path in train_dir.rglob("*"):
            if image_path.is_file():
                arcname = Path("images") / image_path.relative_to(train_dir)
                zf.write(image_path, arcname=str(arcname))

    print(f"\nlearnings .zip built: {output_zip}")
