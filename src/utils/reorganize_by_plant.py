"""
One-off helper: splits a flat images/<Plant_Disease>/ folder (from leaves.zip)
into separate per-plant-type directories, since train.py expects one plant
type per run (matching the subject's ./Apple/ example).

Usage:
    python reorganize_by_plant.py leaves_extracted/images leaves_by_plant
"""

import shutil
import sys
from pathlib import Path


def main():
    source = Path(sys.argv[1])
    dest = Path(sys.argv[2])

    for class_dir in sorted(source.iterdir()):
        if not class_dir.is_dir():
            continue

        plant_type = class_dir.name.split("_", 1)[0]
        dest_class_dir = dest / plant_type / class_dir.name
        dest_class_dir.mkdir(parents=True, exist_ok=True)

        for img in class_dir.iterdir():
            if img.is_file():
                shutil.copy2(img, dest_class_dir / img.name)

        print(f"{class_dir.name} -> {dest_class_dir}")

    print(f"\ndone -> {dest}")


if __name__ == "__main__":
    main()
