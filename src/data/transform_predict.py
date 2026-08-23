"""
transformation.py -- minimal PlantCV-based leaf transformation, used by
predict.py to show a transformed view next to the original image.

Isolates the leaf from its background using a standard PlantCV
segmentation workflow (grayscale channel -> Otsu threshold -> mask
cleanup -> apply mask). This is a single importable function, not a CLI --
unlike augmentation.py it doesn't need to run as a separate process, so
predict.py can just call it directly.

NOT tested end-to-end against a running plantcv install (no environment
with plantcv available while writing this) -- run it and check the output
looks right before relying on it.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from plantcv import plantcv as pcv


def transform_leaf(image_path: Path) -> Image.Image:
    pcv.params.debug = None

    img, _, _ = pcv.readimage(filename=str(image_path))  # BGR, like cv2.imread

    gray = pcv.rgb2gray_lab(rgb_img=img, channel="a")

    # If the mask comes out inverted (background isolated instead of the
    # leaf), flip this to "light" -- which one is correct depends on
    # whether the leaf is darker or lighter than the background in the
    # LAB "a" channel for this dataset.
    bin_mask = pcv.threshold.otsu(gray_img=gray, object_type="dark")

    clean_mask = pcv.gaussian_blur(img=bin_mask,
                                   ksize=(5, 5), sigma_x=0, sigma_y=None)
    masked = pcv.apply_mask(img=img, mask=clean_mask, mask_color="white")

    masked_rgb = masked[:, :, ::-1]  # BGR -> RGB
    return Image.fromarray(masked_rgb.astype(np.uint8))
