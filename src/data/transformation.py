import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
import argparse

IMAGE_EXTENSIONS = [
    ".jpg", ".JPG",
    ".png", ".PNG",
    ".jpeg", ".JPEG"
]

TRANSFORMATION_TYPES = [
    "Original", "Gaussian", "Mask", "ROI", "Analyze", "Pseudolandmarks"
]

pcv.params.debug = "none"


def transform_image(image_path):
    img, path, filename = pcv.readimage(filename=image_path)

    # Gaussian
    gray_img = pcv.rgb2gray_hsv(rgb_img=img, channel="s")
    gaussian = pcv.gaussian_blur(img=gray_img, ksize=(51, 51))

    # Mask
    mask = pcv.threshold.binary(
        gray_img=gaussian, threshold=50, object_type="light"
        )

    # ROI
    roi_img = pcv.apply_mask(img=img, mask=mask, mask_color="white")

    # Analyze object
    analyze_img = pcv.analyze.size(img=img, labeled_mask=mask)

    # Pseudolandmarks
    top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(
        img=img, mask=mask
        )

    landmark_img = img.copy()
    for point in top:
        cv2.circle(
            landmark_img,
            (int(point[0][0]), int(point[0][1])),
            5, (255, 0, 0), -1
        )
    for point in bottom:
        cv2.circle(
            landmark_img,
            (int(point[0][0]), int(point[0][1])),
            5, (0, 255, 0), -1
        )
    for point in center_v:
        cv2.circle(
            landmark_img,
            (int(point[0][0]), int(point[0][1])),
            5, (0, 0, 255), -1
        )

    return img, gaussian, mask, roi_img, analyze_img, landmark_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", help="path to image")
    parser.add_argument("-src", help="src directory")
    parser.add_argument("-dst", help="dest directory")
    parser.add_argument(
        "-type", choices=TRANSFORMATION_TYPES,
        help="apply only this type of transformation to the directory (default: all)"
    )
    args = parser.parse_args()

    if args.src and args.dst:
        os.makedirs(args.dst, exist_ok=True)
        for file in os.listdir(args.src):
            path = os.path.join(args.src, file)
            _, extension = os.path.splitext(file)
            if extension in IMAGE_EXTENSIONS:
                (img, gaussian, mask, roi_img, analyze_img,
                    landmark_img) = transform_image(path)
                base = os.path.splitext(file)[0]

                def out_path(suffix, base=base, extension=extension):
                    return os.path.join(
                        args.dst, base + suffix + extension
                    )

                outputs = {
                    "Original": img,
                    "Gaussian": gaussian,
                    "Mask": mask,
                    "ROI": roi_img,
                    "Analyze": analyze_img,
                    "Pseudolandmarks": landmark_img,
                }

                if args.type:
                    cv2.imwrite(out_path(args.type), outputs[args.type])
                else:
                    for suffix, image in outputs.items():
                        cv2.imwrite(out_path(suffix), image)
    else:
        (img, gaussian, mask, roi_img, analyze_img,
         landmark_img) = transform_image(args.image)

        fig, axes = plt.subplots(1, 6, figsize=(30, 5))
        images = [
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            gaussian,
            mask,
            roi_img,
            analyze_img,
            landmark_img,
        ]

        titles = [
            "Original",
            "Gaussian",
            "Mask",
            "ROI",
            "Analyze",
            "Pseudolandmarks",
        ]

        for ax, image, title in zip(axes, images, titles):
            if title in ["Gaussian", "Mask"]:
                ax.imshow(image, cmap="gray")
            else:
                ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax.set_title(title)
            ax.axis("off")

        plt.show()

        fig2, ax = plt.subplots(figsize=(10, 4))

        colors = ["red", "green", "blue"]
        channel_names = ["Red", "Green", "Blue"]

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for i, (color, name) in enumerate(zip(colors, channel_names)):
            hist, bins = np.histogram(
                img_rgb[:, :, i], bins=256, range=(0, 256)
                )
            ax.plot(hist, color=color, label=name)

        ax.set_xlabel("Pixel intensity")
        ax.set_ylabel("Proportion of pixels")
        ax.set_title("Color histogram")
        ax.legend()

        plt.show()
