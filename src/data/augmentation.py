import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


IMAGE_EXTENSIONS = [
    ".jpg", ".JPG",
    ".png", ".PNG",
    ".jpeg", ".JPEG"
]


def augment_image(image_path: str, show: bool = True, index: int = 0):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Flip
    image_flip = cv2.flip(image, 1)
    image_flip_rgb = cv2.cvtColor(image_flip, cv2.COLOR_BGR2RGB)

    # Rotate
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), 90, 1.0)
    image_rotate = cv2.warpAffine(image, M, (w, h))
    image_rotate_rgb = cv2.cvtColor(image_rotate, cv2.COLOR_BGR2RGB)

    # Skew
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[30, 30], [w - 30, 0], [w, h - 30], [0, h]])

    M = cv2.getPerspectiveTransform(src, dst)
    image_skew = cv2.warpPerspective(image, M, (w, h))
    image_skew_rgb = cv2.cvtColor(image_skew, cv2.COLOR_BGR2RGB)

    # Shear
    M = np.float32([
        [1, 0.3, 0],
        [0, 1, 0]
    ])

    image_shear = cv2.warpAffine(
        image,
        M,
        (w + int(0.3 * h), h)
    )

    image_shear = cv2.resize(image_shear, (w, h))
    image_shear_rgb = cv2.cvtColor(image_shear, cv2.COLOR_BGR2RGB)

    # Crop
    x1 = int(w * 0.1)
    y1 = int(h * 0.1)
    x2 = int(w * 0.9)
    y2 = int(h * 0.9)

    image_crop = image[y1:y2, x1:x2]
    image_crop = cv2.resize(image_crop, (w, h))
    image_crop_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)

    # Distortion
    K = np.float32([
        [w, 0, w / 2],
        [0, h, h / 2],
        [0, 0, 1]
    ])

    d = np.float32([0.3, -0.1, 0, 0])

    image_distortion = cv2.undistort(image, K, d)
    image_distortion_rgb = cv2.cvtColor(image_distortion, cv2.COLOR_BGR2RGB)

    # Save augmented images
    base, extension = os.path.splitext(image_path)

    suffix = "_" + str(index) if index > 0 else ""

    cv2.imwrite(base + "_Flip" + suffix + extension, image_flip)
    cv2.imwrite(base + "_Rotate" + suffix + extension, image_rotate)
    cv2.imwrite(base + "_Skew" + suffix + extension, image_skew)
    cv2.imwrite(base + "_Shear" + suffix + extension, image_shear)
    cv2.imwrite(base + "_Crop" + suffix + extension, image_crop)
    cv2.imwrite(base + "_Distortion" + suffix + extension, image_distortion)

    # Plot only if requested
    if show:
        fig, axes = plt.subplots(1, 7, figsize=(30, 5))

        images = [
            image_rgb,
            image_flip_rgb,
            image_rotate_rgb,
            image_skew_rgb,
            image_shear_rgb,
            image_crop_rgb,
            image_distortion_rgb
        ]

        titles = [
            "Original",
            "Flip",
            "Rotate",
            "Skew",
            "Shear",
            "Crop",
            "Distortion"
        ]

        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")

        plt.show()


def count_images(directory_path: str):

    directory = {}

    for element in os.listdir(directory_path):

        path = os.path.join(directory_path, element)

        if os.path.isdir(path):

            image_list = []
            for file in os.listdir(path):
                _, extension = os.path.splitext(file)
                if extension in IMAGE_EXTENSIONS:
                    chemin = os.path.join(path, file)
                    image_list.append(chemin)
            directory[element] = image_list

    maximum = max(len(images) for images in directory.values())

    for class_name, images in directory.items():
        missing_img = maximum - len(images)
        for i in range(0, missing_img, 6):
            augment_image(images[i % len(images)], show=False, index=i)


if __name__ == "__main__":

    image_path = sys.argv[1]

    if os.path.isdir(image_path):
        count_images(image_path)

    else:
        augment_image(image_path, show=True)
