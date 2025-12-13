#!/usr/bin/env python3
"""
Скрипт для аугментации изображений исходного датасета.

Назначение:
- увеличение объёма обучающей выборки;
- повышение устойчивости моделей к геометрическим и фотометрическим искажениям;
- генерация нескольких аугментированных версий каждого изображения.

Используется библиотека MONAI, предоставляющая набор универсальных трансформаций
для 2D-изображений.
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    Rand2DElastic,
    RandAdjustContrast,
    RandAffine,
    RandCoarseDropout,
    RandFlip,
    RandGaussianNoise,
    RandGaussianSharpen,
    RandGaussianSmooth,
    RandGridDistortion,
    RandHistogramShift,
    RandRotate,
    RandShiftIntensity,
    RandZoom,
    ToTensor,
)


def make_transforms():
    """
    Формирует композицию аугментаций для 2D-изображений.

    Включает пространственные, шумовые и фотометрические преобразования
    с умеренными вероятностями применения.
    """
    return Compose(
        [
            EnsureChannelFirst(channel_dim=-1),
            RandFlip(prob=0.5, spatial_axis=1),
            RandFlip(prob=0.5, spatial_axis=0),
            RandRotate(range_x=15, prob=0.4),
            RandZoom(prob=0.3, min_zoom=0.9, max_zoom=1.1),
            RandAffine(
                prob=0.3,
                translate_range=(10, 10),
                rotate_range=0.1,
                scale_range=(0.1, 0.1),
                shear_range=(0.1, 0.1),
            ),
            Rand2DElastic(spacing=(20, 20), magnitude_range=(1, 2), prob=0.2),
            RandGridDistortion(distort_limit=0.2, prob=0.25),
            RandGaussianNoise(prob=0.3, std=0.02),
            RandAdjustContrast(prob=0.4, gamma=(0.8, 1.2)),
            RandGaussianSmooth(prob=0.25, sigma_x=(0.5, 1.5)),
            RandGaussianSharpen(prob=0.2, sigma1_x=(0.2, 1.0), sigma2_x=(0.5, 1.5)),
            RandHistogramShift(prob=0.2, num_control_points=35),
            RandCoarseDropout(prob=0.15, holes=10, spatial_size=(10, 10)),
            RandShiftIntensity(offsets=0.1, prob=0.2),
            ToTensor(),
        ]
    )


def augment_image(img: np.ndarray, transforms, num_aug: int):
    """
    Генерирует несколько аугментированных версий одного изображения.

    Параметры:
        img: исходное изображение (RGB).
        transforms: композиция аугментаций MONAI.
        num_aug: число аугментированных копий.

    Возвращает:
        Список изображений в формате numpy.ndarray (uint8).
    """
    outs = []
    for _ in range(num_aug):
        t = transforms(img)
        arr = t.numpy() if hasattr(t, "numpy") else np.array(t)

        if arr.ndim == 3:
            img_out = np.transpose(arr, (1, 2, 0))
            if img_out.shape[2] == 1:
                img_out = img_out[:, :, 0]
        else:
            img_out = arr

        if img_out.dtype != np.uint8:
            img_out = (
                (img_out * 255).astype(np.uint8)
                if img_out.max() <= 1.0
                else img_out.clip(0, 255).astype(np.uint8)
            )

        outs.append(img_out)
    return outs


def save_image(path: Path, img: np.ndarray):
    """
    Сохраняет изображение на диск.

    Цветные изображения сохраняются с преобразованием RGB → BGR
    для совместимости с OpenCV.
    """
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img)


def process_folder(input_dir: str, output_dir: str, num_aug: int):
    """
    Выполняет аугментацию всех изображений в папке input_dir
    и сохраняет результаты в output_dir.
    """
    p_in = Path(input_dir)
    p_out = Path(output_dir)
    p_out.mkdir(parents=True, exist_ok=True)

    transforms = make_transforms()

    for fp in sorted(p_in.iterdir()):
        if fp.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            continue

        img = cv2.cvtColor(cv2.imread(str(fp)), cv2.COLOR_BGR2RGB)
        aug_imgs = augment_image(img, transforms, num_aug)

        for i, a in enumerate(aug_imgs, 1):
            save_image(p_out / f"{fp.stem}_aug{i}{fp.suffix}", a)


def visualize_one(original_path: str, aug_paths: list):
    """
    Визуализирует оригинальное изображение и несколько его аугментаций.
    """
    orig = cv2.cvtColor(cv2.imread(original_path), cv2.COLOR_BGR2RGB)
    aug_imgs = [
        cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in aug_paths[:9]
    ]

    fig, axes = plt.subplots(2, 5)
    axes = axes.ravel()

    axes[0].imshow(orig)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for i in range(1, 10):
        if i - 1 < len(aug_imgs):
            axes[i].imshow(aug_imgs[i - 1])
            axes[i].set_title(f"aug{i}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="raw")
    parser.add_argument("--output_dir", type=str, default="augmented")
    parser.add_argument("--num_aug", type=int, default=9)
    args = parser.parse_args()

    process_folder(args.input_dir, args.output_dir, args.num_aug)
