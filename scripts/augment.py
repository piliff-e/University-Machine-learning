#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    Rand2DElastic,
    # Intensity transforms
    RandAdjustContrast,
    RandAffine,
    # RandBiasField,
    # Dropouts
    RandCoarseDropout,
    RandFlip,
    # Spatial
    RandGaussianNoise,
    RandGaussianSharpen,
    RandGaussianSmooth,
    #    RandGibbsNoise,
    RandGridDistortion,
    RandHistogramShift,
    RandRotate,
    RandShiftIntensity,
    RandZoom,
    ToTensor,
)


def make_transforms():
    """
    Полный набор аугментаций для 2D-изображения.
    Все включены, но отключены низкой вероятностью или мягкими параметрами.
    Пользователь может просто закомментировать лишнее.
    """
    return Compose(
        [
            # ——————————————————————————
            # БАЗА: канал первым
            # ——————————————————————————
            EnsureChannelFirst(channel_dim=-1),
            # ——————————————————————————
            # SPATIAL TRANSFORMS
            # ——————————————————————————
            RandFlip(prob=0.5, spatial_axis=1),  # горизонтальный флип
            RandFlip(prob=0.5, spatial_axis=0),  # вертикальный флип
            RandRotate(range_x=15, prob=0.4),  # поворот ±15°
            RandZoom(prob=0.3, min_zoom=0.9, max_zoom=1.1),  # зум
            RandAffine(
                prob=0.3,
                translate_range=(10, 10),
                rotate_range=0.1,
                scale_range=(0.1, 0.1),
                shear_range=(0.1, 0.1),
            ),
            # 2D-эластическая
            Rand2DElastic(
                spacing=(20, 20),
                magnitude_range=(1, 2),
                prob=0.2,
            ),
            # Дисторсии сетки
            RandGridDistortion(
                distort_limit=0.2,
                prob=0.25,
            ),
            # ——————————————————————————
            # NOISE / ARTIFACTS
            # ——————————————————————————
            RandGaussianNoise(prob=0.3, std=0.02),
            # RandRicianNoise(prob=0.1),
            # RandGibbsNoise(prob=0.05),  # имитация undersampling
            # ——————————————————————————
            # INTENSITY / CONTRAST / BLUR
            # ——————————————————————————
            RandAdjustContrast(prob=0.4, gamma=(0.8, 1.2)),
            RandGaussianSmooth(prob=0.25, sigma_x=(0.5, 1.5)),
            RandGaussianSharpen(prob=0.2, sigma1_x=(0.2, 1.0), sigma2_x=(0.5, 1.5)),
            # Сдвиг гистограммы (имитирует другое оборудование)
            RandHistogramShift(prob=0.2, num_control_points=35),
            # Bias field — полезно, если данные МРТ, можно выключить
            # RandBiasField(prob=0.15),
            # ——————————————————————————
            # DROPOUT / CUTOUT
            # ——————————————————————————
            RandCoarseDropout(
                prob=0.15,
                holes=10,
                spatial_size=(10, 10),
            ),
            # ——————————————————————————
            # INTENSITY SHIFT
            # ——————————————————————————
            RandShiftIntensity(offsets=0.1, prob=0.2),
            # ——————————————————————————
            # OUTPUT
            # ——————————————————————————
            ToTensor(),
        ]
    )


def augment_image(img: np.ndarray, transforms, num_aug: int):
    """
    Генерирует num_aug аугментированных версий изображения.

    Возвращает:
        List[numpy.ndarray] — список изображений в формате (H, W, 3) или (H, W), dtype uint8.

    Пояснения по реализации:
    - transforms(img) может вернуть torch.Tensor или numpy-представление; корректно обрабатываем оба варианта.
    - После трансформации тензор имеет форму (C, H, W), поэтому применяем np.transpose(..., (1,2,0)), чтобы получить привычную форму (H, W, C) для отображения и сохранения.
    - Если канал один (C==1), упрощаем до (H, W).
    - MONAI иногда возвращает float в диапазоне [0,1], поэтому умножаем на 255 и приводим к uint8.
      Если значения уже в диапазоне 0..255, просто приводим тип и обрезаем.
    """
    outs = []
    for _ in range(num_aug):
        t = transforms(img)
        # Если MONAI вернул torch.Tensor, берём .numpy(), иначе используем np.array
        arr = t.numpy() if hasattr(t, "numpy") else np.array(t)

        # Преобразование формы: (C,H,W) -> (H,W,C)
        if arr.ndim == 3:
            img_out = np.transpose(arr, (1, 2, 0))
            # Если канал один — сведём к 2D массиву (H,W)
            if img_out.shape[2] == 1:
                img_out = img_out[:, :, 0]
        else:
            img_out = arr

        # Приведение к uint8 0..255:
        # - если значения в 0..1 (float), умножаем на 255
        # - иначе предполагаем, что значения уже в 0..255, просто обрезаем и приводим к uint8
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
    Сохраняет изображение.

    Почему две ветки if:
    - OpenCV (cv2.imwrite) ожидает порядок каналов BGR для цветных изображений.
      В коде используем RGB (более привычный формат для matplotlib и визуализации),
      поэтому при сохранении делаем cv2.cvtColor(img, COLOR_RGB2BGR).
    - Если изображение grayscale (2D) или канал один, сохраняем напрямую.
    """
    if img.ndim == 3 and img.shape[2] == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), img_bgr)
    else:
        cv2.imwrite(str(path), img)


def process_folder(input_dir: str, output_dir: str, num_aug: int):
    """
    Обходит входную папку, аугментирует каждый поддерживаемый файл и сохраняет
    num_aug результатов в выходную папку.
    """
    p_in = Path(input_dir)
    p_out = Path(output_dir)
    p_out.mkdir(parents=True, exist_ok=True)
    transforms = make_transforms()
    for fp in sorted(p_in.iterdir()):
        if fp.suffix.lower() not in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
            ".dcm",
        }:
            continue
        img_bgr = cv2.imread(str(fp))
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        aug_imgs = augment_image(img, transforms, num_aug)
        stem = fp.stem
        ext = fp.suffix.lower()
        for i, a in enumerate(aug_imgs, 1):
            out_name = f"{stem}_aug{i}{ext}"
            save_image(p_out / out_name, a)


def visualize_one(original_path: str, aug_paths: list):
    """
    Показать окно с оригиналом и до 9 аугментированных версий.
    """
    orig_bgr = cv2.imread(original_path)
    orig = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    aug_imgs = [
        cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in aug_paths[:9]
    ]
    fig, axes = plt.subplots(2, 5)  # 2 строки, 5 столбцов
    axes = axes.ravel()  # получить одномерный список осей для удобства индексирования
    axes[0].imshow(orig)
    axes[0].set_title("Original")
    axes[0].axis("off")
    for i in range(1, 10):
        if i - 1 < len(aug_imgs):
            axes[i].imshow(aug_imgs[i - 1])
            axes[i].set_title(f"aug{i}")
            axes[i].axis("off")
        else:
            axes[i].axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, default="raw", help="Directory with original images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="augmented",
        help="Output directory",
    )
    parser.add_argument(
        "--num_aug", type=int, default=9, help="Number of augmentations for each image"
    )
    args = parser.parse_args()
    process_folder(args.input_dir, args.output_dir, args.num_aug)

    """
    # Визуализация 9 аугментированных картинок для первой оригинальной
    first_input = sorted(Path(input_dir).glob("*"))[0]
    aug_list = sorted(
        Path(output_dir).glob(f"{first_input.stem}_aug*{first_input.suffix}")
    )
    visualize_one(str(first_input), aug_list)
    """
