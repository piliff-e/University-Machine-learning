#!/usr/bin/env python3
"""
Функции для извлечения признаков из изображений.

Реализованные признаки:
- HOG (Histogram of Oriented Gradients);
- моменты Ху;
- цветовые гистограммы в пространстве HSV.

Итоговый вектор признаков формируется конкатенацией всех признаков
и используется в классических алгоритмах машинного обучения.
"""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def is_image_path(p: Path) -> bool:
    """
    Проверяет, является ли путь файлом изображения поддерживаемого формата.
    """
    return p.suffix.lower() in IMG_EXTS and p.is_file()


def extract_hu_from_binary(gray_uint8: np.ndarray) -> np.ndarray:
    """
    Вычисляет моменты Ху по бинарному изображению
    с логарифмическим масштабированием.
    """
    moments = cv2.moments(gray_uint8)
    hu = cv2.HuMoments(moments).flatten()

    with np.errstate(divide="ignore", invalid="ignore"):
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)

    return hu


def extract_hsv_hist(rgb_uint8: np.ndarray, bins: int = 16) -> np.ndarray:
    """
    Вычисляет нормированную HSV-гистограмму изображения.
    """
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
    hist = []

    for ch in cv2.split(hsv)[:3]:
        h = cv2.calcHist([ch], [0], None, [bins], [0, 256]).flatten()
        h /= h.sum() + 1e-9
        hist.append(h)

    return np.concatenate(hist)


def extract_hog(gray_uint8: np.ndarray) -> np.ndarray:
    """
    Вычисляет HOG-дескриптор для изображения в оттенках серого.
    """
    img_size = max(gray_uint8.shape[:2])
    cell = max(8, img_size // 32)

    return hog(
        gray_uint8,
        orientations=9,
        pixels_per_cell=(cell, cell),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )


def image_to_feature(rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Преобразует одно изображение в вектор признаков.
    """
    gray = (rgb2gray(rgb_uint8) * 255).astype(np.uint8)

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hu = extract_hu_from_binary(th)

    hog_full = extract_hog(gray)
    hog_upper = extract_hog(gray[: gray.shape[0] // 2, :])

    hsv_hist = extract_hsv_hist(rgb_uint8, bins=16)

    return np.concatenate([hog_full, hog_upper, hu, hsv_hist]).astype(np.float64)


def build_features(X_images: List[np.ndarray]) -> np.ndarray:
    """
    Преобразует список изображений в матрицу признаков.
    """
    return np.vstack([image_to_feature(img) for img in X_images])


def load_images_and_labels(data_dir: str) -> Tuple[List[np.ndarray], List[str]]:
    """
    Загружает изображения и метки классов из директории вида:
    data_dir / <class_name> / image.jpg
    """
    base = Path(data_dir)
    X, y = [], []

    for cls in sorted(d.name for d in base.iterdir() if d.is_dir()):
        for p in sorted((base / cls).iterdir()):
            if not is_image_path(p):
                continue

            im = cv2.imread(str(p))
            if im is None:
                print(f"Warning: cannot read {p}, skipping.")
                continue

            X.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            y.append(cls)

    return X, y
