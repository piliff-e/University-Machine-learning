"""
Единая реализация извлечения признаков для train / test / optuna.

Функции:
- is_image_path(Path) -> bool
- image_to_feature(rgb_uint8) -> 1D numpy array
- build_features(list_of_rgb_images) -> 2D numpy array (n_samples, n_features)
- load_images_and_labels(data_dir) -> X_images_list, y_list  (удобно для train/optuna)
"""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def is_image_path(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS and p.is_file()


def extract_hu_from_binary(gray_uint8: np.ndarray) -> np.ndarray:
    # cv2.HuMoments возвращает 7 чисел
    moments = cv2.moments(gray_uint8)
    hu = cv2.HuMoments(moments).flatten()
    # лог масштабирование (как у тебя было) — стабилизируем мелкие значения
    with np.errstate(divide="ignore", invalid="ignore"):
        hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    return hu_log


def extract_hsv_hist(rgb_uint8: np.ndarray, bins: int = 16) -> np.ndarray:
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
    chans = cv2.split(hsv)
    hist = []
    for ch in chans[:3]:
        h = cv2.calcHist([ch], [0], None, [bins], [0, 256]).flatten()
        # нормировка по сумме (чтобы признаки были сопоставимы между изображениями)
        h = h / (h.sum() + 1e-9)
        hist.append(h)
    return np.concatenate(hist)


def extract_hog(gray_uint8: np.ndarray) -> np.ndarray:
    # Подбор cell size относительно размера изображения — как в твоём коде
    img_size = max(gray_uint8.shape[:2])
    cell = max(8, img_size // 32)
    fd = hog(
        gray_uint8,
        orientations=9,
        pixels_per_cell=(cell, cell),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )
    return fd


def image_to_feature(rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Унифицированная функция: вход RGB uint8 (H,W,3) -> 1D признаки.
    Состав признаков:
      - HOG (полное изображение)
      - HOG (верхняя половина изображения)
      - Hu moments (от бинаризованного изображения через Otsu)
      - HSV histogram (16 bins per channel -> 48 values)
    Возвращаем: numpy 1D (dtype float64)
    """
    # 1) серое
    gray = (rgb2gray(rgb_uint8) * 255).astype(np.uint8)

    # 2) Hu moments (по бинарной маске Otsu)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hu = extract_hu_from_binary(th)

    # 3) HOG full + HOG upper
    hog_full = extract_hog(gray)
    upper = gray[: gray.shape[0] // 2, :]
    hog_upper = extract_hog(upper)
    hogf = np.concatenate([hog_full, hog_upper])

    # 4) HSV hist
    hsv_hist = extract_hsv_hist(rgb_uint8, bins=16)

    feat = np.concatenate([hogf, hu, hsv_hist]).astype(np.float64)
    return feat


def build_features(X_images: List[np.ndarray]) -> np.ndarray:
    feats = [image_to_feature(img) for img in X_images]
    return np.vstack(feats)


def load_images_and_labels(data_dir: str) -> Tuple[List[np.ndarray], List[str]]:
    """
    Загружает изображения и метки из папки data_dir, где ожидаются подпапки с классами.
    Возвращает (images_list, labels_list).
    Использует cv2.imread + BGR->RGB.
    """
    base = Path(data_dir)
    X = []
    y = []
    # сортировка классов для детерминированности
    for cls in sorted([d.name for d in base.iterdir() if d.is_dir()]):
        folder = base / cls
        for p in sorted(folder.iterdir()):
            if not is_image_path(p):
                continue
            im = cv2.imread(str(p))
            if im is None:
                print(f"Warning: cannot read {p}, skipping.")
                continue
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            X.append(im)
            y.append(cls)
    return X, y
