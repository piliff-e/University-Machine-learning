#!/usr/bin/env python3
"""
Скрипт предобработки изображений перед извлечением признаков.

Выполняемые шаги:
- чтение изображений из raw-директории;
- приведение изображений к квадратному формату;
- масштабирование до фиксированного размера (256x256);
- добавление паддинга для сохранения пропорций;
- сохранение обработанных файлов в структуре processed/train/<class>/.

Назначение:
- приведение всех изображений к единому формату;
- обеспечение корректного и сопоставимого вычисления признаков.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def is_image_path(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS and p.is_file()


def get_class_from_filename(fname: str) -> str:
    """
    Класс = строка до первого '_'
    down_3_aug2 -> down
    one_5       -> one
    up_7_aug1   -> up
    """
    stem = Path(fname).stem
    return stem.split("_")[0]


def resize_with_pad(img: np.ndarray, size: int) -> np.ndarray:
    """Resize (с сохранением соотношения сторон) + паддинг до size x size."""
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    top = (size - nh) // 2
    bottom = size - nh - top
    left = (size - nw) // 2
    right = size - nw - left

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )
    return padded


def process_all(raw_dir: str, out_dir: str, img_size: int = 256):
    raw = Path(raw_dir)
    out_root = Path(out_dir)

    count = 0
    per_class = {}

    for p in sorted(raw.iterdir()):
        if not is_image_path(p):
            continue

        cls = get_class_from_filename(p.name)
        dest_dir = out_root / cls
        dest_dir.mkdir(parents=True, exist_ok=True)

        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            print(f"Warning: cannot read {p}, skipping.")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_proc = resize_with_pad(img_rgb, img_size)

        # out_path = dest_dir / p.name  # сохраняем оригинальное имя
        stem = p.stem
        out_name = f"{stem}_proc256{p.suffix.lower()}"
        out_path = dest_dir / out_name

        img_bgr_out = cv2.cvtColor(img_proc, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), img_bgr_out)

        count += 1
        per_class[cls] = per_class.get(cls, 0) + 1

    print(f"Processed {count} images into {out_root.resolve()}")
    for cls, n in per_class.items():
        print(f"  class '{cls}': {n} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="dataset/raw")
    parser.add_argument("--output_dir", type=str, default="dataset/preprocessed")
    parser.add_argument("--img_size", type=int, default=256)
    args = parser.parse_args()

    process_all(args.raw_dir, args.output_dir, img_size=args.img_size)
