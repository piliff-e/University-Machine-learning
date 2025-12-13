#!/usr/bin/env python3
"""
Скрипт предобработки изображений перед извлечением признаков.

Выполняемые шаги:
- чтение изображений из raw-директории;
- приведение изображений к квадратному формату;
- масштабирование до фиксированного размера;
- добавление паддинга для сохранения пропорций;
- сохранение обработанных изображений в структурированном виде.

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
    """
    Проверяет, является ли путь файлом изображения поддерживаемого формата.
    """
    return p.suffix.lower() in IMG_EXTS and p.is_file()


def get_class_from_filename(fname: str) -> str:
    """
    Определяет класс изображения по имени файла.

    Ожидается, что имя файла начинается с имени класса:
    down_3_aug2 -> down
    one_5       -> one
    up_7_aug1   -> up
    """
    return Path(fname).stem.split("_")[0]


def resize_with_pad(img: np.ndarray, size: int) -> np.ndarray:
    """
    Масштабирует изображение с сохранением пропорций
    и добавляет паддинг до размера size × size.
    """
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
    """
    Выполняет предобработку всех изображений из raw_dir
    и сохраняет результаты в out_dir, разложив их по классам.
    """
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

        out_name = f"{p.stem}_proc{img_size}{p.suffix.lower()}"
        out_path = dest_dir / out_name

        cv2.imwrite(str(out_path), cv2.cvtColor(img_proc, cv2.COLOR_RGB2BGR))

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
