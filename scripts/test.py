#!/usr/bin/env python3
"""
Скрипт тестирования обученных моделей на отложенной выборке.

Выполняемые шаги:
- загрузка сохранённых моделей и scaler'а;
- извлечение признаков для тестовых изображений;
- получение предсказаний для каждой модели;
- расчёт метрик качества;
- сохранение результатов в CSV-файл;
- построение и сохранение confusion matrix.

Используется для финальной оценки качества моделей.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from features import image_to_feature, is_image_path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_models(models_dir="models"):
    mdir = Path(models_dir)
    scaler = joblib.load(mdir / "scaler.joblib")
    knn = joblib.load(mdir / "knn.joblib")
    svm = joblib.load(mdir / "svm.joblib")
    rf = joblib.load(mdir / "rf.joblib")
    return scaler, {"kNN": knn, "SVM": svm, "RandomForest": rf}


def load_test_images(test_dir):
    base = Path(test_dir)
    files = []
    labels = []
    images = []
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
            images.append(im)
            labels.append(cls)
            files.append(p.name)
    return files, images, labels


def main(test_dir="dataset/test", out_dir="reports", models_dir="models"):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    scaler, models = load_models(models_dir)
    files, images, labels = load_test_images(test_dir)
    print(f"Loaded {len(images)} test images, classes: {sorted(set(labels))}")

    X = np.vstack([image_to_feature(im) for im in images])
    Xs = scaler.transform(X)

    # predictions
    all_preds = {}
    for name, clf in models.items():
        preds = clf.predict(Xs)
        all_preds[name] = preds
        acc = accuracy_score(labels, preds)
        print(f"{name} test accuracy: {acc:.3f}")
        print("Pred counts:", Counter(preds))
        print(classification_report(labels, preds, digits=3, zero_division=0))
        cm = confusion_matrix(labels, preds, labels=sorted(set(labels)))
        # save confusion matrix plot
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=sorted(set(labels)),
            yticklabels=sorted(set(labels)),
        )
        plt.title(f"Confusion matrix ({name})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(out / f"cm_{name}.png")
        plt.close()

    # save CSV with predictions
    csv_path = out / "test_predictions.csv"
    cols = ["filename", "true_label"] + [f"pred_{name}" for name in models.keys()]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for i, fn in enumerate(files):
            row = [fn, labels[i]] + [all_preds[name][i] for name in models.keys()]
            writer.writerow(row)
    print(f"Saved predictions CSV to {csv_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset/test")
    parser.add_argument("--out_dir", type=str, default="reports")
    parser.add_argument("--models_dir", type=str, default="models")
    args = parser.parse_args()
    main(test_dir=args.data_dir, out_dir=args.out_dir, models_dir=args.models_dir)
