#!/usr/bin/env python3
"""
Скрипт обучения классических моделей машинного обучения.

Выполняемые шаги:
- загрузка изображений обучающей выборки;
- извлечение признаков;
- масштабирование признаков;
- обучение моделей kNN, SVM и Random Forest;
- оценка качества с помощью стратифицированной кросс-валидации;
- сохранение обученных моделей и scaler'а на диск.

Используется как основной этап обучения в pipeline.
"""

import argparse
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import build_features, load_images_and_labels

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def train_and_save_models(X, y, models_dir):
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    knn = KNeighborsClassifier(n_neighbors=5)
    svm = SVC(
        kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=False
    )
    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    for name, clf in [("kNN", knn), ("SVM", svm), ("RandomForest", rf)]:
        scores = cross_val_score(clf, Xs, y, cv=skf, scoring="accuracy", n_jobs=-1)
        print(f"{name} CV accuracy: mean={scores.mean():.3f}, std={scores.std():.3f}")

    # fit on all train
    knn.fit(Xs, y)
    svm.fit(Xs, y)
    rf.fit(Xs, y)

    out = Path(models_dir)
    out.mkdir(exist_ok=True)
    joblib.dump(scaler, out / "scaler.joblib")
    joblib.dump(knn, out / "knn.joblib")
    joblib.dump(svm, out / "svm.joblib")
    joblib.dump(rf, out / "rf.joblib")
    print(f"Saved scaler and models into {out.resolve()}")


def main(data_dir, models_dir):
    Ximgs, y = load_images_and_labels(data_dir)
    print(f"Loaded {len(Ximgs)} images for training, classes: {sorted(set(y))}")
    X = build_features(Ximgs)
    print("Feature matrix shape:", X.shape)
    train_and_save_models(X, y, models_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset/train",
        help="Directory with train data with class subdirectories",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Directory to save models to",
    )

    args = parser.parse_args()
    main(args.data_dir, args.models_dir)
