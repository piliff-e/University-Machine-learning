# University-Machine-learning

## Отчёт по лабораторной работе  
**Тема:** Классические методы машинного обучения (kNN, SVM, RandomForest) без нейронных сетей.

---

## 1. Постановка задачи
- **Задача:** трёхклассовая классификация изображений (классы `down`, `one`, `up`).  
- **Датасет:** набор фотографий (скриншотов) трёх поз/жестов персонажа из видеоигры [Night in the Woods](http://www.nightinthewoods.com); исходные примеры (без аугментаций):

| Класс `down` | Класс `one` | Класс `up` |
|:---:|:---:|:---:|
| руки опущены вниз | поднята только одна рука | руки подняты вверх |
| файл `dataset/raw/down_1.jpeg` | файл `dataset/raw/one_7.jpeg` | файл `dataset/raw/up_2.jpeg` |
| ![](dataset/raw/down_1.jpeg) | ![](dataset/raw/one_7.jpeg) | ![](dataset/raw/up_2.jpeg)  |

- **Методы, изученные в работе:** аугментация данных, kNN, SVM, Random Forest (RF).




---

## 2. Подготовка датасета (частично относится к заданию 3 — аугментация)
- **Аугментации** (набор трансформаций для 2D-изображений): использована библиотека [**MONAI**](https://monai-dev.readthedocs.io/en/latest/index.html). В финальной версии применялись более мягкие/контролируемые аугментации:
  - горизонтальный/вертикальный флип,
  - небольшой поворот (±15°),
  - локальный зум,
  - лёгкий гауссов шум / размытие / резкость,
  - контраст/гистограмма, случайное выбрасывание пикселей.
- **Предобработка:** все изображения приводятся к квадратному виду и единому размеру (256×256) с сохранением соотношения сторон и добавлением паддинга (отступа вокруг изображения) — для того, чтобы признаки (HOG, Hu, HSV) были вычислимы в однотипном пространстве признаков и масштаб признаков не зависел от исходного размера.
- **Команды (pipeline):**
  1. Аугментация:
	 ```
	 python3 scripts/augment.py --input_dir datasets/raw --output_dir dataset/raw --num_aug 9
	 ```
  2. Предобработка (resize + pad -> processed):
     ```
     python3 scripts/preprocess.py --raw_dir datasets/raw --output_dir datasets/processed --img_size 256
     ```
  3. (Ручной шаг) часть файлов из `train` была перемещена в `test`, чтобы получить итоговый небольшой тестовый набор.
- **Примеры файлов ПОСЛЕ предобработки и аугментаций:**

| Класс `down` | Класс `one` | Класс `up` |
|:---:|:---:|:---:|
| руки опущены вниз | поднята только одна рука | руки подняты вверх |
| файл `dataset/processed/train/down/down_1_aug7_proc256.jpeg` | файл `dataset/processed/train/one/one_7_aug9_proc256.jpeg` | файл `dataset/processed/train/up/up_2_aug2_proc256.jpeg` |
| ![](dataset/processed/train/down/down_1_aug7_proc256.jpeg) | ![](dataset/processed/train/one/one_7_aug9_proc256.jpeg) | ![](dataset/processed/train/up/up_2_aug2_proc256.jpeg)  |

---

## 3. Задача 1 — сравнение методов kNN и SVM

### Описание методов:

- **kNN (k-Nearest Neighbors):** присваивание метки по большинству среди k ближайших соседей в пространстве признаков (обычно евклидово расстояние).  
- **SVM (Support Vector Machine, RBF ядро):** максимизация разделяющей полосы с регуляризацией; с RBF-ядром классификация становится нелинейной:
  \[
  K(x, x')=\exp(-\gamma\|x-x'\|^2).
  \]
  Решающая задача включает параметр `C` — жёсткость штрафа за ошибки.

### Реализация и гиперпараметры:
- Библиотека: `scikit-learn`.
- В коде (`train.py`):
  - kNN:

	```
	from sklearn.neighbors import KNeighborsClassifier
	
	KNeighborsClassifier(n_neighbors=5)
	```

  - SVM: 

	```
	from sklearn.svm import SVC
	
	SVC(kernel="rbf", C=1.0, gamma="scale")
	```

- Оценка: 5-fold Stratified cross-validation (`sklearn.model_selection.StratifiedKFold`) — средняя accuracy и std.
- Команды запуска:

	```bash
   python3 scripts/train.py --data_dir datasets/processed/train --models_dir models
   python3 scripts/test.py  --data_dir datasets/processed/test  --models_dir models --out_dir reports
	```
  
### Результаты (финальный датасет из репозитория):

- kNN CV / test accuracy: в эксперименте тестовой accuracy ≈ 0.81  
- SVM CV / test accuracy: ≈ 0.93 / 0.95  
  (точные числа — в разделе “Итоги” и в выводе консоли ниже)

---

## 4. Задача 3 — добавление метода Random Forest (RF)

### Описание метода:
Ансамбль решающих деревьев, каждое дерево обучается на бутстреп-выборке и при разбиении узлов рассматривается случайное подмножество признаков; итог — голосование деревьев.

### Реализация и гиперпараметры:
- Библиотека: `scikit-learn`.
- В коде (`train.py`):

	```
	from sklearn.ensemble import RandomForestClassifier
	
	RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
	```

- Оценка: как и для SVM — 5-fold stratified CV + тест на отложенной выборке.
- **Результаты:** RF показал наилучшие результаты на train и близкие лучшие результаты на тесте (в ряде запусков RF и SVM дают примерно одинаковую высокую точность).

		Примечание: пайплайн запуска для RF тот же, что в разделе 3 (train/test). Всё остальное — повторяется.

---

## 5. Итоговые результаты

- **Вывод:** RandomForest (RF) показал наилучшие результаты среди протестированных методов; SVM также даёт высокие показатели; kNN уступает (но остаётся полезным как простой baseline).
- **Пример CSV (фрагмент `reports/.../test_predictions.csv`):**

| filename                  | true_label | pred_kNN | pred_SVM | pred_RandomForest |
|--------------------------:|:----------:|:--------:|:--------:|:-----------------:|
| down_1_proc256.jpeg       | down       | down     | down     | down              |
| one_7_proc256.jpeg        | one        | one      | one      | one               |
| up_2_proc256.jpeg         | up         | up       | up       | up                |
| down_7_aug4_proc256.jpeg  | down       | down     | down     | down              |

*(Это демонстрационный фрагмент; полный CSV сохраняется в `reports/.../test_predictions.csv`.)*

- **Вывод в консоли (пример, запуск `python3 scripts/test.py` на финальном датасете):**
```text
Loaded 21 test images, classes: ['down', 'one', 'up']
kNN test accuracy: 0.810
SVM test accuracy: 0.952
RandomForest test accuracy: 0.952

Saved predictions CSV to reports/<folder>/test_predictions.csv