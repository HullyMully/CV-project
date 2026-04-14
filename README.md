# Маппинг координат между камерами холодильника

Решение задачи маппинга пиксельных координат из камер `top`/`bottom` на вид с камеры `door2`.

---

## Подход

Для каждой пары камер (`top→door2`, `bottom→door2`) обучаются три модели, и автоматически выбирается лучшая по кросс-валидации на train:

| Модель | Описание |
|--------|----------|
| **Homography** | Проективное преобразование (cv2.findHomography + RANSAC) |
| **Poly2** | Полиномиальная регрессия 2-й степени + Ridge |
| **TPS** | Thin Plate Spline — гибкий нелинейный маппинг |

Выбор TPS обоснован: камеры смотрят под разными углами, объекты в холодильнике находятся на разных глубинах → геометрия нелинейная. TPS точно проходит через все опорные точки и хорошо интерполирует между ними.

---

## Структура

```
solution/
├── train.py          # обучение, сохраняет artifacts/mapper_top.pkl и mapper_bottom.pkl
├── predict.py        # функция predict(x, y, source)
├── evaluate.py       # считает MED на val/, сохраняет results/metrics.json
├── requirements.txt
└── README.md
```

---

## Воспроизведение

### 1. Установка зависимостей

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Обучение

```bash
python train.py
```

Ожидаемый вывод:
```
[top]    загружено ~XXXX точек из 56 сессий
  CV MED [homography  ]: XXX.XX px
  CV MED [poly2       ]: XXX.XX px
  CV MED [tps         ]: XXX.XX px
  ✓ Лучшая модель: tps (CV MED = XXX.XX px)
  Артефакт сохранён: artifacts/mapper_top.pkl

[bottom] загружено ~XXXX точек из 56 сессий
  ...
```

Артефакты сохраняются в `artifacts/`.

### 3. Оценка на валидации

```bash
python evaluate.py
```

Результаты сохраняются в `results/metrics.json`.

---

## Использование predict()

```python
from predict import predict, predict_batch

# Одна точка
x_door2, y_door2 = predict(743.96, 524.59, source="top")
x_door2, y_door2 = predict(512.0,  300.0,  source="bottom")

# Батч точек
results = predict_batch([(743.96, 524.59), (920.42, 343.37)], source="top")
# → [(x1', y1'), (x2', y2')]
```

### Аргументы

| Параметр | Тип | Описание |
|----------|-----|----------|
| `x` | float | X-координата на кадре камеры source |
| `y` | float | Y-координата на кадре камеры source |
| `source` | str | `"top"` или `"bottom"` |

### Возвращает

`(x', y')` — предсказанная пиксельная координата на кадре `door2`

---

## Метрика

**MED (Mean Euclidean Distance)** в пикселях на валидационном сплите:

```
MED = mean( sqrt( (x_pred - x_true)^2 + (y_pred - y_true)^2 ) )
```

Метрики по итогу evaluate.py доступны в `results/metrics.json`.

---

## Важно

- Путь к датасету прописан в `train.py` и `evaluate.py` как `DATA_ROOT`
- Разбиение train/val строго соблюдается из `split.json`
- Модели обучаются только на данных из `"train"`, оцениваются только на `"val"`
