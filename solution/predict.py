"""
predict.py — загружает обученные артефакты и предоставляет функцию predict().

Использование:
    from predict import predict

    # Без session_id → глобальный fallback (SparseTPS)
    x_door2, y_door2 = predict(743.96, 524.59, source="top")

    # С session_id → per-session гомография (точнее, если сессия есть в train)
    x_door2, y_door2 = predict(743.96, 524.59, source="top",
                                session_id="camera_door2_2025-11-27_14-26-36")
"""

import pickle
import numpy as np
from pathlib import Path

# Импорт нужен чтобы pickle мог восстановить классы моделей
from models import (  # noqa: F401
    HomographyMapper, NormalizedPolyMapper, SparseTPS,
    SessionAwareMapper, PolyMapper, TPSMapper,
)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
_model_cache: dict = {}


def _load_model(source: str) -> SessionAwareMapper:
    if source not in _model_cache:
        path = ARTIFACTS_DIR / f"mapper_{source}.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"Артефакт не найден: {path}\n"
                f"Запустите сначала: python train.py"
            )
        with open(path, "rb") as f:
            data = pickle.load(f)
        _model_cache[source] = data["model"]
        print(f"[predict] Загружена модель '{data['model_name']}' для {source}→door2")
    return _model_cache[source]


def predict(x: float, y: float, source: str,
            session_id: str = None) -> tuple[float, float]:
    """
    Маппинг пиксельной координаты из камеры source на вид door2.

    Args:
        x          : координата X на кадре камеры source
        y          : координата Y на кадре камеры source
        source     : "top" или "bottom"
        session_id : имя сессии (опционально, улучшает точность для train-сессий)

    Returns:
        (x', y') — предсказанная пиксельная координата на кадре door2
    """
    if source not in ("top", "bottom"):
        raise ValueError(f"source должен быть 'top' или 'bottom', получен: '{source}'")

    model = _load_model(source)
    pts   = np.array([[x, y]], dtype=np.float32)
    pred  = model.predict(pts, session_id=session_id)

    return float(pred[0, 0]), float(pred[0, 1])


def predict_batch(points: list[tuple[float, float]], source: str,
                  session_id: str = None) -> list[tuple[float, float]]:
    """
    Батч-версия predict для нескольких точек одной сессии.

    Args:
        points     : список (x, y) координат на камере source
        source     : "top" или "bottom"
        session_id : имя сессии (опционально)

    Returns:
        список (x', y') координат на door2
    """
    if source not in ("top", "bottom"):
        raise ValueError(f"source должен быть 'top' или 'bottom', получен: '{source}'")

    model = _load_model(source)
    pts   = np.array(points, dtype=np.float32)
    pred  = model.predict(pts, session_id=session_id)

    return [(float(pred[i, 0]), float(pred[i, 1])) for i in range(len(pred))]


if __name__ == "__main__":
    print("=== Демо предсказаний ===\n")
    examples = [
        (743.96, 524.59, "top"),
        (920.42, 343.37, "top"),
        (512.0,  300.0,  "bottom"),
    ]
    for x, y, src in examples:
        try:
            xp, yp = predict(x, y, source=src)
            print(f"  {src:6s} ({x:8.2f}, {y:8.2f})  →  door2 ({xp:8.2f}, {yp:8.2f})")
        except FileNotFoundError as e:
            print(f"  ОШИБКА: {e}")
            break