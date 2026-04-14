"""
evaluate.py — считает MED (Mean Euclidean Distance) на валидационном сплите.

Запуск:
    python evaluate.py

Выводит метрики в консоль и сохраняет results/metrics.json
"""

import json
import numpy as np
from pathlib import Path

from predict import predict_batch

# ─── Настройки ────────────────────────────────────────────────────────────────
DATA_ROOT  = Path("/Users/hullymully/Documents/development/Projects/CV-project/test-task/")
SPLIT_FILE = DATA_ROOT / "split.json"
RESULTS    = Path("results")
RESULTS.mkdir(exist_ok=True)

SOURCES = ["top", "bottom"]


# ─── Загрузка val-данных ──────────────────────────────────────────────────────

def load_val_pairs(session_dirs: list[Path], source: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Загружает пары точек из val-сессий для заданного source.

    Returns:
        src_pts : (N, 2) — координаты на камере source
        dst_pts : (N, 2) — истинные координаты на door2
    """
    coord_file = f"coords_{source}.json"
    src_list, dst_list = [], []
    session_count = 0

    for session_dir in session_dirs:
        json_path = session_dir / coord_file
        if not json_path.exists():
            print(f"  [WARN] не найден: {json_path}")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            pairs = json.load(f)

        session_count += 1
        for pair in pairs:
            img1 = {p["number"]: (p["x"], p["y"]) for p in pair["image1_coordinates"]}  # door2
            img2 = {p["number"]: (p["x"], p["y"]) for p in pair["image2_coordinates"]}  # source cam

            common = set(img1.keys()) & set(img2.keys())
            for num in sorted(common):
                src_list.append(img2[num])
                dst_list.append(img1[num])

    src_pts = np.array(src_list, dtype=np.float32)
    dst_pts = np.array(dst_list, dtype=np.float32)
    print(f"  [{source}] val: {len(src_pts)} точек из {session_count} сессий")
    return src_pts, dst_pts


# ─── Метрика ──────────────────────────────────────────────────────────────────

def mean_euclidean_distance(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(((pred - true) ** 2).sum(axis=1)).mean())


def percentile_error(pred: np.ndarray, true: np.ndarray, p: int) -> float:
    errors = np.sqrt(((pred - true) ** 2).sum(axis=1))
    return float(np.percentile(errors, p))


# ─── Основная функция ─────────────────────────────────────────────────────────

def evaluate():
    with open(SPLIT_FILE, "r") as f:
        split = json.load(f)

    val_dirs = [DATA_ROOT / p for p in split["val"]]

    print("="*55)
    print("ОЦЕНКА НА ВАЛИДАЦИОННОМ СПЛИТЕ")
    print("="*55)

    all_metrics = {}

    for source in SOURCES:
        print(f"\n[{source} → door2]")
        src_pts, dst_pts = load_val_pairs(val_dirs, source)

        if len(src_pts) == 0:
            print(f"  ОШИБКА: нет данных для {source}")
            continue

        # Предсказание батчем
        points = [(float(src_pts[i, 0]), float(src_pts[i, 1])) for i in range(len(src_pts))]
        preds  = predict_batch(points, source=source)
        pred_arr = np.array(preds, dtype=np.float32)

        # Метрики
        med   = mean_euclidean_distance(pred_arr, dst_pts)
        p50   = percentile_error(pred_arr, dst_pts, 50)
        p90   = percentile_error(pred_arr, dst_pts, 90)
        p95   = percentile_error(pred_arr, dst_pts, 95)

        # Покрытие < N пикселей
        errors = np.sqrt(((pred_arr - dst_pts) ** 2).sum(axis=1))
        cov50  = float((errors < 50).mean() * 100)
        cov100 = float((errors < 100).mean() * 100)

        print(f"  MED (Mean Euclidean Distance) : {med:.2f} px")
        print(f"  Медиана ошибки (P50)          : {p50:.2f} px")
        print(f"  P90                           : {p90:.2f} px")
        print(f"  P95                           : {p95:.2f} px")
        print(f"  % точек с ошибкой < 50 px    : {cov50:.1f}%")
        print(f"  % точек с ошибкой < 100 px   : {cov100:.1f}%")
        print(f"  Всего точек в val             : {len(src_pts)}")

        all_metrics[source] = {
            "MED_px":    round(med, 4),
            "P50_px":    round(p50, 4),
            "P90_px":    round(p90, 4),
            "P95_px":    round(p95, 4),
            "coverage_50px_pct":  round(cov50, 2),
            "coverage_100px_pct": round(cov100, 2),
            "n_points":  int(len(src_pts)),
        }

    # ─── Итоговая сводка ───────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("ИТОГОВЫЕ МЕТРИКИ")
    print("="*55)
    for src, m in all_metrics.items():
        print(f"  {src:6s} → door2 | MED = {m['MED_px']:.2f} px | P90 = {m['P90_px']:.2f} px")

    # Сохраняем
    metrics_path = RESULTS / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\nМетрики сохранены: {metrics_path}")

    return all_metrics


if __name__ == "__main__":
    evaluate()
