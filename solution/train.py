"""
train.py — обучает модели маппинга координат top→door2 и bottom→door2.

Кандидаты (выбирается лучший по 5-fold CV на train):
  1. HomographyMapper      — проективное преобразование (RANSAC)
  2. NormalizedPolyMapper  — полином 2-й степени + нормализация
  3. NormalizedPolyMapper  — полином 3-й степени + нормализация
  4. SparseTPS             — TPS на 150 опорных точках (k-means)

Артефакты сохраняются в artifacts/
"""

import json
import pickle
import numpy as np
from pathlib import Path

from models import HomographyMapper, NormalizedPolyMapper, SparseTPS

# ─── Настройки ────────────────────────────────────────────────────────────────
DATA_ROOT  = Path("/Users/hullymully/Documents/development/Projects/CV-project/test-task/")
SPLIT_FILE = DATA_ROOT / "split.json"
ARTIFACTS  = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

SOURCES = ["top", "bottom"]


# ─── Загрузка данных ──────────────────────────────────────────────────────────

def load_point_pairs(session_dirs: list[Path], source: str) -> tuple[np.ndarray, np.ndarray]:
    coord_file = f"coords_{source}.json"
    src_list, dst_list = [], []

    for session_dir in session_dirs:
        json_path = session_dir / coord_file
        if not json_path.exists():
            print(f"  [WARN] не найден: {json_path}")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            pairs = json.load(f)

        for pair in pairs:
            img1 = {p["number"]: (p["x"], p["y"]) for p in pair["image1_coordinates"]}  # door2
            img2 = {p["number"]: (p["x"], p["y"]) for p in pair["image2_coordinates"]}  # source cam
            common = set(img1.keys()) & set(img2.keys())
            for num in sorted(common):
                src_list.append(img2[num])
                dst_list.append(img1[num])

    src_pts = np.array(src_list, dtype=np.float32)
    dst_pts = np.array(dst_list, dtype=np.float32)
    print(f"  [{source}] загружено {len(src_pts)} точек из {len(session_dirs)} сессий")
    return src_pts, dst_pts


# ─── Метрика ──────────────────────────────────────────────────────────────────

def mean_euclidean_distance(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(((pred - true) ** 2).sum(axis=1)).mean())


# ─── Кросс-валидация ──────────────────────────────────────────────────────────

def cross_val_med(src_pts, dst_pts, model_fn, n_folds=5) -> float:
    n = len(src_pts)
    idx = np.random.permutation(n)
    fold_size = n // n_folds
    meds = []

    for k in range(n_folds):
        val_idx = idx[k * fold_size: (k + 1) * fold_size]
        trn_idx = np.concatenate([idx[:k * fold_size], idx[(k + 1) * fold_size:]])
        if len(trn_idx) < 20:
            continue
        try:
            model = model_fn()
            model.fit(src_pts[trn_idx], dst_pts[trn_idx])
            pred = model.predict(src_pts[val_idx])
            meds.append(mean_euclidean_distance(pred, dst_pts[val_idx]))
        except Exception as e:
            print(f"    [CV fold {k} error]: {e}")
            return float("inf")

    return float(np.mean(meds)) if meds else float("inf")


# ─── Обучение ─────────────────────────────────────────────────────────────────

def train(source: str, session_dirs: list[Path]) -> dict:
    print(f"\n{'='*55}")
    print(f"Обучение маппинга: {source} → door2")
    print(f"{'='*55}")

    src_pts, dst_pts = load_point_pairs(session_dirs, source)

    if len(src_pts) < 20:
        raise ValueError(f"Слишком мало точек для {source}: {len(src_pts)}")

    np.random.seed(42)

    candidates = {
        "homography":  lambda: HomographyMapper(),
        "poly2":       lambda: NormalizedPolyMapper(degree=2, alpha=0.01),
        "poly3":       lambda: NormalizedPolyMapper(degree=3, alpha=0.01),
        "sparse_tps":  lambda: SparseTPS(n_ctrl=150, regularization=1.0),
    }

    results = {}
    for name, fn in candidates.items():
        print(f"  Оцениваю [{name}]...", end=" ", flush=True)
        med = cross_val_med(src_pts, dst_pts, fn, n_folds=5)
        print(f"CV MED = {med:.2f} px")
        results[name] = med

    best_name = min(results, key=results.get)
    best_med  = results[best_name]
    print(f"\n  ✓ Лучшая модель: {best_name} (CV MED = {best_med:.2f} px)")

    best_model = candidates[best_name]()
    best_model.fit(src_pts, dst_pts)

    artifact_path = ARTIFACTS / f"mapper_{source}.pkl"
    with open(artifact_path, "wb") as f:
        pickle.dump({"model": best_model, "model_name": best_name}, f)
    print(f"  Артефакт сохранён: {artifact_path}")

    return {"source": source, "model_name": best_name, "cv_med": round(best_med, 4)}


# ─── Точка входа ──────────────────────────────────────────────────────────────

def main():
    with open(SPLIT_FILE, "r") as f:
        split = json.load(f)

    train_dirs = [DATA_ROOT / p for p in split["train"]]

    summary = []
    for source in SOURCES:
        result = train(source, train_dirs)
        summary.append(result)

    print("\n\n" + "="*55)
    print("ИТОГ ОБУЧЕНИЯ")
    print("="*55)
    for r in summary:
        print(f"  {r['source']:8s} → door2 | модель: {r['model_name']:12s} | CV MED: {r['cv_med']:.2f} px")

    with open(ARTIFACTS / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nГотово! Артефакты в папке artifacts/")


if __name__ == "__main__":
    main()