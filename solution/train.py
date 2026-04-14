"""
train.py — обучает SessionAwareMapper для top→door2 и bottom→door2.

SessionAwareMapper:
  - глобальный SparseTPS на всех train-данных (fallback)
  - per-session гомография для каждой тренировочной сессии

Артефакты сохраняются в artifacts/
"""

import json
import pickle
import numpy as np
from pathlib import Path

from models import HomographyMapper, NormalizedPolyMapper, SparseTPS, SessionAwareMapper

# ─── Настройки ────────────────────────────────────────────────────────────────
DATA_ROOT  = Path("/Users/hullymully/Documents/development/Projects/CV-project/test-task/")
SPLIT_FILE = DATA_ROOT / "split.json"
ARTIFACTS  = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

SOURCES = ["top", "bottom"]


# ─── Загрузка данных ──────────────────────────────────────────────────────────

def load_point_pairs(session_dirs: list[Path], source: str):
    """
    Возвращает:
        src_pts     : (N, 2)
        dst_pts     : (N, 2)
        session_ids : (N,) строковые id сессий
    """
    coord_file = f"coords_{source}.json"
    src_list, dst_list, sid_list = [], [], []

    for session_dir in session_dirs:
        json_path = session_dir / coord_file
        if not json_path.exists():
            print(f"  [WARN] не найден: {json_path}")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            pairs = json.load(f)

        session_id = session_dir.name  # например camera_door2_2025-11-27_14-26-36

        for pair in pairs:
            img1 = {p["number"]: (p["x"], p["y"]) for p in pair["image1_coordinates"]}
            img2 = {p["number"]: (p["x"], p["y"]) for p in pair["image2_coordinates"]}
            common = set(img1.keys()) & set(img2.keys())
            for num in sorted(common):
                src_list.append(img2[num])
                dst_list.append(img1[num])
                sid_list.append(session_id)

    src_pts     = np.array(src_list, dtype=np.float32)
    dst_pts     = np.array(dst_list, dtype=np.float32)
    session_ids = np.array(sid_list)
    print(f"  [{source}] загружено {len(src_pts)} точек из {len(set(sid_list))} сессий")
    return src_pts, dst_pts, session_ids


# ─── Метрика ──────────────────────────────────────────────────────────────────

def mean_euclidean_distance(pred, true):
    return float(np.sqrt(((pred - true) ** 2).sum(axis=1)).mean())


# ─── Кросс-валидация (по сессиям, не по точкам!) ─────────────────────────────

def session_cross_val(src_pts, dst_pts, session_ids, n_folds=5):
    """
    Правильная CV: fold = группа сессий, а не случайные точки.
    Это точнее отражает реальный val-сплит.
    """
    unique_sessions = np.unique(session_ids)
    np.random.shuffle(unique_sessions)
    folds = np.array_split(unique_sessions, n_folds)

    meds = []
    for k, val_sessions in enumerate(folds):
        val_mask = np.isin(session_ids, val_sessions)
        trn_mask = ~val_mask
        if trn_mask.sum() < 20 or val_mask.sum() < 4:
            continue

        model = SessionAwareMapper(n_ctrl=150, regularization=1.0)
        model.fit(src_pts[trn_mask], dst_pts[trn_mask],
                  session_ids=session_ids[trn_mask])

        # Для val-сессий используем ТОЛЬКО глобальный fallback
        # (per-session моделей для val нет — как в реальном сценарии)
        pred = model.predict(src_pts[val_mask], session_id=None)
        meds.append(mean_euclidean_distance(pred, dst_pts[val_mask]))
        print(f"    fold {k+1}/{n_folds}: MED = {meds[-1]:.2f} px")

    return float(np.mean(meds)) if meds else float("inf")


# ─── Обучение ─────────────────────────────────────────────────────────────────

def train(source: str, session_dirs: list[Path]) -> dict:
    print(f"\n{'='*55}")
    print(f"Обучение маппинга: {source} → door2")
    print(f"{'='*55}")

    src_pts, dst_pts, session_ids = load_point_pairs(session_dirs, source)

    if len(src_pts) < 20:
        raise ValueError(f"Слишком мало точек для {source}: {len(src_pts)}")

    np.random.seed(42)

    # Session-CV чтобы понять качество глобального fallback
    print(f"\n  Session-CV (оценка глобального fallback):")
    cv_med = session_cross_val(src_pts, dst_pts, session_ids, n_folds=5)
    print(f"  → CV MED (global fallback) = {cv_med:.2f} px")

    # Финальное обучение на всех данных
    print(f"\n  Финальное обучение на всех {len(src_pts)} точках...")
    model = SessionAwareMapper(n_ctrl=200, regularization=0.5)
    model.fit(src_pts, dst_pts, session_ids=session_ids)

    artifact_path = ARTIFACTS / f"mapper_{source}.pkl"
    with open(artifact_path, "wb") as f:
        pickle.dump({"model": model, "model_name": "session_aware"}, f)
    print(f"  Артефакт сохранён: {artifact_path}")

    return {"source": source, "model_name": "session_aware", "cv_med": round(cv_med, 4)}


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
        print(f"  {r['source']:8s} → door2 | модель: {r['model_name']:14s} | CV MED: {r['cv_med']:.2f} px")

    with open(ARTIFACTS / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nГотово! Артефакты в папке artifacts/")


if __name__ == "__main__":
    main()