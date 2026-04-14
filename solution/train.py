"""
train.py — обучает ZonalEnsemble для top→door2 и bottom→door2.

ZonalEnsemble:
  - делит входное пространство на n_zones зон (k-means)
  - для каждой зоны обучает локальный SparseTPS
  - предсказание = взвешенная сумма по зонам (Gaussian weights)
  - глобальный SparseTPS как fallback

Grid search по n_zones и overlap_sigma.
"""

import json
import pickle
import numpy as np
from pathlib import Path

from models import ZonalEnsemble, SparseTPS

DATA_ROOT  = Path("/Users/hullymully/Documents/development/Projects/CV-project/test-task/")
SPLIT_FILE = DATA_ROOT / "split.json"
ARTIFACTS  = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

SOURCES = ["top", "bottom"]


def load_point_pairs(session_dirs, source):
    coord_file = f"coords_{source}.json"
    src_list, dst_list, sid_list = [], [], []
    for session_dir in session_dirs:
        json_path = session_dir / coord_file
        if not json_path.exists():
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            pairs = json.load(f)
        sid = session_dir.name
        for pair in pairs:
            img1 = {p["number"]: (p["x"], p["y"]) for p in pair["image1_coordinates"]}
            img2 = {p["number"]: (p["x"], p["y"]) for p in pair["image2_coordinates"]}
            for num in sorted(set(img1) & set(img2)):
                src_list.append(img2[num])
                dst_list.append(img1[num])
                sid_list.append(sid)
    src_pts     = np.array(src_list, dtype=np.float32)
    dst_pts     = np.array(dst_list, dtype=np.float32)
    session_ids = np.array(sid_list)
    print(f"  [{source}] загружено {len(src_pts)} точек из {len(set(sid_list))} сессий")
    return src_pts, dst_pts, session_ids


def med(pred, true):
    return float(np.sqrt(((pred - true) ** 2).sum(axis=1)).mean())


def session_cv(src_pts, dst_pts, session_ids, model_fn, n_folds=5):
    unique = np.unique(session_ids)
    np.random.shuffle(unique)
    folds = np.array_split(unique, n_folds)
    meds = []
    for val_sessions in folds:
        val_mask = np.isin(session_ids, val_sessions)
        trn_mask = ~val_mask
        if trn_mask.sum() < 50 or val_mask.sum() < 4:
            continue
        try:
            m = model_fn()
            m.fit(src_pts[trn_mask], dst_pts[trn_mask])
            pred = m.predict(src_pts[val_mask])
            meds.append(med(pred, dst_pts[val_mask]))
        except Exception as e:
            print(f"      [CV error]: {e}")
            return float("inf")
    return float(np.mean(meds)) if meds else float("inf")


def train(source, session_dirs):
    print(f"\n{'='*55}")
    print(f"Обучение маппинга: {source} → door2")
    print(f"{'='*55}")

    src_pts, dst_pts, session_ids = load_point_pairs(session_dirs, source)
    np.random.seed(42)

    # Grid search
    print("\n  Grid search (ZonalEnsemble):")
    grid = [
        {"n_zones": 6,  "n_ctrl_per_zone": 80,  "overlap_sigma": 0.5},
        {"n_zones": 9,  "n_ctrl_per_zone": 80,  "overlap_sigma": 0.4},
        {"n_zones": 9,  "n_ctrl_per_zone": 80,  "overlap_sigma": 0.6},
        {"n_zones": 12, "n_ctrl_per_zone": 60,  "overlap_sigma": 0.4},
        {"n_zones": 16, "n_ctrl_per_zone": 50,  "overlap_sigma": 0.35},
    ]

    best_med_val  = float("inf")
    best_params   = None

    for params in grid:
        fn = lambda p=params: ZonalEnsemble(**p)
        cv_med = session_cv(src_pts, dst_pts, session_ids, fn, n_folds=5)
        marker = " ✓" if cv_med < best_med_val else ""
        print(f"    zones={params['n_zones']:2d}  ctrl={params['n_ctrl_per_zone']:3d}"
              f"  sigma={params['overlap_sigma']:.2f}"
              f"  →  CV MED = {cv_med:.2f} px{marker}")
        if cv_med < best_med_val:
            best_med_val = cv_med
            best_params  = params

    # Сравниваем с глобальным SparseTPS
    cv_global = session_cv(src_pts, dst_pts, session_ids,
                           lambda: SparseTPS(n_ctrl=300, regularization=0.1), n_folds=5)
    print(f"    global SparseTPS(300)         →  CV MED = {cv_global:.2f} px"
          + (" ✓" if cv_global < best_med_val else ""))
    if cv_global < best_med_val:
        best_med_val = cv_global
        best_params  = {"type": "global_tps"}

    print(f"\n  ✓ Лучшие параметры: {best_params}  (CV MED = {best_med_val:.2f} px)")

    # Финальное обучение
    print(f"  Финальное обучение на всех {len(src_pts)} точках...")
    if best_params.get("type") == "global_tps":
        model = SparseTPS(n_ctrl=300, regularization=0.1)
        model_name = "sparse_tps_n300"
    else:
        model = ZonalEnsemble(**best_params)
        z, c, s = best_params["n_zones"], best_params["n_ctrl_per_zone"], best_params["overlap_sigma"]
        model_name = f"zonal_z{z}_c{c}_s{s}"

    model.fit(src_pts, dst_pts)

    artifact_path = ARTIFACTS / f"mapper_{source}.pkl"
    with open(artifact_path, "wb") as f:
        pickle.dump({"model": model, "model_name": model_name}, f)
    print(f"  Артефакт сохранён: {artifact_path}")

    return {"source": source, "model_name": model_name, "cv_med": round(best_med_val, 4)}


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
        print(f"  {r['source']:8s} → door2 | {r['model_name']:35s} | CV MED: {r['cv_med']:.2f} px")

    with open(ARTIFACTS / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nГотово! Артефакты в папке artifacts/")


if __name__ == "__main__":
    main()