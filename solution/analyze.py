"""
analyze.py — анализ ошибок по сессиям и визуализация.
Помогает понять откуда берётся высокий MED.

Запуск: python analyze.py
"""

import json
import numpy as np
from pathlib import Path
from predict import predict_batch, _load_model

DATA_ROOT  = Path("/Users/hullymully/Documents/development/Projects/CV-project/test-task/")
SPLIT_FILE = DATA_ROOT / "split.json"
SOURCES    = ["top", "bottom"]


def load_by_session(session_dirs, source):
    coord_file = f"coords_{source}.json"
    result = {}
    for session_dir in session_dirs:
        json_path = session_dir / coord_file
        if not json_path.exists():
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            pairs = json.load(f)
        src_list, dst_list = [], []
        for pair in pairs:
            img1 = {p["number"]: (p["x"], p["y"]) for p in pair["image1_coordinates"]}
            img2 = {p["number"]: (p["x"], p["y"]) for p in pair["image2_coordinates"]}
            for num in sorted(set(img1) & set(img2)):
                src_list.append(img2[num])
                dst_list.append(img1[num])
        if src_list:
            result[session_dir.name] = (
                np.array(src_list, dtype=np.float32),
                np.array(dst_list, dtype=np.float32),
            )
    return result


def main():
    with open(SPLIT_FILE) as f:
        split = json.load(f)

    val_dirs   = [DATA_ROOT / p for p in split["val"]]
    train_dirs = [DATA_ROOT / p for p in split["train"]]

    for source in SOURCES:
        print(f"\n{'='*60}")
        print(f"Анализ: {source} → door2")
        print(f"{'='*60}")

        model = _load_model(source)

        # ── Ошибки по val-сессиям ─────────────────────────────────────────────
        val_sessions = load_by_session(val_dirs, source)
        print(f"\n  MED по val-сессиям:")
        session_meds = {}
        for sid, (src, dst) in val_sessions.items():
            pts   = [(float(src[i,0]), float(src[i,1])) for i in range(len(src))]
            preds = predict_batch(pts, source=source)
            pred_arr = np.array(preds, dtype=np.float32)
            errors = np.sqrt(((pred_arr - dst)**2).sum(axis=1))
            session_meds[sid] = float(errors.mean())

        for sid, m in sorted(session_meds.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(m / 20)
            print(f"    {sid[-19:]}  MED={m:7.1f}px  {bar}")

        overall = np.mean(list(session_meds.values()))
        print(f"\n  Среднее по сессиям: {overall:.2f} px")
        print(f"  Худшая сессия    : {max(session_meds.values()):.2f} px")
        print(f"  Лучшая сессия    : {min(session_meds.values()):.2f} px")
        print(f"  Разброс (std)    : {np.std(list(session_meds.values())):.2f} px")

        # ── Анализ ошибок по зонам изображения ───────────────────────────────
        print(f"\n  MED по зонам кадра door2 (3200×1800):")
        all_src, all_dst, all_pred = [], [], []
        for sid, (src, dst) in val_sessions.items():
            pts   = [(float(src[i,0]), float(src[i,1])) for i in range(len(src))]
            preds = predict_batch(pts, source=source)
            all_src.append(src)
            all_dst.append(dst)
            all_pred.append(np.array(preds, dtype=np.float32))

        all_dst  = np.vstack(all_dst)
        all_pred = np.vstack(all_pred)
        all_err  = np.sqrt(((all_pred - all_dst)**2).sum(axis=1))

        # Делим на 3×3 зоны по door2-координатам
        W, H = 3200, 1800
        zones = {"верх-лево": [], "верх-центр": [], "верх-право": [],
                 "центр-лево": [], "центр": [], "центр-право": [],
                 "низ-лево": [], "низ-центр": [], "низ-право": []}

        for i, (x, y) in enumerate(all_dst):
            col = 0 if x < W/3 else (1 if x < 2*W/3 else 2)
            row = 0 if y < H/3 else (1 if y < 2*H/3 else 2)
            key = [["верх-лево","верх-центр","верх-право"],
                   ["центр-лево","центр","центр-право"],
                   ["низ-лево","низ-центр","низ-право"]][row][col]
            zones[key].append(all_err[i])

        for zone, errs in zones.items():
            if errs:
                print(f"    {zone:15s}: MED={np.mean(errs):7.1f}px  (n={len(errs)})")

        # ── Train-сессии: насколько хорошо модель запомнила их ────────────────
        train_sessions = load_by_session(train_dirs[:5], source)  # первые 5
        print(f"\n  MED на первых 5 train-сессиях (проверка переобучения):")
        for sid, (src, dst) in train_sessions.items():
            pts   = [(float(src[i,0]), float(src[i,1])) for i in range(len(src))]
            preds = predict_batch(pts, source=source)
            pred_arr = np.array(preds, dtype=np.float32)
            errors = np.sqrt(((pred_arr - dst)**2).sum(axis=1))
            print(f"    {sid[-19:]}  MED={errors.mean():.1f}px")


if __name__ == "__main__":
    main()
