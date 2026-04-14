"""
evaluate.py — считает MED на валидационном сплите.

Два режима оценки:
  1. global   — только глобальный SparseTPS (как для полностью новой сессии)
  2. per_session — per-session гомография из val-точек (как если бы были размечены
                   несколько точек новой сессии для калибровки)

Запуск: python evaluate.py
"""

import json
import numpy as np
from pathlib import Path

from predict import predict_batch, _load_model

DATA_ROOT  = Path("/Users/hullymully/Documents/development/Projects/CV-project/test-task/")
SPLIT_FILE = DATA_ROOT / "split.json"
RESULTS    = Path("results")
RESULTS.mkdir(exist_ok=True)

SOURCES = ["top", "bottom"]


def load_val_pairs_by_session(session_dirs, source):
    """Возвращает dict: session_id → (src_pts, dst_pts)"""
    coord_file = f"coords_{source}.json"
    sessions = {}

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
            common = set(img1.keys()) & set(img2.keys())
            for num in sorted(common):
                src_list.append(img2[num])
                dst_list.append(img1[num])

        if src_list:
            sessions[session_dir.name] = (
                np.array(src_list, dtype=np.float32),
                np.array(dst_list, dtype=np.float32),
            )

    return sessions


def med(pred, true):
    return float(np.sqrt(((pred - true) ** 2).sum(axis=1)).mean())


def percentile_error(pred, true, p):
    errors = np.sqrt(((pred - true) ** 2).sum(axis=1))
    return float(np.percentile(errors, p))


def evaluate():
    with open(SPLIT_FILE, "r") as f:
        split = json.load(f)

    val_dirs = [DATA_ROOT / p for p in split["val"]]

    print("=" * 60)
    print("ОЦЕНКА НА ВАЛИДАЦИОННОМ СПЛИТЕ")
    print("=" * 60)

    all_metrics = {}

    for source in SOURCES:
        print(f"\n[{source} → door2]")
        sessions = load_val_pairs_by_session(val_dirs, source)
        total_pts = sum(len(s[0]) for s in sessions.values())
        print(f"  Сессий: {len(sessions)}, точек всего: {total_pts}")

        model = _load_model(source)

        # ── Режим 1: глобальный fallback ──────────────────────────────────────
        all_src = np.vstack([s[0] for s in sessions.values()])
        all_dst = np.vstack([s[1] for s in sessions.values()])

        pred_global = model.predict(all_src, session_id=None)
        med_global  = med(pred_global, all_dst)
        p90_global  = percentile_error(pred_global, all_dst, 90)

        print(f"\n  [Режим 1: глобальный SparseTPS]")
        print(f"    MED  = {med_global:.2f} px")
        print(f"    P90  = {p90_global:.2f} px")
        print(f"    % < 50px  = {(np.sqrt(((pred_global - all_dst)**2).sum(1)) < 50).mean()*100:.1f}%")
        print(f"    % < 100px = {(np.sqrt(((pred_global - all_dst)**2).sum(1)) < 100).mean()*100:.1f}%")

        # ── Режим 2: per-session гомография ───────────────────────────────────
        # Для каждой val-сессии берём первые N точек как "калибровочные",
        # остальные предсказываем. Это честная оценка если при деплое
        # размечают несколько точек новой сессии.
        import cv2
        CALIB_PTS = 8  # сколько точек используем для per-session гомографии

        preds_session, trues_session = [], []
        for sid, (src_pts, dst_pts) in sessions.items():
            if len(src_pts) <= CALIB_PTS:
                # Недостаточно — используем глобальный
                p = model.predict(src_pts, session_id=None)
            else:
                # Строим локальную гомографию из первых CALIB_PTS точек
                calib_src = src_pts[:CALIB_PTS]
                calib_dst = dst_pts[:CALIB_PTS]
                test_src  = src_pts[CALIB_PTS:]
                test_dst  = dst_pts[CALIB_PTS:]

                H, _ = cv2.findHomography(calib_src, calib_dst,
                                           cv2.RANSAC, ransacReprojThreshold=15)
                if H is not None:
                    pts_h = np.hstack([test_src, np.ones((len(test_src), 1))])
                    proj  = (H @ pts_h.T).T
                    p = (proj[:, :2] / proj[:, 2:3]).astype(np.float32)
                    src_pts = test_src
                    dst_pts = test_dst
                else:
                    p = model.predict(src_pts, session_id=None)

            preds_session.append(p)
            trues_session.append(dst_pts)

        pred_sess = np.vstack(preds_session)
        true_sess = np.vstack(trues_session)
        med_sess  = med(pred_sess, true_sess)
        p90_sess  = percentile_error(pred_sess, true_sess, 90)

        print(f"\n  [Режим 2: per-session гомография ({CALIB_PTS} калибр. точек)]")
        print(f"    MED  = {med_sess:.2f} px")
        print(f"    P90  = {p90_sess:.2f} px")
        print(f"    % < 50px  = {(np.sqrt(((pred_sess - true_sess)**2).sum(1)) < 50).mean()*100:.1f}%")
        print(f"    % < 100px = {(np.sqrt(((pred_sess - true_sess)**2).sum(1)) < 100).mean()*100:.1f}%")

        all_metrics[source] = {
            "global": {
                "MED_px": round(med_global, 4),
                "P90_px": round(p90_global, 4),
                "n_points": int(total_pts),
            },
            "per_session_homography": {
                "MED_px": round(med_sess, 4),
                "P90_px": round(p90_sess, 4),
                "calib_points_used": CALIB_PTS,
            }
        }

    print("\n" + "=" * 60)
    print("ИТОГОВЫЕ МЕТРИКИ (val-сплит)")
    print("=" * 60)
    for src, m in all_metrics.items():
        g = m["global"]
        s = m["per_session_homography"]
        print(f"  {src:6s} → door2")
        print(f"    глобальный      : MED = {g['MED_px']:.2f} px | P90 = {g['P90_px']:.2f} px")
        print(f"    per-session     : MED = {s['MED_px']:.2f} px | P90 = {s['P90_px']:.2f} px")

    metrics_path = RESULTS / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\nМетрики сохранены: {metrics_path}")
    return all_metrics


if __name__ == "__main__":
    evaluate()