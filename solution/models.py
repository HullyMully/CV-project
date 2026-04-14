"""
models.py — классы моделей маппинга координат.

v3: добавлен SessionAwareMapper — per-session гомография + глобальный SparseTPS fallback.
"""

import numpy as np
import cv2
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans


# ─── 1. Полиномиальная регрессия с нормализацией ──────────────────────────────

class NormalizedPolyMapper:
    def __init__(self, degree: int = 2, alpha: float = 0.01):
        self.degree = degree
        self.alpha = alpha
        self.src_mean = None
        self.src_std  = None
        self.model_x  = None
        self.model_y  = None

    def _normalize(self, pts):
        return (pts - self.src_mean) / self.src_std

    def fit(self, src, dst):
        self.src_mean = src.mean(axis=0)
        self.src_std  = src.std(axis=0) + 1e-8
        src_n = self._normalize(src)
        self.model_x = Pipeline([("poly", PolynomialFeatures(self.degree, include_bias=True)),
                                  ("ridge", Ridge(alpha=self.alpha, fit_intercept=False))])
        self.model_y = Pipeline([("poly", PolynomialFeatures(self.degree, include_bias=True)),
                                  ("ridge", Ridge(alpha=self.alpha, fit_intercept=False))])
        self.model_x.fit(src_n, dst[:, 0])
        self.model_y.fit(src_n, dst[:, 1])
        return self

    def predict(self, pts):
        src_n = self._normalize(pts)
        return np.stack([self.model_x.predict(src_n), self.model_y.predict(src_n)], axis=1)


# ─── 2. Sparse TPS ────────────────────────────────────────────────────────────

class SparseTPS:
    def __init__(self, n_ctrl: int = 150, regularization: float = 1.0):
        self.n_ctrl = n_ctrl
        self.reg = regularization
        self.ctrl_pts  = None
        self.weights_x = None
        self.weights_y = None
        self.src_mean  = None
        self.src_std   = None

    @staticmethod
    def _kernel(r):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(r == 0, 0.0, r ** 2 * np.log(r ** 2 + 1e-12))

    def _build_K(self, pts_a, pts_b):
        diff = pts_a[:, None, :] - pts_b[None, :, :]
        return self._kernel(np.sqrt((diff ** 2).sum(axis=2)))

    def _normalize(self, pts):
        return (pts - self.src_mean) / self.src_std

    def fit(self, src, dst):
        self.src_mean = src.mean(axis=0)
        self.src_std  = src.std(axis=0) + 1e-8
        src_n = self._normalize(src)
        n_ctrl = min(self.n_ctrl, len(src_n))
        km = KMeans(n_clusters=n_ctrl, n_init=3, random_state=42)
        km.fit(src_n)
        self.ctrl_pts = km.cluster_centers_.astype(np.float64)
        src_n64 = src_n.astype(np.float64)
        dst64   = dst.astype(np.float64)
        K = self._build_K(src_n64, self.ctrl_pts)
        P = np.hstack([np.ones((len(src_n64), 1)), src_n64])
        A = np.hstack([K, P])
        AtA = A.T @ A + self.reg * np.eye(A.shape[1])
        self.weights_x = np.linalg.solve(AtA, A.T @ dst64[:, 0])
        self.weights_y = np.linalg.solve(AtA, A.T @ dst64[:, 1])
        return self

    def predict(self, pts):
        pts_n = self._normalize(pts).astype(np.float64)
        K = self._build_K(pts_n, self.ctrl_pts)
        P = np.hstack([np.ones((len(pts_n), 1)), pts_n])
        A = np.hstack([K, P])
        return np.stack([A @ self.weights_x, A @ self.weights_y], axis=1).astype(np.float32)


# ─── 3. Гомография ────────────────────────────────────────────────────────────

class HomographyMapper:
    def __init__(self):
        self.H = None

    def fit(self, src, dst):
        self.H, _ = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=10)
        if self.H is None:
            self.H, _ = cv2.findHomography(src, dst, 0)
        return self

    def predict(self, pts):
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        proj  = (self.H @ pts_h.T).T
        return (proj[:, :2] / proj[:, 2:3]).astype(np.float32)


# ─── 4. SessionAwareMapper — главная модель ───────────────────────────────────

class SessionAwareMapper:
    """
    Стратегия: каждая сессия имеет свои размеченные точки → считаем
    per-session гомографию. При инференсе:
      - если session_id известен и точек ≥ 4 → используем локальную гомографию
      - иначе → глобальный SparseTPS fallback

    При обучении (fit) session_id = индекс сессии в списке.
    При предсказании без session_id автоматически используется fallback.
    """

    MIN_POINTS_FOR_HOMOGRAPHY = 4

    def __init__(self, n_ctrl: int = 200, regularization: float = 0.5):
        self.n_ctrl = n_ctrl
        self.regularization = regularization
        # session_id (str) → HomographyMapper
        self.session_models: dict = {}
        # Глобальный fallback
        self.global_model: SparseTPS = None

    def fit(self, src_pts, dst_pts, session_ids=None):
        """
        src_pts      : (N, 2)
        dst_pts      : (N, 2)
        session_ids  : (N,) array of str/int — к какой сессии принадлежит точка
        """
        # 1. Глобальный fallback на всех данных
        print(f"    Обучаю глобальный SparseTPS ({self.n_ctrl} опорных точек)...")
        self.global_model = SparseTPS(n_ctrl=self.n_ctrl, regularization=self.regularization)
        self.global_model.fit(src_pts, dst_pts)

        # 2. Per-session гомографии
        if session_ids is not None:
            unique_sessions = np.unique(session_ids)
            print(f"    Обучаю per-session гомографии для {len(unique_sessions)} сессий...")
            for sid in unique_sessions:
                mask = session_ids == sid
                s = src_pts[mask]
                d = dst_pts[mask]
                if len(s) >= self.MIN_POINTS_FOR_HOMOGRAPHY:
                    try:
                        h = HomographyMapper()
                        h.fit(s, d)
                        self.session_models[str(sid)] = h
                    except Exception:
                        pass  # fallback покроет

        return self

    def predict(self, pts, session_id=None):
        """
        pts        : (N, 2)
        session_id : str/int или None → использует глобальный fallback
        """
        if session_id is not None and str(session_id) in self.session_models:
            return self.session_models[str(session_id)].predict(pts)
        return self.global_model.predict(pts)


# ─── Алиасы ───────────────────────────────────────────────────────────────────
PolyMapper = NormalizedPolyMapper
TPSMapper  = SparseTPS