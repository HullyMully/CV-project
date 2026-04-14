"""
models.py — классы моделей маппинга координат.
Вынесены отдельно чтобы pickle корректно загружал их из любого модуля.

Улучшения v2:
- NormalizedPolyMapper: нормализация координат перед полиномом → нет ill-conditioned
- SparseTPS: TPS на ~150 опорных точках (k-means) вместо всех → стабильно и быстро
- HomographyMapper: базовая линия
"""

import numpy as np
import cv2
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans


# ─── 1. Полиномиальная регрессия с нормализацией ──────────────────────────────

class NormalizedPolyMapper:
    """
    Полиномиальная регрессия степени degree с нормализацией координат.
    Нормализация убирает ill-conditioned матрицы и улучшает точность.
    """

    def __init__(self, degree: int = 2, alpha: float = 0.01):
        self.degree = degree
        self.alpha = alpha
        self.src_mean = None
        self.src_std  = None
        self.model_x  = None
        self.model_y  = None

    def _normalize(self, pts: np.ndarray) -> np.ndarray:
        return (pts - self.src_mean) / self.src_std

    def fit(self, src: np.ndarray, dst: np.ndarray):
        self.src_mean = src.mean(axis=0)
        self.src_std  = src.std(axis=0) + 1e-8

        src_n = self._normalize(src)

        self.model_x = Pipeline([
            ("poly",  PolynomialFeatures(self.degree, include_bias=True)),
            ("ridge", Ridge(alpha=self.alpha, fit_intercept=False)),
        ])
        self.model_y = Pipeline([
            ("poly",  PolynomialFeatures(self.degree, include_bias=True)),
            ("ridge", Ridge(alpha=self.alpha, fit_intercept=False)),
        ])
        self.model_x.fit(src_n, dst[:, 0])
        self.model_y.fit(src_n, dst[:, 1])
        return self

    def predict(self, pts: np.ndarray) -> np.ndarray:
        src_n = self._normalize(pts)
        px = self.model_x.predict(src_n)
        py = self.model_y.predict(src_n)
        return np.stack([px, py], axis=1)


# ─── 2. Sparse TPS (TPS на кластерных центрах) ────────────────────────────────

class SparseTPS:
    """
    Thin Plate Spline на n_ctrl опорных точках (k-means центроиды).
    Стабилен даже на тысячах обучающих точек.
    Координаты нормализуются внутри.
    """

    def __init__(self, n_ctrl: int = 150, regularization: float = 1.0):
        self.n_ctrl = n_ctrl
        self.reg = regularization
        self.ctrl_pts  = None
        self.weights_x = None
        self.weights_y = None
        self.src_mean  = None
        self.src_std   = None

    @staticmethod
    def _kernel(r: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(r == 0, 0.0, r ** 2 * np.log(r ** 2 + 1e-12))

    def _build_K(self, pts_a: np.ndarray, pts_b: np.ndarray) -> np.ndarray:
        diff = pts_a[:, None, :] - pts_b[None, :, :]
        r = np.sqrt((diff ** 2).sum(axis=2))
        return self._kernel(r)

    def _normalize(self, pts: np.ndarray) -> np.ndarray:
        return (pts - self.src_mean) / self.src_std

    def fit(self, src: np.ndarray, dst: np.ndarray):
        self.src_mean = src.mean(axis=0)
        self.src_std  = src.std(axis=0) + 1e-8
        src_n = self._normalize(src)

        n_ctrl = min(self.n_ctrl, len(src_n))
        km = KMeans(n_clusters=n_ctrl, n_init=3, random_state=42)
        km.fit(src_n)
        self.ctrl_pts = km.cluster_centers_.astype(np.float64)

        n = len(self.ctrl_pts)
        src_n64 = src_n.astype(np.float64)
        dst64   = dst.astype(np.float64)

        K_train = self._build_K(src_n64, self.ctrl_pts)
        P_train = np.hstack([np.ones((len(src_n64), 1)), src_n64])
        A = np.hstack([K_train, P_train])

        reg_matrix = self.reg * np.eye(A.shape[1])
        AtA = A.T @ A + reg_matrix
        self.weights_x = np.linalg.solve(AtA, A.T @ dst64[:, 0])
        self.weights_y = np.linalg.solve(AtA, A.T @ dst64[:, 1])
        return self

    def predict(self, pts: np.ndarray) -> np.ndarray:
        pts_n = self._normalize(pts).astype(np.float64)
        K = self._build_K(pts_n, self.ctrl_pts)
        P = np.hstack([np.ones((len(pts_n), 1)), pts_n])
        A = np.hstack([K, P])
        pred_x = A @ self.weights_x
        pred_y = A @ self.weights_y
        return np.stack([pred_x, pred_y], axis=1).astype(np.float32)


# ─── 3. Гомография (базовая линия) ────────────────────────────────────────────

class HomographyMapper:
    """Обёртка над cv2.findHomography (проективное преобразование)."""

    def __init__(self):
        self.H = None

    def fit(self, src: np.ndarray, dst: np.ndarray):
        self.H, _ = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=10)
        if self.H is None:
            self.H, _ = cv2.findHomography(src, dst, 0)
        return self

    def predict(self, pts: np.ndarray) -> np.ndarray:
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        proj  = (self.H @ pts_h.T).T
        return (proj[:, :2] / proj[:, 2:3]).astype(np.float32)


# ─── Алиасы для обратной совместимости ───────────────────────────────────────
PolyMapper = NormalizedPolyMapper
TPSMapper  = SparseTPS