"""
models.py — классы моделей маппинга координат. v4: ZonalEnsemble.

Ключевая идея: разбить пространство входных координат на зоны
и обучить отдельный SparseTPS для каждой зоны с перекрытием (overlap).
Предсказание = взвешенная сумма моделей ближайших зон.
"""

import numpy as np
import cv2
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans


# ─── SparseTPS ────────────────────────────────────────────────────────────────

class SparseTPS:
    def __init__(self, n_ctrl: int = 150, regularization: float = 0.1):
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


# ─── NormalizedPolyMapper ─────────────────────────────────────────────────────

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
        return np.stack([self.model_x.predict(src_n),
                         self.model_y.predict(src_n)], axis=1)


# ─── HomographyMapper ─────────────────────────────────────────────────────────

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


# ─── ZonalEnsemble — главная модель ──────────────────────────────────────────

class ZonalEnsemble:
    """
    Разбивает входное пространство (координаты source-камеры) на n_zones зон
    через k-means. Для каждой зоны обучает отдельный SparseTPS на точках
    этой зоны + соседних (overlap через sigma-взвешивание).

    Предсказание: взвешенная сумма по всем зонам, вес = exp(-d²/2σ²).
    """

    def __init__(self, n_zones: int = 9, n_ctrl_per_zone: int = 80,
                 regularization: float = 0.1, overlap_sigma: float = 0.4):
        self.n_zones        = n_zones
        self.n_ctrl         = n_ctrl_per_zone
        self.reg            = regularization
        self.sigma          = overlap_sigma   # в нормализованных единицах
        self.zone_centers   = None            # (n_zones, 2) нормализованные
        self.zone_models    = []              # список SparseTPS
        self.src_mean       = None
        self.src_std        = None
        # Глобальный fallback
        self.global_model   = None

    def _normalize(self, pts):
        return (pts - self.src_mean) / self.src_std

    def _zone_weights(self, pts_n):
        """
        pts_n : (M, 2) нормализованные координаты точек запроса
        returns: (M, n_zones) веса (сумма по зонам = 1 для каждой точки)
        """
        # d[i, z] = расстояние от точки i до центра зоны z
        diff = pts_n[:, None, :] - self.zone_centers[None, :, :]  # (M, Z, 2)
        d2   = (diff ** 2).sum(axis=2)                             # (M, Z)
        w    = np.exp(-d2 / (2 * self.sigma ** 2))
        w   /= w.sum(axis=1, keepdims=True) + 1e-12
        return w

    def fit(self, src, dst):
        # Нормализация
        self.src_mean = src.mean(axis=0)
        self.src_std  = src.std(axis=0) + 1e-8
        src_n = self._normalize(src)

        # 1. Глобальная модель (fallback + стабилизация)
        print(f"      Глобальная модель...", end=" ", flush=True)
        self.global_model = SparseTPS(n_ctrl=300, regularization=0.1)
        self.global_model.fit(src, dst)
        print("OK")

        # 2. K-means по входным координатам → центры зон
        n_zones = min(self.n_zones, len(src_n) // 20)
        km = KMeans(n_clusters=n_zones, n_init=5, random_state=42)
        km.fit(src_n)
        self.zone_centers = km.cluster_centers_.astype(np.float32)  # (Z, 2)
        self.n_zones = n_zones

        # 3. Для каждой зоны — обучаем локальный SparseTPS
        # Точки берём с мягким взвешиванием: все точки, но ближние имеют больший вес
        print(f"      Зональные модели ({n_zones} зон)...", end=" ", flush=True)
        self.zone_models = []

        # Расстояния от каждой точки до каждой зоны
        diff = src_n[:, None, :] - self.zone_centers[None, :, :]  # (N, Z, 2)
        d2   = (diff ** 2).sum(axis=2)                             # (N, Z)

        for z in range(n_zones):
            # Берём точки, для которых эта зона — одна из 3 ближайших
            nearest = np.argsort(d2, axis=1)[:, :3]  # (N, 3)
            mask = (nearest == z).any(axis=1)

            # Минимум 30 точек, иначе берём ближайшие 30
            if mask.sum() < 30:
                mask = np.argsort(d2[:, z])[:30]
                mask_idx = mask
            else:
                mask_idx = np.where(mask)[0]

            zone_src = src[mask_idx]
            zone_dst = dst[mask_idx]

            n_ctrl = min(self.n_ctrl, len(zone_src))
            try:
                m = SparseTPS(n_ctrl=n_ctrl, regularization=self.reg)
                m.fit(zone_src, zone_dst)
                self.zone_models.append(m)
            except Exception:
                # Fallback: используем глобальную
                self.zone_models.append(None)

        print("OK")
        return self

    def predict(self, pts):
        pts_n = self._normalize(pts)
        weights = self._zone_weights(pts_n)  # (M, Z)

        # Собираем предсказания всех зон: (M, Z, 2)
        M = len(pts)
        preds = np.zeros((M, self.n_zones, 2), dtype=np.float32)
        for z, model in enumerate(self.zone_models):
            if model is not None:
                preds[:, z, :] = model.predict(pts)
            else:
                preds[:, z, :] = self.global_model.predict(pts)

        # Взвешенная сумма: (M, 2)
        result = (preds * weights[:, :, None]).sum(axis=1)
        return result.astype(np.float32)


# ─── Алиасы для обратной совместимости ───────────────────────────────────────
PolyMapper        = NormalizedPolyMapper
TPSMapper         = SparseTPS
SessionAwareMapper = ZonalEnsemble  # совместимость с predict.py