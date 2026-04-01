"""
Point cloud classifier for pipeline use - Random Forest with handcrafted geometric features.

Optimized for real-time inference in the perception pipeline.
Default: 19 features, 100 trees, no KDTree/ConvexHull dependencies.

Feature modes:
  - 19 features: compact, pipeline fast path (0.69ms/cluster)
  - 23 features: extended (+nn_dist_std, cross_section_mean, z_range_cleaned, hull_point_ratio)
  - 35 features: full set including redundant derived ratios

Usage:
  python classifier_pipeline.py --features 19
  python classifier_pipeline.py --features 23 --force-extract
  python classifier_pipeline.py --features 19 --n-estimators 100 --max-depth 20
"""

import argparse
import logging
import os
import pickle
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import KDTree, ConvexHull
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from data_loader import load_dataset, get_dataset_summary, CLASS_MAP, INV_CLASS_MAP
from data_augmenter import augment_dataset

logger = logging.getLogger(__name__)


# --- COMPACT (19 features)
FEATURES_COMPACT = [
    "num_points",
    "xy_spread",
    "xy_area",
    "z_range",
    "height_to_footprint",
    "density",
    "z_25",
    "z_75",
    "z_std",
    # PCA features
    "xy_aspect_ratio",
    "scattering",
    "linearity",
    "planarity",
    # Vertical profile — ped↔bike separation (6 features, CV 0.8027)
    "layer_frac_0",
    "layer_frac_1",
    "layer_frac_2",
    "layer_frac_3",
    "layer_frac_4",
    "cross_section_var",
]

# --- EXTENDED (23 features) — adds KDTree + ConvexHull, ~1ms/cluster ---
FEATURES_EXTENDED = FEATURES_COMPACT + [
    "nn_dist_std",        # KDTree: organized surface vs clutter
    "cross_section_mean", # complements cross_section_var
    "z_range_cleaned",    # robust z_range (95th-5th percentile)
    "hull_point_ratio",   # ConvexHull: solid object vs scattered noise
]

# --- FULL SET (35 features) — adds redundant globals + cluster quality ---
FEATURE_NAMES_35 = [
    # Size & shape (8)
    "num_points", "x_range", "y_range", "z_range",
    "xy_spread", "xy_area", "bbox_volume", "height_to_footprint",
    # Density (1)
    "density",
    # Z-profile (4)
    "z_std", "z_median", "z_25", "z_75",
    # PCA shape descriptors (6)
    "ev_norm_0", "ev_norm_1", "ev_norm_2",
    "linearity", "planarity", "scattering",
    # Vertical structure (8)
    "vertical_uniformity", "cross_section_var", "cross_section_mean",
    "layer_frac_0", "layer_frac_1", "layer_frac_2",
    "layer_frac_3", "layer_frac_4",
    # Local structure (4)
    "nn_dist_std", "xy_aspect_ratio", "bottom_to_top_ratio", "hull_point_ratio",
    # Cluster quality (4)
    "n_components", "largest_component_ratio", "ground_gap_ratio", "z_range_cleaned",
]

# Mode registry
_MODE_TO_NAMES = {
    19: FEATURES_COMPACT,
    23: FEATURES_EXTENDED,
    35: FEATURE_NAMES_35,
}

def set_feature_mode(mode: int) -> None:
    """Set global feature mode (19, 23, 35). Updates FEATURE_NAMES accordingly."""
    global FEATURE_MODE, FEATURE_NAMES
    if mode not in _MODE_TO_NAMES:
        raise ValueError(f"Invalid feature mode: {mode}. Use one of {list(_MODE_TO_NAMES.keys())}.")
    FEATURE_MODE = mode
    FEATURE_NAMES = _MODE_TO_NAMES[mode]
    logger.info(f"Feature mode set to {FEATURE_MODE} ({len(FEATURE_NAMES)} features)")


def _fast_percentile(z: np.ndarray, q: float) -> float:
    """Approximate percentile via np.partition. O(n) average vs O(n log n) sort.

    For cluster sizes >~500 points this is measurably faster than np.percentile.
    Exact when k lands on an integer index; off by at most one rank otherwise —
    irrelevant for RF feature discretization.

    :param z: 1D array of values.
    :param q: Percentile in [0, 100].
    :return: Approximate q-th percentile value.
    """
    n = len(z)
    if n == 0:
        return 0.0
    if n == 1:
        return float(z[0])
    k = int(q / 100.0 * (n - 1))
    k = min(max(k, 0), n - 1)
    return float(np.partition(z, k)[k])


def _extract_compact(xyz: np.ndarray) -> np.ndarray:
    """Fast path: 11 features. No KDTree, no loops, no ConvexHull.

    Pure min/max/partition ops + two tiny eigvalsh calls (2x2, 3x3).
    Target: <0.3ms per cluster.
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    num_points = len(xyz)

    # Z is correct after ground alignment — only xy needs PCA alignment
    z_range = z.max() - z.min()

    # 2D PCA on xy — align to object's principal horizontal axes
    if num_points > 3:
        xy = xyz[:, :2]
        xy_centered = xy - xy.mean(axis=0)
        cov_2d = xy_centered.T @ xy_centered / num_points
        eigvals_2d, eigvecs_2d = np.linalg.eigh(cov_2d)
        # Rotate xy to object frame
        xy_rotated = xy_centered @ eigvecs_2d
        x_range = xy_rotated[:, 1].max() - xy_rotated[:, 1].min()  # length (larger eigval)
        y_range = xy_rotated[:, 0].max() - xy_rotated[:, 0].min()  # width (smaller eigval)
    else:
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()

    xy_spread = np.sqrt(x_range ** 2 + y_range ** 2)
    xy_area = max(x_range * y_range, 1e-9)
    bbox_volume = max(xy_area * z_range, 1e-9)
    height_to_footprint = z_range / max(xy_spread, 1e-6)
    density = num_points / max(xy_area * np.sqrt(z_range + 1e-6), 1e-6)

    z_25 = _fast_percentile(z, 25)
    z_75 = _fast_percentile(z, 75)
    z_std = z.std()

    # 2D PCA — 2x2 covariance, microseconds
    if num_points > 3:
        ev_2d = np.clip(eigvals_2d, 1e-9, None)
        xy_aspect_ratio = ev_2d[1] / ev_2d[0]  # length/width
    else:
        xy_aspect_ratio = 1.0

    # 3D PCA — 3x3 covariance, microseconds
    centered = xyz - xyz.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.clip(eigenvalues, 1e-9, None)
    scattering = eigenvalues[0] / eigenvalues[2]
    linearity = (eigenvalues[2] - eigenvalues[1]) / eigenvalues[2]
    planarity = (eigenvalues[1] - eigenvalues[0]) / eigenvalues[2]
    # Vertical layer fractions — 5-bin histogram
    if z_range > 0.01:
        z_normalized = (z - z.min()) / z_range
        layer_fractions = []
        for i in range(5):
            lo, hi = i / 5.0, (i + 1) / 5.0
            frac = np.sum((z_normalized >= lo) & (z_normalized < hi)) / num_points
            layer_fractions.append(frac)
    else:
        layer_fractions = [0.2] * 5

    # Cross-section variance
    if z_range > 0.1 and num_points > 20:
        slice_spreads = []
        for i in range(5):
            lo = z.min() + i * z_range / 5
            hi = z.min() + (i + 1) * z_range / 5
            mask = (z >= lo) & (z < hi)
            if mask.sum() > 2:
                if num_points > 3:
                    sl_xy = xy_centered[mask] @ eigvecs_2d
                    sx = sl_xy[:, 1].max() - sl_xy[:, 1].min()
                    sy = sl_xy[:, 0].max() - sl_xy[:, 0].min()
                else:
                    sx = x[mask].max() - x[mask].min()
                    sy = y[mask].max() - y[mask].min()
                slice_spreads.append(np.sqrt(sx ** 2 + sy ** 2))
            else:
                slice_spreads.append(0.0)
        cross_section_var = np.std(slice_spreads)
    else:
        cross_section_var = 0.0

    return np.array([
        num_points, xy_spread, xy_area, z_range,
        height_to_footprint, density, z_25, z_75,
        z_std, xy_aspect_ratio, scattering,
        linearity, planarity,
        *layer_fractions,
        cross_section_var,
    ], dtype=np.float32)


def extract_features(points: np.ndarray) -> np.ndarray:
    """Extract geometric features from a single point cloud cluster.

    :param points: (N, 3+) array. Only first 3 columns (x, y, z) are used.
    :return: Feature vector sized by current FEATURE_MODE.
    """
    xyz = points[:, :3]
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    num_points = len(xyz)
    z_range = (z.max() - z.min()) if num_points > 1 else 0.0

    # --- 19 features: fast path ---
    if FEATURE_MODE == 19:
        return _extract_compact(xyz)

    # --- 23 features: compact + 4 extras ---
    if FEATURE_MODE == 23:
        compact = _extract_compact(xyz)

        # nn_dist_std
        if num_points > 6:
            tree = KDTree(xyz)
            dists, _ = tree.query(xyz, k=min(6, num_points))
            nn_dist_std = dists[:, 1:].mean(axis=1).std()
        else:
            nn_dist_std = 0.0

        # cross_section_mean — PCA-aligned to match compact path
        if z_range > 0.1 and num_points > 20:
            if num_points > 3:
                xy_centered = xyz[:, :2] - xyz[:, :2].mean(axis=0)
                cov_2d = xy_centered.T @ xy_centered / num_points
                _, eigvecs_2d = np.linalg.eigh(cov_2d)
            else:
                xy_centered = xyz[:, :2] - xyz[:, :2].mean(axis=0)
                eigvecs_2d = np.eye(2)
            slice_spreads = []
            for i in range(5):
                lo = z.min() + i * z_range / 5
                hi = z.min() + (i + 1) * z_range / 5
                mask = (z >= lo) & (z < hi)
                if mask.sum() > 2:
                    sl_xy = xy_centered[mask] @ eigvecs_2d
                    sx = sl_xy[:, 1].max() - sl_xy[:, 1].min()
                    sy = sl_xy[:, 0].max() - sl_xy[:, 0].min()
                    slice_spreads.append(np.sqrt(sx ** 2 + sy ** 2))
                else:
                    slice_spreads.append(0.0)
            cross_section_mean = np.mean(slice_spreads)
        else:
            cross_section_mean = 0.0

        # z_range_cleaned
        if num_points > 10:
            z_range_cleaned = _fast_percentile(z, 95) - _fast_percentile(z, 5)
        else:
            z_range_cleaned = z_range

        # hull_point_ratio
        if num_points > 10:
            try:
                hull = ConvexHull(xyz)
                hull_point_ratio = len(hull.vertices) / num_points
            except Exception:
                hull_point_ratio = 1.0
        else:
            hull_point_ratio = 1.0

        return np.concatenate([compact, [nn_dist_std, cross_section_mean, z_range_cleaned, hull_point_ratio]])

    # --- 35 features: full set ---
    # 2D PCA on xy — yaw-invariant horizontal dimensions
    if num_points > 3:
        xy_centered = xyz[:, :2] - xyz[:, :2].mean(axis=0)
        cov_2d = xy_centered.T @ xy_centered / num_points
        eigvals_2d, eigvecs_2d = np.linalg.eigh(cov_2d)
        xy_rotated = xy_centered @ eigvecs_2d
        x_range = xy_rotated[:, 1].max() - xy_rotated[:, 1].min()  # length
        y_range = xy_rotated[:, 0].max() - xy_rotated[:, 0].min()  # width
    else:
        xy_centered = xyz[:, :2] - xyz[:, :2].mean(axis=0)
        eigvals_2d = np.array([1e-9, 1e-9])
        eigvecs_2d = np.eye(2)
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()

    xy_spread = np.sqrt(x_range ** 2 + y_range ** 2)
    xy_area = max(x_range * y_range, 1e-9)
    bbox_volume = max(xy_area * z_range, 1e-9)
    height_to_footprint = z_range / max(np.sqrt(xy_area), 1e-6)
    density = num_points / max(xy_area * np.sqrt(z_range + 1e-6), 1e-6)

    z_std = z.std()
    z_median = np.median(z)
    z_25 = _fast_percentile(z, 25)
    z_75 = _fast_percentile(z, 75)

    centered = xyz - xyz.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.clip(eigenvalues, 1e-9, None)
    ev_sum = eigenvalues.sum()
    ev_norm_0 = eigenvalues[0] / ev_sum
    ev_norm_1 = eigenvalues[1] / ev_sum
    ev_norm_2 = eigenvalues[2] / ev_sum
    linearity = (eigenvalues[2] - eigenvalues[1]) / eigenvalues[2]
    planarity = (eigenvalues[1] - eigenvalues[0]) / eigenvalues[2]
    scattering = eigenvalues[0] / eigenvalues[2]

    if z_range > 0.01:
        z_normalized = (z - z.min()) / z_range
        layer_fractions = []
        for i in range(5):
            lo, hi = i / 5.0, (i + 1) / 5.0
            frac = np.sum((z_normalized >= lo) & (z_normalized < hi)) / num_points
            layer_fractions.append(frac)
    else:
        layer_fractions = [0.2] * 5
    vertical_uniformity = np.std(layer_fractions)

    if z_range > 0.1 and num_points > 20:
        slice_spreads = []
        for i in range(5):
            lo = z.min() + i * z_range / 5
            hi = z.min() + (i + 1) * z_range / 5
            smask = (z >= lo) & (z < hi)
            if smask.sum() > 2:
                sl_xy = xy_centered[smask] @ eigvecs_2d
                sx = sl_xy[:, 1].max() - sl_xy[:, 1].min()
                sy = sl_xy[:, 0].max() - sl_xy[:, 0].min()
                slice_spreads.append(np.sqrt(sx ** 2 + sy ** 2))
            else:
                slice_spreads.append(0.0)
        cross_section_var = np.std(slice_spreads)
        cross_section_mean = np.mean(slice_spreads)
    else:
        cross_section_var = 0.0
        cross_section_mean = 0.0

    # Local: nn_dist_std
    if num_points > 6:
        tree = KDTree(xyz)
        dists, _ = tree.query(xyz, k=min(6, num_points))
        nn_dist_std = dists[:, 1:].mean(axis=1).std()
    else:
        nn_dist_std = 0.0

    # Local: xy_aspect_ratio — reuse 2D PCA from above
    ev_2d = np.clip(eigvals_2d, 1e-9, None)
    xy_aspect_ratio = ev_2d[1] / ev_2d[0]

    # Local: bottom_to_top_ratio
    bottom_to_top_ratio = layer_fractions[0] / max(layer_fractions[4], 1e-6)

    # Local: hull_point_ratio
    if num_points > 10:
        try:
            hull = ConvexHull(xyz)
            hull_point_ratio = len(hull.vertices) / num_points
        except Exception:
            hull_point_ratio = 1.0
    else:
        hull_point_ratio = 1.0

    # Quality: n_components + largest_component_ratio
    if num_points > 10:
        dists_3, _ = tree.query(xyz, k=min(3, num_points))
        median_nn = np.median(dists_3[:, 1])
        db = DBSCAN(eps=median_nn * 2.5, min_samples=3)
        labels_db = db.fit_predict(xyz)
        valid_labels = labels_db[labels_db >= 0]
        if len(valid_labels) > 0:
            unique_labels, counts = np.unique(valid_labels, return_counts=True)
            n_components = float(len(unique_labels))
            largest_component_ratio = float(counts.max()) / num_points
        else:
            n_components = 1.0
            largest_component_ratio = 1.0
    else:
        n_components = 1.0
        largest_component_ratio = 1.0

    # Quality: ground_gap_ratio
    if z_range > 0.1:
        z_5 = _fast_percentile(z, 5)
        z_10 = _fast_percentile(z, 10)
        ground_gap_ratio = (z_10 - z_5) / z_range
    else:
        ground_gap_ratio = 0.0

    # Quality: z_range_cleaned
    if num_points > 10:
        z_range_cleaned = _fast_percentile(z, 95) - _fast_percentile(z, 5)
    else:
        z_range_cleaned = z_range

    return np.array([
        num_points, x_range, y_range, z_range,
        xy_spread, xy_area, bbox_volume, height_to_footprint,
        density,
        z_std, z_median, z_25, z_75,
        ev_norm_0, ev_norm_1, ev_norm_2,
        linearity, planarity, scattering,
        vertical_uniformity, cross_section_var, cross_section_mean,
        layer_fractions[0], layer_fractions[1], layer_fractions[2],
        layer_fractions[3], layer_fractions[4],
        nn_dist_std, xy_aspect_ratio, bottom_to_top_ratio, hull_point_ratio,
        n_components, largest_component_ratio, ground_gap_ratio, z_range_cleaned,
    ], dtype=np.float32)

def extract_dataset_features(
    points_list: List[np.ndarray],
    labels: List[int],
    cache_path: str = None,
    force_extract: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features for an entire dataset. Caches to numpy files.

    :param points_list: List of (N_i, 3) point cloud arrays.
    :param labels: List of integer class labels.
    :param cache_path: If provided, save/load features here.
    :param force_extract: If True, re-extract even if cache exists.
    :return: (X, y) where X is (n_samples, n_features) and y is (n_samples,).
    """
    y = np.array(labels, dtype=np.int32)

    # Try loading from cache
    if cache_path and os.path.exists(cache_path) and not force_extract:
        logger.info(f"Loading cached features from {cache_path}")
        data = np.load(cache_path)
        X = data["X"]
        y_cached = data["y"]
        if X.shape[0] == len(labels) and X.shape[1] == len(FEATURE_NAMES):
            logger.info(f"Cache valid: {X.shape}")
            return X, y_cached
        else:
            logger.warning(
                f"Cache mismatch: cached {X.shape} vs expected "
                f"({len(labels)}, {len(FEATURE_NAMES)}). Re-extracting."
            )

    # Extract features with latency instrumentation
    n_features = len(FEATURE_NAMES)
    X = np.zeros((len(points_list), n_features), dtype=np.float32)
    extract_times = []

    for i, pts in enumerate(points_list):
        t0 = time.perf_counter()
        X[i] = extract_features(pts)
        extract_times.append(time.perf_counter() - t0)
        if (i + 1) % 5000 == 0:
            logger.info(f"Extracted features: {i + 1}/{len(points_list)}")

    et = np.array(extract_times) * 1000  # ms
    logger.info(
        f"Extract latency: mean={et.mean():.3f}ms  "
        f"median={np.median(et):.3f}ms  p95={_fast_percentile(et, 95):.3f}ms  "
        f"max={et.max():.3f}ms"
    )

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    logger.info(f"Feature matrix shape: {X.shape}")

    # Save to cache
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, X=X, y=y)
        logger.info(f"Features cached to {cache_path}")

    return X, y


def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    n_estimators: int = 100,
    max_depth: int = 20,
    groups: np.ndarray = None,
) -> Dict:
    """Stratified k-fold CV — development metrics. Uses StratifiedGroupKFold when groups provided."""
    if groups is not None:
        skf = StratifiedGroupKFold(n_splits=n_folds)
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    class_names = [INV_CLASS_MAP[i] for i in sorted(INV_CLASS_MAP.keys())]

    fold_results = []
    all_val_preds = np.zeros_like(y)
    all_val_true = np.zeros_like(y)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        all_val_preds[val_idx] = y_pred
        all_val_true[val_idx] = y_val

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, labels=list(range(4)), zero_division=0
        )
        macro_f1 = f1_score(y_val, y_pred, average="macro")

        fold_results.append({
            "fold": fold_idx + 1,
            "macro_f1": macro_f1,
            "per_class_f1": dict(zip(class_names, f1)),
            "per_class_prec": dict(zip(class_names, prec)),
            "per_class_rec": dict(zip(class_names, rec)),
        })

        logger.info(
            f"Fold {fold_idx + 1}/{n_folds}: macro-F1={macro_f1:.4f} | "
            + " | ".join([f"{name}={f:.3f}" for name, f in zip(class_names, f1)])
        )

    avg_macro_f1 = np.mean([r["macro_f1"] for r in fold_results])
    std_macro_f1 = np.std([r["macro_f1"] for r in fold_results])

    print("\n" + "=" * 70)
    print(f"CV RESULTS ({n_folds}-fold stratified, {n_estimators} trees, depth {max_depth})")
    print("=" * 70)
    print(f"Macro F1: {avg_macro_f1:.4f} ± {std_macro_f1:.4f}")
    print()

    for cname in class_names:
        avg_f1 = np.mean([r["per_class_f1"][cname] for r in fold_results])
        std_f1 = np.std([r["per_class_f1"][cname] for r in fold_results])
        avg_prec = np.mean([r["per_class_prec"][cname] for r in fold_results])
        avg_rec = np.mean([r["per_class_rec"][cname] for r in fold_results])
        print(f"  {cname:>12s}: F1={avg_f1:.4f}±{std_f1:.4f}  P={avg_prec:.4f}  R={avg_rec:.4f}")

    print("=" * 70)

    cm = confusion_matrix(all_val_true, all_val_preds, labels=list(range(4)))
    print("\nCV Confusion Matrix (aggregated):")
    print(pd.DataFrame(cm, index=class_names, columns=class_names))

    return {
        "fold_results": fold_results,
        "avg_macro_f1": avg_macro_f1,
        "std_macro_f1": std_macro_f1,
        "cv_confusion_matrix": cm,
    }


def train_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 20,
) -> RandomForestClassifier:
    """Train final model on full training set."""
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    logger.info(f"Final model trained: {n_estimators} trees, depth {max_depth}")
    return rf


def evaluate_on_test(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_dir: str = "results",
) -> Dict:
    """Evaluate on held-out test set — the submission metrics."""
    os.makedirs(save_dir, exist_ok=True)
    class_names = [INV_CLASS_MAP[i] for i in sorted(INV_CLASS_MAP.keys())]

    # --- Latency measurement: single-cluster ---
    n_sample = min(200, len(X_test))
    sample_idx = np.random.RandomState(42).choice(len(X_test), n_sample, replace=False)

    t0 = time.perf_counter()
    for idx in sample_idx:
        model.predict(X_test[idx].reshape(1, -1))
    t_predict = (time.perf_counter() - t0) / n_sample

    print(f"\n--- Latency ({FEATURE_MODE} features, n={n_sample} clusters) ---")
    print(f"  RF predict (single):  {t_predict*1000:.3f} ms/cluster")

    # --- Latency measurement: batch 450 clusters (simulated frame) ---
    batch_idx = np.random.RandomState(42).choice(len(X_test), 450, replace=False)
    X_batch = X_test[batch_idx]
    t0 = time.perf_counter()
    _ = model.predict(X_batch)
    t_batch = (time.perf_counter() - t0) * 1000
    print(f"  RF predict (batch 450): {t_batch:.1f}ms")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print("\n" + "=" * 70)
    print(f"TEST SET RESULTS ({FEATURE_MODE} features)")
    print("=" * 70)
    print(report)

    cm = confusion_matrix(y_test, y_pred, labels=list(range(4)))
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    print("Confusion Matrix (raw counts):")
    print(pd.DataFrame(cm, index=class_names, columns=class_names))
    print()
    print("Confusion Matrix (normalized by true label):")
    print(pd.DataFrame(
        np.round(cm_normalized, 3),
        index=class_names,
        columns=class_names,
    ))

    # --- Plot confusion matrices ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_title(f"Confusion Matrix — {FEATURE_MODE} features (counts)", fontsize=12)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(4):
        for j in range(4):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    im = ax.imshow(cm_normalized, cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title(f"Confusion Matrix — {FEATURE_MODE} features (normalized)", fontsize=12)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(4):
        for j in range(4):
            color = "white" if cm_normalized[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_normalized[i, j]:.3f}",
                    ha="center", va="center", color=color, fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"confusion_matrix_{FEATURE_MODE}feat.png"), dpi=120)
    plt.close()
    logger.info(f"Saved confusion_matrix_{FEATURE_MODE}feat.png")

    # --- Feature importance ---
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(
        range(len(FEATURE_NAMES)),
        importances[sorted_idx],
        color="steelblue",
    )
    ax.set_yticks(range(len(FEATURE_NAMES)))
    ax.set_yticklabels([FEATURE_NAMES[i] for i in sorted_idx], fontsize=8)
    ax.invert_yaxis()
    ax.set_title(f"Feature Importance — {FEATURE_MODE} features (Gini)", fontsize=12)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"feature_importance_{FEATURE_MODE}feat.png"), dpi=120)
    plt.close()
    logger.info(f"Saved feature_importance_{FEATURE_MODE}feat.png")

    # --- Per-class confidence analysis ---
    print("\nPer-class prediction confidence (mean predicted probability):")
    for cls_id, cname in INV_CLASS_MAP.items():
        mask = y_test == cls_id
        if mask.sum() > 0:
            correct_mask = mask & (y_pred == cls_id)
            wrong_mask = mask & (y_pred != cls_id)

            correct_conf = y_proba[correct_mask, cls_id].mean() if correct_mask.sum() > 0 else 0.0
            wrong_conf = y_proba[wrong_mask].max(axis=1).mean() if wrong_mask.sum() > 0 else 0.0

            print(
                f"  {cname:>12s}: correct={correct_conf:.3f}, "
                f"misclassified={wrong_conf:.3f}, "
                f"n_correct={correct_mask.sum()}, n_wrong={wrong_mask.sum()}"
            )

    return {
        "y_pred": y_pred,
        "y_proba": y_proba,
        "confusion_matrix": cm,
        "confusion_matrix_normalized": cm_normalized,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Point cloud classifier")
    parser.add_argument(
        "--features", type=int, choices=[19, 23, 35], default=19,
        help="Feature mode: 19 (compact), 23 (extended)",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=100,
        help="Number of RF trees (default: 100)",
    )
    parser.add_argument(
        "--max-depth", type=int, default=20,
        help="Max tree depth (default: 20)",
    )
    parser.add_argument(
        "--force-extract", action="store_true",
        help="Force re-extraction even if cached features exist",
    )
    args = parser.parse_args()

    set_feature_mode(args.features)

    # --- Load data ---
    logger.info("Loading training data...")
    train_points, train_labels, train_paths = load_dataset("data/train")
    get_dataset_summary(train_labels)

    logger.info("Loading test data...")
    test_points, test_labels, test_paths = load_dataset("data/test")
    get_dataset_summary(test_labels)

    # --- Extract features (with caching) ---
    cache_dir = "cache"
    train_cache = os.path.join(cache_dir, f"train_{FEATURE_MODE}feat.npz")
    test_cache = os.path.join(cache_dir, f"test_{FEATURE_MODE}feat.npz")

    # --- Augment training data (disabled - degraded test by 3pp due to CV data leak) ---
    logger.info("Augmenting training data with partial views...")
    train_points, train_labels, train_groups = augment_dataset(train_points, train_labels)

    logger.info(f"Extracting training features ({FEATURE_MODE} features per cluster)...")
    X_train, y_train = extract_dataset_features(
        train_points, train_labels,
        cache_path=train_cache,
        force_extract=args.force_extract,
    )

    logger.info(f"Extracting test features ({FEATURE_MODE} features per cluster)...")
    X_test, y_test = extract_dataset_features(
        test_points, test_labels,
        cache_path=test_cache,
        force_extract=args.force_extract,
    )

    # --- Cross-validation (development) ---
    logger.info("Running 5-fold stratified CV...")
    cv_results = run_cross_validation(
        X_train, y_train,
        n_folds=5,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )

    # --- Train final model ---
    logger.info("Training final model on full training set...")
    model = train_final_model(
        X_train, y_train,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )

    # --- Evaluate on test ---
    logger.info("Evaluating on test set...")
    test_results = evaluate_on_test(model, X_test, y_test)

    # --- Save model ---
    os.makedirs("models", exist_ok=True)
    model_path = f"models/rf_classifier_{FEATURE_MODE}feat.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_names": FEATURE_NAMES,
            "feature_mode": FEATURE_MODE,
        }, f)
    logger.info(f"Model saved to {model_path}")

    # Also save as default for other scripts
    default_path = "models/rf_classifier.pkl"
    with open(default_path, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_names": FEATURE_NAMES,
            "feature_mode": FEATURE_MODE,
        }, f)
    logger.info(f"Also saved as {default_path}")

    logger.info("Done.")