"""
Ablation study: compare 19, 23, and 35 feature sets.

Extracts all 35 features into CSV, then evaluates RF on each subset.
Run once to generate CSVs, subsequent runs load from cache.

Usage:
  python ablation_study.py
  python ablation_study.py --force-extract
"""

import argparse
import logging
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from data_loader import load_dataset, CLASS_MAP, INV_CLASS_MAP

logger = logging.getLogger(__name__)

CLASS_NAMES = ["background", "bicyclist", "car", "pedestrian"]

# =====================================================================
# Feature name lists
# =====================================================================

FEATURES_19 = [
    "num_points", "xy_spread", "xy_area", "z_range",
    "height_to_footprint", "density", "z_25", "z_75",
    "z_std", "xy_aspect_ratio", "scattering", "linearity", "planarity",
    "layer_frac_0", "layer_frac_1", "layer_frac_2",
    "layer_frac_3", "layer_frac_4", "cross_section_var",
]

FEATURES_23 = FEATURES_19 + [
    "nn_dist_std", "cross_section_mean", "z_range_cleaned", "hull_point_ratio",
]

FEATURES_35 = [
    "num_points", "x_range", "y_range", "z_range",
    "xy_spread", "xy_area", "bbox_volume", "height_to_footprint",
    "density",
    "z_std", "z_median", "z_25", "z_75",
    "ev_norm_0", "ev_norm_1", "ev_norm_2",
    "linearity", "planarity", "scattering",
    "vertical_uniformity", "cross_section_var", "cross_section_mean",
    "layer_frac_0", "layer_frac_1", "layer_frac_2",
    "layer_frac_3", "layer_frac_4",
    "nn_dist_std", "xy_aspect_ratio", "bottom_to_top_ratio", "hull_point_ratio",
    "n_components", "largest_component_ratio", "ground_gap_ratio", "z_range_cleaned",
]


# =====================================================================
# Feature extraction (all 35 features in one pass)
# =====================================================================

def _fast_percentile(z, q):
    n = len(z)
    if n == 0:
        return 0.0
    if n == 1:
        return float(z[0])
    k = int(q / 100.0 * (n - 1))
    k = min(max(k, 0), n - 1)
    return float(np.partition(z, k)[k])


def extract_all_features(xyz):
    """Extract all 35 features from a single cluster.
    Returns dict with all feature names as keys.
    """
    from scipy.spatial import KDTree, ConvexHull
    from sklearn.cluster import DBSCAN

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    num_points = len(xyz)

    # --- PCA-aligned bbox ---
    z_range = z.max() - z.min() if num_points > 1 else 0.0

    if num_points > 3:
        xy_centered = xyz[:, :2] - xyz[:, :2].mean(axis=0)
        cov_2d = xy_centered.T @ xy_centered / num_points
        eigvals_2d, eigvecs_2d = np.linalg.eigh(cov_2d)
        xy_rotated = xy_centered @ eigvecs_2d
        x_range = xy_rotated[:, 1].max() - xy_rotated[:, 1].min()
        y_range = xy_rotated[:, 0].max() - xy_rotated[:, 0].min()
    else:
        xy_centered = xyz[:, :2] - xyz[:, :2].mean(axis=0)
        eigvals_2d = np.array([1e-9, 1e-9])
        eigvecs_2d = np.eye(2)
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()

    xy_spread = np.sqrt(x_range**2 + y_range**2)
    xy_area = max(x_range * y_range, 1e-9)
    bbox_volume = max(xy_area * z_range, 1e-9)
    height_to_footprint = z_range / max(xy_spread, 1e-6)
    density = num_points / bbox_volume

    # --- Z-profile ---
    z_std = z.std()
    z_median = np.median(z)
    z_25 = _fast_percentile(z, 25)
    z_75 = _fast_percentile(z, 75)

    # --- 3D PCA ---
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

    # --- xy_aspect_ratio ---
    ev_2d = np.clip(eigvals_2d, 1e-9, None)
    xy_aspect_ratio = ev_2d[1] / ev_2d[0]

    # --- Layer fractions ---
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

    # --- Cross-section (PCA-aligned) ---
    if z_range > 0.1 and num_points > 20:
        slice_spreads = []
        for i in range(5):
            lo = z.min() + i * z_range / 5
            hi = z.min() + (i + 1) * z_range / 5
            mask = (z >= lo) & (z < hi)
            if mask.sum() > 2:
                sl_xy = xy_centered[mask] @ eigvecs_2d
                sx = sl_xy[:, 1].max() - sl_xy[:, 1].min()
                sy = sl_xy[:, 0].max() - sl_xy[:, 0].min()
                slice_spreads.append(np.sqrt(sx**2 + sy**2))
            else:
                slice_spreads.append(0.0)
        cross_section_var = np.std(slice_spreads)
        cross_section_mean = np.mean(slice_spreads)
    else:
        cross_section_var = 0.0
        cross_section_mean = 0.0

    # --- nn_dist_std ---
    if num_points > 6:
        tree = KDTree(xyz)
        dists, _ = tree.query(xyz, k=min(6, num_points))
        nn_dist_std = dists[:, 1:].mean(axis=1).std()
    else:
        tree = None
        nn_dist_std = 0.0

    # --- bottom_to_top_ratio ---
    bottom_to_top_ratio = layer_fractions[0] / max(layer_fractions[4], 1e-6)

    # --- hull_point_ratio ---
    if num_points > 10:
        try:
            hull = ConvexHull(xyz)
            hull_point_ratio = len(hull.vertices) / num_points
        except Exception:
            hull_point_ratio = 1.0
    else:
        hull_point_ratio = 1.0

    # --- n_components + largest_component_ratio ---
    if num_points > 10 and tree is not None:
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

    # --- ground_gap_ratio ---
    if z_range > 0.1:
        z_5 = _fast_percentile(z, 5)
        z_10 = _fast_percentile(z, 10)
        ground_gap_ratio = (z_10 - z_5) / z_range
    else:
        ground_gap_ratio = 0.0

    # --- z_range_cleaned ---
    if num_points > 10:
        z_range_cleaned = _fast_percentile(z, 95) - _fast_percentile(z, 5)
    else:
        z_range_cleaned = z_range

    return {
        "num_points": num_points,
        "x_range": x_range,
        "y_range": y_range,
        "z_range": z_range,
        "xy_spread": xy_spread,
        "xy_area": xy_area,
        "bbox_volume": bbox_volume,
        "height_to_footprint": height_to_footprint,
        "density": density,
        "z_std": z_std,
        "z_median": z_median,
        "z_25": z_25,
        "z_75": z_75,
        "ev_norm_0": ev_norm_0,
        "ev_norm_1": ev_norm_1,
        "ev_norm_2": ev_norm_2,
        "linearity": linearity,
        "planarity": planarity,
        "scattering": scattering,
        "xy_aspect_ratio": xy_aspect_ratio,
        "vertical_uniformity": vertical_uniformity,
        "cross_section_var": cross_section_var,
        "cross_section_mean": cross_section_mean,
        "layer_frac_0": layer_fractions[0],
        "layer_frac_1": layer_fractions[1],
        "layer_frac_2": layer_fractions[2],
        "layer_frac_3": layer_fractions[3],
        "layer_frac_4": layer_fractions[4],
        "nn_dist_std": nn_dist_std,
        "bottom_to_top_ratio": bottom_to_top_ratio,
        "hull_point_ratio": hull_point_ratio,
        "n_components": n_components,
        "largest_component_ratio": largest_component_ratio,
        "ground_gap_ratio": ground_gap_ratio,
        "z_range_cleaned": z_range_cleaned,
    }


def build_stats_dataframe(points_list, labels, paths=None):
    """Extract all 35 features for a dataset into a DataFrame."""
    rows = []
    n = len(points_list)
    for i, (pts, label) in enumerate(zip(points_list, labels)):
        xyz = pts[:, :3]
        feats = extract_all_features(xyz)
        feats["label"] = label
        if paths:
            feats["path"] = paths[i]
        rows.append(feats)
        if (i + 1) % 5000 == 0:
            print(f"  Extracted {i + 1}/{n}...")
    return pd.DataFrame(rows)


# =====================================================================
# Evaluation
# =====================================================================

def evaluate_feature_set(train_df, test_df, feature_cols, label,
                         n_estimators=100, max_depth=20):
    """Train RF on given features, evaluate on test, return macro F1."""
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["label"].values
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["label"].values

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    # CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1s = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        rf_cv = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=5, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        rf_cv.fit(X_train[train_idx], y_train[train_idx])
        y_val_pred = rf_cv.predict(X_train[val_idx])
        cv_f1s.append(f1_score(y_train[val_idx], y_val_pred, average="macro"))

    cm = confusion_matrix(y_test, y_pred, labels=list(range(4)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Confidence
    correct_conf, wrong_conf = [], []
    for i in range(len(y_test)):
        if y_pred[i] == y_test[i]:
            correct_conf.append(y_proba[i, y_pred[i]])
        else:
            wrong_conf.append(y_proba[i, y_pred[i]])

    print(f"\n{'=' * 70}")
    print(f"FEATURE SET: {label} ({len(feature_cols)} features)")
    print(f"{'=' * 70}")
    print(f"CV Macro F1: {np.mean(cv_f1s):.4f} ± {np.std(cv_f1s):.4f}")
    print(f"Test Macro F1: {macro_f1:.4f}")
    print()
    print(classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, digits=4,
    ))
    print("Confusion Matrix (normalized):")
    print(pd.DataFrame(
        np.round(cm_norm, 3), index=CLASS_NAMES, columns=CLASS_NAMES,
    ))
    print(f"\nConfidence: correct={np.mean(correct_conf):.3f}, "
          f"misclassified={np.mean(wrong_conf):.3f}")
    print(f"{'=' * 70}")

    return {
        "label": label,
        "n_features": len(feature_cols),
        "cv_f1": np.mean(cv_f1s),
        "cv_std": np.std(cv_f1s),
        "test_f1": macro_f1,
        "cm_norm": cm_norm,
        "correct_conf": np.mean(correct_conf),
        "wrong_conf": np.mean(wrong_conf),
    }


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Ablation study")
    parser.add_argument("--force-extract", action="store_true",
                        help="Force re-extraction even if CSVs exist")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=20)
    args = parser.parse_args()

    train_csv = "cache/ablation_train_stats.csv"
    test_csv = "cache/ablation_test_stats.csv"
    os.makedirs("cache", exist_ok=True)

    # --- Build CSVs ---
    if not os.path.exists(train_csv) or args.force_extract:
        print("Extracting training features...")
        train_points, train_labels, train_paths = load_dataset("data/train")
        train_df = build_stats_dataframe(train_points, train_labels, train_paths)
        train_df.to_csv(train_csv, index=False)
        print(f"Saved {train_csv} ({len(train_df)} rows)")
    else:
        print(f"Loading cached {train_csv}")
        train_df = pd.read_csv(train_csv)

    if not os.path.exists(test_csv) or args.force_extract:
        print("Extracting test features...")
        test_points, test_labels, test_paths = load_dataset("data/test")
        test_df = build_stats_dataframe(test_points, test_labels, test_paths)
        test_df.to_csv(test_csv, index=False)
        print(f"Saved {test_csv} ({len(test_df)} rows)")
    else:
        print(f"Loading cached {test_csv}")
        test_df = pd.read_csv(test_csv)

    print(f"\nTrain: {len(train_df)} samples, Test: {len(test_df)} samples")

    # --- Run ablation ---
    results = []
    for feat_list, name in [
        (FEATURES_19, "19-feat (compact)"),
        (FEATURES_23, "23-feat (extended)"),
        (FEATURES_35, "35-feat (full)"),
    ]:
        r = evaluate_feature_set(
            train_df, test_df, feat_list, name,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
        )
        results.append(r)

    # --- Summary table ---
    print(f"\n{'=' * 70}")
    print("ABLATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Feature Set':<25s} {'N':>4s} {'CV F1':>12s} {'Test F1':>10s} "
          f"{'Correct':>9s} {'Wrong':>9s}")
    print("-" * 70)
    for r in results:
        print(f"{r['label']:<25s} {r['n_features']:>4d} "
              f"{r['cv_f1']:.4f}±{r['cv_std']:.4f} {r['test_f1']:>10.4f} "
              f"{r['correct_conf']:>9.3f} {r['wrong_conf']:>9.3f}")
    print(f"{'=' * 70}")

    # --- Key confusion pairs ---
    print(f"\nKEY CONFUSION PAIRS (test set, normalized)")
    print(f"{'Feature Set':<25s} {'car→bg':>8s} {'bg→car':>8s} "
          f"{'ped→bike':>10s} {'bike→ped':>10s}")
    print("-" * 65)
    for r in results:
        cm = r["cm_norm"]
        print(f"{r['label']:<25s} "
              f"{cm[2, 0]:>8.1%} {cm[0, 2]:>8.1%} "
              f"{cm[3, 1]:>10.1%} {cm[1, 3]:>10.1%}")
    print(f"{'=' * 65}")
