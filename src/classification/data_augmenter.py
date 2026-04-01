"""
Smart point cloud augmenter for classification training.

Only augments "full" objects — skips sparse/partial scans.
Class-specific cuts based on object geometry:
  - Car: vertical halves, 1/3 and 2/3 front/back/left/right
  - Pedestrian: above/below hip, vertical half
  - Bicyclist: vertical half, horizontal 1/3 and 2/3

Usage:
    from data_augmenter import augment_dataset
    aug_points, aug_labels = augment_dataset(points_list, labels)

    # Or with custom config:
    aug_points, aug_labels = augment_dataset(
        points_list, labels, min_points=100, balance=True
    )
"""

import logging
from typing import List, Tuple, Dict
import os
import numpy as np

from data_loader import CLASS_MAP, INV_CLASS_MAP

logger = logging.getLogger(__name__)


# =====================================================================
# Fullness checks — is this object complete enough to augment?
# =====================================================================

def _is_full_object(xyz: np.ndarray, class_name: str) -> bool:
    """Check if point cloud represents a complete object worth augmenting.

    Criteria per class based on training data statistics:
    - Enough points (above class median)
    - Sufficient vertical coverage (not just roof or feet)
    - Sufficient horizontal coverage (not just one side)
    - Reasonable density distribution across vertical bins

    :param xyz: (N, 3) point coordinates.
    :param class_name: One of 'car', 'pedestrian', 'bicyclist', 'background'.
    :return: True if object appears complete.
    """
    n = len(xyz)
    z_range = xyz[:, 2].max() - xyz[:, 2].min()
    x_range = xyz[:, 0].max() - xyz[:, 0].min()
    y_range = xyz[:, 1].max() - xyz[:, 1].min()
    xy_spread = np.sqrt(x_range**2 + y_range**2)

    # Vertical bin occupancy — must have points across full height
    if z_range > 0.1:
        z_norm = (xyz[:, 2] - xyz[:, 2].min()) / z_range
        bins = [((z_norm >= i/4) & (z_norm < (i+1)/4)).sum() for i in range(4)]
        occupied_bins = sum(1 for b in bins if b > n * 0.05)  # >5% of points
    else:
        occupied_bins = 0

    if class_name == "car":
        # Full car: >150 pts (median=148), z>0.8m, xy>2.5m, 3+ vertical bins
        return n >= 150 and z_range >= 0.8 and xy_spread >= 2.5 and occupied_bins >= 3

    elif class_name == "pedestrian":
        # Full ped: >90 pts (median=93), z>1.0m, xy<2.0m, 3+ vertical bins
        return n >= 90 and z_range >= 1.0 and xy_spread < 2.0 and occupied_bins >= 3

    elif class_name == "bicyclist":
        # Full bike: >70 pts (median=72), z>1.0m, 3+ vertical bins
        return n >= 70 and z_range >= 1.0 and occupied_bins >= 3

    return False


# =====================================================================
# Cut functions — class-specific slicing strategies
# =====================================================================

def _cut(xyz: np.ndarray, pts: np.ndarray, axis: int,
         lo_frac: float, hi_frac: float, min_points: int = 30) -> np.ndarray:
    """Generic axis-aligned cut.

    :param xyz: (N, 3) coordinates.
    :param pts: (N, K) full point array (preserves all columns).
    :param axis: 0=x, 1=y, 2=z.
    :param lo_frac: Lower fraction [0, 1).
    :param hi_frac: Upper fraction (0, 1].
    :param min_points: Minimum points for valid cut.
    :return: Sliced points array, or None if too few points.
    """
    v = xyz[:, axis]
    v_min, v_max = v.min(), v.max()
    v_range = v_max - v_min
    if v_range < 0.1:
        return None

    lo = v_min + lo_frac * v_range
    hi = v_min + hi_frac * v_range
    mask = (v >= lo) & (v < hi)

    if mask.sum() < min_points:
        return None
    return pts[mask]

def _get_height_axis(xyz):
    """Find the 'up' direction of an object using PCA.
    Smallest eigenvector = shortest axis = height for cars."""
    centered = xyz - xyz.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns ascending — [0] is smallest = height axis
    # Ensure it points "up" (positive z component)
    height_vec = eigvecs[:, 0]
    if height_vec[2] < 0:
        height_vec = -height_vec
    return height_vec

def _cut_along_axis(xyz, pts, axis_vec, lo_frac, hi_frac, min_points=30):
    """Cut along arbitrary axis direction, not just x/y/z."""
    projections = (xyz - xyz.mean(axis=0)) @ axis_vec
    p_min, p_max = projections.min(), projections.max()
    p_range = p_max - p_min
    if p_range < 0.1:
        return None
    lo = p_min + lo_frac * p_range
    hi = p_min + hi_frac * p_range
    mask = (projections >= lo) & (projections < hi)
    if mask.sum() < min_points:
        return None
    return pts[mask]

def _augment_car(xyz: np.ndarray, pts: np.ndarray) -> List[np.ndarray]:
    """Car-specific augmentation cuts.

    Vertical (x/y) cuts simulate partial visibility from sensor angle:
    - Left/right halves
    - Front/back halves
    - 1/3 slices (front only, back only, left only, right only)
    - 2/3 slices (front 2/3, back 2/3, left 2/3, right 2/3)
    """
    augmented = []

    # Determine longer horizontal axis for front/back vs left/right
    x_range = xyz[:, 0].max() - xyz[:, 0].min()
    y_range = xyz[:, 1].max() - xyz[:, 1].min()
    long_axis = 0 if x_range >= y_range else 1
    short_axis = 1 if long_axis == 0 else 0

    # Halves along long axis (front/back)
    for lo, hi in [(0.0, 0.5), (0.5, 1.0)]:
        result = _cut(xyz, pts, long_axis, lo, hi)
        if result is not None:
            augmented.append(result)

    # Halves along short axis (left/right)
    for lo, hi in [(0.0, 0.5), (0.5, 1.0)]:
        result = _cut(xyz, pts, short_axis, lo, hi)
        if result is not None:
            augmented.append(result)

    # 1/3 slices along long axis
    for lo, hi in [(0.0, 0.33), (0.67, 1.0)]:
        result = _cut(xyz, pts, long_axis, lo, hi)
        if result is not None:
            augmented.append(result)

    # 2/3 slices along long axis
    for lo, hi in [(0.0, 0.67), (0.33, 1.0)]:
        result = _cut(xyz, pts, long_axis, lo, hi)
        if result is not None:
            augmented.append(result)

    # 1/3 slices along short axis
    for lo, hi in [(0.0, 0.33), (0.67, 1.0)]:
        result = _cut(xyz, pts, short_axis, lo, hi)
        if result is not None:
            augmented.append(result)

    # 2/3 slices along short axis
    for lo, hi in [(0.0, 0.67), (0.33, 1.0)]:
        result = _cut(xyz, pts, short_axis, lo, hi)
        if result is not None:
            augmented.append(result)

    # Horizontal cuts along PCA height axis (handles LiDAR tilt)
    height_axis = _get_height_axis(xyz)

    # Roof only (top 1/3) — simulates seeing only car top from high sensor
    result = _cut_along_axis(xyz, pts, height_axis, 0.67, 1.0)
    if result is not None:
        augmented.append(result)

    # Body without roof (bottom 2/3) — simulates roof occlusion
    result = _cut_along_axis(xyz, pts, height_axis, 0.0, 0.67)
    if result is not None:
        augmented.append(result)

    # Top half of car (top 1/2)
    result = _cut_along_axis(xyz, pts, height_axis, 0.5, 1.0)
    if result is not None:
        augmented.append(result)

    # Bottom half of car (bottom 1/2)
    result = _cut_along_axis(xyz, pts, height_axis, 0.0, 0.5)
    if result is not None:
        augmented.append(result)

    return augmented


def _augment_pedestrian(xyz, pts):
    augmented = []
    # --- Axis-aligned cuts (arbitrary sensor angles) ---
    # Raw z cuts
    for lo, hi in [(0.33, 1.0), (0.0, 0.67)]:
        result = _cut(xyz, pts, 2, lo, hi)
        if result is not None:
            augmented.append(result)

    return augmented


def _augment_bicyclist(xyz, pts):

    augmented = []

    x_range = xyz[:,0].max() - xyz[:,0].min()
    y_range = xyz[:,1].max() - xyz[:,1].min()

    long_axis = 0 if x_range >= y_range else 1

    for lo, hi in [(0.0,0.67),(0.33,1.0)]:
        result = _cut(xyz, pts, long_axis, lo, hi)
        if result is not None:
            augmented.append(result)

    # vertical cuts
    for lo, hi in [(0.3,1.0),(0.0,0.7)]:
        result = _cut(xyz, pts, 2, lo, hi)
        if result is not None:
            augmented.append(result)

    return augmented

# =====================================================================
# Main augmentation interface
# =====================================================================

_CLASS_AUGMENTERS = {
    "car": _augment_car,
    "pedestrian": _augment_pedestrian,
    "bicyclist": _augment_bicyclist,
}


def augment_dataset(
    points_list: List[np.ndarray],
    labels: List[int],
    min_points: int = 30,
    balance: bool = True,
    max_aug_per_class: int = None,
    seed: int = 42,
    save_dir: str = None,
) -> Tuple[List[np.ndarray], List[int]]:
    """Augment training dataset with class-specific partial views.

    Only augments objects that pass fullness checks.
    Does not augment background.

    :param points_list: List of (N, K) point cloud arrays.
    :param labels: List of integer class labels.
    :param min_points: Minimum points for a valid augmented sample.
    :param balance: If True, cap augmented samples per class to avoid
                    overwhelming the original distribution.
    :param max_aug_per_class: Override max augmented samples per class.
                              If None, defaults to original class count.
    :param seed: Random seed for reproducible subsampling.
    :param save_dir: save the augmented pointclouds to storage.
    :return: (augmented_points_list, augmented_labels) — originals + augmented.
    """
    rng = np.random.RandomState(seed)

    # Count originals per class
    original_counts = {}
    for label in labels:
        name = INV_CLASS_MAP[label]
        original_counts[name] = original_counts.get(name, 0) + 1

    # Collect augmented samples per class
    aug_by_class: Dict[str, List[Tuple[np.ndarray, int]]] = {
        "car": [], "pedestrian": [], "bicyclist": [],
    }

    n_full = 0
    n_partial = 0
    n_augmented = 0

    for parent_idx, (pts, label) in enumerate(zip(points_list, labels)):
        class_name = INV_CLASS_MAP[label]
        if class_name not in _CLASS_AUGMENTERS:
            continue

        xyz = pts[:, :3]

        if not _is_full_object(xyz, class_name):
            n_partial += 1
            continue

        n_full += 1
        augmenter = _CLASS_AUGMENTERS[class_name]
        cuts = augmenter(xyz, pts)

        for cut_pts in cuts:
            if len(cut_pts) >= min_points:
                aug_by_class[class_name].append((cut_pts, label, parent_idx))
                n_augmented += 1

    # Balance: cap per class
    if balance:
        for class_name in aug_by_class:
            cap = max_aug_per_class or original_counts.get(class_name, 1000)
            samples = aug_by_class[class_name]
            if len(samples) > cap:
                idx = rng.choice(len(samples), cap, replace=False)
                aug_by_class[class_name] = [samples[i] for i in idx]

    # Merge: originals + augmented, with group IDs for GroupKFold
    all_points = list(points_list)
    all_labels = list(labels)
    all_groups = list(range(len(points_list)))  # each original = unique group

    for class_name, samples in aug_by_class.items():
        for cut_pts, label, group_id in samples:
            all_points.append(cut_pts)
            all_labels.append(label)
            all_groups.append(group_id)

    # Report
    print(f"\n{'='*60}")
    print("AUGMENTATION SUMMARY")
    print(f"{'='*60}")
    print(f"Full objects found: {n_full}")
    print(f"Partial/sparse skipped: {n_partial}")
    print(f"Total cuts generated: {n_augmented}")
    for class_name in ["car", "pedestrian", "bicyclist"]:
        n_orig = original_counts.get(class_name, 0)
        n_aug = len(aug_by_class.get(class_name, []))
        print(f"  {class_name:>12s}: {n_orig} original + {n_aug} augmented = {n_orig + n_aug}")
    print(f"Total dataset: {len(points_list)} → {len(all_points)}")
    print(f"{'='*60}\n")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for i, (pts, label) in enumerate(zip(all_points[len(points_list):], all_labels[len(labels):])):
            cls_name = INV_CLASS_MAP[label]
            cls_dir = os.path.join(save_dir, cls_name)
            os.makedirs(cls_dir, exist_ok=True)
            pts.astype(np.float32).tofile(os.path.join(cls_dir, f"aug_{i:06d}.bin"))

    return all_points, all_labels, all_groups


if __name__ == "__main__":
    """Quick test — load training data and augment."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    from data_loader import load_dataset, get_dataset_summary

    print("Loading training data...")
    points, labels, paths = load_dataset("data/train")
    get_dataset_summary(labels)

    print("\nAugmenting...")
    aug_points, aug_labels, aug_groups = augment_dataset(points, labels, save_dir="data/train_augmented")
    get_dataset_summary(aug_labels)

    # Spot check augmented samples
    print("\nSpot check — augmented sample shapes:")
    n_orig = len(points)
    for i in range(min(5, len(aug_points) - n_orig)):
        idx = n_orig + i
        cls = INV_CLASS_MAP[aug_labels[idx]]
        print(f"  [{idx}] {cls}: {aug_points[idx].shape}")