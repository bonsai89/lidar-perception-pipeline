"""
Data loader for Seoul Robotics point cloud classification.
Loads binary point cloud files from class-structured directories.

Directory structure:
    data/{train,test}/{background,bicyclist,car,pedestrian}/*.bin
"""

import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Explicit mapping — not inferred from directory listing order,
# which can vary across OS/filesystem.
CLASS_MAP: Dict[str, int] = {
    "background": 0,
    "bicyclist": 1,
    "car": 2,
    "pedestrian": 3,
}
INV_CLASS_MAP: Dict[int, str] = {v: k for k, v in CLASS_MAP.items()}


def load_point_cloud(file_path: str, num_attributes: int = 3) -> Optional[np.ndarray]:
    """Load a single binary point cloud file.

    :param file_path: Path to .bin file.
    :param num_attributes: Floats per point (3 for x,y,z; 5 for x,y,z,intensity,ring).
    :return: (N, num_attributes) float32 array, or None if file is corrupt/empty.
    """
    try:
        raw = np.fromfile(file_path, dtype=np.float32)

        if raw.size == 0:
            logger.warning(f"Empty file, skipping: {file_path}")
            return None

        if raw.size % num_attributes != 0:
            logger.error(
                f"File size ({raw.size} floats) not divisible by {num_attributes}: {file_path}"
            )
            return None

        return raw.reshape(-1, num_attributes)

    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return None


def load_dataset(
    data_folder: str,
    num_attributes: int = 3,
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """Load all point cloud samples from a class-structured directory.

    :param data_folder: Root folder (e.g., 'data/train').
    :param num_attributes: Floats per point.
    :return: (points_list, labels, file_paths)
             points_list: list of (N_i, num_attributes) arrays
             labels: list of integer class labels
             file_paths: list of source file paths for traceability
    """
    points_list: List[np.ndarray] = []
    labels: List[int] = []
    file_paths: List[str] = []

    # Only look for known class directories — ignore stray files/folders
    for class_name, class_id in CLASS_MAP.items():
        class_dir = os.path.join(data_folder, class_name)

        if not os.path.isdir(class_dir):
            logger.warning(f"Class directory not found, skipping: {class_dir}")
            continue

        bin_files = sorted([
            f for f in os.listdir(class_dir)
            if f.endswith(".bin")
        ])

        loaded, skipped = 0, 0
        for fname in bin_files:
            fpath = os.path.join(class_dir, fname)
            pts = load_point_cloud(fpath, num_attributes)

            if pts is not None:
                points_list.append(pts)
                labels.append(class_id)
                file_paths.append(fpath)
                loaded += 1
            else:
                skipped += 1

        logger.info(
            f"[{class_name:>12s}] loaded: {loaded:5d}, skipped: {skipped:2d}"
        )

    logger.info(
        f"Total samples loaded: {len(points_list)} from {data_folder}"
    )

    return points_list, labels, file_paths


def get_dataset_summary(labels: List[int]) -> Dict[str, int]:
    """Print and return per-class sample counts.

    :param labels: List of integer labels.
    :return: Dict mapping class name to count.
    """
    summary = {}
    for class_name, class_id in CLASS_MAP.items():
        count = labels.count(class_id)
        summary[class_name] = count

    logger.info("Dataset distribution:")
    for name, count in summary.items():
        logger.info(f"  {name:>12s}: {count:5d}")

    return summary


def get_class_metrics(points_list, labels):
    """Per-class physical statistics from labeled data.

    Prints: point count, bbox dims, z_range, xy_spread, density.
    Helps explain confusion patterns (ped↔bike, car→bg).
    """
    from collections import defaultdict

    stats = defaultdict(lambda: {
        "n_points": [], "z_range": [], "xy_spread": [],
        "x_range": [], "y_range": [],
        "z_max": [], "z_mean": [], "density": [],
    })

    for pts, label in zip(points_list, labels):
        name = INV_CLASS_MAP[label]
        xyz = pts[:, :3]
        mn, mx = xyz.min(axis=0), xyz.max(axis=0)
        ranges = mx - mn
        vol = max(ranges[0] * ranges[1] * ranges[2], 1e-6)

        stats[name]["n_points"].append(len(xyz))
        stats[name]["x_range"].append(ranges[0])
        stats[name]["y_range"].append(ranges[1])
        stats[name]["z_range"].append(ranges[2])
        stats[name]["xy_spread"].append(np.sqrt(ranges[0] ** 2 + ranges[1] ** 2))
        stats[name]["z_max"].append(mx[2])
        stats[name]["z_mean"].append(xyz[:, 2].mean())
        stats[name]["density"].append(len(xyz) / vol)

    print(f"\n{'Class':>12s} {'Count':>6s} | {'n_pts':>14s} | "
          f"{'z_range':>14s} | {'xy_spread':>14s} | {'density':>14s}")
    print("-" * 90)

    for name in CLASS_MAP:
        s = stats[name]
        n = len(s["n_points"])

        def fmt(vals):
            a = np.array(vals)
            return f"{a.mean():7.1f} ±{a.std():6.1f}"

        print(f"{name:>12s} {n:6d} | {fmt(s['n_points'])} | "
              f"{fmt(s['z_range'])} | {fmt(s['xy_spread'])} | {fmt(s['density'])}")

    # Overlap analysis — the key confusion pairs
    for name in CLASS_MAP:
        s = stats[name]
        pts = np.array(s["n_points"])
        print(f"  {name:>12s}: median={np.median(pts):.0f}, "
              f"p10={np.percentile(pts, 10):.0f}, p90={np.percentile(pts, 90):.0f}")

    print(f"\n--- Overlap analysis (why confusion happens) ---")
    for a, b in [("pedestrian", "bicyclist"), ("car", "background")]:
        print(f"\n  {a} vs {b}:")
        for feat in ["n_points", "z_range", "xy_spread", "density"]:
            va, vb = np.array(stats[a][feat]), np.array(stats[b][feat])
            # Overlap = how much the ranges intersect
            lo = max(va.min(), vb.min())
            hi = min(va.max(), vb.max())
            if hi > lo:
                pct_a = ((va >= lo) & (va <= hi)).mean() * 100
                pct_b = ((vb >= lo) & (vb <= hi)).mean() * 100
                print(f"    {feat:>12s}: {a}={va.mean():.2f}±{va.std():.2f}  "
                      f"{b}={vb.mean():.2f}±{vb.std():.2f}  "
                      f"overlap: {pct_a:.0f}%/{pct_b:.0f}%")
            else:
                print(f"    {feat:>12s}: NO OVERLAP (clean separation)")

    return dict(stats)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Quick sanity check — load train set and print summary
    train_points, train_labels, train_paths = load_dataset("data/train")
    get_dataset_summary(train_labels)
    get_class_metrics(train_points, train_labels)

    test_points, test_labels, test_paths = load_dataset("data/test")
    get_dataset_summary(test_labels)
    get_class_metrics(test_points, test_labels)

    # Spot check: print shape of first sample per class
    for class_name, class_id in CLASS_MAP.items():
        idx = train_labels.index(class_id)
        pts = train_points[idx]
        logger.info(
            f"Sample {class_name}: {pts.shape[0]} points, "
            f"x=[{pts[:,0].min():.2f}, {pts[:,0].max():.2f}], "
            f"y=[{pts[:,1].min():.2f}, {pts[:,1].max():.2f}], "
            f"z=[{pts[:,2].min():.2f}, {pts[:,2].max():.2f}]"
        )
