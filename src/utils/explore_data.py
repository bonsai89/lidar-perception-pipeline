"""
Explore the optional challenge data.
10 sequential scenes, each with x, y, z, intensity, ring.

Run from project root:
  python optional_challenge/explore_optional.py
"""

import logging
import os
import sys

import numpy as np

# Add project root to path so we can import shared modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "optional_challenge_data")


def explore_scenes(data_dir: str = DATA_DIR) -> None:
    """Load and print basic stats for each scene."""

    data_dir = os.path.abspath(data_dir)
    files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith(".bin")
    ])

    print(f"Data directory: {data_dir}")
    print(f"Found {len(files)} scene files\n")

    for fname in files:
        fpath = os.path.join(data_dir, fname)
        points = np.fromfile(fpath, dtype=np.float32).reshape(-1, 5)

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        intensity = points[:, 3]
        ring = points[:, 4]

        print(f"{'=' * 60}")
        print(f"Scene: {fname}")
        print(f"{'=' * 60}")
        print(f"  Points: {len(points)}")
        print(f"  X range: [{x.min():.2f}, {x.max():.2f}] (span: {x.max()-x.min():.2f})")
        print(f"  Y range: [{y.min():.2f}, {y.max():.2f}] (span: {y.max()-y.min():.2f})")
        print(f"  Z range: [{z.min():.2f}, {z.max():.2f}] (span: {z.max()-z.min():.2f})")
        print(f"  Intensity: [{intensity.min():.1f}, {intensity.max():.1f}], mean={intensity.mean():.1f}")
        print(f"  Ring: [{int(ring.min())}, {int(ring.max())}], unique={len(np.unique(ring))}")
        print(f"  Point density: {len(points) / ((x.max()-x.min()) * (y.max()-y.min())):.1f} pts/m²")
        print()

    # Cross-scene comparison
    print(f"\n{'=' * 60}")
    print("CROSS-SCENE SUMMARY")
    print(f"{'=' * 60}")

    all_counts = []
    all_z_min = []
    all_z_max = []
    for fname in files:
        fpath = os.path.join(data_dir, fname)
        pts = np.fromfile(fpath, dtype=np.float32).reshape(-1, 5)
        all_counts.append(len(pts))
        all_z_min.append(pts[:, 2].min())
        all_z_max.append(pts[:, 2].max())

    print(f"  Point counts: min={min(all_counts)}, max={max(all_counts)}, mean={np.mean(all_counts):.0f}")
    print(f"  Z min across scenes: [{min(all_z_min):.2f}, {max(all_z_min):.2f}]")
    print(f"  Z max across scenes: [{min(all_z_max):.2f}, {max(all_z_max):.2f}]")
    print(f"  Consistent ground plane: {'Yes' if max(all_z_min) - min(all_z_min) < 0.5 else 'No'}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    explore_scenes()
