"""
Ground removal for infrastructure-mounted LiDAR.

Approach: Calibrate once, then fast per-frame processing.

1. CALIBRATION (first frame only):
   - RANSAC finds ground plane normal (handles tilted sensor)
   - Compute rotation matrix to align ground normal with z-axis
   - Compute ground height in rotated frame

2. PER FRAME (fast, O(N)):
   - Rotate all points using calibration matrix
   - Simple z-threshold separates ground from non-ground
   - No RANSAC, no refinement, no wall leakage

Why this works:
   Infrastructure sensor is fixed — same position, same tilt for all frames.
   Calibration transfers across frames. After rotation, ground is horizontal
   and z-threshold cleanly separates it from walls (which are now vertical).

Performance: ~5-10ms per frame vs 6-7s with per-frame RANSAC.

Run from project root:
  python optional_challenge/ground_removal.py
"""

import logging
import os
import sys
import time
from typing import Tuple, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "optional_challenge_data")


def ransac_ground_plane(
    points: np.ndarray,
    max_iterations: int = 1000,
    distance_threshold: float = 0.3,
) -> Tuple[np.ndarray, float]:
    """Find ground plane using RANSAC. Used only for calibration.

    :param points: (N, 3+) array with at least x, y, z.
    :param max_iterations: RANSAC iterations.
    :param distance_threshold: Max distance from plane to be ground.
    :return: (normal, d) where plane equation is normal.dot(p) + d = 0
    """
    xyz = points[:, :3]
    n_points = len(xyz)

    best_n_inliers = 0
    best_normal = None
    best_d = None

    for _ in range(max_iterations):
        idx = np.random.choice(n_points, 3, replace=False)
        p1, p2, p3 = xyz[idx[0]], xyz[idx[1]], xyz[idx[2]]

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)

        if norm_len < 1e-10:
            continue

        normal = normal / norm_len
        d = -np.dot(normal, p1)

        # Ground plane should be roughly horizontal
        # normal should have dominant z-component (>0.7)
        if abs(normal[2]) < 0.7:
            continue

        # Ensure normal points upward (positive z)
        if normal[2] < 0:
            normal = -normal
            d = -d

        distances = np.abs(xyz @ normal + d)
        n_inliers = (distances < distance_threshold).sum()

        if n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            best_normal = normal
            best_d = d

    if best_normal is None:
        logger.warning("RANSAC failed, defaulting to z-up plane")
        best_normal = np.array([0.0, 0.0, 1.0])
        best_d = -np.percentile(xyz[:, 2], 20)

    logger.info(
        f"RANSAC plane: normal={best_normal.round(4)}, d={best_d:.4f}, "
        f"inliers={best_n_inliers}/{n_points} ({100*best_n_inliers/n_points:.1f}%)"
    )

    return best_normal, best_d


def compute_rotation_matrix(source_normal: np.ndarray, target_normal: np.ndarray) -> np.ndarray:
    """Compute rotation matrix that aligns source_normal to target_normal.

    Uses Rodrigues' rotation formula.

    :param source_normal: (3,) unit vector — ground plane normal (tilted).
    :param target_normal: (3,) unit vector — desired alignment, typically [0, 0, 1].
    :return: (3, 3) rotation matrix R such that R @ source_normal ≈ target_normal.
    """
    source_normal = source_normal / np.linalg.norm(source_normal)
    target_normal = target_normal / np.linalg.norm(target_normal)

    v = np.cross(source_normal, target_normal)
    c = np.dot(source_normal, target_normal)

    # If already aligned (or anti-aligned)
    if np.linalg.norm(v) < 1e-10:
        if c > 0:
            return np.eye(3)
        else:
            # 180 degree rotation — pick any perpendicular axis
            perp = np.array([1, 0, 0]) if abs(source_normal[0]) < 0.9 else np.array([0, 1, 0])
            axis = np.cross(source_normal, perp)
            axis = axis / np.linalg.norm(axis)
            return 2 * np.outer(axis, axis) - np.eye(3)

    # Skew-symmetric cross-product matrix
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])

    R = np.eye(3) + vx + vx @ vx / (1 + c)

    return R


class GroundCalibration:
    """Calibration data for ground removal.

    Computed once from the first frame, reused for all subsequent frames.
    """

    def __init__(
        self,
        rotation_matrix: np.ndarray,
        ground_height: float,
        original_normal: np.ndarray,
        original_d: float,
    ):
        self.rotation_matrix = rotation_matrix      # (3, 3) rotation to align ground to z
        self.ground_height = ground_height           # ground z-value in rotated frame
        self.original_normal = original_normal       # plane normal before rotation
        self.original_d = original_d                 # plane d before rotation

    def __repr__(self):
        return (
            f"GroundCalibration(ground_z={self.ground_height:.3f}, "
            f"normal={self.original_normal.round(4)})"
        )


def calibrate_ground(
    points: np.ndarray,
    max_iterations: int = 1000,
    distance_threshold: float = 0.3,
    near_range: float = 10.0,
) -> GroundCalibration:
    """Run calibration on one frame to determine ground plane.

    Strategy:
    1. Use only points within near_range of sensor origin for RANSAC.
       Near the sensor, ground dominates — dense concentric scan lines
       on the road directly below. No car roofs, no distant buildings.
    2. RANSAC on nearby points gives a clean ground normal (sensor tilt).
    3. Compute rotation matrix to make ground horizontal.
    4. Set ground height = low percentile of rotated z across FULL scene.
       This finds the actual road surface level, not whatever RANSAC locked onto.

    Call this ONCE on the first frame. Reuse the result for all frames.

    :param points: (N, 5) first frame's points.
    :param max_iterations: RANSAC iterations.
    :param distance_threshold: RANSAC distance threshold.
    :param near_range: Only use points within this XY distance for RANSAC (meters).
    :return: GroundCalibration object.
    """
    t0 = time.time()

    xyz = points[:, :3]

    # Filter extreme z-outliers
    valid = (xyz[:, 2] > -10) & (xyz[:, 2] < 30)
    valid_xyz = xyz[valid]

    # Step 1: Select only nearby points for RANSAC
    # Near the sensor, ground is dominant — clean plane fitting
    xy_dist = np.sqrt(valid_xyz[:, 0]**2 + valid_xyz[:, 1]**2)
    near_mask = xy_dist < near_range
    near_xyz = valid_xyz[near_mask]

    logger.info(
        f"Calibration: using {near_mask.sum()} nearby points "
        f"(within {near_range}m) out of {len(valid_xyz)} total"
    )

    # Step 2: RANSAC on nearby points — gets clean ground normal
    normal, d = ransac_ground_plane(near_xyz, max_iterations, distance_threshold)

    # Step 3: Compute rotation to align normal with z-up
    z_up = np.array([0.0, 0.0, 1.0])
    R = compute_rotation_matrix(normal, z_up)

    # Step 4: Find actual ground height in rotated frame
    # Rotate ALL valid points, then find the lowest flat region
    rotated_z = valid_xyz @ R[2, :]  # only need z component

    # Ground is the lowest horizontal surface.
    # Use a low percentile — not min (that's noise), not median (that's mid-scene).
    # 5th percentile captures the road surface level.
    ground_height = np.percentile(rotated_z, 5)

    # Verify: check that a reasonable fraction of points are near this height
    near_ground = np.abs(rotated_z - ground_height) < 0.2
    ground_fraction = near_ground.sum() / len(rotated_z)
    logger.info(
        f"Ground height (rotated): {ground_height:.3f}, "
        f"points within 0.2m: {near_ground.sum()} ({100*ground_fraction:.1f}%)"
    )

    elapsed = time.time() - t0
    logger.info(
        f"Calibration done in {elapsed:.2f}s: "
        f"normal={normal.round(4)}, ground_z={ground_height:.3f}"
    )

    return GroundCalibration(
        rotation_matrix=R,
        ground_height=ground_height,
        original_normal=normal,
        original_d=d,
    )


def remove_ground(
    points: np.ndarray,
    calibration: GroundCalibration,
    cell_size: float = 0.5,
    ground_threshold_above: float = 0.15,
    min_cell_points: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ground removal: rotation + polar grid with calibration-validated local height.

    Hybrid approach:
    1. Rotate to make ground horizontal
    2. Polar grid cells find local ground height (5th percentile of z)
    3. VALIDATE: only trust cells where local height is within distance-scaled
       deviation of calibration ground height. Cells that deviate = occluded
       by object, fall back to calibration height.
    4. Threshold above local (or fallback) ground height

    Why polar grid:
    - Matches sensor's radial scan pattern — natural resolution scaling
    - Fine near sensor (dense data), coarse at range (sparse data)
    - Reduces occlusion corruption vs Cartesian grid

    :param points: (N, 5) array with x, y, z, intensity, ring.
    :param calibration: Pre-computed GroundCalibration.
    :param cell_size: Grid cell size (meters). Not used — polar bins instead.
    :param ground_threshold_above: Max height above local ground (meters).
    :param min_cell_points: Min points per cell to estimate ground.
    :return: (ground_points, non_ground_points, ground_mask)
    """
    t0 = time.time()

    xyz = points[:, :3]
    R = calibration.rotation_matrix
    cal_ground = calibration.ground_height

    # Step 1: Rotate — only need z component
    rotated_z = xyz @ R[2, :]

    # Also need rotated xy for grid binning
    rotated_x = xyz @ R[0, :]
    rotated_y = xyz @ R[1, :]

    # Step 2: Polar grid binning
    r = np.sqrt(rotated_x ** 2 + rotated_y ** 2)
    theta = np.arctan2(rotated_y, rotated_x)

    r_bin_size = 5.0
    theta_bin_size = np.radians(5)

    r_bin = (r / r_bin_size).astype(np.int32)
    theta_bin = ((theta + np.pi) / theta_bin_size).astype(np.int32)
    n_r = r_bin.max() + 1
    cell_id = r_bin + theta_bin * n_r

    # Step 3: Find local ground per cell + validate against calibration
    max_cell_id = cell_id.max() + 1
    cell_ground = np.full(max_cell_id, cal_ground, dtype=np.float64)  # default = calibration

    # Sort for fast grouped access
    sort_idx = np.argsort(cell_id)
    sorted_cells = cell_id[sort_idx]
    sorted_z = rotated_z[sort_idx]

    unique_cells, cell_starts = np.unique(sorted_cells, return_index=True)
    cell_ends = np.append(cell_starts[1:], len(sorted_cells))

    for i, cid in enumerate(unique_cells):
        n_pts = cell_ends[i] - cell_starts[i]
        if n_pts >= min_cell_points:
            cell_z = sorted_z[cell_starts[i]:cell_ends[i]]
            k = max(1, int(len(cell_z) * 0.05))
            local_height = np.partition(cell_z, k)[k]

            # VALIDATION: trust this cell only if its ground height
            # is close to calibration. Otherwise it's an object.
            # Distance-scaled deviation: tight near sensor, loose at far range
            r_center = (cid % n_r + 0.5) * r_bin_size
            allowed_deviation = min(0.5 + r_center * 0.08, 2.0)
            if abs(local_height - cal_ground) <= allowed_deviation:
                cell_ground[cid] = local_height
            # else: keeps default = cal_ground (rejects object)

    # Step 4: Assign local ground height to each point
    local_ground = cell_ground[cell_id]

    # Step 5: Ground = within threshold above local ground
    # Below margin is small (0.1m noise tolerance)
    # Above margin is tight (0.15m avoids bumpers)
    ground_mask = (
        (rotated_z >= local_ground - 0.1) &
        (rotated_z <= local_ground + ground_threshold_above)
    )

    ground_points = points[ground_mask]
    non_ground_points = points[~ground_mask]

    elapsed = time.time() - t0
    logger.info(
        f"Ground removal: {ground_mask.sum()} ground ({100*ground_mask.sum()/len(points):.1f}%), "
        f"{(~ground_mask).sum()} non-ground, {elapsed*1000:.1f}ms"
    )

    return ground_points, non_ground_points, ground_mask


# =====================================================================
# Visualization
# =====================================================================


def visualize_ground_removal(
    data_dir: str,
    files: list,
    calibration: GroundCalibration,
) -> None:
    """Interactive VisPy visualizer for ground removal results.

    Pre-computes all frames (fast with calibration), then displays.
    Green = ground, height-colored = non-ground.
    Press N/B to navigate, G to toggle ground/non-ground/both.
    """
    import vispy
    from vispy.scene import visuals
    from vispy.scene.cameras import TurntableCamera
    from vispy.scene import SceneCanvas

    # --- Pre-compute all frames (should be fast now) ---
    print("Pre-computing ground removal for all frames...")
    cached_frames = []
    for i, fname in enumerate(files):
        fpath = os.path.join(data_dir, fname)
        pts = np.fromfile(fpath, dtype=np.float32).reshape(-1, 5)
        ground_pts, non_ground_pts, ground_mask = remove_ground(pts, calibration)

        xyz = pts[:, :3]
        n = len(pts)
        colors = np.zeros((n, 4), dtype=np.float32)

        # Non-ground: color by height
        ng_z = xyz[~ground_mask, 2]
        if len(ng_z) > 0:
            z_min_ng = max(np.min(ng_z), -3)
            z_max_ng = min(np.max(ng_z), 20)
            z_norm = np.clip(ng_z, z_min_ng, z_max_ng)
            z_norm = (z_norm - z_min_ng) / max(z_max_ng - z_min_ng, 1e-6) * 0.8 + 0.1
            colors[~ground_mask, 0] = z_norm
            colors[~ground_mask, 1] = 1 - z_norm
            colors[~ground_mask, 2] = 1 - z_norm * z_norm
            colors[~ground_mask, 3] = 0.8

        # Ground: solid green
        colors[ground_mask, 0] = 0.1
        colors[ground_mask, 1] = 0.8
        colors[ground_mask, 2] = 0.1
        colors[ground_mask, 3] = 0.6

        cached_frames.append({
            "fname": fname,
            "xyz": xyz,
            "colors": colors,
            "ground_mask": ground_mask,
            "n_ground": ground_mask.sum(),
            "n_total": n,
        })
        print(f"  [{i+1}/{len(files)}] {fname}: done")

    print("Launching viewer...")

    # --- Viewer ---
    frame_index = [0]
    show_mode = [0]
    mode_names = ["both", "ground only", "non-ground only"]

    canvas = SceneCanvas(keys="interactive", show=True, size=(1600, 900))
    grid = canvas.central_widget.add_grid()
    view = vispy.scene.widgets.ViewBox(
        parent=canvas.scene,
        camera=TurntableCamera(distance=80.0),
    )
    grid.add_widget(view)

    scatter = visuals.Markers()
    view.add(scatter)
    visuals.XYZAxis(parent=view.scene)

    info_text = visuals.Text(
        text="", color="white", font_size=12,
        anchor_x="left", anchor_y="top",
        parent=canvas.scene,
    )
    info_text.pos = (10, 20)

    def update_display():
        frame = cached_frames[frame_index[0]]
        mode = show_mode[0]

        if mode == 0:
            show_xyz = frame["xyz"]
            show_colors = frame["colors"]
        elif mode == 1:
            show_xyz = frame["xyz"][frame["ground_mask"]]
            show_colors = frame["colors"][frame["ground_mask"]]
        else:
            show_xyz = frame["xyz"][~frame["ground_mask"]]
            show_colors = frame["colors"][~frame["ground_mask"]]

        scatter.set_data(show_xyz, face_color=show_colors, edge_color=show_colors, size=1.0)

        canvas.title = (
            f"Frame {frame_index[0] + 1}/{len(files)} — {mode_names[mode]} — "
            f"[N]ext [B]ack [G]toggle"
        )
        info_text.text = (
            f"File: {frame['fname']}\n"
            f"Total: {frame['n_total']}, Ground: {frame['n_ground']}, "
            f"Non-ground: {frame['n_total'] - frame['n_ground']}\n"
            f"Showing: {mode_names[mode]} ({len(show_xyz)} points)\n"
            f"[N] next  [B] back  [G] toggle ground/non-ground"
        )

    def on_key(event):
        if event.key == "N":
            frame_index[0] = (frame_index[0] + 1) % len(files)
            update_display()
        elif event.key == "B":
            frame_index[0] = (frame_index[0] - 1) % len(files)
            update_display()
        elif event.key == "G":
            show_mode[0] = (show_mode[0] + 1) % 3
            update_display()

    canvas.events.key_press.connect(on_key)

    def on_draw(event):
        if canvas.events.key_press.blocked():
            canvas.events.key_press.unblock()

    canvas.events.draw.connect(on_draw)

    update_display()
    canvas.app.run()


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    data_dir = os.path.abspath(DATA_DIR)
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".bin")])

    if not files:
        logger.error(f"No .bin files found in {data_dir}")
        sys.exit(1)

    # --- Calibrate on first frame ---
    fpath = os.path.join(data_dir, files[0])
    logger.info(f"Calibrating on {files[0]}...")
    first_frame = np.fromfile(fpath, dtype=np.float32).reshape(-1, 5)
    calibration = calibrate_ground(first_frame)
    print(f"\nCalibration: {calibration}\n")

    # --- Run on all frames ---
    print(f"{'=' * 70}")
    print("GROUND REMOVAL RESULTS (calibrate once, apply to all)")
    print(f"{'=' * 70}")
    for fname in files:
        fpath = os.path.join(data_dir, fname)
        pts = np.fromfile(fpath, dtype=np.float32).reshape(-1, 5)
        g_pts, ng_pts, g_mask = remove_ground(pts, calibration)
        print(
            f"  {fname}: {g_mask.sum()} ground ({100*g_mask.sum()/len(pts):.1f}%), "
            f"ground Z=[{g_pts[:,2].min():.2f}, {g_pts[:,2].max():.2f}]"
        )
    print(f"{'=' * 70}")

    # --- Visualize (comment out to skip) ---
    visualize_ground_removal(data_dir, files, calibration)
