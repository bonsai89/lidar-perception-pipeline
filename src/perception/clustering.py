"""
BEV clustering, classification, split/merge logic, and interactive 3D viewer.

Core functions used by both the standalone viewer and the pipeline:
  - filter_points: clip point cloud in ground-aligned frame
  - cluster_frame: BEV grid -> connected components -> cluster extraction
  - split_merged_clusters: PCA gap-finding + track-guided splitting
  - merge_engulfed_clusters: absorb small clusters inside larger ones
  - classify_clusters: batch RF classification with physical sanity checks
  - build_bbox_lines: 3D bounding box wireframes for VisPy

Standalone viewer controls:
  G = toggle view mode (all / clusters / non-ground / ground)
  N/B = navigate frames
  T = toggle bounding boxes
  C/V = cycle clusters, X = deselect, D = dump details

Run from project root:
  python optional_challenge/scene_classifier_deep_dive.py
"""

import logging
import os
import sys
import time
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from classifier_pipeline import extract_features, set_feature_mode, _fast_percentile

import numpy as np

from optional_challenge.ground_plane_removal_rotation_ransac import (
    calibrate_ground,
    remove_ground,
    GroundCalibration,
    DATA_DIR,
)

logger = logging.getLogger(__name__)


def filter_points(
        points: np.ndarray,
        calibration: GroundCalibration,
        max_range: float = 80.0,
        max_height_above_ground: float = 3.0,
        min_height_above_ground: float = -1.0,
):
    """Clip point cloud in rotated frame where ground is horizontal.

    Must clip AFTER rotation because sensor is tilted.
    In raw sensor frame, z and xy_range are skewed by tilt.
    In rotated frame, z is true height above ground, xy is true horizontal range.

    :param points: (N, 5) points in original sensor frame.
    :param calibration: Ground calibration with rotation matrix and ground height.
    :param max_range: Max horizontal distance from sensor in rotated frame (meters).
    :param max_height_above_ground: Max height above ground plane to keep (meters).
    :param min_height_above_ground: Min height above ground plane to keep (meters).
    :return: (clipped_points, mask) — clipped points in ORIGINAL frame, mask into original.
    """
    xyz = points[:, :3]
    R = calibration.rotation_matrix

    # Rotate to aligned frame
    rotated = xyz @ R.T
    rot_x, rot_y, rot_z = rotated[:, 0], rotated[:, 1], rotated[:, 2]

    # Range in rotated horizontal plane
    rot_range = np.sqrt(rot_x ** 2 + rot_y ** 2)

    # Height relative to ground
    height_above_ground = rot_z - calibration.ground_height

    mask = (
            (rot_range <= max_range) &
            (height_above_ground >= min_height_above_ground) &
            (height_above_ground <= max_height_above_ground)
    )

    logger.info(
        f"Clip: {len(points)} → {mask.sum()} "
        f"(range: -{(rot_range > max_range).sum()}, "
        f"height: -{((height_above_ground < min_height_above_ground) | (height_above_ground > max_height_above_ground)).sum()})"
    )

    return points[mask], mask


# =====================================================================
# BEV Grid Connected Components Clustering
# =====================================================================


def create_bev_grid(
        xyz: np.ndarray,
        cell_size: float = 0.22,
        min_points_cell: int = 2,
        min_height: float = 0.3,
) -> tuple:
    """Step 1: Project 3D points to 2D Bird's Eye View occupancy grid.

    Each cell is cell_size × cell_size meters.
    A cell is "occupied" if it has enough points AND vertical extent.
    Flat cells (ground fragments, noise) are rejected.

    :param xyz: (N, 3) point coordinates.
    :param cell_size: Grid resolution (meters). 0.15m matches sensor density.
    :param min_points_cell: Minimum points in cell to be occupied.
    :param min_height: Minimum z-range in cell to be occupied (meters).
                       Rejects flat ground fragments (height < 0.3m).
    :return: (grid, cx, cy, origin_x, origin_y, n_cx, n_cy)
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    origin_x, origin_y = x.min(), y.min()
    cx = ((x - origin_x) / cell_size).astype(np.int32)
    cy = ((y - origin_y) / cell_size).astype(np.int32)

    n_cx = cx.max() + 1
    n_cy = cy.max() + 1

    # Per-cell: count points, compute height (max_z - min_z)
    cell_id = cy * n_cx + cx
    max_cell_id = n_cy * n_cx

    cell_count = np.zeros(max_cell_id, dtype=np.int32)
    cell_z_min = np.full(max_cell_id, np.inf)
    cell_z_max = np.full(max_cell_id, -np.inf)

    np.add.at(cell_count, cell_id, 1)
    np.minimum.at(cell_z_min, cell_id, z)
    np.maximum.at(cell_z_max, cell_id, z)

    cell_height = cell_z_max - cell_z_min

    # Occupancy: enough points (no height filter — edges of objects are thin)
    # Height filtering happens at cluster level in extract_clusters()
    cell_occupied = cell_count >= min_points_cell

    # Build 2D grid
    grid = cell_occupied.reshape(n_cy, n_cx)

    n_total_occupied = (cell_count >= 1).sum()
    n_filtered = grid.sum()
    logger.info(
        f"BEV grid: {n_cx}x{n_cy} cells ({cell_size}m), "
        f"{n_total_occupied} raw occupied, "
        f"{n_filtered} after height filter (min_h={min_height}m)"
    )

    return grid, cx, cy, origin_x, origin_y, n_cx, n_cy


def connected_components(grid: np.ndarray, morph_radius: int = 0) -> tuple:
    """Step 2: Connected component labeling on occupancy grid.

    Optional morphological opening (disabled by default, morph_radius=0).
    8-connectivity flood-fill labels connected regions.

    :param grid: (H, W) bool occupancy grid.
    :param morph_radius: 0 = no morphology. 1+ = erode/dilate with that radius.
    :return: (label_grid, n_components)
    """
    from scipy.ndimage import label

    grid_to_label = grid

    # Morphological opening (disabled by default)
    # Uncomment to break thin bridges between objects
    # from scipy.ndimage import binary_erosion, binary_dilation
    # if morph_radius > 0:
    #     kernel = np.ones((2*morph_radius+1, 2*morph_radius+1), dtype=bool)
    #     eroded = binary_erosion(grid, structure=kernel)
    #     grid_to_label = binary_dilation(eroded, structure=kernel)

    # 4-connectivity labeling (shared edges only, no diagonals)
    # Reduces bridging between nearby objects vs 8-connectivity
    structure_4 = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=int)
    label_grid, n_components = label(grid_to_label, structure=structure_4)

    logger.info(f"Connected components: {n_components}")

    return label_grid, n_components


def labels_to_points(
        label_grid: np.ndarray,
        cx: np.ndarray,
        cy: np.ndarray,
) -> np.ndarray:
    """Step 3: Map grid labels back to each 3D point.

    Each point gets the label of its BEV cell.

    :param label_grid: (H, W) component labels from connected_components.
    :param cx: (N,) cell x-indices per point.
    :param cy: (N,) cell y-indices per point.
    :return: (N,) int labels per point. 0 = no cluster.
    """
    return label_grid[cy, cx]


def extract_clusters(
        points: np.ndarray,
        labels: np.ndarray,
        calibration: GroundCalibration,
        min_points: int = 20,
        max_points: int = 15000,
        max_height: float = 5.0,
        max_width: float = 15.0,
) -> list:
    """Step 4: Extract and filter individual clusters.

    Rejects:
    - Too small (<10 pts) → noise, stray points
    - Too large (>5000 pts) → building facades, walls
    - Too tall (>5m) → buildings, trees
    - Too wide (>15m) → merged structures

    :param points: (N, 5) full point cloud.
    :param labels: (N,) cluster labels (0 = no cluster).
    :param min_points: Min cluster size.
    :param max_points: Max cluster size.
    :param max_height: Max z-range within cluster (meters).
    :param max_width: Max xy-spread (meters).
    :return: List of cluster dicts.
    """
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]  # skip 0 (empty)

    clusters = []
    rejected = {"small": 0, "large": 0, "tall": 0, "wide": 0}

    for lbl in unique_labels:
        mask = labels == lbl
        n_pts = mask.sum()

        if n_pts < min_points:
            rejected["small"] += 1
            continue
        if n_pts > max_points:
            rejected["large"] += 1
            continue

        cluster_pts = points[mask]
        xyz = cluster_pts[:, :3]

        z_min = xyz[:, 2].min()
        z_max = xyz[:, 2].max()
        z_range = z_max - z_min

        # Two-stage ground trim (robust to slopes and partial objects)
        z_min = xyz[:, 2].min()

        z_above_ground = xyz[:, 2].max() - calibration.ground_height

        # Only trim if cluster is near the ground
        if (z_min - calibration.ground_height) < 0.25 and z_range > 0.5:
            z_cut = _fast_percentile(xyz[:, 2], 5)
            keep = xyz[:, 2] > z_cut

            cluster_pts = cluster_pts[keep]
            xyz = cluster_pts[:, :3]
            n_pts = len(cluster_pts)

            if n_pts < min_points:
                rejected["small"] += 1
                continue

            z_range = xyz[:, 2].max() - xyz[:, 2].min()

        if z_range > max_height:
            rejected["tall"] += 1
            continue

        if n_pts > 3:
            xy_c = xyz[:, :2] - xyz[:, :2].mean(axis=0)
            cov_2d = xy_c.T @ xy_c / n_pts
            _, evecs = np.linalg.eigh(cov_2d)
            xy_rot = xy_c @ evecs
            x_rng = xy_rot[:, 1].max() - xy_rot[:, 1].min()
            y_rng = xy_rot[:, 0].max() - xy_rot[:, 0].min()
            xy_spread = np.sqrt(x_rng ** 2 + y_rng ** 2)
        else:
            xy_spread = np.sqrt(
                (xyz[:, 0].max() - xyz[:, 0].min()) ** 2 +
                (xyz[:, 1].max() - xyz[:, 1].min()) ** 2
            )
        if xy_spread > max_width:
            rejected["wide"] += 1
            continue

        clusters.append({
            "points": cluster_pts,
            "xyz": xyz,
            "centroid": xyz.mean(axis=0),
            "label_id": int(lbl),
            "n_points": n_pts,
            "z_range": z_range,
            "xy_spread": xy_spread,
            "z_above_ground": z_above_ground,
        })

    logger.info(
        f"Clusters: {len(clusters)} valid, {len(unique_labels)} total "
        f"(rejected: {rejected})"
    )

    return clusters

def classify_clusters(clusters, model):
    """Batch classification with physical sanity checks."""
    if not clusters:
        return [], []

    CLASS_NAMES = {0: "background", 1: "bicyclist", 2: "car", 3: "pedestrian"}

    n = len(clusters)
    n_feat = len(model.feature_importances_)
    X = np.zeros((n, n_feat), dtype=np.float32)
    for i, cluster in enumerate(clusters):
        X[i] = extract_features(cluster["points"])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    preds = model.predict(X)
    probas = model.predict_proba(X)

    object_clusters = []
    background_clusters = []

    for i, cluster in enumerate(clusters):
        pred = int(preds[i])
        cluster["pred"] = pred
        cluster["s2_class"] = CLASS_NAMES[pred]
        cluster["s2_confidence"] = float(probas[i, pred])
        cluster["s2_pred"] = pred
        cluster["s2_probas"] = probas[i].tolist()

        # # Low confidence → demote to background
        # if pred > 0 and cluster["s2_confidence"] < 0.5:
        #     cluster["s2_class"] = "background"
        #     cluster["pred"] = 0
            
        # Physical sanity check
        xy = cluster.get("xy_spread", 0)
        zr = cluster.get("z_range", 0)
        z_ag = cluster.get("z_above_ground", 999)
        if cluster["s2_class"] == "pedestrian" and (xy > 1.4 or z_ag < 0.7 or zr > 2.2):
            cluster["s2_class"] = "background"
            cluster["pred"] = 0
        elif cluster["s2_class"] == "bicyclist" and (xy > 2.6 or xy < 0.8 or z_ag < 0.7 or zr > 2.2):
            cluster["s2_class"] = "background"
            cluster["pred"] = 0
        elif cluster["s2_class"] == "car" and (xy > 7 or z_ag < 0.5 or zr > 2.3):
            cluster["s2_class"] = "background"
            cluster["pred"] = 0

        if cluster["pred"] > 0:
            object_clusters.append(cluster)
        else:
            background_clusters.append(cluster)

    return object_clusters, background_clusters

def merge_engulfed_clusters(clusters):
    """Merge small clusters whose bbox is fully inside a larger cluster's bbox."""
    if len(clusters) < 2:
        return clusters

    n = len(clusters)
    # Precompute bboxes
    bboxes = []
    for cl in clusters:
        xyz = cl["xyz"]
        bboxes.append({
            "min": xyz.min(axis=0),
            "max": xyz.max(axis=0),
        })

    absorbed = set()
    for i in range(n):
        if i in absorbed:
            continue
        for j in range(n):
            if j == i or j in absorbed:
                continue
            # Is j fully inside i?
            if (bboxes[j]["min"][0] >= bboxes[i]["min"][0] and
                bboxes[j]["max"][0] <= bboxes[i]["max"][0] and
                bboxes[j]["min"][1] >= bboxes[i]["min"][1] and
                bboxes[j]["max"][1] <= bboxes[i]["max"][1] and
                bboxes[j]["min"][2] >= bboxes[i]["min"][2] and
                bboxes[j]["max"][2] <= bboxes[i]["max"][2]):
                # Only absorb if j is significantly smaller
                if clusters[j]["n_points"] < clusters[i]["n_points"] * 0.5:
                    absorbed.add(j)
                    # Merge points
                    clusters[i]["points"] = np.vstack([clusters[i]["points"], clusters[j]["points"]])
                    clusters[i]["xyz"] = np.vstack([clusters[i]["xyz"], clusters[j]["xyz"]])
                    clusters[i]["n_points"] = len(clusters[i]["xyz"])
                    clusters[i]["centroid"] = clusters[i]["xyz"].mean(axis=0)
                    clusters[i]["z_range"] = clusters[i]["xyz"][:, 2].max() - clusters[i]["xyz"][:, 2].min()
                    clusters[i]["xy_spread"] = np.sqrt(
                        (clusters[i]["xyz"][:, 0].max() - clusters[i]["xyz"][:, 0].min())**2 +
                        (clusters[i]["xyz"][:, 1].max() - clusters[i]["xyz"][:, 1].min())**2)
                    # Update bbox for i since it grew
                    bboxes[i]["min"] = clusters[i]["xyz"].min(axis=0)
                    bboxes[i]["max"] = clusters[i]["xyz"].max(axis=0)

    return [cl for i, cl in enumerate(clusters) if i not in absorbed]

def cluster_frame(
        non_ground: np.ndarray,
        calibration: GroundCalibration,
        cell_size: float = 0.15,
        min_points: int = 20,
        max_points: int = 15000,
) -> tuple:
    """Full BEV clustering pipeline for one frame's non-ground points.

    Step 1: Project to BEV occupancy grid
    Step 2: Connected components (flood fill)
    Step 3: Map labels back to 3D points
    Step 4: Extract and filter clusters

    :param non_ground: (M, 5) non-ground points (already clipped).
    :param calibration: Ground calibration (kept for interface consistency).
    :param cell_size: BEV grid resolution (meters).
    :param min_points: Min cluster size after filtering.
    :param max_points: Max cluster size.
    :return: (clusters, labels)
    """
    t0 = time.time()

    xyz = non_ground[:, :3]

    # Step 1: BEV grid
    grid, cx, cy, ox, oy, ncx, ncy = create_bev_grid(xyz, cell_size)

    # Step 2: Connected components
    label_grid, n_components = connected_components(grid)

    # Step 3: Map back to points
    labels = labels_to_points(label_grid, cx, cy)

    # Step 4: Extract + filter
    clusters = extract_clusters(non_ground, labels, calibration, min_points, max_points)

    elapsed = time.time() - t0
    logger.info(f"BEV clustering: {len(clusters)} clusters, {elapsed * 1000:.1f}ms")

    return clusters, labels


# =====================================================================
# Bounding Box Line Builder
# =====================================================================

def build_bbox_lines(clusters, cluster_classes, cluster_confs, selected=-1):
    """Build 3D bounding box wireframes for VisPy Line visual.

    Each box = 12 edges = 24 vertices connected pairwise.

    :param clusters: list of cluster dicts with "xyz" key.
    :param cluster_classes: list of class name strings per cluster.
    :param cluster_confs: list of confidence floats per cluster.
    :param selected: index of selected cluster (-1 = none).
                     Selected box draws white+bright, others dim slightly.
    :return: (positions (N,3), colors (N,4), connect (M,2))
    """
    CLASS_COLORS_BBOX = {
        "car":        np.array([0.2, 0.6, 1.0, 1.0]),
        "pedestrian": np.array([1.0, 0.3, 0.3, 1.0]),
        "bicyclist":  np.array([1.0, 0.8, 0.0, 1.0]),
        "background": np.array([0.7, 0.7, 0.7, 0.7]),
    }

    # 12 edges indexed into 8 corners of an axis-aligned box
    #   4---7        top face: 4-5-6-7
    #  /|  /|        bottom:   0-1-2-3
    # 5---6 |        verticals: 0-4, 1-5, 2-6, 3-7
    # | 0-|-3
    # |/  |/
    # 1---2
    EDGES = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
    ]

    all_pos = []
    all_col = []
    all_conn = []
    offset = 0

    for i, cl in enumerate(clusters):
        xyz = cl["xyz"]
        mn = xyz.min(axis=0)
        mx = xyz.max(axis=0)

        corners = np.array([
            [mn[0], mn[1], mn[2]],  # 0
            [mx[0], mn[1], mn[2]],  # 1
            [mx[0], mx[1], mn[2]],  # 2
            [mn[0], mx[1], mn[2]],  # 3
            [mn[0], mn[1], mx[2]],  # 4
            [mx[0], mn[1], mx[2]],  # 5
            [mx[0], mx[1], mx[2]],  # 6
            [mn[0], mx[1], mx[2]],  # 7
        ])

        cls_name = cluster_classes[i] if i < len(cluster_classes) else "background"
        if cls_name == "background":
            continue
        conf = cluster_confs[i] if i < len(cluster_confs) else 0.0
        color = CLASS_COLORS_BBOX.get(cls_name, CLASS_COLORS_BBOX["background"]).copy()

        # Dashed-look: bright magenta for split clusters
        if cl.get("track_guided", False) and i != selected:
            color = np.array([0.0, 1.0, 1.0, 1.0])  # cyan = track-guided split
        elif cl.get("was_split", False) and i != selected:
            color = np.array([1.0, 0.0, 1.0, 1.0])  # magenta = PCA split

        # Selected cluster: white, full alpha
        if i == selected:
            color = np.array([1.0, 1.0, 1.0, 1.0])
        elif selected >= 0:
            # Something else selected — dim this box
            color[3] *= 0.25

        for e0, e1 in EDGES:
            all_pos.append(corners[e0])
            all_pos.append(corners[e1])
            all_col.append(color)
            all_col.append(color)
            all_conn.append([offset, offset + 1])
            offset += 2

    if not all_pos:
        return np.empty((0, 3)), np.empty((0, 4)), np.empty((0, 2), dtype=np.int32)

    return (
        np.array(all_pos, dtype=np.float32),
        np.array(all_col, dtype=np.float32),
        np.array(all_conn, dtype=np.int32),
    )


def find_split_points(projections, min_gap_pts=20):
    """Find natural gaps in 1D projected points.
    Returns split thresholds, or empty list if no clear gap."""
    bins = np.linspace(projections.min(), projections.max(), 30)
    hist, edges = np.histogram(projections, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2

    # Find bins below 20% of mean — these are gaps
    threshold = max(hist.mean() * 0.2, 1)
    gap_mask = hist < threshold

    # Get gap centers as split points
    splits = []
    in_gap = False
    gap_start = 0
    for i, is_gap in enumerate(gap_mask):
        if is_gap and not in_gap:
            gap_start = i
            in_gap = True
        elif not is_gap and in_gap:
            # split at center of gap
            splits.append(centers[(gap_start + i) // 2])
            in_gap = False

    return splits

def detect_merged_clusters(clusters):
    """Flag clusters that look like merged pedestrians using PCA.

    Criteria: tall (z_range 1.0-2.0m), high linearity, enough points.
    Returns list of (cluster_idx, linearity, ev0_direction) tuples.
    """
    candidates = []
    for i, cl in enumerate(clusters):
        xyz = cl["xyz"]
        n = len(xyz)
        z_range = cl["z_range"]

        # Only tall, reasonably-sized clusters
        if z_range < 1.0 or z_range > 2.2 or n < 100 or n > 2000:
            continue

        # PCA
        centered = xyz - xyz.mean(axis=0)
        cov = centered.T @ centered / n
        eigvals, eigvecs = np.linalg.eigh(cov)
        # eigh returns ascending, flip to descending
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]
        # Principal axis must be mostly horizontal (small z-component)
        # Two peds side by side → elongation is in xy plane
        # Single tall ped → elongation is vertical, skip
        z_component = abs(eigvecs[2, 0])  # z-weight of first eigenvector
        if z_component > 0.5:
            continue
        total = eigvals.sum() + 1e-12
        linearity = (eigvals[0] - eigvals[1]) / (total)

        if linearity > 0.3 and n > 150:
            candidates.append({
                "idx": i,
                "linearity": linearity,
                "direction": eigvecs[:, 0],  # principal axis
                "n_points": n,
                "z_range": z_range,
                "xy_spread": cl["xy_spread"],
            })

    return candidates

def split_merged_clusters(clusters, prev_tracks=None):
    """Detect and split merged pedestrian clusters along principal axis.

    Returns new cluster list with merged ones replaced by their pieces.
    """
    candidates = detect_merged_clusters(clusters)
    if not candidates:
        return clusters

    split_indices = {mc["idx"] for mc in candidates}
    new_clusters = []

    for i, cl in enumerate(clusters):
        # Track-guided split: if tracker knows there were 2+ objects here
        if prev_tracks and cl.get("z_range", 0) > 1.0 and cl.get("n_points", 0) > 100 and cl.get("n_points", 0) < 2000:
            centroid = cl["centroid"]
            nearby = [t for t in prev_tracks
                      if t.get("class_label") != "background"
                      and np.linalg.norm(np.array(t["position"]) - centroid) < cl["xy_spread"] + 1.0]
            if len(nearby) >= 2:
                xyz = cl["xyz"]
                pts = cl["points"]
                # Split axis = line connecting two nearest track positions
                p1 = np.array(nearby[0]["position"])
                p2 = np.array(nearby[1]["position"])
                direction = p2 - p1
                norm = np.linalg.norm(direction)
                if norm > 0.3:
                    direction /= norm
                    midpoint = (p1 + p2) / 2
                    projections = (xyz - midpoint) @ direction
                    # Split at 0 (midpoint between the two tracks)
                    valid_pieces = []
                    for sign_mask in [projections <= 0, projections > 0]:
                        if sign_mask.sum() < 20:
                            continue
                        sub_xyz = xyz[sign_mask]
                        sub_pts = pts[sign_mask]
                        sub_xy_spread = np.sqrt(
                            (sub_xyz[:, 0].max() - sub_xyz[:, 0].min()) ** 2 +
                            (sub_xyz[:, 1].max() - sub_xyz[:, 1].min()) ** 2)
                        if sub_xy_spread < 0.3 or sub_xy_spread > 1.5:
                            continue
                        valid_pieces.append((sub_pts, sub_xyz))
                    if len(valid_pieces) >= 2 and all(len(p) <= 1000 for p, _ in valid_pieces):
                        sizes = [len(p) for p, _ in valid_pieces]
                        if min(sizes) < 0.25 * max(sizes):
                            new_clusters.append(cl)
                            continue
                        for sub_pts, sub_xyz in valid_pieces:
                            new_clusters.append({
                                "points": sub_pts,
                                "xyz": sub_xyz,
                                "centroid": sub_xyz.mean(axis=0),
                                "label_id": cl["label_id"],
                                "n_points": len(sub_pts),
                                "z_range": sub_xyz[:, 2].max() - sub_xyz[:, 2].min(),
                                "xy_spread": np.sqrt(
                                    (sub_xyz[:, 0].max() - sub_xyz[:, 0].min()) ** 2 +
                                    (sub_xyz[:, 1].max() - sub_xyz[:, 1].min()) ** 2),
                                "was_split": True,
                                "track_guided": True,
                            })
                        print(f"    Track-guided split cluster {i}: {cl['n_points']}pts → "
                              f"{' + '.join(str(len(p)) for p, _ in valid_pieces)}")
                        continue  # skip PCA path, already handled

        if i not in split_indices:
            new_clusters.append(cl)
            continue

        # Split along principal axis
        xyz = cl["xyz"]
        pts = cl["points"]
        centroid = xyz.mean(axis=0)

        # Get principal direction
        mc = next(m for m in candidates if m["idx"] == i)
        direction = mc["direction"]

        # Project onto principal axis, split at median
        projections = (xyz - centroid) @ direction
        splits = find_split_points(projections)

        if not splits:
            new_clusters.append(cl)
            continue

        boundaries = [-np.inf] + splits + [np.inf]
        valid_pieces = []
        for k in range(len(boundaries) - 1):
            mask = (projections >= boundaries[k]) & (projections < boundaries[k + 1])
            if mask.sum() < 20:
                continue
            sub_xyz = xyz[mask]
            sub_pts = pts[mask]

            sub_z_range = sub_xyz[:, 2].max() - sub_xyz[:, 2].min()
            sub_xy_spread = np.sqrt(
                (sub_xyz[:, 0].max() - sub_xyz[:, 0].min()) ** 2 +
                (sub_xyz[:, 1].max() - sub_xyz[:, 1].min()) ** 2
            )

            # 1. Must be tall enough
            if sub_z_range < 0.8:
                continue

            # 2. Must be narrow — pedestrian-like footprint
            if sub_xy_spread > 1.5:
                continue

            # 3. Aspect ratio — taller than wide
            if sub_z_range / (sub_xy_spread + 1e-6) < 0.8:
                continue

            # 4. Vertical density uniformity — not plate/roof-like
            n_bins = 4
            z_vals = sub_xyz[:, 2]
            z_lo = np.percentile(z_vals, 5)
            z_hi = np.percentile(z_vals, 95)
            if z_hi - z_lo < 0.3:
                continue  # truly flat
            counts = np.histogram(z_vals, bins=np.linspace(z_lo, z_hi, n_bins + 1))[0]
            counts = counts.astype(float)
            mean_count = counts.mean()
            if mean_count > 0 and counts.std() / mean_count > 1.5:
                continue

            # 5. Must have minimum width — too narrow is a pole/edge, not a person
            if sub_xy_spread < 0.3:
                continue

            valid_pieces.append((sub_pts, sub_xyz))

        if len(valid_pieces) < 2:
            new_clusters.append(cl)  # need at least 2 valid pieces to split
        elif any(len(p) > 1000 for p, _ in valid_pieces):
            new_clusters.append(cl)  # piece too large, not pedestrian
        elif min(len(p) for p, _ in valid_pieces) < 0.25 * max(len(p) for p, _ in valid_pieces):
            new_clusters.append(cl)  # too uneven, not real split
        else:
            # Collect which points were claimed by valid pieces
            claimed = np.zeros(len(pts), dtype=bool)
            for k in range(len(boundaries) - 1):
                mask = (projections >= boundaries[k]) & (projections < boundaries[k + 1])
                # Check if this segment produced a valid piece
                for sub_pts, sub_xyz in valid_pieces:
                    if len(sub_pts) == mask.sum() and mask.sum() > 0:
                        claimed |= mask
                        break

            # Add valid pieces
            for sub_pts, sub_xyz in valid_pieces:
                new_clusters.append({
                    "points": sub_pts,
                    "xyz": sub_xyz,
                    "centroid": sub_xyz.mean(axis=0),
                    "label_id": cl["label_id"],
                    "n_points": len(sub_pts),
                    "z_range": sub_xyz[:, 2].max() - sub_xyz[:, 2].min(),
                    "xy_spread": np.sqrt(
                        (sub_xyz[:, 0].max() - sub_xyz[:, 0].min()) ** 2 +
                        (sub_xyz[:, 1].max() - sub_xyz[:, 1].min()) ** 2),
                    "was_split": True,
                })

            # Leftover → separate cluster
            leftover_mask = ~claimed
            if leftover_mask.sum() > 1000:
                new_clusters.append(cl)
                continue
            if leftover_mask.sum() >= 20:
                lo_pts = pts[leftover_mask]
                lo_xyz = xyz[leftover_mask]
                new_clusters.append({
                    "points": lo_pts,
                    "xyz": lo_xyz,
                    "centroid": lo_xyz.mean(axis=0),
                    "label_id": cl["label_id"],
                    "n_points": len(lo_pts),
                    "z_range": lo_xyz[:, 2].max() - lo_xyz[:, 2].min(),
                    "xy_spread": np.sqrt(
                        (lo_xyz[:, 0].max() - lo_xyz[:, 0].min()) ** 2 +
                        (lo_xyz[:, 1].max() - lo_xyz[:, 1].min()) ** 2),
                })

            print(f"    Split cluster {i}: {cl['n_points']}pts → "
                  f"{' + '.join(str(len(p)) for p, _ in valid_pieces)}"
                  f"{f' + {leftover_mask.sum()} leftover' if leftover_mask.sum() >= 20 else ''}")


    return new_clusters

# =====================================================================
# Visualization
# =====================================================================

def visualize(data_dir: str, files: list, calibration: GroundCalibration,
              model_path: str = "models/rf_classifier_19feat.pkl") -> None:
    """VisPy viewer with clustering results + 3D bounding boxes.
    Each cluster = class color. Ground = green. Unclustered = dim gray.
    G = toggle view mode
    N/B = navigate frames
    T = toggle bounding boxes
    C/V = cycle clusters, X = deselect
    D = dump cluster details
    """
    import vispy
    from vispy.scene import visuals
    from vispy.scene.cameras import TurntableCamera
    from vispy.scene import SceneCanvas

    # Load classifier
    print("Pre-computing all frames...")

    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]
    set_feature_mode(saved["feature_mode"])
    print(f"Loaded model: {saved['feature_mode']} features from {model_path}")
    CLASS_NAMES = {0: "background", 1: "bicyclist", 2: "car", 3: "pedestrian"}

    cached = []
    for i, fname in enumerate(files):
        fpath = os.path.join(data_dir, fname)
        pts = np.fromfile(fpath, dtype=np.float32).reshape(-1, 5)

        t_total = time.time()

        # Clip in rotated frame (sensor is tilted)
        t0 = time.time()
        clipped, clip_mask = filter_points(pts, calibration)
        t_clip = (time.time() - t0) * 1000

        # Ground removal on clipped
        t0 = time.time()
        _, non_ground, ground_mask = remove_ground(clipped, calibration)
        t_ground = (time.time() - t0) * 1000

        # BEV clustering on non-ground
        t0 = time.time()
        clusters, cluster_labels = cluster_frame(non_ground, calibration)
        t_cluster = (time.time() - t0) * 1000

        # Split merged clusters
        t0 = time.time()
        clusters = split_merged_clusters(clusters)
        t_split = (time.time() - t0) * 1000

        # Merge engulfed clusters
        t0 = time.time()
        clusters = merge_engulfed_clusters(clusters)
        t_merge = (time.time() - t0) * 1000

        # Classify
        t0 = time.time()
        object_clusters, background_clusters = classify_clusters(clusters, model)
        t_classify = (time.time() - t0) * 1000

        t_total_ms = (time.time() - t_total) * 1000

        print(f"  [{i + 1}/{len(files)}] {fname}: {len(clipped)} pts, "
              f"{ground_mask.sum()} ground, {len(clusters)} clusters")
        print(f"    Latency: clip={t_clip:.1f}ms  ground={t_ground:.1f}ms  "
              f"cluster={t_cluster:.1f}ms  split={t_split:.1f}ms  "
              f"merge={t_merge:.1f}ms  classify={t_classify:.1f}ms  "
              f"total={t_total_ms:.1f}ms")
        all_classified = object_clusters + background_clusters

        CLASS_NAMES = {0: "background", 1: "bicyclist", 2: "car", 3: "pedestrian"}
        cluster_data = []
        for cluster in all_classified:
            probs = cluster.get("s2_probas", [0.25, 0.25, 0.25, 0.25])
            sorted_idx = np.argsort(probs)[::-1]
            cluster_data.append({
                "cls_name": cluster["s2_class"],
                "conf": cluster["s2_confidence"],
                "second_cls": CLASS_NAMES[sorted_idx[1]],
                "second_conf": float(probs[sorted_idx[1]]),
                "n_points": cluster["n_points"],
                "z_range": cluster["z_range"],
                "xy_spread": cluster["xy_spread"],
                "centroid": cluster["centroid"].copy(),
            })
        preds = np.array([c["pred"] for c in all_classified])
        clusters = all_classified

        # Build color array for clipped point cloud
        xyz = clipped[:, :3]
        n = len(clipped)
        colors = np.zeros((n, 4), dtype=np.float32)

        # Ground: green
        colors[ground_mask, 0] = 0.1
        colors[ground_mask, 1] = 0.8
        colors[ground_mask, 2] = 0.1
        colors[ground_mask, 3] = 0.6

        # Non-ground: dim gray
        ng_indices = np.where(~ground_mask)[0]
        colors[ng_indices, :3] = 0.2
        colors[ng_indices, 3] = 0.3

        # Color clusters by predicted class
        CLASS_COLORS = {
            "car": np.array([0.2, 0.6, 1.0]),
            "pedestrian": np.array([1.0, 0.3, 0.3]),
            "bicyclist": np.array([1.0, 0.8, 0.0]),
            "background": np.array([0.7, 0.7, 0.7]),
        }

        cluster_mask = np.zeros(n, dtype=bool)
        class_counts = {}
        for j, cluster in enumerate(clusters):
            lbl = cluster["label_id"]
            cl_mask_in_ng = cluster_labels == lbl
            cl_indices = ng_indices[cl_mask_in_ng]
            cls_name = CLASS_NAMES[preds[j]]
            colors[cl_indices, :3] = CLASS_COLORS.get(cls_name, np.array([0.5, 0.5, 0.5]))
            colors[cl_indices, 3] = 1.0
            cluster_mask[cl_indices] = True
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        cached.append({
            "fname": fname,
            "xyz": xyz,
            "colors": colors,
            "ground_mask": ground_mask,
            "cluster_mask": cluster_mask,
            "n_total": n,
            "n_ground": ground_mask.sum(),
            "n_clusters": len(clusters),
            "cluster_data": cluster_data,
            "class_counts": class_counts,
            # For bounding boxes
            "clusters_raw": clusters,
            "cluster_classes": [cd["cls_name"] for cd in cluster_data],
            "cluster_confs": [cd["conf"] for cd in cluster_data],
        })
        print(
            f"  [{i + 1}/{len(files)}] {fname}: {n} pts, "
            f"{ground_mask.sum()} ground, {len(clusters)} clusters"
        )

    print("Launching viewer...")

    frame_idx = [0]
    show_mode = [0]
    mode_names = ["all", "clusters only", "non-ground", "ground only"]
    selected_cluster = [-1]
    show_bbox = [True]

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

    # Bounding box wireframe visual
    bbox_visual = visuals.Line(parent=view.scene, width=2.0, antialias=True)

    info = visuals.Text(
        text="", color="white", font_size=12,
        anchor_x="left", anchor_y="top",
        parent=canvas.scene,
    )
    info.pos = (10, 20)

    def update():
        f = cached[frame_idx[0]]
        mode = show_mode[0]

        if mode == 0:
            show_xyz = f["xyz"]
            show_col = f["colors"].copy()
        elif mode == 1:
            mask = f["cluster_mask"]
            show_xyz = f["xyz"][mask]
            show_col = f["colors"][mask].copy()
        elif mode == 2:
            mask = ~f["ground_mask"]
            show_xyz = f["xyz"][mask]
            show_col = f["colors"][mask].copy()
        else:
            mask = f["ground_mask"]
            show_xyz = f["xyz"][mask]
            show_col = np.zeros((mask.sum(), 4), dtype=np.float32)
            show_col[:, 1] = 0.8
            show_col[:, 3] = 0.6

        # Highlight selected cluster
        sel = selected_cluster[0]
        sel_text = ""
        if sel >= 0 and sel < len(f.get("cluster_data", [])):
            cd = f["cluster_data"][sel]
            sel_text = (
                f"\n\n--- CLUSTER #{sel}/{len(f['cluster_data'])} ---\n"
                f"  Class: {cd['cls_name']}\n"
                f"  Confidence: {cd['conf']:.3f}\n"
                f"  2nd: {cd['second_cls']} ({cd['second_conf']:.3f})\n"
                f"  Points: {cd['n_points']}\n"
                f"  Z-range: {cd['z_range']:.2f}m\n"
                f"  XY-spread: {cd['xy_spread']:.2f}m"
            )
            # Dim everything, highlight selected white
            show_col[:, 3] *= 0.2
            centroid = cd["centroid"]
            dists = np.linalg.norm(show_xyz[:, :2] - centroid[:2], axis=1)
            highlight = dists < max(cd["xy_spread"], 1.0)
            show_col[highlight, :3] = 1.0
            show_col[highlight, 3] = 1.0

        scatter.set_data(show_xyz, face_color=show_col, edge_color=show_col, size=1.0)

        # --- Bounding boxes ---
        clusters_raw = f.get("clusters_raw", [])
        cls_names = f.get("cluster_classes", [])
        cls_confs = f.get("cluster_confs", [])

        if show_bbox[0] and clusters_raw and mode in (0, 1, 2):
            bbox_pos, bbox_col, bbox_conn = build_bbox_lines(
                clusters_raw, cls_names, cls_confs, selected=sel
            )
            if len(bbox_pos) > 0:
                bbox_visual.set_data(pos=bbox_pos, color=bbox_col, connect=bbox_conn)
                bbox_visual.visible = True
            else:
                bbox_visual.visible = False
        else:
            bbox_visual.visible = False

        counts_str = ", ".join(f"{k}={v}" for k, v in f.get("class_counts", {}).items())
        bbox_str = "ON" if show_bbox[0] else "OFF"
        canvas.title = (
            f"Frame {frame_idx[0] + 1}/{len(files)} — {mode_names[mode]} — "
            f"{f['n_clusters']} clusters — bbox:{bbox_str}"
        )
        info.text = (
            f"File: {f['fname']}\n"
            f"Points: {len(show_xyz)} | Clusters: {f['n_clusters']}\n"
            f"Classes: {counts_str}\n"
            f"Mode: {mode_names[mode]} | BBox: {bbox_str}\n\n"
            f"LEGEND: Blue=car Red=ped Yellow=bike Gray=bg\n"
            f"[N]ext [B]ack [G]toggle [T]bbox [C]next [V]prev [X]desel [D]dump"
            f"{sel_text}"
        )

    def on_key(event):
        if event.key == "N":
            frame_idx[0] = (frame_idx[0] + 1) % len(files)
            selected_cluster[0] = -1
            update()
        elif event.key == "B":
            frame_idx[0] = (frame_idx[0] - 1) % len(files)
            selected_cluster[0] = -1
            update()
        elif event.key == "G":
            show_mode[0] = (show_mode[0] + 1) % len(mode_names)
            update()
        elif event.key == "T":
            show_bbox[0] = not show_bbox[0]
            update()
        elif event.key == "D":
            f = cached[frame_idx[0]]
            print(f"\n{'=' * 80}")
            print(f"Frame {frame_idx[0] + 1}: {f['fname']} — {len(f['cluster_data'])} clusters")
            print(f"{'=' * 80}")
            for j, cd in enumerate(f["cluster_data"]):
                print(f"  [{j:3d}] {cd['cls_name']:12s} conf={cd['conf']:.3f} "
                      f"2nd={cd['second_cls']:12s}({cd['second_conf']:.3f}) "
                      f"n={cd['n_points']:5d} z={cd['z_range']:.2f}m "
                      f"xy={cd['xy_spread']:.2f}m "
                      f"pos=[{cd['centroid'][0]:.1f}, {cd['centroid'][1]:.1f}, {cd['centroid'][2]:.1f}]")
            print(f"{'=' * 80}")
        elif event.key == "C":
            f = cached[frame_idx[0]]
            n = len(f["cluster_data"])
            if n > 0:
                selected_cluster[0] = (selected_cluster[0] + 1) % n
                cd = f["cluster_data"][selected_cluster[0]]
                print(f"  [{selected_cluster[0]:3d}] {cd['cls_name']:12s} conf={cd['conf']:.3f} "
                      f"2nd={cd['second_cls']}({cd['second_conf']:.3f}) "
                      f"n={cd['n_points']:5d} z={cd['z_range']:.2f}m xy={cd['xy_spread']:.2f}m")
                update()
        elif event.key == "V":
            f = cached[frame_idx[0]]
            n = len(f["cluster_data"])
            if n > 0:
                selected_cluster[0] = (selected_cluster[0] - 1) % n
                cd = f["cluster_data"][selected_cluster[0]]
                print(f"  [{selected_cluster[0]:3d}] {cd['cls_name']:12s} conf={cd['conf']:.3f} "
                      f"2nd={cd['second_cls']}({cd['second_conf']:.3f}) "
                      f"n={cd['n_points']:5d} z={cd['z_range']:.2f}m xy={cd['xy_spread']:.2f}m")
                update()
        elif event.key == "X":
            selected_cluster[0] = -1
            update()

    canvas.events.key_press.connect(on_key)
    canvas.events.draw.connect(lambda e: canvas.events.key_press.unblock()
    if canvas.events.key_press.blocked() else None)
    update()
    canvas.app.run()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="BEV clustering + classification viewer")
    parser.add_argument(
        "--features", type=int, choices=[19, 23, 35], default=19,
        help="Feature mode — auto-loads matching model",
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Override model path (default: models/rf_classifier_{N}feat.pkl)",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(DATA_DIR)
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".bin")])

    if not files:
        logger.error(f"No .bin files in {data_dir}")
        sys.exit(1)

    # Calibrate
    logger.info(f"Calibrating on {files[0]}...")
    first_frame = np.fromfile(os.path.join(data_dir, files[0]),
                              dtype=np.float32).reshape(-1, 5)
    calibration = calibrate_ground(first_frame)
    print(f"Calibration: {calibration}\n")

    # Visualize with model
    model_path = args.model_path or f"models/rf_classifier_{args.features}feat.pkl"
    visualize(data_dir, files, calibration, model_path=model_path)