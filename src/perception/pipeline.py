"""
Full perception pipeline for optional challenge.

Ground removal → Clipping → BEV Clustering → Single-stage RF Classification → Tracking

Uses the single-stage 4-class RF trained on the main task data.
Tracker operates on car/pedestrian/bicyclist clusters.

Run from project root:
  python optional_challenge/pipeline.py
  python optional_challenge/pipeline.py --force-rerun
  python optional_challenge/pipeline.py --no-viz
"""

import logging
import os
import sys
import time
import pickle

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from optional_challenge.ground_plane_removal_rotation_ransac import (
    calibrate_ground,
    remove_ground,
    GroundCalibration,
    DATA_DIR,
)
from optional_challenge.scene_classifier_deep_dive import (
    filter_points,
    cluster_frame,
    build_bbox_lines,
    split_merged_clusters,
    merge_engulfed_clusters,
    classify_clusters,
)
from classifier_pipeline import extract_features, set_feature_mode
from data_loader import INV_CLASS_MAP

logger = logging.getLogger(__name__)

# Class mapping: 0=background, 1=bicyclist, 2=car, 3=pedestrian
CLASS_NAMES = {0: "background", 1: "bicyclist", 2: "car", 3: "pedestrian"}


def load_model(model_path: str = "models/rf_classifier.pkl"):
    """Load pre-trained single-stage 4-class model.

    :param model_path: Path to pickled model.
    :return: Trained RF classifier.
    """
    with open(model_path, "rb") as f:
        data = pickle.load(f)

    model = data["model"]
    feat_mode = data.get("feature_mode", "unknown")
    logger.info(f"Loaded single-stage model: {feat_mode} features")
    return model

def process_frame_full(
    points: np.ndarray,
    calibration: GroundCalibration,
    model,
) -> dict:
    """Full pipeline for one frame.

    :param points: (N, 5) raw frame.
    :param calibration: Ground calibration.
    :param model: Single-stage RF classifier.
    :return: Dict with all pipeline outputs.
    """
    t0 = time.time()

    # Clip
    clipped, clip_mask = filter_points(points, calibration)

    # Ground removal
    ground_pts, non_ground, ground_mask = remove_ground(clipped, calibration)

    # BEV clustering
    clusters, cluster_labels = cluster_frame(non_ground, calibration)

    # Split merged pedestrian clusters
    clusters = split_merged_clusters(clusters)

    # Merge engulfed clusters
    clusters = merge_engulfed_clusters(clusters)

    # Single-stage classification
    object_clusters, background_clusters = classify_clusters(clusters, model)

    elapsed = time.time() - t0
    logger.info(f"Full pipeline: {elapsed*1000:.0f}ms")

    return {
        "clipped": clipped,
        "ground_mask": ground_mask,
        "non_ground": non_ground,
        "cluster_labels": cluster_labels,
        "all_clusters": clusters,
        "object_clusters": object_clusters,
        "background_clusters": background_clusters,
        "elapsed_ms": elapsed * 1000,
    }


# =====================================================================
# Visualization
# =====================================================================

# Class colors for visualization
CLASS_COLORS = {
    "car": np.array([0.2, 0.6, 1.0]),        # blue
    "pedestrian": np.array([1.0, 0.3, 0.3]),  # red
    "bicyclist": np.array([1.0, 0.8, 0.0]),   # yellow
    "background": np.array([0.7, 0.7, 0.7]),  # dark gray
}


def visualize_pipeline(cached_frames: list, track_results: list = None) -> None:
    """Full pipeline visualizer using cached results + tracking.

    Colors by track ID (consistent across frames). Large white centroid
    markers on tracked objects. Info panel with track list.

    G = toggle: all / tracked only / objects only / all clusters / non-ground / ground
    N/B = navigate frames
    """
    import vispy
    from vispy.scene import visuals
    from vispy.scene.cameras import TurntableCamera
    from vispy.scene import SceneCanvas

    # Build track ID → consistent color mapping
    all_track_ids = set()
    if track_results:
        for tr in track_results:
            for t in tr["tracks"]:
                all_track_ids.add(t["track_id"])

    np.random.seed(42)
    track_colors = {}
    for tid in sorted(all_track_ids):
        track_colors[tid] = np.random.rand(3) * 0.6 + 0.4

    # Build per-frame track lookup
    frame_track_map = []
    if track_results:
        for tr in track_results:
            frame_track_map.append(tr["tracks"])
    else:
        frame_track_map = [[] for _ in cached_frames]

    print("Building visualization from cached results...")
    viz_frames = []
    for i, frame in enumerate(cached_frames):
        clipped = frame["clipped"]
        xyz = clipped[:, :3]
        n = len(clipped)
        colors = np.zeros((n, 4), dtype=np.float32)

        # Ground: green
        gm = frame["ground_mask"]
        colors[gm, 0] = 0.1
        colors[gm, 1] = 0.8
        colors[gm, 2] = 0.1
        colors[gm, 3] = 0.5

        # Non-ground unclustered: dim
        ng_indices = np.where(~gm)[0]
        colors[ng_indices, :3] = 0.15
        colors[ng_indices, 3] = 0.2

        # Background clusters: dark gray
        cluster_labels = frame["cluster_labels"]
        object_mask = np.zeros(n, dtype=bool)
        tracked_mask = np.zeros(n, dtype=bool)

        for cluster in frame["background_clusters"]:
            lbl = cluster["label_id"]
            cl_mask = cluster_labels == lbl
            cl_indices = ng_indices[cl_mask]
            colors[cl_indices, :3] = CLASS_COLORS["background"]
            colors[cl_indices, 3] = 0.7

        # Match object clusters to tracks by centroid proximity
        tracks_this_frame = frame_track_map[i]
        cluster_to_track = {}
        for cluster in frame["object_clusters"]:
            centroid = cluster["centroid"]
            best_dist = 2.0
            best_track = None
            for t in tracks_this_frame:
                dist = np.linalg.norm(centroid - t["position"])
                if dist < best_dist:
                    best_dist = dist
                    best_track = t
            if best_track is not None:
                cluster_to_track[cluster["label_id"]] = best_track

        # Object clusters: colored by CLASS
        for cluster in frame["object_clusters"]:
            lbl = cluster["label_id"]
            cl_mask = cluster_labels == lbl
            cl_indices = ng_indices[cl_mask]

            cls = cluster.get("s2_class_tracked") or cluster.get("s2_class", "background")
            color = CLASS_COLORS.get(cls, np.array([0.5, 0.5, 0.5]))

            if lbl in cluster_to_track:
                tracked_mask[cl_indices] = True

            colors[cl_indices, :3] = color
            colors[cl_indices, 3] = 1.0
            object_mask[cl_indices] = True

        # Counts
        class_counts = {}
        n_tracked = 0
        for cluster in frame["object_clusters"]:
            cls = cluster.get("s2_class", "unknown")
            class_counts[cls] = class_counts.get(cls, 0) + 1
            if cluster["label_id"] in cluster_to_track:
                n_tracked += 1

        viz_frames.append({
            "fname": frame["fname"],
            "xyz": xyz,
            "colors": colors,
            "ground_mask": gm,
            "object_mask": object_mask,
            "tracked_mask": tracked_mask,
            "n_total": n,
            "n_objects": len(frame["object_clusters"]),
            "n_tracked": n_tracked,
            "n_confirmed": len(tracks_this_frame),
            "n_background": len(frame["background_clusters"]),
            "class_counts": class_counts,
            "track_labels": [
                {
                    "track_id": cluster_to_track[c["label_id"]]["track_id"],
                    "class": cluster_to_track[c["label_id"]].get("class_label", "?"),
                    "position": c["centroid"],
                    "speed": cluster_to_track[c["label_id"]].get("speed", 0),
                }
                for c in frame["object_clusters"]
                if c["label_id"] in cluster_to_track
            ],
            "all_clusters": frame["all_clusters"],
            "cluster_classes": [c.get("s2_class_tracked") or c.get("s2_class", "background") for c in frame["all_clusters"]],
            "cluster_confs": [c.get("s2_confidence", 0.0) for c in frame["all_clusters"]],
        })

        counts_str = ", ".join(f"{k}={v}" for k, v in class_counts.items())
        print(
            f"  [{i+1}/{len(cached_frames)}] {frame['fname']}: "
            f"{counts_str}, tracked={n_tracked}/{len(frame['object_clusters'])}"
        )

    print("Launching viewer...")

    frame_idx = [0]
    show_mode = [0]
    show_bbox = [True]
    mode_names = ["all", "tracked only", "objects only", "all clusters", "non-ground", "ground only"]

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

    track_markers = visuals.Markers()
    view.add(track_markers)
    bbox_visual = visuals.Line(parent=view.scene, width=2.0, antialias=True)

    info = visuals.Text(
        text="", color="white", font_size=10,
        anchor_x="left", anchor_y="top",
        parent=canvas.scene,
    )
    info.pos = (10, 80)

    def update():
        f = viz_frames[frame_idx[0]]
        mode = show_mode[0]

        if mode == 0:
            show_xyz = f["xyz"]
            show_col = f["colors"]
        elif mode == 1:
            mask = f["tracked_mask"]
            show_xyz = f["xyz"][mask]
            show_col = f["colors"][mask]
        elif mode == 2:
            mask = f["object_mask"]
            show_xyz = f["xyz"][mask]
            show_col = f["colors"][mask]
        elif mode == 3:
            mask = f["colors"][:, 3] > 0.3
            show_xyz = f["xyz"][mask]
            show_col = f["colors"][mask]
        elif mode == 4:
            mask = ~f["ground_mask"]
            show_xyz = f["xyz"][mask]
            show_col = f["colors"][mask]
        else:
            mask = f["ground_mask"]
            show_xyz = f["xyz"][mask]
            show_col = np.zeros((mask.sum(), 4), dtype=np.float32)
            show_col[:, 1] = 0.8
            show_col[:, 3] = 0.6

        if len(show_xyz) == 0:
            scatter.set_data(np.zeros((1, 3), dtype=np.float32),
                             face_color=np.array([[0, 0, 0, 0]]),
                             edge_color=np.array([[0, 0, 0, 0]]), size=0.1)
        else:
            scatter.set_data(show_xyz, face_color=show_col, edge_color=show_col, size=1.0)

        # Track centroid markers
        labels = f["track_labels"]
        if labels and mode in [0, 1, 2]:
            positions = np.array([lbl["position"] for lbl in labels])
            positions[:, 2] += 0.5
            marker_colors = np.array([
                np.append(track_colors.get(lbl["track_id"], np.array([1.0, 1.0, 1.0])), 1.0)
                for lbl in labels
            ], dtype=np.float32)
            track_markers.set_data(
                positions, face_color=marker_colors,
                edge_color="black", size=8.0, edge_width=2.0,
            )
        else:
            track_markers.set_data(np.array([[0, 0, -100]]), size=0)

        # Bounding boxes
        vf = viz_frames[frame_idx[0]]
        if show_bbox[0] and vf.get("all_clusters") and mode != 3:
            bbox_pos, bbox_col, bbox_conn = build_bbox_lines(
                vf["all_clusters"], vf["cluster_classes"], vf["cluster_confs"])
            if len(bbox_pos) > 0:
                bbox_visual.set_data(pos=bbox_pos, color=bbox_col, connect=bbox_conn)
                bbox_visual.visible = True
            else:
                bbox_visual.visible = False
        else:
            bbox_visual.visible = False

        # Info panel with track list
        counts_str = ", ".join(f"{k}={v}" for k, v in f["class_counts"].items())
        track_lines = []
        if labels and mode in [0, 1, 2]:
            sorted_labels = sorted(labels, key=lambda x: x["track_id"])
            for lbl in sorted_labels[:15]:
                tid = lbl["track_id"]
                cls = lbl["class"][:3].upper()
                spd = lbl["speed"]
                track_lines.append(f"  T{tid:3d} {cls} {spd:.1f}m/s")
            if len(sorted_labels) > 15:
                track_lines.append(f"  ... +{len(sorted_labels)-15} more")

        track_text = "\n".join(track_lines) if track_lines else ""
        canvas.title = (
            f"Frame {frame_idx[0]+1}/{len(viz_frames)} — {mode_names[mode]} — "
            f"[N]ext [B]ack [G]toggle [T] toggle Bounding box"
        )
        info.text = (
            f"File: {f['fname']}\n"
            f"Objects: {f['n_objects']} | Tracked: {f['n_tracked']} | "
            f"Confirmed: {f['n_confirmed']}\n"
            f"Classes: {counts_str}\n"
            f"Mode: {mode_names[mode]} ({len(show_xyz)} pts)\n"
            f"---\n"
            f"Tracks:\n"
            f"{track_text}"
        )

    def on_key(event):
        if event.key == "N":
            frame_idx[0] = (frame_idx[0] + 1) % len(viz_frames)
            update()
        elif event.key == "B":
            frame_idx[0] = (frame_idx[0] - 1) % len(viz_frames)
            update()
        elif event.key == "G":
            show_mode[0] = (show_mode[0] + 1) % len(mode_names)
            update()
        elif event.key == "T":
            show_bbox[0] = not show_bbox[0]
            update()

    canvas.events.key_press.connect(on_key)
    canvas.events.draw.connect(lambda e: canvas.events.key_press.unblock()
                               if canvas.events.key_press.blocked() else None)
    update()
    canvas.app.run()


# =====================================================================
# Caching
# =====================================================================

CACHE_PATH = "cache/pipeline_results.pkl"


def make_cacheable(result: dict, frame_idx: int, fname: str) -> dict:
    """Convert pipeline result to a pickle-safe dict."""
    def slim_cluster(c):
        return {
            "centroid": c["centroid"],
            "n_points": c["n_points"],
            "z_range": c["z_range"],
            "xy_spread": c["xy_spread"],
            "label_id": c["label_id"],
            "pred": c.get("pred"),
            "s2_class": c.get("s2_class"),
            "s2_confidence": c.get("s2_confidence"),
            "s2_pred": c.get("s2_pred"),
            "points": c["points"],
            "xyz": c["xyz"],
            "was_split": c.get("was_split", False),
            "track_guided": c.get("track_guided", False),
            "s2_class_tracked": c.get("s2_class_tracked"),
        }

    return {
        "frame_idx": frame_idx,
        "fname": fname,
        "clipped": result["clipped"],
        "ground_mask": result["ground_mask"],
        "cluster_labels": result["cluster_labels"],
        "object_clusters": [slim_cluster(c) for c in result["object_clusters"]],
        "background_clusters": [slim_cluster(c) for c in result["background_clusters"]],
        "all_clusters": [slim_cluster(c) for c in result["all_clusters"]],
        "elapsed_ms": result["elapsed_ms"],
    }


def run_pipeline_with_tracking(data_dir, files, calibration, model):
    from optional_challenge.tracker import MultiObjectTracker

    timestamps = []
    for fname in files:
        timestamps.append(int(fname.replace(".bin", "")))
    dt = np.median(np.diff(timestamps)) / 1e9 if len(timestamps) > 1 else 0.1

    tracker = MultiObjectTracker(max_misses=3, confirm_hits=2, gate_distance=5.0, dt=dt)

    cached_frames = []
    track_results = []
    prev_tracks = None

    for i, fname in enumerate(files):
        fpath = os.path.join(data_dir, fname)
        pts = np.fromfile(fpath, dtype=np.float32).reshape(-1, 5)

        # Clip + ground removal
        clipped, clip_mask = filter_points(pts, calibration)
        ground_pts, non_ground, ground_mask = remove_ground(clipped, calibration)

        # Cluster
        clusters, cluster_labels = cluster_frame(non_ground, calibration)

        # Split with track guidance
        clusters = split_merged_clusters(clusters, prev_tracks=prev_tracks)

        # Merge engulfed clusters
        clusters = merge_engulfed_clusters(clusters)

        # Classify
        object_clusters, background_clusters = classify_clusters(clusters, model)

        # Track
        tracks = tracker.step(object_clusters)
        confirmed = tracker.get_confirmed_tracks()

        # Save tracks for next frame's split
        prev_tracks = [
            {"position": t.position.copy(), "class_label": t.cluster.get("s2_class", "unknown")}
            for t in confirmed
            if t.cluster.get("s2_class", "unknown") != "background"
        ]

        # Cache frame (from run_pipeline_cached)
        result = {
            "clipped": clipped,
            "ground_mask": ground_mask,
            "non_ground": non_ground,
            "cluster_labels": cluster_labels,
            "all_clusters": clusters,
            "object_clusters": object_clusters,
            "background_clusters": background_clusters,
            "elapsed_ms": 0,
            "fname": fname,
        }
        cached_result = make_cacheable(result, i, fname)
        cached_frames.append(cached_result)

        # Track results (from run_tracker_on_cached)
        track_info = []
        for t in confirmed:
            track_info.append({
                "track_id": t.track_id,
                "position": t.position.copy(),
                "velocity": t.velocity.copy(),
                "speed": t.speed,
                "class_label": t.class_label,
                "class_confidence": t.class_confidence,
                "hits": t.hits,
                "age": t.age,
                "n_points": t.cluster.get("n_points", 0),
            })

        track_results.append({
            "frame_idx": i,
            "fname": fname,
            "n_detections": len(object_clusters),
            "n_confirmed": len(confirmed),
            "n_all_tracks": len(tracks),
            "tracks": track_info,
        })

        # Print progress
        class_counts = {}
        for c in object_clusters:
            class_counts[c["s2_class"]] = class_counts.get(c["s2_class"], 0) + 1
        counts_str = ", ".join(f"{k}={v}" for k, v in class_counts.items()) or "none"
        print(f"  [{i + 1}/{len(files)}] {fname}: "
              f"{len(object_clusters)} objects ({counts_str}), "
              f"{len(confirmed)} tracks")

    # Save cache
    os.makedirs("cache", exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cached_frames, f)

    return cached_frames, track_results


def run_pipeline_cached(
    data_dir: str,
    files: list,
    calibration: GroundCalibration,
    model,
    force_rerun: bool = False,
) -> list:
    """Run pipeline on all frames with caching.

    :param force_rerun: If True, ignore cache and reprocess.
    :return: List of cached result dicts, one per frame.
    """
    os.makedirs("cache", exist_ok=True)

    if os.path.exists(CACHE_PATH) and not force_rerun:
        logger.info(f"Loading cached pipeline results from {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            cached = pickle.load(f)
        logger.info(f"Loaded {len(cached)} cached frames")
        return cached

    logger.info("Running full pipeline on all frames...")
    cached = []
    for i, fname in enumerate(files):
        fpath = os.path.join(data_dir, fname)
        pts = np.fromfile(fpath, dtype=np.float32).reshape(-1, 5)

        result = process_frame_full(pts, calibration, model)
        cached_result = make_cacheable(result, i, fname)
        cached.append(cached_result)

        class_counts = {}
        for c in result["object_clusters"]:
            cls = c["s2_class"]
            class_counts[cls] = class_counts.get(cls, 0) + 1
        counts_str = ", ".join(f"{k}={v}" for k, v in class_counts.items())
        print(
            f"  [{i+1}/{len(files)}] {fname}: "
            f"{len(result['object_clusters'])} objects ({counts_str}), "
            f"{len(result['background_clusters'])} background, "
            f"{result['elapsed_ms']:.0f}ms"
        )

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cached, f)
    logger.info(f"Pipeline results cached to {CACHE_PATH}")

    return cached


# =====================================================================
# Tracker integration
# =====================================================================

def run_tracker_on_cached(cached_frames: list) -> list:
    """Run tracker on cached pipeline results.

    :param cached_frames: List of cached result dicts.
    :return: List of per-frame track results.
    """
    from optional_challenge.tracker import MultiObjectTracker

    # Compute dt from filenames (timestamps)
    timestamps = []
    for f in cached_frames:
        ts = int(f["fname"].replace(".bin", ""))
        timestamps.append(ts)

    if len(timestamps) > 1:
        dt_ns = np.median(np.diff(timestamps))
        dt = dt_ns / 1e9
    else:
        dt = 0.1

    logger.info(f"Estimated dt: {dt:.4f}s ({1/dt:.1f} Hz)")

    tracker = MultiObjectTracker(
        max_misses=3,
        confirm_hits=2,
        gate_distance=5.0,
        dt=dt,
    )

    track_results = []

    for i, frame in enumerate(cached_frames):
        detections = frame["object_clusters"]
        tracks = tracker.step(detections)
        confirmed = tracker.get_confirmed_tracks()

        track_info = []
        for t in confirmed:
            track_info.append({
                "track_id": t.track_id,
                "position": t.position.copy(),
                "velocity": t.velocity.copy(),
                "speed": t.speed,
                "class_label": t.cluster.get("s2_class", "unknown"),
                "class_confidence": t.cluster.get("s2_confidence", 0.0),
                "hits": t.hits,
                "age": t.age,
                "n_points": t.cluster.get("n_points", 0),
            })

        track_results.append({
            "frame_idx": i,
            "fname": frame["fname"],
            "n_detections": len(detections),
            "n_confirmed": len(confirmed),
            "n_all_tracks": len(tracks),
            "tracks": track_info,
        })

        class_counts = {}
        for t in track_info:
            cls = t["class_label"]
            class_counts[cls] = class_counts.get(cls, 0) + 1
        counts_str = ", ".join(f"{k}={v}" for k, v in class_counts.items()) or "none"
        print(
            f"  Frame {i+1}: {len(detections)} detections → "
            f"{len(confirmed)} confirmed tracks ({counts_str})"
        )

    return track_results


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Full perception pipeline")
    parser.add_argument("--force-rerun", action="store_true",
                        help="Ignore cache, reprocess all frames")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualization")
    parser.add_argument("--model-path", type=str, default="models/rf_classifier_19feat.pkl",
                        help="Path to trained model")
    args = parser.parse_args()

    data_dir = os.path.abspath(DATA_DIR)
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".bin")])

    if not files:
        logger.error(f"No .bin files in {data_dir}")
        sys.exit(1)

    # Load single-stage model
    model_path = args.model_path
    if not os.path.exists(model_path):
        model_path = "models/rf_classifier.pkl"

    model = load_model(model_path)

    # Set feature mode from loaded model
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    set_feature_mode(data["feature_mode"])

    # Calibrate ground
    logger.info(f"Calibrating on {files[0]}...")
    first_frame = np.fromfile(os.path.join(data_dir, files[0]),
                              dtype=np.float32).reshape(-1, 5)
    calibration = calibrate_ground(first_frame)
    print(f"Calibration: {calibration}\n")

    # --- Run pipeline (cached) ---
    print(f"{'=' * 70}")
    print("PIPELINE")
    print(f"{'=' * 70}")
    cached_frames, track_results = run_pipeline_with_tracking(
        data_dir, files, calibration, model)
    print(f"{'=' * 70}")

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    all_track_ids = set()
    for tr in track_results:
        for t in tr["tracks"]:
            all_track_ids.add(t["track_id"])
    print(f"Total unique tracks across {len(files)} frames: {len(all_track_ids)}")

    final_frame = track_results[-1]
    class_counts = {}
    for t in final_frame["tracks"]:
        cls = t["class_label"]
        class_counts[cls] = class_counts.get(cls, 0) + 1
    print(f"Final frame confirmed tracks: {class_counts}")
    print(f"{'=' * 70}")

    # --- Visualize ---
    if not args.no_viz:
        visualize_pipeline(cached_frames, track_results)
