"""
Visual analysis of point cloud classification results.

Generates diagnostic images showing:
  1. High-confidence misclassifications (model confidently wrong — most valuable)
  2. Low-confidence correct predictions (borderline cases — almost wrong)
  3. High-confidence correct predictions (sanity check — what "good" looks like)

Each image: 2-panel view (top-down XY + side XZ) with metadata overlay.

Usage:
  python visualize_errors_interactive.py --features 19 --n-per-bucket 40
  python visualize_errors_interactive.py --features 35 --n-per-bucket 20 --output-dir results/visual_35feat
"""

import argparse
import logging
import os
import pickle
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from data_loader import load_dataset, get_dataset_summary, INV_CLASS_MAP
from classifier_main_task import (
    set_feature_mode,
    extract_dataset_features,
    FEATURE_MODE,
)

logger = logging.getLogger(__name__)


def plot_cluster(
    points: np.ndarray,
    true_label: str,
    pred_label: str,
    confidence: float,
    num_points: int,
    sample_idx: int,
    save_path: str,
) -> None:
    """Render a single cluster as 2-panel diagnostic image.

    Left: top-down (XY). Right: side view (XZ).
    Minimal chrome, max information density.
    """
    xyz = points[:, :3]
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    is_correct = (true_label == pred_label)
    title_color = "green" if is_correct else "red"
    status = "CORRECT" if is_correct else "WRONG"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor("black")
    for ax in axes:
        ax.set_facecolor("black")
        ax.tick_params(colors="white", labelsize=7)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")

    # --- Top-down: XY ---
    ax = axes[0]
    ax.scatter(x, y, s=1.5, c=z, cmap="viridis", alpha=0.8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Top-down (XY)", fontsize=10)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)

    # --- Side view: XZ ---
    ax = axes[1]
    ax.scatter(x, z, s=1.5, c=z, cmap="viridis", alpha=0.8)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("Side (XZ)", fontsize=10)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)

    fig.suptitle(
        f"[{status}]  true={true_label}  pred={pred_label}  "
        f"conf={confidence:.3f}  n={num_points}  idx={sample_idx}",
        fontsize=11,
        color=title_color,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def run_visual_analysis(
    test_points: List[np.ndarray],
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    output_dir: str = "results/visual_analysis",
    n_per_bucket: int = 40,
) -> None:
    """Generate diagnostic images for classification analysis.

    Three buckets:
      1. misclassified/ — sorted by descending confidence (confident mistakes first)
      2. correct_low_conf/ — sorted by ascending confidence (barely right)
      3. correct_high_conf/ — sorted by descending confidence (sanity check, capped at n//4)

    :param test_points: List of (N_i, 3) point cloud arrays.
    :param y_test: (n_samples,) true labels.
    :param y_pred: (n_samples,) predicted labels.
    :param y_proba: (n_samples, n_classes) prediction probabilities.
    :param output_dir: Root directory for output images.
    :param n_per_bucket: Max images per confusion pair (misclassified) or per class (correct).
    """
    os.makedirs(output_dir, exist_ok=True)
    class_names = {k: v for k, v in INV_CLASS_MAP.items()}
    n_samples = len(y_test)

    # Precompute per-sample metadata
    pred_conf = np.array([y_proba[i, y_pred[i]] for i in range(n_samples)])
    is_correct = (y_test == y_pred)

    # =========================================================
    # 1. MISCLASSIFIED — grouped by (true, pred) pair
    #    Sorted by confidence DESC — high-confidence mistakes first
    # =========================================================
    misc_dir = os.path.join(output_dir, "misclassified")
    misc_indices = np.where(~is_correct)[0]
    logger.info(f"Misclassified: {len(misc_indices)} total samples")

    # Group by (true_class, pred_class)
    confusion_pairs = {}
    for idx in misc_indices:
        true_name = class_names[y_test[idx]]
        pred_name = class_names[y_pred[idx]]
        key = f"{true_name}_as_{pred_name}"
        if key not in confusion_pairs:
            confusion_pairs[key] = []
        confusion_pairs[key].append(idx)

    total_misc = 0
    for pair_name, indices in sorted(confusion_pairs.items()):
        pair_dir = os.path.join(misc_dir, pair_name)
        os.makedirs(pair_dir, exist_ok=True)

        # Sort by confidence descending — confident mistakes first
        indices_sorted = sorted(indices, key=lambda i: pred_conf[i], reverse=True)
        indices_sorted = indices_sorted[:n_per_bucket]

        for rank, idx in enumerate(indices_sorted):
            pts = test_points[idx]
            fname = f"conf{pred_conf[idx]:.2f}_n{len(pts)}_idx{idx}.png"
            plot_cluster(
                points=pts,
                true_label=class_names[y_test[idx]],
                pred_label=class_names[y_pred[idx]],
                confidence=pred_conf[idx],
                num_points=len(pts),
                sample_idx=idx,
                save_path=os.path.join(pair_dir, fname),
            )
        total_misc += len(indices_sorted)
        logger.info(f"  {pair_name}: {len(indices_sorted)} images (of {len(indices)})")

    # =========================================================
    # 2. CORRECT LOW-CONFIDENCE — per class
    #    Sorted by confidence ASC — least confident correct first
    # =========================================================
    low_conf_dir = os.path.join(output_dir, "correct_low_conf")
    correct_indices = np.where(is_correct)[0]

    total_low = 0
    for cls_id, cls_name in sorted(class_names.items()):
        cls_dir = os.path.join(low_conf_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)

        cls_correct = correct_indices[y_test[correct_indices] == cls_id]
        # Sort by confidence ascending — borderline cases first
        cls_sorted = sorted(cls_correct, key=lambda i: pred_conf[i])
        cls_sorted = cls_sorted[:n_per_bucket]

        for idx in cls_sorted:
            pts = test_points[idx]
            fname = f"conf{pred_conf[idx]:.2f}_n{len(pts)}_idx{idx}.png"
            plot_cluster(
                points=pts,
                true_label=cls_name,
                pred_label=cls_name,
                confidence=pred_conf[idx],
                num_points=len(pts),
                sample_idx=idx,
                save_path=os.path.join(cls_dir, fname),
            )
        total_low += len(cls_sorted)
        logger.info(f"  correct_low_conf/{cls_name}: {len(cls_sorted)} images")

    # =========================================================
    # 3. CORRECT HIGH-CONFIDENCE — per class (sanity check)
    #    Fewer images — just verify the model is right for the right reasons
    # =========================================================
    high_conf_dir = os.path.join(output_dir, "correct_high_conf")
    n_high = max(n_per_bucket // 4, 5)

    total_high = 0
    for cls_id, cls_name in sorted(class_names.items()):
        cls_dir = os.path.join(high_conf_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)

        cls_correct = correct_indices[y_test[correct_indices] == cls_id]
        # Sort by confidence descending
        cls_sorted = sorted(cls_correct, key=lambda i: pred_conf[i], reverse=True)
        cls_sorted = cls_sorted[:n_high]

        for idx in cls_sorted:
            pts = test_points[idx]
            fname = f"conf{pred_conf[idx]:.2f}_n{len(pts)}_idx{idx}.png"
            plot_cluster(
                points=pts,
                true_label=cls_name,
                pred_label=cls_name,
                confidence=pred_conf[idx],
                num_points=len(pts),
                sample_idx=idx,
                save_path=os.path.join(cls_dir, fname),
            )
        total_high += len(cls_sorted)
        logger.info(f"  correct_high_conf/{cls_name}: {len(cls_sorted)} images")

    logger.info(
        f"Visual analysis complete: {total_misc} misclassified + "
        f"{total_low} low-conf correct + {total_high} high-conf correct = "
        f"{total_misc + total_low + total_high} total images"
    )
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Visual analysis of classification results")
    parser.add_argument(
        "--features", type=int, choices=[19, 23, 35], default=19,
        help="Feature mode (must match a trained model)",
    )
    parser.add_argument(
        "--n-per-bucket", type=int, default=40,
        help="Max images per confusion pair or per class (default: 40)",
    )
    args = parser.parse_args()

    set_feature_mode(args.features)

    output_dir = f"results/visual_analysis_{args.features}feat"
    model_path = f"models/rf_classifier_{args.features}feat.pkl"

    # --- Load model ---
    if not os.path.exists(model_path):
        # Fall back to default
        model_path = "models/rf_classifier.pkl"
    logger.info(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]
    logger.info(f"Model loaded: {saved.get('feature_mode', '?')} features")

    # --- Load test data ---
    logger.info("Loading test data...")
    test_points, test_labels, test_paths = load_dataset("data/test")
    get_dataset_summary(test_labels)

    # --- Extract features ---
    cache_path = f"cache/main_test_{args.features}feat.npz"
    logger.info(f"Extracting test features ({args.features} features)...")
    X_test, y_test = extract_dataset_features(
        test_points, test_labels,
        cache_path=cache_path,
    )

    # --- Predict ---
    logger.info("Running predictions...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # --- Generate visual analysis ---
    logger.info(f"Generating visual analysis (n_per_bucket={args.n_per_bucket})...")
    run_visual_analysis(
        test_points=test_points,
        y_test=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        output_dir=output_dir,
        n_per_bucket=args.n_per_bucket,
    )

    logger.info("Done.")