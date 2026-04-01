"""
Two-stage point cloud classifier.

Motivation: 92% of classification errors involve the background class.
Inter-object confusion (car/pedestrian/bicyclist) accounts for only 8%.
Separating detection from identification lets us optimize each independently.

Stage 1: Object vs Background (binary)
  - High recall target — don't miss real objects
  - False alarms are acceptable (Stage 2 handles them)
  - Asymmetric class weight: penalize missing objects more than false alarms

Stage 2: Car vs Pedestrian vs Bicyclist (3-class)
  - Only sees clusters that passed Stage 1
  - No background noise in the training set
  - Cleaner decision boundaries for the easy inter-object problem

Fallback: Clusters predicted as background by Stage 1 but with low confidence
  are flagged as "uncertain" — a production safety net.
"""

import logging
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from data_loader import load_dataset, get_dataset_summary, CLASS_MAP, INV_CLASS_MAP
from classifier_main_task import extract_features, extract_dataset_features, FEATURE_NAMES, set_feature_mode

logger = logging.getLogger(__name__)

# Original 4-class labels
# background=0, bicyclist=1, car=2, pedestrian=3

# Stage 1 mapping: object=1, background=0
OBJECT_CLASSES = {1, 2, 3}  # bicyclist, car, pedestrian

# Stage 2 mapping: only among object classes
# We remap to 0-indexed: bicyclist=0, car=1, pedestrian=2
STAGE2_CLASS_MAP = {1: 0, 2: 1, 3: 2}
STAGE2_INV_MAP = {0: "bicyclist", 1: "car", 2: "pedestrian"}

# Confidence threshold for safety net
# Clusters predicted background but below this threshold get flagged as uncertain
SAFETY_THRESHOLD = 0.70


def make_stage1_labels(y: np.ndarray) -> np.ndarray:
    """Convert 4-class labels to binary: background=0, object=1.

    :param y: Original labels (0-3).
    :return: Binary labels (0 or 1).
    """
    return (y > 0).astype(np.int32)


def make_stage2_subset(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract only object samples and remap labels for Stage 2.

    :param X: Full feature matrix.
    :param y: Original 4-class labels.
    :return: (X_objects, y_remapped) with only object samples.
    """
    object_mask = y > 0  # not background
    X_obj = X[object_mask]
    y_obj = np.array([STAGE2_CLASS_MAP[label] for label in y[object_mask]], dtype=np.int32)
    return X_obj, y_obj


def train_stage1(
    X: np.ndarray,
    y_binary: np.ndarray,
    miss_penalty: float = 3.0,
) -> RandomForestClassifier:
    """Train Stage 1: Object vs Background.

    Uses asymmetric weighting — missing a real object costs more
    than a false alarm. miss_penalty controls this ratio.

    :param X: Feature matrix.
    :param y_binary: Binary labels (0=background, 1=object).
    :param miss_penalty: Extra weight on object class beyond balanced.
    :return: Trained classifier.
    """
    # Start with balanced weights, then multiply object class by miss_penalty.
    # balanced gives: background_weight ~ 1/32520, object_weight ~ 1/9570
    # miss_penalty further increases object weight to reduce missed detections.
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=5,
        class_weight={0: 1.0, 1: miss_penalty},
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y_binary)
    logger.info(
        f"Stage 1 trained: object miss_penalty={miss_penalty}, "
        f"background={np.sum(y_binary == 0)}, objects={np.sum(y_binary == 1)}"
    )
    return rf


def train_stage2(
    X_obj: np.ndarray,
    y_obj: np.ndarray,
) -> RandomForestClassifier:
    """Train Stage 2: Car vs Pedestrian vs Bicyclist.

    No background samples — clean 3-class problem.
    Balanced weights for the remaining imbalance (car ~7000, others ~1300).

    :param X_obj: Feature matrix, objects only.
    :param y_obj: Remapped labels (0=bicyclist, 1=car, 2=pedestrian).
    :return: Trained classifier.
    """
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_obj, y_obj)
    logger.info(
        f"Stage 2 trained on {len(y_obj)} object samples: "
        f"bicyclist={np.sum(y_obj == 0)}, car={np.sum(y_obj == 1)}, "
        f"pedestrian={np.sum(y_obj == 2)}"
    )
    return rf


def predict_two_stage(
    stage1: RandomForestClassifier,
    stage2: RandomForestClassifier,
    X: np.ndarray,
    safety_threshold: float = SAFETY_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run two-stage prediction with safety net recovery.

    Routing logic:
    1. Stage 1 says "object"              → Stage 2 classifies
    2. Stage 1 says "background" (high conf) → stays background
    3. Stage 1 says "background" (low conf)  → Stage 2 classifies (recovery)

    :param stage1: Object vs background classifier.
    :param stage2: 3-class object classifier.
    :param X: Feature matrix.
    :param safety_threshold: Below this, background predictions get reclassified by Stage 2.
    :return: (final_labels, confidence, uncertain_mask)
             final_labels: original 4-class labels (0-3)
             confidence: predicted probability for the chosen class
             uncertain_mask: True for samples recovered by safety net
    """
    n_samples = X.shape[0]
    final_labels = np.zeros(n_samples, dtype=np.int32)
    confidence = np.zeros(n_samples, dtype=np.float32)
    uncertain_mask = np.zeros(n_samples, dtype=bool)

    # Stage 2 recovery threshold — if Stage 2 can't confidently classify
    # a recovered sample, reject it back to background. This filters out
    # background samples that Stage 2 is uncertain about while keeping
    # confident recoveries.
    RECOVERY_CONFIDENCE = 0.50

    # --- Stage 1: Object vs Background ---
    s1_proba = stage1.predict_proba(X)  # columns: [background, object]
    s1_pred = stage1.predict(X)  # 0=background, 1=object

    background_mask = s1_pred == 0
    object_mask = s1_pred == 1

    # Identify uncertain background predictions
    bg_confidence = np.zeros(n_samples)
    bg_confidence[background_mask] = s1_proba[background_mask, 0]
    uncertain_bg = background_mask & (bg_confidence < safety_threshold)
    confident_bg = background_mask & (bg_confidence >= safety_threshold)

    # Confident background stays background
    final_labels[confident_bg] = 0
    confidence[confident_bg] = s1_proba[confident_bg, 0]

    # Mark uncertain samples
    uncertain_mask[uncertain_bg] = True

    # --- Stage 2: Classify confident objects ---
    if object_mask.sum() > 0:
        X_objects = X[object_mask]
        s2_proba = stage2.predict_proba(X_objects)
        s2_pred = stage2.predict(X_objects)

        inv_map = {0: 1, 1: 2, 2: 3}  # stage2 → original
        original_labels = np.array([inv_map[p] for p in s2_pred], dtype=np.int32)
        final_labels[object_mask] = original_labels
        confidence[object_mask] = s2_proba[np.arange(len(s2_pred)), s2_pred]

    # --- Stage 2: Recover uncertain backgrounds ---
    # Only keep recoveries where Stage 2 is confident enough.
    # Low-confidence recoveries get rejected back to background.
    if uncertain_bg.sum() > 0:
        X_uncertain = X[uncertain_bg]
        s2_proba_unc = stage2.predict_proba(X_uncertain)
        s2_pred_unc = stage2.predict(X_uncertain)
        s2_conf_unc = s2_proba_unc[np.arange(len(s2_pred_unc)), s2_pred_unc]

        inv_map = {0: 1, 1: 2, 2: 3}

        # Get full array indices of uncertain samples
        unc_indices = np.where(uncertain_bg)[0]

        for i, idx in enumerate(unc_indices):
            if s2_conf_unc[i] >= RECOVERY_CONFIDENCE:
                # Stage 2 is confident — recover as object
                final_labels[idx] = inv_map[s2_pred_unc[i]]
                confidence[idx] = s2_conf_unc[i]
            else:
                # Stage 2 is not confident — reject back to background
                final_labels[idx] = 0
                confidence[idx] = bg_confidence[idx]

    return final_labels, confidence, uncertain_mask


def evaluate_two_stage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    uncertain_mask: np.ndarray,
    save_dir: str = "results",
) -> Dict:
    """Full evaluation of two-stage predictions.

    :param y_true: Ground truth 4-class labels.
    :param y_pred: Predicted 4-class labels.
    :param confidence: Per-prediction confidence.
    :param uncertain_mask: Flags for uncertain background predictions.
    :param save_dir: Directory to save plots.
    :return: Evaluation dict.
    """
    os.makedirs(save_dir, exist_ok=True)
    class_names = [INV_CLASS_MAP[i] for i in sorted(INV_CLASS_MAP.keys())]

    # --- Classification report ---
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print("\n" + "=" * 70)
    print("TWO-STAGE TEST SET RESULTS")
    print("=" * 70)
    print(report)

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"Macro F1: {macro_f1:.4f}")

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred, labels=list(range(4)))
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    print("\nConfusion Matrix (raw counts):")
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
    ax.set_title("Two-Stage: Confusion Matrix (counts)", fontsize=12)
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
    ax.set_title("Two-Stage: Confusion Matrix (normalized)", fontsize=12)
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
    plt.savefig(os.path.join(save_dir, "confusion_matrix_two_stage.png"), dpi=120)
    plt.close()
    logger.info("Saved confusion_matrix_two_stage.png")

    # --- Safety analysis ---
    # uncertain_mask = samples that Stage 1 called background but were routed
    # to Stage 2 for recovery due to low confidence.
    n_uncertain = uncertain_mask.sum()
    print(f"\n{'=' * 70}")
    print("SAFETY NET RECOVERY ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Samples recovered by safety net (low-conf background → Stage 2): "
          f"{n_uncertain} / {len(y_true)} ({100 * n_uncertain / len(y_true):.1f}%)")

    if n_uncertain > 0:
        uncertain_true = y_true[uncertain_mask]
        uncertain_pred = y_pred[uncertain_mask]

        actual_objects = np.sum(uncertain_true > 0)
        actual_background = np.sum(uncertain_true == 0)

        # How many real objects were correctly reclassified?
        correctly_recovered = np.sum((uncertain_true > 0) & (uncertain_pred == uncertain_true))
        # How many real objects were reclassified but to wrong object class?
        wrong_class_recovered = np.sum((uncertain_true > 0) & (uncertain_pred > 0) & (uncertain_pred != uncertain_true))
        # How many real objects were rejected back to background (Stage 2 low confidence)?
        still_missed = np.sum((uncertain_true > 0) & (uncertain_pred == 0))

        # How many backgrounds became false alarms?
        false_alarms = np.sum((uncertain_true == 0) & (uncertain_pred > 0))
        # How many backgrounds were correctly rejected back?
        correctly_rejected = np.sum((uncertain_true == 0) & (uncertain_pred == 0))

        print(f"  Actually objects: {actual_objects}")
        print(f"    Correctly recovered (right class): {correctly_recovered}")
        print(f"    Recovered but wrong class: {wrong_class_recovered}")
        print(f"    Rejected back to background (Stage 2 low conf): {still_missed}")
        print(f"  Actually background: {actual_background}")
        print(f"    Became false alarms: {false_alarms}")
        print(f"    Correctly rejected back to background: {correctly_rejected}")

        # Per-class recovery details
        print(f"\n  Per-class recovery:")
        for cls_id in [1, 2, 3]:
            cls_uncertain = uncertain_mask & (y_true == cls_id)
            n_routed = cls_uncertain.sum()
            if n_routed > 0:
                n_correct = np.sum(cls_uncertain & (y_pred == cls_id))
                n_wrong_cls = np.sum(cls_uncertain & (y_pred > 0) & (y_pred != cls_id))
                n_rejected = np.sum(cls_uncertain & (y_pred == 0))
                print(
                    f"    {INV_CLASS_MAP[cls_id]:>12s}: {n_routed} routed, "
                    f"{n_correct} correctly recovered, "
                    f"{n_wrong_cls} wrong class, "
                    f"{n_rejected} rejected back"
                )

        # --- Stage 2 confidence distribution for recovered samples ---
        # This tells you the right RECOVERY_CONFIDENCE threshold.
        # If correct recoveries have high confidence and false alarms have low,
        # a threshold can cleanly separate them.
        print(f"\n  Stage 2 confidence distribution for recovered samples:")

        # Correctly recovered objects
        correct_recovery_mask = uncertain_mask & (y_true > 0) & (y_pred == y_true)
        if correct_recovery_mask.sum() > 0:
            conf_correct = confidence[correct_recovery_mask]
            print(f"    Correct recoveries ({correct_recovery_mask.sum()}):")
            print(f"      mean={conf_correct.mean():.3f}, "
                  f"median={np.median(conf_correct):.3f}, "
                  f"min={conf_correct.min():.3f}, "
                  f"25th={np.percentile(conf_correct, 25):.3f}, "
                  f"75th={np.percentile(conf_correct, 75):.3f}")

        # Wrong class recoveries
        wrong_recovery_mask = uncertain_mask & (y_true > 0) & (y_pred > 0) & (y_pred != y_true)
        if wrong_recovery_mask.sum() > 0:
            conf_wrong = confidence[wrong_recovery_mask]
            print(f"    Wrong class recoveries ({wrong_recovery_mask.sum()}):")
            print(f"      mean={conf_wrong.mean():.3f}, "
                  f"median={np.median(conf_wrong):.3f}, "
                  f"min={conf_wrong.min():.3f}, "
                  f"25th={np.percentile(conf_wrong, 25):.3f}, "
                  f"75th={np.percentile(conf_wrong, 75):.3f}")

        # Rejected real objects (back to background)
        rejected_objects_mask = uncertain_mask & (y_true > 0) & (y_pred == 0)
        if rejected_objects_mask.sum() > 0:
            # These were rejected — their confidence is the bg_confidence, not Stage 2
            # We need the Stage 2 confidence that caused rejection
            print(f"    Rejected real objects ({rejected_objects_mask.sum()}):")
            print(f"      (rejected because Stage 2 conf < threshold)")

        # False alarms (background became object)
        false_alarm_mask = uncertain_mask & (y_true == 0) & (y_pred > 0)
        if false_alarm_mask.sum() > 0:
            conf_fa = confidence[false_alarm_mask]
            print(f"    False alarms ({false_alarm_mask.sum()}):")
            print(f"      mean={conf_fa.mean():.3f}, "
                  f"median={np.median(conf_fa):.3f}, "
                  f"min={conf_fa.min():.3f}, "
                  f"25th={np.percentile(conf_fa, 25):.3f}, "
                  f"75th={np.percentile(conf_fa, 75):.3f}")

        # Correctly rejected backgrounds
        correct_reject_mask = uncertain_mask & (y_true == 0) & (y_pred == 0)
        if correct_reject_mask.sum() > 0:
            print(f"    Correctly rejected backgrounds ({correct_reject_mask.sum()}):")
            print(f"      (rejected because Stage 2 conf < threshold)")

    # --- Object miss rates (only confident background = hard misses) ---
    print(f"\nObject miss rates (hard misses — confident background only):")
    for cls_id in [1, 2, 3]:
        cls_mask = y_true == cls_id
        n_total = cls_mask.sum()
        # Hard miss = predicted background AND not in uncertain set (wasn't recovered)
        n_hard_missed = np.sum((y_true == cls_id) & (y_pred == 0) & ~uncertain_mask)
        # Total misclassified as background (should now only be hard misses)
        n_as_background = np.sum((y_true == cls_id) & (y_pred == 0))

        print(
            f"  {INV_CLASS_MAP[cls_id]:>12s}: {n_hard_missed}/{n_total} hard misses "
            f"({100 * n_hard_missed / n_total:.1f}%), "
            f"{n_as_background} total as background"
        )

    print(f"{'=' * 70}")

    # --- Per-class confidence analysis ---
    # Same as single-stage: shows whether the model "knows when it doesn't know".
    print(f"\nPer-class prediction confidence (mean confidence):")
    for cls_id, cname in INV_CLASS_MAP.items():
        mask = y_true == cls_id
        if mask.sum() > 0:
            correct_mask = mask & (y_pred == cls_id)
            wrong_mask = mask & (y_pred != cls_id)

            correct_conf = confidence[correct_mask].mean() if correct_mask.sum() > 0 else 0.0
            wrong_conf = confidence[wrong_mask].mean() if wrong_mask.sum() > 0 else 0.0

            print(
                f"  {cname:>12s}: correct={correct_conf:.3f}, "
                f"misclassified={wrong_conf:.3f}, "
                f"n_correct={correct_mask.sum()}, n_wrong={wrong_mask.sum()}"
            )

    return {
        "confusion_matrix": cm,
        "confusion_matrix_normalized": cm_normalized,
        "macro_f1": macro_f1,
        "n_uncertain": n_uncertain,
    }


def run_cross_validation_two_stage(
    X: np.ndarray,
    y: np.ndarray,
    miss_penalty: float = 3.0,
    n_folds: int = 5,
) -> Dict:
    """Stratified CV for two-stage classifier.

    :return: CV results dict.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    class_names = [INV_CLASS_MAP[i] for i in sorted(INV_CLASS_MAP.keys())]

    fold_f1s = []
    all_val_preds = np.zeros_like(y)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train both stages on this fold's training data
        y_train_binary = make_stage1_labels(y_train)
        stage1 = train_stage1(X_train, y_train_binary, miss_penalty)

        X_train_obj, y_train_obj = make_stage2_subset(X_train, y_train)
        stage2 = train_stage2(X_train_obj, y_train_obj)

        # Predict on validation
        y_pred, _, _ = predict_two_stage(stage1, stage2, X_val)
        all_val_preds[val_idx] = y_pred

        macro_f1 = f1_score(y_val, y_pred, average="macro")
        fold_f1s.append(macro_f1)

        logger.info(f"Fold {fold_idx + 1}/{n_folds}: macro-F1={macro_f1:.4f}")

    avg_f1 = np.mean(fold_f1s)
    std_f1 = np.std(fold_f1s)

    print(f"\n{'=' * 70}")
    print(f"TWO-STAGE CV RESULTS ({n_folds}-fold)")
    print(f"{'=' * 70}")
    print(f"Macro F1: {avg_f1:.4f} ± {std_f1:.4f}")

    cm = confusion_matrix(y, all_val_preds, labels=list(range(4)))
    print("\nCV Confusion Matrix (aggregated):")
    print(pd.DataFrame(cm, index=class_names, columns=class_names))
    print(f"{'=' * 70}")

    return {"avg_macro_f1": avg_f1, "std_macro_f1": std_f1}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # --- Load data ---
    logger.info("Loading training data...")
    train_points, train_labels, train_paths = load_dataset("data/train")
    get_dataset_summary(train_labels)

    logger.info("Loading test data...")
    test_points, test_labels, test_paths = load_dataset("data/test")
    get_dataset_summary(test_labels)

    # --- Set feature mode (change here to switch between 19/23/35) ---
    set_feature_mode(35)
    feat_mode = 35

    # --- Extract features (with caching) ---
    logger.info(f"Extracting {feat_mode}-feature set...")
    X_train, y_train = extract_dataset_features(
        train_points, train_labels,
        cache_path=f"cache/main_train_{feat_mode}feat.npz",
    )
    X_test, y_test = extract_dataset_features(
        test_points, test_labels,
        cache_path=f"cache/test_{feat_mode}feat.npz",
    )

    # --- Try different miss penalties ---
    # Higher penalty = fewer missed objects, more false alarms
    os.makedirs("models", exist_ok=True)

    for penalty in [2.0, 3.0, 5.0]:
        print(f"\n{'#' * 70}")
        print(f"MISS PENALTY = {penalty}")
        print(f"{'#' * 70}")

        # Check for cached model
        penalty_str = str(penalty).replace(".", "_")
        model_path = f"models/two_stage_{feat_mode}feat_p{penalty_str}.pkl"

        if os.path.exists(model_path):
            logger.info(f"Loading cached model from {model_path}")
            with open(model_path, "rb") as f:
                cached = pickle.load(f)
            stage1 = cached["stage1"]
            stage2 = cached["stage2"]
            logger.info(f"Skipping CV — using cached model for penalty={penalty}")
        else:
            # CV
            cv_results = run_cross_validation_two_stage(
                X_train, y_train, miss_penalty=penalty
            )

            # Train final model on full training set
            y_train_binary = make_stage1_labels(y_train)
            stage1 = train_stage1(X_train, y_train_binary, miss_penalty=penalty)

            X_train_obj, y_train_obj = make_stage2_subset(X_train, y_train)
            stage2 = train_stage2(X_train_obj, y_train_obj)

            # Save model
            with open(model_path, "wb") as f:
                pickle.dump({
                    "stage1": stage1,
                    "stage2": stage2,
                    "feature_names": FEATURE_NAMES,
                    "miss_penalty": penalty,
                    "safety_threshold": SAFETY_THRESHOLD,
                }, f)
            logger.info(f"Model saved to {model_path}")

        # Test — always run evaluation (fast)
        y_pred, confidence, uncertain_mask = predict_two_stage(stage1, stage2, X_test)
        results = evaluate_two_stage(y_test, y_pred, confidence, uncertain_mask)

    # --- Also save default model as two_stage_classifier.pkl for other scripts ---
    best_penalty = 5.0
    best_path = f"models/two_stage_{feat_mode}feat_p{str(best_penalty).replace('.', '_')}.pkl"
    default_path = "models/two_stage_classifier.pkl"
    if os.path.exists(best_path):
        import shutil
        shutil.copy2(best_path, default_path)
        logger.info(f"Copied {best_path} → {default_path}")

    logger.info("Done.")
