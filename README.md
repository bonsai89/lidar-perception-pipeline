# LiDAR Point Cloud Perception Pipeline

End-to-end 3D perception pipeline for infrastructure-mounted LiDAR: ground removal, clustering, classification, and multi-object tracking.

Built as a self-contained Python implementation demonstrating classical perception techniques on real 128-beam LiDAR data from an urban intersection scene.

<p align="center">
  <img src="pipeline_demo.gif" alt="Pipeline Demo" width="720">
</p>

<p align="center"><i>10 sequential LiDAR frames processed through the full pipeline. Blue = car, red = pedestrian, yellow = bicyclist, gray = background, green = ground. Bounding boxes show classified objects, dot markers show tracked object IDs.</i></p>

> 📹 [Full resolution video (pipeline_demo.mp4)](pipeline_demo.mp4)

*See [docs/architecture.md](docs/architecture.md) for the full pipeline diagram and feature group breakdown.*

---

## What This Project Does

Takes raw LiDAR point clouds (~184k points per frame) and produces classified, tracked objects (cars, pedestrians, bicyclists) with persistent IDs across 10 sequential frames.

**Pipeline stages:**

1. **Ground Removal** — RANSAC calibration + polar grid with adaptive thresholds
2. **BEV Clustering** — Bird's Eye View grid projection + connected components
3. **Classification** — Random Forest on 23 handcrafted geometric features (0.82 macro-F1)
4. **Multi-Object Tracking** — Kalman filter + Hungarian assignment with Mahalanobis gating

Each stage was developed iteratively. The [reports](docs/reports/) document 6 ground removal iterations, multiple clustering approaches, and detailed ablation studies showing the contribution of each feature.

---

## Key Technical Highlights

**Ground Removal — Calibrate Once, Apply Fast**

The sensor is infrastructure-mounted (fixed position, fixed tilt). RANSAC runs once on nearby points to find the ground normal, then a rotation matrix aligns the ground plane with the z-axis. Per-frame ground removal uses a polar grid with distance-adaptive deviation thresholds. Calibration transfers across all frames. Latency: ~30-80ms per frame vs 6-7s with per-frame RANSAC.

**PCA Yaw-Invariant Features**

Objects face arbitrary directions relative to the sensor. Raw bounding box dimensions (x_range, y_range) change with orientation — a car at 45° looks square. 2D PCA alignment rotates each cluster's horizontal coordinates to its principal axes before measuring dimensions, making features orientation-independent.

**Vertical Profile Features for Pedestrian–Bicyclist Separation**

These two classes have 100% overlap on basic geometric features (z_range, xy_spread, point count). The key discriminator is the vertical point distribution: pedestrians have roughly uniform density from feet to head; bicyclists have more points at wheel level and shoulder level with a gap in between. Five-bin vertical layer fractions capture this pattern.

**Adaptive Cluster Split/Merge**

BEV connected components sometimes merge nearby pedestrians into one cluster. PCA gap-finding detects elongated clusters with density gaps along the principal axis and splits them. After frame 3, the tracker's confirmed positions guide splits even when point density gaps have closed. Small noise clusters fully contained within larger clusters are absorbed before classification.

**Kalman Filter Tracking with Temporal Class Voting**

Constant-velocity Kalman filter with Mahalanobis-distance gating and class-aware association. Each track accumulates confidence-weighted class votes over time with a 0.15 hysteresis threshold to prevent frame-to-frame label flipping. Tentative tracks die after 1 miss (kills false alarms fast); confirmed tracks survive 3 misses (bridges temporary occlusion).

---

## Results

### Classification (23 features, 100-tree Random Forest)

| Class | Precision | Recall | F1 | Support |
|:---|---:|---:|---:|---:|
| background | 0.953 | 0.932 | 0.942 | 8135 |
| bicyclist | 0.601 | 0.855 | 0.706 | 330 |
| car | 0.800 | 0.813 | 0.806 | 1755 |
| pedestrian | 0.718 | 0.756 | 0.737 | 320 |
| **macro avg** | **0.768** | **0.839** | **0.798** | **10540** |

Test macro-F1: **0.82** (23 features). Feature extraction: **~1ms/cluster**. RF prediction: **~3ms/batch of 450 clusters**.

### Feature Ablation

| Features | N | Test Macro-F1 | Key Addition |
|:---|---:|---:|:---|
| Bounding box + height | 9 | 0.731 | Baseline |
| + PCA shape | 11 | 0.777 | scattering separates cars from walls |
| + Vertical profile | 19 | 0.800 | layer fractions separate ped from bike |
| + Local structure | 23 | 0.815 | nn_dist_std, hull_point_ratio |
| + Redundant derived | 35 | 0.822 | Marginal — precomputed ratios |

### Tracking (10 frames, urban intersection)

| Metric | Value |
|:---|:---|
| Total unique tracks | 197 |
| Final frame confirmed | 184 (150 ped, 21 car, 13 bike) |
| Tracking ratio (frame 3+) | 90-99% of detections matched |
| Tracker latency | < 50ms for 10 cached frames |

### Pipeline Latency

| Stage | Time per Frame |
|:---|---:|
| Ground removal | 30-80ms |
| BEV clustering | 290-770ms |
| Classification (23-feat) | 500-1200ms |
| Tracking | < 50ms |
| **Total** | **~1-2s** |

With the 19-feature compact mode: ~570ms per frame (1.7 Hz). See [Production Improvements](#production-improvements) for the path to real-time.

---

## Project Structure

```
├── src/
│   ├── classification/
│   │   ├── feature_classifier.py      # Main RF classifier with 19/23/35 feature modes
│   │   ├── pipeline_classifier.py     # Optimized classifier for pipeline use
│   │   ├── two_stage_classifier.py    # Object-vs-background then 3-class experiment
│   │   ├── ablation_study.py          # Systematic feature set comparison
│   │   └── data_augmenter.py          # PCA-aware partial-view augmentation
│   │
│   ├── perception/
│   │   ├── ground_removal.py          # RANSAC calibration + polar grid removal
│   │   ├── clustering.py              # BEV grid, connected components, split/merge
│   │   └── pipeline.py                # Full pipeline orchestration + visualization
│   │
│   ├── tracking/
│   │   └── kalman_tracker.py          # Kalman filter + Hungarian assignment
│   │
│   └── utils/
│       ├── data_loader.py             # Binary point cloud I/O + class statistics
│       ├── visualize.py               # Basic 3D point cloud viewer
│       ├── error_analysis.py          # Interactive misclassification analysis
│       └── explore_data.py            # Data exploration tools
│
├── docs/
│   └── reports/
│       ├── classification_report.md   # Feature engineering methodology + results
│       └── perception_pipeline_report.md  # Pipeline design, iterations, evaluation
│
├── results/                           # Confusion matrices, feature importance plots
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/yourusername/lidar-perception-pipeline.git
cd lidar-perception-pipeline
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, numpy, scipy, scikit-learn, matplotlib, pandas. Optional: vispy + PyQt5 for 3D visualization.

**Data:** This project was developed on proprietary LiDAR data and the point cloud files are not included. The code expects binary float32 files with the following structure:
- Classification data: `data/{train,test}/{background,bicyclist,car,pedestrian}/*.bin` (3 floats per point: x, y, z)
- Sequential frames: `data/sequential_scenes/*.bin` (5 floats per point: x, y, z, intensity, ring)

---

## Usage

### Classification

```bash
# Train and evaluate with 23-feature mode
python -m src.classification.feature_classifier --features 23

# Run feature ablation study
python -m src.classification.ablation_study

# Two-stage classifier experiment
python -m src.classification.two_stage_classifier

# Visual error analysis
python -m src.utils.error_analysis --features 19 --n-per-bucket 40
```

### Full Perception Pipeline

```bash
# Run full pipeline with tracking and 3D visualization
python -m src.perception.pipeline

# Force reprocess (ignore cache)
python -m src.perception.pipeline --force-rerun

# Run without visualization
python -m src.perception.pipeline --no-viz
```

### Viewer Controls

**Pipeline viewer:** `N`/`B` navigate frames, `G` toggle view mode, `T` toggle bounding boxes

---

## Design Decisions and Tradeoffs

**Why Random Forest over gradient boosting or deep learning?**

With only 1,300 samples in minority classes and median cluster sizes of 70-150 points, data-hungry models risk overfitting. RF with `class_weight='balanced'` handles the 25:1 class imbalance cleanly. Handcrafted features encoding known physical differences (cars are wide and flat, pedestrians are tall and narrow) are more data-efficient than learned features for this sample size. XGBoost would likely yield 1-3pp improvement but RF was sufficient and faster to iterate on.

**Why BEV clustering over DBSCAN?**

DBSCAN on 140k non-ground points is O(N²) even with KDTree acceleration — minutes per frame. BEV grid projection is O(N), connected components on the grid is O(grid_size). From an elevated infrastructure sensor, objects separate naturally in the horizontal plane. BEV preserves this separation. Total clustering time: ~300ms vs minutes.

**Why polar grid for ground removal instead of Cartesian?**

The LiDAR scans radially — dense near the sensor, sparse far away. A polar grid matches this pattern naturally: fine resolution near the sensor (dense data), coarse at range (sparse data). A Cartesian grid has the wrong resolution everywhere. Additionally, a vehicle in a polar grid only occludes a narrow angular span, so adjacent wedges still see the ground beside it.

---

## Production Improvements

The current Python implementation runs at ~1-2s per frame. Path to real-time (< 100ms):

**Feature extraction** (current bottleneck): Move to C++ with Eigen for PCA, nanoflann for KDTree, pre-allocated buffers. Parallelize across clusters with a thread pool. Target: ~3ms for 300 clusters.

**BEV clustering**: Replace Python scatter operations (np.add.at) with C++ single-pass accumulation or np.bincount. Target: ~10ms.

**Ground removal**: Parallelize the polar cell loop with OpenMP. Target: ~5ms.

**Detection alternative**: Replace handcrafted features + RF with CenterPoint or PointPillars via TensorRT for learned 3D detection. Target: ~25ms inference.

**Background subtraction**: For fixed infrastructure sensors, accumulate a reference map of the empty scene. Per frame, compare each point against the reference — O(1) per point. Static background removed in ~1ms without RANSAC or grid search.

---

## Detailed Reports

The [docs/reports/](docs/reports/) directory contains detailed write-ups with figures covering:

- Data analysis and class overlap statistics
- Feature engineering rationale for all 35 features
- Ablation study showing each feature group's contribution
- 6 iterations of ground removal development with failure analysis
- BEV clustering design decisions (4-connectivity, cell size tuning, morphological operations)
- Tracking parameter tuning and data association design
- Known limitations and failure cases with visual examples

---

## Author

**Nithilan Karunakaran**

Background in 3D perception for autonomous driving. Previously at Toyota Technological Institute (camera-LiDAR-radar fusion, semantic segmentation, 5 publications) and TierIV (Autoware perception pipeline, ROS2, multi-sensor calibration).

---

## License

This project is released for portfolio and educational purposes. The underlying algorithms (RANSAC, Kalman filter, Random Forest, connected components) are well-established in the robotics perception literature.
