# Pipeline Architecture

## Data Flow

```
Raw LiDAR Frame (184k points, 5 channels: x, y, z, intensity, ring)
    │
    ▼
┌─────────────────────────────┐
│  1. GROUND REMOVAL          │
│  ─────────────────────────  │
│  Calibration (frame 1 only):│
│    RANSAC on nearby points  │
│    → ground normal          │
│    → rotation matrix        │
│    → ground height          │
│                             │
│  Per frame:                 │
│    Rotate points            │
│    Polar grid binning       │
│    Per-cell 5th percentile  │
│    Adaptive threshold       │
│                             │
│  Output: ground mask        │
│  Latency: 30-80ms          │
└─────────────┬───────────────┘
              │ ~140k non-ground points
              ▼
┌─────────────────────────────┐
│  2. CLIPPING                │
│  ─────────────────────────  │
│  Range filter: 80m          │
│  Height filter: -1m to 3m   │
│  (in rotated frame)         │
│                             │
│  Output: clipped cloud      │
│  Latency: ~10ms            │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  3. BEV CLUSTERING          │
│  ─────────────────────────  │
│  Project to 2D grid (0.15m) │
│  4-connectivity CC labeling │
│  Extract + filter clusters  │
│  Ground trim per cluster    │
│                             │
│  Output: ~300 clusters      │
│  Latency: 290-770ms        │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  4. SPLIT / MERGE           │
│  ─────────────────────────  │
│  PCA gap-finding split      │
│  Track-guided split (fr 3+) │
│  Engulfed cluster merge     │
│                             │
│  Latency: 20-130ms         │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  5. CLASSIFICATION          │
│  ─────────────────────────  │
│  23 geometric features/     │
│  cluster (PCA-aligned)      │
│  Random Forest (100 trees)  │
│  Physical sanity checks     │
│                             │
│  Output: ~160 objects,      │
│          ~140 background    │
│  Latency: 500-1200ms       │
└─────────────┬───────────────┘
              │ object clusters only
              ▼
┌─────────────────────────────┐
│  6. TRACKING                │
│  ─────────────────────────  │
│  Kalman predict (CV model)  │
│  Mahalanobis gating         │
│  Class-aware association    │
│  Hungarian assignment       │
│  Temporal class voting      │
│  Birth/confirm/coast/death  │
│                             │
│  Output: confirmed tracks   │
│          with persistent IDs│
│  Latency: <50ms            │
└─────────────────────────────┘
```

## Feature Groups (23-feature mode)

```
Size & Shape (4):     num_points, xy_spread, xy_area, z_range
Derived Ratios (2):   height_to_footprint, density
Z-Profile (3):        z_25, z_75, z_std
PCA Shape (4):        xy_aspect_ratio, scattering, linearity, planarity
Vertical Profile (6): layer_frac_0..4, cross_section_var
Local Structure (4):  nn_dist_std, cross_section_mean, z_range_cleaned, hull_point_ratio
```

All horizontal measurements (xy_spread, xy_area, cross-sections) are computed
in PCA-aligned coordinates for yaw invariance.
