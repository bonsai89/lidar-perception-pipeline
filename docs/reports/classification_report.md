# Task 1: Point Cloud Classification

So the first task was to classify preclustered point cloud data into four classes: car, pedestrian, bicyclist, and background.

&nbsp;

## Data Observations

I visually evaluated the provided point cloud data. The data seems to be scene sequences of extracted objects, but orientation is arbitrary.

Also I noticed many frames have occlusion, the background class seems to be everything which doesn't fit into the other 3 classes. Pillars, tree trunks and side shrubs, pavements, etc. there is no clear structure to it. And it appears to be predominantly stationary unlike the other 3 classes.

Pedestrian class sometimes has more than 2 pedestrians in a single frame, walking together. And sometimes there is noise or overhangs in car and bicycle pointclouds.

So it is can be defined as a 3 class problem - car/pedestrian/bicycle - what is not in these classes becomes background.

Another aspect I notice is data distribution across the 4 classes. Background has lot of data (32500), followed by car class (7000), bicycle and pedestrian have only ~1300 data points.

From the data, the lidar seems to be a velodyne type lidar since the rings are horizontal, stacked and mounting position is on the side of the road and at an elevated position - so it mostly catches one half of the object unless it is a big object like car it covers almost 2/3rds. This creates occlusion.

I also noticed that in each class there are pointclouds which are extremely sparse and look like noise. So I decided to programmatically check the point cloud features to help decide the classification approach.

I computed n_points, z_range, xy_spread, density per class and compared their mean and std.

*Table 1: Per-class feature statistics*

| Class | Count | n_pts | z_range | xy_spread | density |
|:---|---:|---:|---:|---:|---:|
| background | 32520 | 173.3 ± 320.6 | 2.0 ± 1.2 | 4.2 ± 3.1 | 1211.5 ± 86256.3 |
| bicyclist | 1305 | 299.5 ± 569.5 | 1.5 ± 0.4 | 1.4 ± 0.6 | 175.7 ± 318.5 |
| car | 7000 | 372.5 ± 792.3 | 1.3 ± 0.4 | 3.8 ± 1.5 | 41.4 ± 111.5 |
| pedestrian | 1265 | 214.4 ± 317.9 | 1.5 ± 0.4 | 1.2 ± 1.1 | 284.5 ± 335.3 |

Point variance across classes seems to be very high:
- Car: p10=27, median=148, p90=814
- Pedestrian: p10=23, median=93, p90=559
- Bicyclist: p10=16, median=72, p90=900

Also looking at the median we can say that most clusters are sparse.

The data showed 100% overlap on z_range, xy_spread, density, n_points for pedestrians vs bike.
Similarly car vs background also showed close to 100% overlap on these features.

Considering these metrics and also the dataset size, I could think of two alternatives for classification: random forests or PointNet. Looking at the data size for each class in the training set, PointNet would need considerable data augmentation for pedestrian and bicycle classes. Also since median ranges are tiny 63 to 150 and p10 as low as 15 points. We would have to pad or subsample every cluster since PointNet needs fixed size input. With 15 points, there's almost no geometry for a network to learn from.

So I decided to go ahead with Random forests.

&nbsp;

---

&nbsp;

## Random Forest

The first step is to design handcrafted features that would capture the natural structure of each class.

### Training Setup

I set up the random forest with 100 trees, max depth 20, minimum sample leafs 5, class was set to balanced as it automatically upweights minority classes.

I used 5-fold stratified cross-validation. Stratified ensures each fold preserves the class distribution.
Final model was trained on full training set (42,090 samples), evaluated on held-out test set (10,540 samples).

### Evaluation Framework

I calculated per class precision, recall, F1 which are standard evaluation metrics. And added aggregate evaluation metrics weighted F1 and accuracy. Since dataset is heavily imbalanced towards background using only weightedF1 or accuracy was biased.

For example, the RF learned to predict the background well it falsely pushed accuracy and weightedF1, so I added macroF1 which is unweighted and treats all class equally.

I also added confusion matrix to understand the error distribution across classes.

Finally, added prediction confidence and latency as in autonomous stacks real-time requirements are non-negotiable.

&nbsp;

---

&nbsp;

## Feature Engineering

I designed the features keeping the physical differences between the classes in mind. Car is typically wide and low, while pedestrian is narrow and tall. Bicyclist is somewhere in between.

I added simple bounding box features like **num_points, x_range, y_range, z_range, xy_spread, xy_area, density**.

I also added height based features as car, pedestrian, and bicyclist's vertical shape profile distribution is unique and separating.

**height_to_footprint:** Pedestrian is tall and narrow vs car flat and wide.

**z_25, z_75:** Where the bulk of the object sits vertically. A car's z_25 and z_75 are close together while pedestrian's are spread apart.

**z_std:** How much vertical spread the points have. A car roof is flat so it has low z_std.

On training and evaluation with this feature set I got macroF1 on test set 0.7306 and the confusion matrix showed majority of the errors were coming from car being classified as background and background getting classified as car. And some of the pedestrian and bicycles getting mixed up because a bicyclist is basically a pedestrian on a steel frame, extremely similar features.

&nbsp;

*Table 2: Confusion Matrix (normalized) - 9 features*

| True \ Predicted | background | bicyclist | car | pedestrian |
|:---|---:|---:|---:|---:|
| **background** (8135) | 89.3% | 2.0% | 7.3% | 1.5% |
| **bicyclist** (330) | 9.1% | 81.2% | 0.6% | 9.1% |
| **car** (1755) | 18.8% | 1.5% | 79.2% | 0.5% |
| **pedestrian** (320) | 8.4% | 21.9% | 0.6% | 69.1% |

&nbsp;

The issue is that a car and a wall segment can have similar bounding box dimensions. But their point distributions are fundamentally different - car points fill a 3D volume, wall points lie on a flat surface. I have faced similar problem during my time at Toyota technological institute when I used PCA based features to differentiate shape features.

### PCA Features

**xy_aspect_ratio:** ratio of the two horizontal eigenvalues. I thought this would help separate pedestrians and bicycles because pedestrians have similar xy ratio but bicycle is extended in the bicycle's direction. (Note: but it didn't help much to boost the macroF1, but I left it in there just in case.)

**scattering** is smallest eigenvalue divided by largest. A car's points fill a 3D volume - all three eigenvalues are significant, scattering is moderate to high. A wall or roadside concrete divider has points on a flat surface so one eigenvalue is near zero, scattering is very low.

**linearity** is the gap between the two largest eigenvalues relative to the largest. High linearity means points are stretched along one direction like a traffic pole or fence edge.

**planarity** is the gap between the two smallest eigenvalues relative to the largest. High planarity means points lie on a sheet like a wall or car roof section.

(Note: adding linearity and planarity added only marginal gains, not as much as adding scattering, scattering did most of the heavy lifting.)

### Impact of PCA Features

**xy_aspect_ratio + scattering:**
- car→bg misclassification: 18.8% → 16.4%
- bg→car: 7.3% → 6.3%
- ped→bike: 21.9% → 16.9%
- bike→ped: 9.1% → 5.2%

**linearity + planarity:**
- car→bg: 16.4% → 16.8%
- ped→bike: 16.9% → 15.6%
- bike→ped: unchanged

After these features, a total of 13, the macroF1 rose to 0.78 from 0.73. Still the biggest misclassification zones are car and background, pedestrian and bicyclists.

&nbsp;

### Vertical Layer Features

After visually analysing the pointclouds more closely, I realized that pedestrian and bicyclist even though they have similar bounding boxes and pointcloud density their horizontal layer by layer point distribution is unique. A pedestrian has roughly uniform density from feet to head. A bicyclist has more points at wheel level and shoulder level with a gap in between. So I added 5-bin vertical layer fractions - each captures the proportion of points in a 20% height slice. **(layer_frac_0 to 4)**

I also added **cross_section_var** which measures how much the horizontal footprint changes across height. For eg, a pedestrian stays roughly the same width, a car tapers, a bicyclist widens at the handlebars.

### Impact of Layer Features

**layer_frac_0 through layer_frac_4:**
- bg→car: 6.3% → 4.4%
- car→bg: 16.8% → 17.8% (degraded)
- ped→bike: 15.6% → 15.0%
- bike→ped: 5.2% → 3.9%

**cross_section_var:**
- bg→car: 4.4% → 4.3% (unchanged)
- car→bg: 17.8% → 17.8% (unchanged)
- ped-bike was also unchanged.

After these features, a total of 19, the macroF1 rose to 0.80 from 0.78.

&nbsp;

---

&nbsp;

## Results - 19 Features

At 19 features the classifier achieves 0.80 test macro-F1. Background and car are strong - 93% and 81% recall respectively. Bicyclist recall is high at 85% but precision is only 60%, meaning many things get called bicyclist that aren't. Pedestrian is the weakest class at 76% recall and 72% precision.

The classification runs at 0.69ms per cluster extraction with zero external dependencies. The confidence gap between correct predictions (0.84 mean) and misclassifications (0.60 mean) is clean and consistent, meaning the model knows when it's uncertain.

The remaining confusion has two sources.

First, pedestrian↔bicyclist at 15%/4.2% - these classes have 100% overlap on z_range, xy_spread, density, and n_points. No single-frame geometric feature can fully separate them.

Second, car→background at 17.8% - sparse partial cars with 27 points (p10) are geometrically indistinguishable from typical background clusters.

### Room for Improvement

Data augmentation with partial views with more data would really help RF to learn partial pointcloud classes which are really sparse. Also using temporal awareness, like information from previous frames would really help solidify classification accuracy.

&nbsp;

*Table 3: CV Results - 19 Features*

| Class | F1 | Precision | Recall |
|:---|---:|---:|---:|
| background | 0.9466 ± 0.0007 | 0.9519 | 0.9413 |
| bicyclist | 0.7253 ± 0.0204 | 0.6493 | 0.8222 |
| car | 0.8180 ± 0.0049 | 0.8224 | 0.8139 |
| pedestrian | 0.7185 ± 0.0222 | 0.6960 | 0.7431 |
| **Macro F1** | **0.8021 ± 0.0084** | | |

&nbsp;

*Table 4: Test Set Results - 19 Features*

| Class | Precision | Recall | F1 | Support |
|:---|---:|---:|---:|---:|
| background | 0.9533 | 0.9318 | 0.9424 | 8135 |
| bicyclist | 0.6013 | 0.8545 | 0.7059 | 330 |
| car | 0.7998 | 0.8125 | 0.8061 | 1755 |
| pedestrian | 0.7181 | 0.7562 | 0.7367 | 320 |
| **macro avg** | **0.7681** | **0.8388** | **0.7978** | **10540** |

&nbsp;

*Table 5: Confusion Matrix - 19 Features (raw counts)*

| True \ Predicted | background | bicyclist | car | pedestrian |
|:---|---:|---:|---:|---:|
| background | 7580 | 123 | 353 | 79 |
| bicyclist | 32 | 282 | 2 | 14 |
| car | 312 | 15 | 1426 | 2 |
| pedestrian | 27 | 49 | 2 | 242 |

&nbsp;

*Table 6: Confusion Matrix - 19 Features (normalized)*

| True \ Predicted | background | bicyclist | car | pedestrian |
|:---|---:|---:|---:|---:|
| background | 0.932 | 0.015 | 0.043 | 0.010 |
| bicyclist | 0.097 | 0.855 | 0.006 | 0.042 |
| car | 0.178 | 0.009 | 0.813 | 0.001 |
| pedestrian | 0.084 | 0.153 | 0.006 | 0.756 |

&nbsp;

*Table 7: Prediction Confidence - 19 Features*

| Class | Correct | Misclassified | n_correct | n_wrong |
|:---|---:|---:|---:|---:|
| background | 0.883 | 0.598 | 7551 | 584 |
| bicyclist | 0.817 | 0.592 | 277 | 53 |
| car | 0.868 | 0.621 | 1438 | 317 |
| pedestrian | 0.842 | 0.602 | 246 | 74 |

**Latency:** 0.69ms per cluster (19 features)

&nbsp;

---

&nbsp;

## Visual Error Analysis

I visually checked the misclassified classes and noticed that majority of the misclassification is due to sparse and shapeless pointclouds. While there were some clear bicycle like shapes in background which was predicted as bicycle.

<img src="docs/figures/image1.png" width="280">

*Figure 1: Misclassified example - side view (XZ plane)*

&nbsp;

While there were some clear misclassifications, here are some samples:

<img src="docs/figures/image4.png" width="320">

*Figure 2: Examples from background-car misclassification - the pointclouds clearly look like car*

&nbsp;

<img src="docs/figures/image6.png" width="320">

*Figure 3: Background-car confusion examples*

&nbsp;

I thought the information in xz plane is not fully captured by the features and also in cluster properties like how many clusters, their size distributions, etc. So I explored a broader set of 35 features including individual axis ranges, normalized eigenvalues, DBSCAN-based fragmentation metrics, and various derived ratios. Through ablation I identified that most were redundant - mathematically derivable from existing features. The full 35-feature set scores marginally higher (0.82 vs 0.81) simply because precomputed ratios save the RF from learning them internally, but it added no new discriminative information.

&nbsp;

### Ablation Summary

*Table 8: Feature ablation progression*

| Step | Features | N | CV Macro-F1 | Test Macro-F1 | What changed |
|:---|:---|---:|---:|---:|:---|
| Baseline | bbox + height | 9 | 0.7403 | 0.7306 | num_points, xy_spread, xy_area, z_range, height_to_footprint, density, z_25, z_75, z_std |
| +PCA shape | +xy_aspect, scattering | 11 | 0.7799 | 0.7766 | +4.6pp, car↔bg separation |
| +PCA extras | +linearity, planarity | 13 | 0.7801 | 0.7794 | +0.3pp, marginal |
| +Vertical | +layer_frac_0-4 | 18 | 0.7996 | 0.7987 | +1.9pp, ped↔bike separation |
| +Cross section | +cross_section_var | 19 | 0.8088 | 0.8002 | +0.2pp, pipeline feature set |
| +Local/hull | +nn_dist_std, cross_section_mean, z_range_cleaned, hull_point_ratio | 23 | 0.8149 | 0.8151 | +1.5pp, targets car→bg and sparse clusters |
| +Redundant | +12 derived ratios, DBSCAN | 35 | 0.8213 | 0.8220 | +0.7pp, precomputed ratios save tree splits |

&nbsp;

For completeness I also tried RF with 300 trees to see if it squeezes out any additional performance but the classification quality and metrics remained the same while latency increased. So I reverted back to 100.

&nbsp;

---

&nbsp;

## Two Stage Classifier

Looking at the confusion matrix since majority of the confusion involves the background class, I thought it would be good idea to try a two stage approach. First separate objects from background, then classify objects into car/pedestrian/bicyclist.

I used RF for both stages, for the first stage RF with asymmetric class weights penalizing wrong classification as object and balanced for the second. I tried with three levels of miss penalties for wrong classification 2, 3, & 5.

The results showed a clear tradeoff. Two-stage with miss_penalty=2.0 improved car recall from 81% to 89% and halved the car→background confusion from 18% to 9%. Fewer real objects get lost.

But background precision dropped, bg→car went from 4% to 9%. The safety net recovered 234 correct objects but created 685 false alarms.

Higher penalty (p=3.0) pushed car recall to 91% but macro-F1 dropped further to 0.75. The single-stage classifier at 0.80 macro-F1 handled the background class better because it considers all four options simultaneously rather than making a binary cut first.

The two-stage recall improvement is attractive for a perception system where missing a pedestrian is worse than a false alarm, but for the optional challenge I decided to try with single stage since we can use tracking data to augment classification accuracy.

I paused the classifier here and decided to move to the optional challenge with the core 19 feature set for classification as the 35 feature set added latency but not much improvement, the real test is whether these classifications hold up in a full perception pipeline with clustering, tracking, and real sensor data.

&nbsp;

---

&nbsp;

## Post Optional Challenge Tuning and Additional Features

*(Adding here for continuity.)*

While working on the optional challenge I uncovered several edge cases which reiterated my thoughts on not capturing the internal cluster structure and distributions, so I added back features from the 35 feature set one by one, the least redundant ones which the RF can't internally learn and they proved to be helpful without blowing up the latency.

**nn_dist_std** measures how uniform the nearest-neighbor distances are. Car's surface panels had organized, regular point spacing - low variance. Background clutter (scattered reflections, vegetation) has irregular spacing - high variance. This directly targets the car→bg confusion.

**cross_section_mean** complements cross_section_var by giving the absolute footprint size per height slice, not just how much it changes. A car has large cross sections, a pedestrian has small ones. The variance alone can't distinguish a large object with constant cross section from a small one with constant cross section.

**z_range_cleaned** uses 5th-95th percentile instead of raw min/max. Background clusters often have one stray point far above or below the main body, inflating z_range to look like a pedestrian or car. Cleaning removes these outliers. Helped remove leaking ground points in clusters to a great extent.

<img src="docs/figures/image8.png" width="320">

*Figure 4: Ground point leakage in clusters - z_range_cleaned addresses this*

&nbsp;

**hull_point_ratio** is the fraction of points on the convex hull. Solid objects like cars have most points inside the hull so low ratio. Scattered background fragments have most points on the hull so high ratio.

<img src="docs/figures/image10.png" width="320">

*Figure 5: Convex hull ratio - solid objects vs scattered fragments*

&nbsp;

After adding these features macroF1 slightly went up to 0.82, but I was able to see clear better classifications in the optional challenge scenes.

&nbsp;

---

&nbsp;

## Data Augmentation

Another aspect the optional challenge revealed which was already partially revealed in the first task was that many clusters only have part of the object, like half a car, half a pedestrian, only top half of a car, etc. There is a lot of occlusion in the clusters mainly due to lidar angle and distance of objects vary from the sensor. So I decided to do data-augmentation especially targeted at cars which are cut off in middle and pedestrians whose torso is missing, or only shoulder to head is seen.

<img src="docs/figures/image13.png" width="260">

*Figure 6: Occlusion examples - partial objects in LiDAR data*

&nbsp;

So I designed data augmentation specifically for the pedestrian, bicyclist, and car classes.

<img src="docs/figures/image16.png" width="250">

*Figure 7: Augmentation cut patterns per class*

&nbsp;

<img src="docs/figures/image19.png" width="320">

*Figure 8: PCA-aware cuts for cars*

For cars due to their arbitrary alignment with the lidar, I decided to add PCA aware cuts on top of arbitrary angle cuts to get the half car, top, bottom horizontal cuts.

&nbsp;

On cross-validation the augmented models showed +2.5pp improvement across all feature modes (19, 23, 35). But on the held-out test set all three degraded by roughly 3pp but with the actual optional challenge data it was able to correctly classify partially occluded cars and pedestrians.

<img src="docs/figures/image21.png" width="320">

*Figure 9: Augmentation results - partially occluded objects correctly classified*

&nbsp;

---

&nbsp;

## PCA Yaw-Invariance

While working on the optional challenge I noticed that the same car scanned from different angles produced very different x_range and y_range values. A car at 45 degrees to the sensor had nearly equal x and y ranges making it look like a square footprint instead of a rectangle. This inflated xy_area by roughly 2.4x and distorted density calculations.

The root cause is that ground alignment only fixes pitch and roll, not yaw. Objects in the scene can face any direction relative to the sensor axes.

So I replaced axis-aligned bounding box measurements with 2D PCA-aligned measurements. For each cluster I compute a 2x2 eigendecomposition on the horizontal plane and rotate the xy coordinates to align with the cluster's own principal axes before measuring dimensions. This makes xy_spread, xy_area, density, and height_to_footprint invariant to object orientation.

Pillar and blocks wrongly classified as pedestrian.

<img src="docs/figures/image24.png" width="320">

*Figure 10: Pillars and blocks wrongly classified as pedestrian*

&nbsp;

---

&nbsp;

## Split and Merge Clusters

As previously noticed in the classification problem there are clusters with ground leaking into, or other objects partially appearing in the scene, like tree foliage, occluded pedestrian standing near car, pedestrians walking side by side, etc. This affected classification and often led to misclassification.

<img src="docs/figures/image25.png" width="320">

*Figure 11: Group of pedestrians misclassified as a car*

&nbsp;

So, after initial clustering of the scene, I built a two-path splitting system to handle this.

### PCA Gap-Finding Split

For clusters that look suspicious - z_range between 1.0 and 2.2m (pedestrian height), 100-2000 points, and high PCA linearity (>0.3 meaning elongated in one direction) with the principal axis mostly horizontal (not vertical, which would just be a tall single person) - I project all points onto the principal axis and look for density gaps. Using a 30-bin histogram along the projection, bins below 20% of the mean density indicate a natural gap between two objects. The cluster is split at these gap centers.

Each split piece goes through validation to make sure it looks like a real object and not a fragment of a car or wall:

- z_range > 0.5m (tall enough, not a flat plate)
- xy_spread between 0.3m and 1.5m (pedestrian-sized footprint)
- z_range / xy_spread > 0.8 (taller than wide)
- Vertical density coefficient of variation < 1.5 (points distributed across height, not concentrated in one layer)
- Need at least 2 valid pieces to proceed with the split
- min piece size > 25% of max piece size (roughly even split, not extracting a tiny fragment)
- All pieces under 1000 points (prevents splitting walls or facades)

If validation fails the original cluster is kept intact.

&nbsp;

### Track-Guided Split (frames 3+)

Once the tracker has confirmed tracks from previous frames, I use their positions to guide splits. If a cluster has 2 or more confirmed non-background tracks nearby (within xy_spread + 1.0m), I split along the axis connecting the two track positions at their midpoint. This is more reliable than PCA gap-finding because the tracker already knows these are separate objects - even if the point density gap has closed due to the objects moving closer together.

Track-guided splits skip the PCA linearity check since the tracker's temporal evidence is stronger than single-frame geometry. The same piece validation still applies.

<div style="display: flex; align-items: center; gap: 10px;">
  <img src="docs/figures/image26.png" width="320">
  <img src="docs/figures/image27.png" width="320">
</div>

*Figure 12: Pink - PCA split clusters, Cyan - track-guided splits*

&nbsp;

### Merge Engulfed Clusters

The opposite problem also occurred, a small cluster of noise points sits entirely inside a larger cluster's bounding box. After clustering but before classification, I checked all cluster pairs. If cluster B's 3D bounding box is fully contained within cluster A's bounding box, and B has less than 50% of A's points, B gets absorbed into A. This prevents tiny fragments from getting their own classification and creating false detections.

The ordering matters. In the pipeline the flow is: cluster → split → merge engulfed → classify → track. Split runs first to break merged pedestrians apart. Then merge absorbs small interior fragments. Then the classifier sees clean, properly separated clusters.
<div style="display: flex; align-items: center; gap: 10px;">
  <img src="docs/figures/image28.png" width="300">
  <img src="docs/figures/image29.png" width="300">
</div>

*Figure 13: Small clusters fully engulfed in larger clusters are absorbed into the larger cluster*

&nbsp;

---

&nbsp;

## Edge Cases

There are cases where the objects are still wrongly classified which need much closer fine-tuning with lot more data for training, especially covering the edge case scenarios. Here are some examples:

<img src="docs/figures/image32.png" width="320">

*Figure 14: Left - wrongly predicted as car, the pedestrians don't align on a common principal axis so the split clustering misses it. Right - pedestrians get classified as bicyclists which is a very common misclassification prone to the geometric similarity between the classes.*

&nbsp;

---

&nbsp;

## Confidence Gap

One consistent finding across all feature modes is the prediction confidence gap. Correct predictions average 0.87 confidence while misclassifications average 0.60. This gap holds regardless of whether we use 19, 23, or 35 features.

*Table 9: Confidence gap across feature modes*

| Feature Set | Correct conf | Misclassified conf | Gap |
|:---|---:|---:|---:|
| 19-feat | 0.876 | 0.599 | 0.277 |
| 23-feat | 0.875 | 0.596 | 0.279 |
| 35-feat | 0.873 | 0.598 | 0.275 |

But for the actual optional pipeline implementation I didn't use the confidence because of low confidence true positives in the data. As I previously noted, there are genuinely bicycle-like or car-like objects in background so the learning is inherently biased and confidence doesn't reflect prediction accuracy.

Adding more features does not help the model become more certain about the hard cases. The boundary between classes is fundamentally ambiguous in geometric feature space - a sparse half-car with 27 points genuinely looks like background clutter, and a pedestrian genuinely looks like a bicyclist in a single frame. This is the Bayes error rate of the feature space, not a model limitation.

&nbsp;

---

&nbsp;

## Future Directions

Closely looking at the data, there are cases where background or 'non object' points corrupt the classification while visually we can clearly see the object - car/pedestrian etc. So I think a PointNet based solution would improve classification for visually recognizable clusters.

But I think it would still suffer misclassification in cases where we have very sparse points. Also it might suffer with background classification as it is quite varied. RF seems to be really good at background classification.

**Combined solution that could potentially excel in performance:** A hybrid architecture where RF's handcrafted features and PointNet's learned embeddings are concatenated as input to a final classification head. Combining RF's strength on background rejection with PointNet's strength on object surface pattern recognition.
