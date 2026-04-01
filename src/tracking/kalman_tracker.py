"""
Multi-object tracker using Kalman filter + Hungarian assignment.

Designed for infrastructure LiDAR perception pipeline:
- Constant velocity model (sufficient for 10Hz, 10 frames)
- Hungarian algorithm for data association
- Track management: birth, confirmation, death

Operates on Stage 1 filtered object clusters (centroids + metadata).

Run from project root:
  python optional_challenge/tracker.py
"""

import logging
from typing import List, Dict, Tuple, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


class KalmanTrack:
    """Single object track with Kalman filter state estimation.

    State: [x, y, z, vx, vy, vz] — position + velocity in 3D.
    Measurement: [x, y, z] — cluster centroid.

    Constant velocity model: position updates by velocity each frame.
    """

    _next_id = 1  # class-level track ID counter

    def __init__(
        self,
        centroid: np.ndarray,
        cluster: dict,
        dt: float = 0.1,
    ):
        """Initialize track from first detection.

        :param centroid: (3,) initial position [x, y, z].
        :param cluster: Full cluster dict from pipeline.
        :param dt: Time step between frames (seconds). ~0.1 for 10Hz LiDAR.
        """
        self.track_id = KalmanTrack._next_id
        KalmanTrack._next_id += 1

        self.dt = dt
        self.age = 1                    # frames since creation
        self.hits = 1                   # frames with matched detection
        self.misses = 0                 # consecutive frames without match
        self.confirmed = False          # promoted after N hits
        self.dead = False               # marked for removal

        # Latest cluster data
        self.cluster = cluster
        self.class_label = None         # filled by Stage 2
        self.class_confidence = 0.0
        self.class_history = []         # for temporal voting
        self.class_votes = {}
        cls = cluster.get("s2_class", "unknown")
        conf = cluster.get("s2_confidence", 0.5)
        self.class_votes[cls] = {"total_conf": conf, "count": 1}
        self.class_label = cls
        self.class_confidence = 1.0
        cluster["s2_class_tracked"] = self.class_label

        # --- Kalman filter state ---
        # State: [x, y, z, vx, vy, vz]
        self.x = np.array([
            centroid[0], centroid[1], centroid[2],
            0.0, 0.0, 0.0,  # initial velocity = 0
        ], dtype=np.float64)

        # State transition: constant velocity
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # Measurement matrix: observe position only
        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1

        # Covariance
        self.P = np.diag([
            0.5, 0.5, 0.5,  # position uncertainty
            10.0, 10.0, 10.0  # velocity uncertainty
        ])

        # Process noise
        q_pos = 0.5   # position noise
        q_vel = 2.0   # velocity noise (objects can accelerate)
        self.Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel])

        # Measurement noise
        r = 0.3  # ~30cm measurement uncertainty (cluster centroid jitter)
        self.R = np.diag([r, r, r])

    @property
    def position(self) -> np.ndarray:
        """Current estimated position [x, y, z]."""
        return self.x[:3]

    @property
    def velocity(self) -> np.ndarray:
        """Current estimated velocity [vx, vy, vz]."""
        return self.x[3:]

    @property
    def speed(self) -> float:
        """Speed magnitude (m/s)."""
        return np.linalg.norm(self.velocity)

    def predict(self) -> np.ndarray:
        """Kalman predict step. Returns predicted position.

        :return: (3,) predicted measurement [x, y, z].
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.H @ self.x

    def update(self, centroid: np.ndarray, cluster: dict) -> None:
        """Kalman update step with matched detection.

        :param centroid: (3,) measured position.
        :param cluster: Full cluster dict.
        """
        z = centroid
        y = z - self.H @ self.x                       # innovation
        S = self.H @ self.P @ self.H.T + self.R       # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)      # Kalman gain

        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

        # Update metadata
        self.cluster = cluster
        self.hits += 1
        self.misses = 0
        cls = cluster.get("s2_class", "unknown")
        conf = cluster.get("s2_confidence", 0.5)
        if cls not in self.class_votes:
            self.class_votes[cls] = {"total_conf": 0.0, "count": 0}
        self.class_votes[cls]["total_conf"] += conf
        self.class_votes[cls]["count"] += 1

        best_cls = max(self.class_votes, key=lambda k: self.class_votes[k]["total_conf"] / self.class_votes[k]["count"])
        best_avg = self.class_votes[best_cls]["total_conf"] / self.class_votes[best_cls]["count"]
        current_entry = self.class_votes.get(self.class_label, {"total_conf": 0, "count": 1})
        current_avg = current_entry["total_conf"] / current_entry["count"]

        if best_cls != self.class_label and best_avg - current_avg > 0.15:
            self.class_label = best_cls

        entry = self.class_votes[self.class_label]
        self.class_confidence = entry["total_conf"] / entry["count"]
        cluster["s2_class_tracked"] = self.class_label

    def mark_missed(self) -> None:
        """Called when no detection matched this track."""
        self.misses += 1
        self.age += 1

    def __repr__(self):
        status = "DEAD" if self.dead else ("CONFIRMED" if self.confirmed else "tentative")
        cls = self.class_label or "unclassified"
        return (
            f"Track(id={self.track_id}, {status}, {cls}, "
            f"pos=[{self.position[0]:.1f},{self.position[1]:.1f},{self.position[2]:.1f}], "
            f"speed={self.speed:.2f}m/s, hits={self.hits}, misses={self.misses})"
        )


class MultiObjectTracker:
    """Multi-object tracker with Kalman filters and Hungarian assignment.

    Track lifecycle:
    1. BIRTH: Unmatched detection → new tentative track
    2. CONFIRM: Track with hits >= confirm_hits → confirmed
    3. COAST: No match → predict only, increment misses
    4. DEATH: misses >= max_misses → track killed

    :param max_misses: Kill track after this many consecutive misses.
    :param confirm_hits: Confirm track after this many total hits.
    :param gate_distance: Max distance for valid association (meters).
    :param dt: Time step between frames.
    """

    def __init__(
        self,
        max_misses: int = 3,
        confirm_hits: int = 2,
        gate_distance: float = 3.0,
        dt: float = 0.1,
    ):
        self.max_misses = max_misses
        self.confirm_hits = confirm_hits
        self.gate_distance = gate_distance
        self.dt = dt

        self.tracks: List[KalmanTrack] = []
        self.frame_count = 0

    def step(self, detections: List[dict]) -> List[KalmanTrack]:
        """Process one frame of detections.

        :param detections: List of cluster dicts with 'centroid' key.
        :return: List of all active tracks (confirmed + tentative).
        """
        self.frame_count += 1

        # --- Predict all tracks ---
        predicted_positions = []
        for track in self.tracks:
            pred = track.predict()
            predicted_positions.append(pred)

        # --- Build cost matrix ---
        n_tracks = len(self.tracks)
        n_dets = len(detections)

        if n_tracks == 0 and n_dets == 0:
            return self.tracks

        if n_tracks > 0 and n_dets > 0:
            cost_matrix = np.zeros((n_tracks, n_dets), dtype=np.float64)

            for i, pred in enumerate(predicted_positions):
                for j, det in enumerate(detections):
                    z = det["centroid"]
                    track = self.tracks[i]

                    y = z - track.H @ track.x
                    S = track.H @ track.P @ track.H.T + track.R
                    d2 = float(y.T @ np.linalg.solve(S, y))

                    if d2 > 7.81:
                        cost_matrix[i, j] = 1e6
                    else:
                        track_cls = track.class_label
                        det_cls = det.get("s2_class", "unknown")

                        if track_cls is not None and det_cls != track_cls:
                            cost_matrix[i, j] = 1e6
                        else:
                            cost_matrix[i, j] = np.sqrt(d2)

            # --- Hungarian assignment ---
            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            matched_tracks = set()
            matched_dets = set()

            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] < self.gate_distance:
                    # Valid match
                    self.tracks[row].update(
                        detections[col]["centroid"],
                        detections[col],
                    )
                    matched_tracks.add(row)
                    matched_dets.add(col)

            # --- Unmatched tracks → coast ---
            for i in range(n_tracks):
                if i not in matched_tracks:
                    self.tracks[i].mark_missed()

            # --- Unmatched detections → new tracks ---
            for j in range(n_dets):
                if j not in matched_dets:
                    new_track = KalmanTrack(
                        detections[j]["centroid"],
                        detections[j],
                        dt=self.dt,
                    )
                    self.tracks.append(new_track)

        elif n_dets > 0:
            # No existing tracks — all detections become new tracks
            for det in detections:
                new_track = KalmanTrack(det["centroid"], det, dt=self.dt)
                self.tracks.append(new_track)

        else:
            # No detections — all tracks coast
            for track in self.tracks:
                track.mark_missed()

        # --- Update track status ---
        for track in self.tracks:
            if track.hits >= self.confirm_hits:
                track.confirmed = True
            if not track.confirmed and track.misses >= 1:
                track.dead = True
            elif track.confirmed and track.misses >= self.max_misses:
                track.dead = True

        # --- Remove dead tracks ---
        n_before = len(self.tracks)
        self.tracks = [t for t in self.tracks if not t.dead]
        n_killed = n_before - len(self.tracks)

        # Stats
        n_confirmed = sum(1 for t in self.tracks if t.confirmed)
        n_tentative = sum(1 for t in self.tracks if not t.confirmed)

        logger.info(
            f"Tracker frame {self.frame_count}: "
            f"{n_dets} detections, "
            f"{n_confirmed} confirmed, {n_tentative} tentative, "
            f"{n_killed} killed"
        )

        return self.tracks

    def get_confirmed_tracks(self) -> List[KalmanTrack]:
        """Return only confirmed tracks."""
        return [t for t in self.tracks if t.confirmed]

    def get_all_tracks(self) -> List[KalmanTrack]:
        """Return all active tracks."""
        return self.tracks


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
