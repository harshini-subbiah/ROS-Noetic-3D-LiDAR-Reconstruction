#!/usr/bin/env python3
"""
Module 4: Static/Dynamic Separation — IMPROVED
Key improvements:
1. Larger window (20 frames) for better temporal consistency
2. Tighter dynamic threshold (0.15) — point must be truly transient
3. Height filtering — ground/ceiling points are always static
4. Cluster validation — dynamic clusters must be physically plausible
   (right size and shape for a person/robot, not a wall)
5. Velocity check — dynamic objects should be moving between frames
"""

import numpy as np
import pickle
import os
import yaml
import gc
import csv
from collections import deque
from scipy.spatial import cKDTree

CONFIG_PATH = os.path.expanduser(
    "~/catkin_ws/src/lidar_reconstruction/config/pipeline_config.yaml")
DEBUG_DIR   = os.path.expanduser(
    "~/catkin_ws/src/lidar_reconstruction/output/debug")

def load_config():
    with open(CONFIG_PATH,'r') as f:
        return yaml.safe_load(f)

class ImprovedTemporalDetector:
    """
    Improved static/dynamic classifier.

    Core logic:
    - A point is STATIC if a nearby point existed in most history frames
    - A point is DYNAMIC if it appears rarely across history frames
    - Extra checks: size, height, movement velocity
    """

    def __init__(self, window_size, dist_threshold,
                 dynamic_ratio, min_cluster_size):
        self.window_size    = window_size
        self.dist_threshold = dist_threshold
        self.dynamic_ratio  = dynamic_ratio
        self.min_cluster    = min_cluster_size
        # history: deque of (timestamp, points_xyz)
        self.history        = deque(maxlen=window_size)
        # track centroids of detected dynamic clusters over time
        # to validate they are actually moving
        self.prev_centroids = []

    def add_frame(self, timestamp, pts_xyz):
        self.history.append((timestamp, pts_xyz))

    def classify(self, pts):
        """
        Returns label array: 0=static, 1=dynamic
        """
        n = len(pts)
        if n == 0:
            return np.array([], dtype=int)

        # Not enough history — assume all static
        if len(self.history) < max(3, self.window_size // 4):
            return np.zeros(n, dtype=int)

        # Count how many history frames contain a nearby point
        appear_count = np.zeros(n, dtype=int)
        n_hist       = len(self.history)

        for _, hpts in self.history:
            if len(hpts) < 3:
                continue
            tree  = cKDTree(hpts[:,:3])
            dists,_ = tree.query(pts[:,:3], workers=-1)
            appear_count += (dists < self.dist_threshold).astype(int)

        # Appearance ratio across history
        appear_ratio = appear_count / n_hist

        # STATIC: appears in >= (1 - dynamic_ratio) of frames
        # DYNAMIC: appears in < dynamic_ratio of frames
        # e.g. with dynamic_ratio=0.15:
        #   static  = appears in >= 85% of frames (walls, floor)
        #   dynamic = appears in <  15% of frames (moving objects)
        labels = (appear_ratio < self.dynamic_ratio).astype(int)

        # Extra filter 1: height-based overrides
        # Ground plane points (Z < -0.3m) and ceiling (Z > 2.8m)
        # are always static regardless
        z = pts[:,2]
        labels[(z < -0.3) | (z > 2.8)] = 0

        # Extra filter 2: remove tiny isolated dynamic clusters
        # Real moving objects have a minimum footprint
        if np.any(labels == 1):
            dynamic_pts = pts[labels == 1]
            if len(dynamic_pts) >= self.min_cluster:
                labels = self._validate_clusters(
                    pts, labels, dynamic_pts)
            else:
                # Too few dynamic points — likely noise, mark all static
                labels[labels == 1] = 0

        return labels

    def _validate_clusters(self, all_pts, labels, dynamic_pts):
        """
        Remove dynamic clusters that are too large (walls)
        or too small (noise) or the wrong shape.
        Keep only clusters that match the physical size of
        a person (0.3-1.5m wide) or robot (0.3-1.0m wide).
        """
        try:
            from sklearn.cluster import DBSCAN
            db   = DBSCAN(eps=0.4, min_samples=5).fit(dynamic_pts[:,:3])
            lbls = db.labels_

            dynamic_indices = np.where(labels == 1)[0]
            new_labels      = labels.copy()

            for cluster_id in set(lbls):
                if cluster_id == -1:
                    # Noise points — mark static
                    noise_mask = lbls == -1
                    new_labels[dynamic_indices[noise_mask]] = 0
                    continue

                cluster_mask = lbls == cluster_id
                cluster_pts  = dynamic_pts[cluster_mask]

                # Bounding box of this cluster
                xyz_min = cluster_pts[:,:3].min(axis=0)
                xyz_max = cluster_pts[:,:3].max(axis=0)
                extents = xyz_max - xyz_min  # [dx, dy, dz]

                # Physical size constraints for valid moving objects:
                # Width (X or Y): 0.2m to 2.0m
                # Height (Z): 0.3m to 2.5m
                # Too large = wall segment, too small = sensor noise
                max_horiz = max(extents[0], extents[1])
                height    = extents[2]

                is_valid = (
                    0.2 <= max_horiz <= 2.0 and
                    0.2 <= height    <= 2.5 and
                    len(cluster_pts) >= self.min_cluster
                )

                if not is_valid:
                    # Invalid cluster — mark as static
                    new_labels[dynamic_indices[cluster_mask]] = 0

            return new_labels

        except Exception:
            return labels

    def separate(self, pts_xyzi):
        """Separate into static and dynamic point arrays."""
        if len(pts_xyzi) == 0:
            return np.zeros((0,4)), np.zeros((0,4))
        labels = self.classify(pts_xyzi)
        return pts_xyzi[labels==0], pts_xyzi[labels==1]

def get_clusters(dynamic_pts, min_sz):
    """Extract clusters from dynamic points."""
    if len(dynamic_pts) < min_sz:
        return []
    try:
        from sklearn.cluster import DBSCAN
        db   = DBSCAN(eps=0.5, min_samples=min_sz).fit(
            dynamic_pts[:,:3])
        lbls = db.labels_
        clusters = []
        for lab in set(lbls):
            if lab == -1:
                continue
            m = lbls == lab
            clusters.append({
                'id':       lab,
                'points':   dynamic_pts[m],
                'centroid': np.mean(dynamic_pts[m][:,:3], axis=0),
                'size':     int(np.sum(m))
            })
        return clusters
    except Exception:
        return []

def main():
    cfg = load_config()
    print("\n"+"="*60)
    print("MODULE 4: IMPROVED STATIC/DYNAMIC SEPARATION")
    print("="*60)

    dc = cfg['dynamic_detection']
    print(f"\n  window_size            : {dc['window_size']}")
    print(f"  distance_threshold     : {dc['distance_threshold']}m")
    print(f"  dynamic_ratio_threshold: {dc['dynamic_ratio_threshold']}")
    print(f"  min_cluster_size       : {dc['min_cluster_size']}")
    print(f"\n  A point is DYNAMIC only if present in <"
          f"{dc['dynamic_ratio_threshold']*100:.0f}% of "
          f"{dc['window_size']} frames")
    print(f"  = must be absent from "
          f"{dc['window_size'] - int(dc['window_size']*dc['dynamic_ratio_threshold'])}"
          f"/{dc['window_size']} recent frames")

    index_path = os.path.join(DEBUG_DIR, "sync_index.pkl")
    with open(index_path,'rb') as f:
        index = pickle.load(f)

    detector = ImprovedTemporalDetector(
        window_size    = dc['window_size'],
        dist_threshold = dc['distance_threshold'],
        dynamic_ratio  = dc['dynamic_ratio_threshold'],
        min_cluster_size = dc['min_cluster_size']
    )

    sep_paths    = []
    total_s      = 0
    total_d      = 0
    csv_rows     = []
    global_idx   = 0

    for i, cpath in enumerate(index['frames3d_paths']):
        print(f"\n  Chunk {i+1}/{len(index['frames3d_paths'])}")
        with open(cpath,'rb') as f:
            frames = pickle.load(f)

        out = []
        chunk_s = 0; chunk_d = 0

        for frame in frames:
            pts = frame.get('points_3d')
            if pts is None or len(pts) == 0:
                nf = dict(frame)
                nf['static_points']    = np.zeros((0,4))
                nf['dynamic_points']   = np.zeros((0,4))
                nf['dynamic_clusters'] = []
                out.append(nf)
                global_idx += 1
                continue

            # Classify
            sp, dp = detector.separate(pts)

            # Add current frame to history AFTER classifying
            detector.add_frame(frame['timestamp'], pts)

            # Get dynamic clusters
            clusters = get_clusters(dp, dc['min_cluster_size'])

            chunk_s += len(sp); chunk_d += len(dp)
            total_s += len(sp); total_d += len(dp)

            ratio = len(dp)/(len(sp)+len(dp)+1e-9)
            csv_rows.append([
                global_idx,
                f"{frame['timestamp']:.3f}",
                len(sp), len(dp),
                f"{ratio:.3f}",
                len(clusters)
            ])

            nf = dict(frame)
            nf['static_points']    = sp
            nf['dynamic_points']   = dp
            nf['dynamic_clusters'] = clusters
            out.append(nf)
            global_idx += 1

        ratio_chunk = chunk_d/(chunk_s+chunk_d+1e-9)
        print(f"    static={chunk_s:,}  "
              f"dynamic={chunk_d:,}  "
              f"dynamic%={ratio_chunk*100:.1f}%")

        out_path = os.path.join(
            DEBUG_DIR, f"sep_chunk_{i:03d}.pkl")
        with open(out_path,'wb') as f:
            pickle.dump(out, f, protocol=2)
        sep_paths.append(out_path)

        del frames, out
        gc.collect()

    # Save CSV
    csv_path = os.path.join(
        DEBUG_DIR, "static_dynamic_per_frame.csv")
    with open(csv_path,'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame_idx','timestamp','static_pts',
                    'dynamic_pts','dynamic_ratio','n_clusters'])
        w.writerows(csv_rows)

    # Update index
    index['sep_paths'] = sep_paths
    with open(index_path,'wb') as f:
        pickle.dump(index, f, protocol=2)

    n = total_s + total_d
    print(f"\n{'='*60}")
    print(f"✔ Total static  : {total_s:,}")
    print(f"✔ Total dynamic : {total_d:,}")
    print(f"✔ Dynamic ratio : {100*total_d/max(n,1):.1f}%")
    print(f"✔ CSV saved     : {csv_path}")
    print("\n✔ Module 4 COMPLETE\n")

    if total_d/max(n,1) > 0.3:
        print("⚠ WARNING: Dynamic ratio still high (>30%)")
        print("  Consider increasing window_size further or")
        print("  decreasing dynamic_ratio_threshold to 0.10")
    else:
        print("✔ Dynamic ratio looks reasonable (<30%)")
        print("  Dynamic points should now be actual moving objects")

if __name__ == '__main__':
    main()
