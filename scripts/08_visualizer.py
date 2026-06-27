#!/usr/bin/env python3
"""
Module 8: Visualization System
- Publishes static + dynamic maps to /world frame
- Keeps publishing continuously so RViz stays interactive
- Streams per-frame data for live playback
- Works both standalone and with live rosbag
"""

import rospy
import numpy as np
import os
import pickle
import yaml
import struct
import threading
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point

CONFIG_PATH = os.path.expanduser(
    "~/catkin_ws/src/lidar_reconstruction/config/pipeline_config.yaml")
DEBUG_DIR   = os.path.expanduser(
    "~/catkin_ws/src/lidar_reconstruction/output/debug")
STATIC_DIR  = os.path.expanduser(
    "~/catkin_ws/src/lidar_reconstruction/output/static_map")
DYNAMIC_DIR = os.path.expanduser(
    "~/catkin_ws/src/lidar_reconstruction/output/dynamic_map")

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def numpy_to_pc2(points, r, g, b, frame="world", stamp=None):
    """Convert Nx3 or Nx5 numpy array to PointCloud2 with flat color."""
    if stamp is None:
        stamp = rospy.Time.now()
    n = len(points)
    if n == 0:
        msg = PointCloud2()
        msg.header = Header(frame_id=frame, stamp=stamp)
        return msg

    xyz = points[:, :3].astype(np.float32)

    # Encode RGB as packed float
    rgb_int = (int(r) << 16) | (int(g) << 8) | int(b)
    rgb_f   = struct.unpack('f', struct.pack('I', rgb_int))[0]
    rgb_col = np.full(n, rgb_f, dtype=np.float32)

    data = np.zeros(n, dtype=[
        ('x','f4'), ('y','f4'), ('z','f4'), ('rgb','f4')
    ])
    data['x']   = xyz[:, 0]
    data['y']   = xyz[:, 1]
    data['z']   = xyz[:, 2]
    data['rgb'] = rgb_col

    msg = PointCloud2()
    msg.header     = Header(frame_id=frame, stamp=stamp)
    msg.height     = 1
    msg.width      = n
    msg.is_dense   = False
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step   = 16 * n
    msg.fields     = [
        PointField('x',   0, PointField.FLOAT32, 1),
        PointField('y',   4, PointField.FLOAT32, 1),
        PointField('z',   8, PointField.FLOAT32, 1),
        PointField('rgb', 12, PointField.FLOAT32, 1),
    ]
    msg.data = data.tobytes()
    return msg

def make_bbox_marker(cluster, marker_id, frame="world"):
    m = Marker()
    m.header.frame_id = frame
    m.header.stamp    = rospy.Time.now()
    m.ns              = "dynamic_objects"
    m.id              = marker_id
    m.type            = Marker.CUBE
    m.action          = Marker.ADD
    pts = cluster['points'][:, :3]
    c   = cluster['centroid']
    ext = pts.max(axis=0) - pts.min(axis=0) + 0.1
    m.pose.position.x  = float(c[0])
    m.pose.position.y  = float(c[1])
    m.pose.position.z  = float(c[2])
    m.pose.orientation.w = 1.0
    m.scale.x = float(max(ext[0], 0.2))
    m.scale.y = float(max(ext[1], 0.2))
    m.scale.z = float(max(ext[2], 0.2))
    m.color   = ColorRGBA(1.0, 0.3, 0.0, 0.45)
    m.lifetime = rospy.Duration(0.8)
    return m

class ReconstructionVisualizer:
    def __init__(self):
        rospy.init_node('reconstruction_visualizer', anonymous=True)

        # Full map publishers (latched — persist in RViz after publish)
        self.pub_static   = rospy.Publisher(
            '/static_map',   PointCloud2, queue_size=1, latch=True)
        self.pub_dynamic  = rospy.Publisher(
            '/dynamic_map',  PointCloud2, queue_size=1, latch=True)
        self.pub_combined = rospy.Publisher(
            '/combined_map', PointCloud2, queue_size=1, latch=True)

        # Per-frame live publishers (not latched — update each frame)
        self.pub_frame_static  = rospy.Publisher(
            '/frame_static_points',  PointCloud2, queue_size=1)
        self.pub_frame_dynamic = rospy.Publisher(
            '/frame_dynamic_points', PointCloud2, queue_size=1)
        self.pub_markers = rospy.Publisher(
            '/dynamic_object_markers', MarkerArray, queue_size=1)

        rospy.sleep(0.5)  # let publishers register
        print("  ✔ Publishers ready")

        # Load maps
        self.static_arr  = self._load_map(
            os.path.join(STATIC_DIR,  "static_semantic_map.npy"),  "Static")
        self.dynamic_arr = self._load_map(
            os.path.join(DYNAMIC_DIR, "dynamic_semantic_map.npy"), "Dynamic")
        self.combined_arr = self._load_map(
            os.path.join(DEBUG_DIR,   "combined_map.npy"),          "Combined")

        # Center everything at origin for easy navigation
        if len(self.static_arr) > 0:
            self.center = self.static_arr[:, :3].mean(axis=0)
            self.center[2] = 0  # keep Z as-is
        else:
            self.center = np.zeros(3)

        self._center_map(self.static_arr)
        self._center_map(self.dynamic_arr)
        self._center_map(self.combined_arr)

    def _load_map(self, path, name):
        if os.path.exists(path):
            arr = np.load(path)
            print(f"  ✔ {name}: {len(arr):,} points loaded")
            return arr
        print(f"  ⚠ {name}: not found at {path}")
        return np.zeros((0, 5))

    def _center_map(self, arr):
        if len(arr) > 0:
            arr[:, :3] -= self.center

    def publish_full_maps(self):
        """Publish complete maps (latched — stay in RViz until overwritten)."""
        stamp = rospy.Time.now()
        print("\n  Publishing full maps to RViz...")

        msg = numpy_to_pc2(self.static_arr,  100, 180, 255, stamp=stamp)
        self.pub_static.publish(msg)
        print(f"  ✔ /static_map  → {len(self.static_arr):,} pts (blue)")

        msg = numpy_to_pc2(self.dynamic_arr, 255, 80, 20, stamp=stamp)
        self.pub_dynamic.publish(msg)
        print(f"  ✔ /dynamic_map → {len(self.dynamic_arr):,} pts (orange-red)")

        msg = numpy_to_pc2(self.combined_arr, 200, 200, 200, stamp=stamp)
        self.pub_combined.publish(msg)
        print(f"  ✔ /combined_map→ {len(self.combined_arr):,} pts (grey)")

    def _republish_loop(self, interval_sec=5.0):
        """
        Republish full maps every N seconds so RViz never loses them
        even when navigating or after rosbag restarts.
        """
        rate = rospy.Rate(1.0 / interval_sec)
        while not rospy.is_shutdown():
            self.publish_full_maps()
            rate.sleep()

    def stream_frames(self, frames, rate_hz=5.0):
        """Stream per-frame data for live playback animation."""
        print(f"\n  Streaming {len(frames)} frames at {rate_hz} Hz...")
        print("  (drag/rotate freely in RViz — maps stay visible)")
        rate = rospy.Rate(rate_hz)

        for idx, frame in enumerate(frames):
            if rospy.is_shutdown():
                break

            stamp = rospy.Time.now()

            # Static frame points
            sp = frame.get('static_semantic',
                 frame.get('static_points', np.zeros((0, 4))))
            if sp is not None and len(sp) > 0:
                sp_centered = sp.copy()
                sp_centered[:, :3] -= self.center
                msg = numpy_to_pc2(sp_centered, 100, 180, 255, stamp=stamp)
                self.pub_frame_static.publish(msg)

            # Dynamic frame points
            dp = frame.get('dynamic_semantic',
                 frame.get('dynamic_points', np.zeros((0, 4))))
            if dp is not None and len(dp) > 0:
                dp_centered = dp.copy()
                dp_centered[:, :3] -= self.center
                msg = numpy_to_pc2(dp_centered, 255, 60, 0, stamp=stamp)
                self.pub_frame_dynamic.publish(msg)

            # Dynamic bounding box markers
            clusters = frame.get('dynamic_clusters', [])
            if clusters:
                ma = MarkerArray()
                for j, cl in enumerate(clusters[:20]):
                    cl_c = dict(cl)
                    cl_c['centroid'] = cl['centroid'] - self.center
                    cl_c['points']   = cl['points'].copy()
                    cl_c['points'][:, :3] -= self.center
                    ma.markers.append(make_bbox_marker(cl_c, j))
                self.pub_markers.publish(ma)

            if idx % 50 == 0:
                print(f"  Frame {idx}/{len(frames)}")

            rate.sleep()

        print("  ✔ Streaming complete — full maps remain in RViz")
        print("  Press Ctrl+C to exit or leave running")

def generate_matplotlib_plots():
    """Generate offline plots — no ROS needed."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    plot_dir = os.path.join(DEBUG_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    static_path  = os.path.join(STATIC_DIR,  "static_semantic_map.npy")
    dynamic_path = os.path.join(DYNAMIC_DIR, "dynamic_semantic_map.npy")

    if not os.path.exists(static_path):
        print("  ⚠ No static map found — skipping plots")
        return

    static_map  = np.load(static_path)
    dynamic_map = np.load(dynamic_path) if os.path.exists(dynamic_path) \
                  else np.zeros((0, 5))

    # Center
    if len(static_map) > 0:
        ctr = static_map[:, :3].mean(axis=0); ctr[2] = 0
        static_map[:, :3]  -= ctr
        if len(dynamic_map) > 0:
            dynamic_map[:, :3] -= ctr

    print("  Generating plots...")

    # ── Plot 1: Top-down ──────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor('#1e1e2e')
    fig.suptitle("Dual Semantic 3D Map — Top-Down View",
                 color='white', fontsize=15, fontweight='bold')

    def sample(arr, n=6000):
        return arr[::max(1, len(arr)//n)] if len(arr) > 0 else arr

    for ax in axes:
        ax.set_facecolor('#0d0d1a')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')

    s = sample(static_map)
    axes[0].scatter(s[:,0], s[:,1], c='#64b4ff', s=0.8, alpha=0.7)
    axes[0].set_title("Static Map", color='white', fontsize=12)
    axes[0].set_xlabel("X (m)"); axes[0].set_ylabel("Y (m)")
    axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.2, color='#444444')

    d = sample(dynamic_map, 2000)
    axes[1].scatter(d[:,0], d[:,1], c='#ff5014', s=2.0, alpha=0.9)
    axes[1].set_title("Dynamic Map", color='white', fontsize=12)
    axes[1].set_xlabel("X (m)"); axes[1].set_ylabel("Y (m)")
    axes[1].set_aspect('equal'); axes[1].grid(True, alpha=0.2, color='#444444')

    axes[2].scatter(s[:,0], s[:,1], c='#64b4ff', s=0.5, alpha=0.5, label='Static')
    axes[2].scatter(d[:,0], d[:,1], c='#ff5014', s=2.0, alpha=0.9, label='Dynamic')
    axes[2].set_title("Combined", color='white', fontsize=12)
    axes[2].set_xlabel("X (m)"); axes[2].set_ylabel("Y (m)")
    axes[2].set_aspect('equal'); axes[2].grid(True, alpha=0.2, color='#444444')
    leg = axes[2].legend(markerscale=6, facecolor='#1e1e2e')

    plt.tight_layout()
    p = os.path.join(plot_dir, "topdown_map.png")
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✔ {p}")

    # ── Plot 2: 3D perspective ─────────────────────────────────
    fig = plt.figure(figsize=(16, 7), facecolor='#1e1e2e')
    fig.suptitle("3D Semantic Map — Perspective",
                 color='white', fontsize=14, fontweight='bold')

    for i, (arr, color, title) in enumerate([
            (static_map,  '#64b4ff', "Static Map (blue)"),
            (dynamic_map, '#ff5014', "Dynamic Map (red)")]):
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        ax.set_facecolor('#0d0d1a')
        s2 = arr[::max(1, len(arr)//3000)] if len(arr) > 0 else arr
        if len(s2) > 0:
            ax.scatter(s2[:,0], s2[:,1], s2[:,2],
                       c=color, s=0.5, alpha=0.6)
        ax.set_title(title, color='white')
        ax.set_xlabel('X', color='white')
        ax.set_ylabel('Y', color='white')
        ax.set_zlabel('Z', color='white')
        ax.tick_params(colors='white')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    plt.tight_layout()
    p = os.path.join(plot_dir, "3d_map.png")
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✔ {p}")

    # ── Plot 3: Class distribution ─────────────────────────────
    CLASSES = ['road','sidewalk','building','wall','fence','pole',
               'traffic light','traffic sign','vegetation','terrain','sky',
               'person','rider','car','truck','bus','train','motorcycle','bicycle']
    COLORS_F = np.array([
        [0.502,0.251,0.502],[0.957,0.137,0.910],[0.275,0.275,0.275],
        [0.400,0.400,0.612],[0.745,0.600,0.600],[0.600,0.600,0.600],
        [0.980,0.667,0.118],[0.863,0.863,0.000],[0.420,0.557,0.137],
        [0.596,0.984,0.596],[0.275,0.510,0.706],[0.863,0.078,0.235],
        [1.000,0.000,0.000],[0.000,0.000,0.557],[0.000,0.000,0.275],
        [0.000,0.235,0.392],[0.000,0.314,0.392],[0.000,0.000,0.902],
        [0.467,0.043,0.125]
    ])

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor='#1e1e2e')
    fig.suptitle("Semantic Class Distribution",
                 color='white', fontsize=14, fontweight='bold')

    from collections import Counter
    for ax, arr, title in [
            (axes[0], static_map,  "Static Map Classes"),
            (axes[1], dynamic_map, "Dynamic Map Classes")]:
        ax.set_facecolor('#1e1e2e')
        if len(arr) == 0:
            ax.text(0.5,0.5,'No data',ha='center',va='center',color='white')
            ax.set_title(title, color='white')
            continue
        labs = arr[:,4].astype(int)
        counts = Counter(labs)
        top = sorted(counts.items(), key=lambda x:-x[1])[:8]
        names  = [CLASSES[k] if 0<=k<len(CLASSES) else 'unknown' for k,_ in top]
        vals   = [v for _,v in top]
        colors = [COLORS_F[k] if 0<=k<len(COLORS_F) else [0.5,0.5,0.5]
                  for k,_ in top]
        wedges, texts, autotexts = ax.pie(
            vals, labels=names, colors=colors,
            autopct='%1.1f%%', startangle=90,
            textprops={'color':'white', 'fontsize':9})
        for at in autotexts:
            at.set_color('white')
        ax.set_title(title, color='white', fontsize=12)

    plt.tight_layout()
    p = os.path.join(plot_dir, "class_distribution.png")
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✔ {p}")
    print(f"  ✔ All plots saved to: {plot_dir}")

def main():
    print("\n" + "="*60)
    print("MODULE 8: VISUALIZATION")
    print("="*60)

    # Always generate offline plots first
    print("\n[1/2] Generating matplotlib plots...")
    generate_matplotlib_plots()

    # ROS visualization
    print("\n[2/2] Starting ROS visualization...")
    print("  Make sure roscore is running!")

    try:
        viz = ReconstructionVisualizer()

        # Publish full maps immediately
        viz.publish_full_maps()

        # Start background thread to keep republishing maps
        # (so they stay visible no matter what you do in RViz)
        t = threading.Thread(target=viz._republish_loop, args=(8.0,), daemon=True)
        t.start()
        print("\n  ✔ Map republish thread started (every 8s)")
        print("  ✔ You can freely rotate/zoom/pan in RViz")
        print("  ✔ Maps will NOT disappear while this node runs")

        # Stream per-frame data
        fused_path = os.path.join(DEBUG_DIR, "fused_frames.pkl")
        if os.path.exists(fused_path):
            print("\n  Loading frames for live streaming...")
            with open(fused_path, 'rb') as f:
                frames = pickle.load(f)
            viz.stream_frames(frames, rate_hz=5.0)
        else:
            print("  No fused_frames.pkl found — publishing maps only")

        print("\n  Streaming done. Maps are latched — staying in RViz.")
        print("  Ctrl+C to exit.\n")
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        print(f"\n  ⚠ ROS error: {e}")
        print("  → Offline plots were generated regardless.")
        print("  → Start roscore and rerun to enable RViz publishing.")

if __name__ == '__main__':
    main()
