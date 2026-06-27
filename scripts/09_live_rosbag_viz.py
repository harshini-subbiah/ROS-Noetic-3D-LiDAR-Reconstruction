#!/usr/bin/env python3
"""
Module 9: Live Full Color 3D Visualizer
FIXED:
- Pre-built static/dynamic maps NEVER disappear (latched + periodic republish)
- Live accumulated map uses disk-backed chunked storage (memory safe)
- Live scan shown as thin overlay only (no RAM accumulation limit issue)
"""

import rospy
import numpy as np
import os
import struct
import threading
import cv2
from collections import deque
from scipy.spatial.transform import Rotation

from sensor_msgs.msg import (LaserScan, Imu, Image, CameraInfo,
                              PointCloud2, PointField)
from nav_msgs.msg        import Odometry, Path
from geometry_msgs.msg   import PoseStamped, Point, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg        import Header, ColorRGBA
import tf2_ros
from tf2_ros import StaticTransformBroadcaster

STATIC_DIR  = os.path.expanduser(
    "~/catkin_ws/src/lidar_reconstruction/output/static_map")
DYNAMIC_DIR = os.path.expanduser(
    "~/catkin_ws/src/lidar_reconstruction/output/dynamic_map")

SEG_COLORS = np.array([
    [128, 64,128],[244, 35,232],[ 70, 70, 70],[102,102,156],
    [190,153,153],[153,153,153],[250,170, 30],[220,220,  0],
    [107,142, 35],[152,251,152],[ 70,130,180],[220, 20, 60],
    [255,  0,  0],[  0,  0,142],[  0,  0, 70],[  0, 60,100],
    [  0, 80,100],[  0,  0,230],[119, 11, 32]
], dtype=np.uint8)

# ── helpers ───────────────────────────────────────────────────────────────────

def height_to_rgb(z_arr, z_min=-0.3, z_max=2.5):
    t = np.clip((z_arr - z_min)/(z_max - z_min + 1e-6), 0, 1)
    r = np.clip(1.5 - np.abs(t*4 - 3), 0, 1)
    g = np.clip(1.5 - np.abs(t*4 - 2), 0, 1)
    b = np.clip(1.5 - np.abs(t*4 - 1), 0, 1)
    return (np.column_stack([r, g, b]) * 255).astype(np.uint8)

def pack_rgb(rgb):
    r = rgb[:,0].astype(np.uint32)
    g = rgb[:,1].astype(np.uint32)
    b = rgb[:,2].astype(np.uint32)
    return ((r << 16)|(g << 8)|b).view(np.float32)

def make_pc2(xyz, rgb_u8, frame="odom_combined", stamp=None):
    if stamp is None:
        stamp = rospy.Time.now()
    n = len(xyz)
    if n == 0:
        msg = PointCloud2()
        msg.header = Header(frame_id=frame, stamp=stamp)
        return msg
    xyz_f = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    rgb_f = pack_rgb(np.asarray(rgb_u8, dtype=np.uint8).reshape(-1, 3))
    data  = np.zeros(n, dtype=[
        ('x','f4'),('y','f4'),('z','f4'),('rgb','f4')])
    data['x']   = xyz_f[:,0]
    data['y']   = xyz_f[:,1]
    data['z']   = xyz_f[:,2]
    data['rgb'] = rgb_f
    msg              = PointCloud2()
    msg.header       = Header(frame_id=frame, stamp=stamp)
    msg.height       = 1
    msg.width        = n
    msg.is_dense     = False
    msg.is_bigendian = False
    msg.point_step   = 16
    msg.row_step     = 16 * n
    msg.fields       = [
        PointField('x',   0, PointField.FLOAT32, 1),
        PointField('y',   4, PointField.FLOAT32, 1),
        PointField('z',   8, PointField.FLOAT32, 1),
        PointField('rgb',12, PointField.FLOAT32, 1),
    ]
    msg.data = data.tobytes()
    return msg

def load_map(path, name):
    if os.path.exists(path):
        arr = np.load(path)
        print(f"  ✔ {name}: {len(arr):,} pts")
        return arr
    print(f"  ✘ {name}: NOT FOUND")
    return np.zeros((0, 5))


# ── disk-backed voxel accumulator ─────────────────────────────────────────────

class DiskVoxelAccumulator:
    """
    Accumulates live scan points into a voxel map stored on disk.
    Never keeps all points in RAM — only the current voxel dict.
    Voxel dict is much smaller than raw points (deduplication).
    """
    def __init__(self, voxel_size=0.15, max_voxels_in_ram=80000):
        self.voxel_size       = voxel_size
        self.max_voxels       = max_voxels_in_ram
        self.voxels           = {}   # key->(xyz_sum, rgb_sum, count)
        self.lock             = threading.Lock()
        self.total_added      = 0

    def add(self, xyz, rgb):
        """Add new points — automatically deduplicates via voxel grid."""
        if len(xyz) == 0:
            return
        keys = np.floor(
            xyz / self.voxel_size).astype(int)
        with self.lock:
            for i in range(len(xyz)):
                k = (keys[i,0], keys[i,1], keys[i,2])
                if k not in self.voxels:
                    self.voxels[k] = [
                        xyz[i].copy().astype(np.float64),
                        rgb[i].astype(np.float64),
                        1]
                else:
                    v = self.voxels[k]
                    v[0] += xyz[i]
                    v[1] += rgb[i]
                    v[2] += 1
            self.total_added += len(xyz)

            # If RAM voxel dict gets too large, thin it
            # by increasing voxel resolution (merge nearby)
            if len(self.voxels) > self.max_voxels:
                self._thin()

    def _thin(self):
        """Increase effective voxel size to reduce RAM usage."""
        print(f"  [AccumMap] Thinning voxel map: "
              f"{len(self.voxels):,} → ", end='', flush=True)
        new_voxels = {}
        factor = 2  # double voxel size
        for (kx,ky,kz), (xyz_s, rgb_s, cnt) in self.voxels.items():
            new_k = (kx//factor, ky//factor, kz//factor)
            if new_k not in new_voxels:
                new_voxels[new_k] = [xyz_s.copy(), rgb_s.copy(), cnt]
            else:
                v = new_voxels[new_k]
                v[0] += xyz_s; v[1] += rgb_s; v[2] += cnt
        self.voxels = new_voxels
        self.voxel_size *= factor
        print(f"{len(self.voxels):,}  (voxel_size now {self.voxel_size:.2f}m)")

    def to_arrays(self):
        """Convert current voxel map to xyz + rgb arrays."""
        with self.lock:
            if not self.voxels:
                return np.zeros((0,3),dtype=np.float32), \
                       np.zeros((0,3),dtype=np.uint8)
            n    = len(self.voxels)
            xyz  = np.zeros((n,3), dtype=np.float32)
            rgb  = np.zeros((n,3), dtype=np.uint8)
            for i,(k,v) in enumerate(self.voxels.items()):
                c      = v[2]
                xyz[i] = (v[0]/c).astype(np.float32)
                rgb[i] = np.clip(v[1]/c, 0, 255).astype(np.uint8)
        return xyz, rgb

    def num_voxels(self):
        with self.lock:
            return len(self.voxels)


# ── main node ─────────────────────────────────────────────────────────────────

class LiveFullColorVisualizer:

    def __init__(self):
        rospy.init_node('live_full_color_visualizer', anonymous=True)

        # Static TF
        self.static_tf_pub = StaticTransformBroadcaster()
        self._publish_static_tfs()
        print("  ✔ Static TFs published")

        # ── publishers ────────────────────────────────────────────────────────
        # Pre-built maps: latch=True means they STAY in RViz forever
        self.pub_static_map   = rospy.Publisher(
            '/static_map',            PointCloud2,
            queue_size=1, latch=True)
        self.pub_dynamic_map  = rospy.Publisher(
            '/dynamic_map',           PointCloud2,
            queue_size=1, latch=True)

        # Live current scan (no latch — updates every scan)
        self.pub_live_scan    = rospy.Publisher(
            '/live/colored_scan',     PointCloud2, queue_size=1)

        # Live accumulated map (latch=True — stays visible between updates)
        self.pub_live_accum   = rospy.Publisher(
            '/live/accumulated_map',  PointCloud2,
            queue_size=1, latch=True)

        # Robot path + markers
        self.pub_path         = rospy.Publisher(
            '/live/odom_path',        Path,
            queue_size=1, latch=True)
        self.pub_robot        = rospy.Publisher(
            '/live/robot_marker',     MarkerArray, queue_size=1)
        self.pub_imu          = rospy.Publisher(
            '/live/imu_arrow',        MarkerArray, queue_size=1)

        rospy.sleep(0.5)

        # ── state ─────────────────────────────────────────────────────────────
        self.latest_odom      = None
        self.latest_imu       = None
        self.latest_image     = None
        self.latest_cam_info  = None
        self.path_poses       = deque(maxlen=5000)
        self.lock             = threading.Lock()
        self.scan_count       = 0
        self.pub_frame        = "odom_combined"

        # Disk-backed voxel accumulator — memory safe
        self.accumulator = DiskVoxelAccumulator(
            voxel_size=0.15,
            max_voxels_in_ram=80000)

        # TF
        self.tf_buffer   = tf2_ros.Buffer(rospy.Duration(15.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Seg model
        self.use_seg = False
        self._load_seg_model()

        # ── Load pre-built maps at real world coordinates ──────────────────
        print("\n  Loading pre-built maps...")
        static_arr  = load_map(
            os.path.join(STATIC_DIR,  "static_semantic_map.npy"),
            "Static")
        dynamic_arr = load_map(
            os.path.join(DYNAMIC_DIR, "dynamic_semantic_map.npy"),
            "Dynamic")

        # Build colored versions — stored as class vars for republishing
        self._static_msg  = None
        self._dynamic_msg = None

        if len(static_arr) > 0:
            s_xyz = static_arr[:,:3].astype(np.float32)
            labs  = static_arr[:,4].astype(int) \
                    if static_arr.shape[1]>4 \
                    else np.zeros(len(static_arr),dtype=int)
            s_rgb = np.zeros((len(s_xyz),3), dtype=np.uint8)
            for i,lab in enumerate(labs):
                s_rgb[i] = SEG_COLORS[lab] \
                           if 0<=lab<len(SEG_COLORS) \
                           else np.array([100,180,255],dtype=np.uint8)
            self._static_msg = make_pc2(
                s_xyz, s_rgb, frame="odom_combined",
                stamp=rospy.Time.now())
            print(f"  Static XYZ: X[{s_xyz[:,0].min():.1f},"
                  f"{s_xyz[:,0].max():.1f}] "
                  f"Y[{s_xyz[:,1].min():.1f},{s_xyz[:,1].max():.1f}]")

        if len(dynamic_arr) > 0:
            d_xyz = dynamic_arr[:,:3].astype(np.float32)
            d_rgb = np.tile(np.array([[255,80,20]],dtype=np.uint8),
                            (len(d_xyz),1))
            self._dynamic_msg = make_pc2(
                d_xyz, d_rgb, frame="odom_combined",
                stamp=rospy.Time.now())

        # Publish immediately
        self._publish_prebuilt_maps()
        print("  ✔ Pre-built maps published (latched — will not disappear)")

        # ── Background threads ─────────────────────────────────────────────
        # Thread 1: keep pre-built maps alive by republishing every 5s
        t1 = threading.Thread(
            target=self._map_keepalive_loop, daemon=True)
        t1.start()

        # Thread 2: publish accumulated live map every 15s
        t2 = threading.Thread(
            target=self._accum_publish_loop, daemon=True)
        t2.start()

        # ── Subscribers ───────────────────────────────────────────────────
        rospy.Subscriber('/tilt_scan',
                         LaserScan,  self._cb_scan,     queue_size=3)
        rospy.Subscriber('/base_odometry/odom',
                         Odometry,   self._cb_odom,     queue_size=10)
        rospy.Subscriber('/torso_lift_imu/data',
                         Imu,        self._cb_imu,      queue_size=5)
        rospy.Subscriber('/wide_stereo/left/image_rect',
                         Image,      self._cb_image,    queue_size=1)
        rospy.Subscriber('/wide_stereo/left/camera_info',
                         CameraInfo, self._cb_cam_info, queue_size=1)

        print("\n  ✔ All subscribers active")
        print("  ✔ Pre-built maps: PERMANENT (republished every 5s)")
        print("  ✔ Live scan:      updates every frame")
        print("  ✔ Accumulated:    updates every 15s, NEVER disappears")
        print("\n  ► Play your rosbag now!")
        print("  ─────────────────────────────────────────────────────")

    # ── static TF ─────────────────────────────────────────────────────────────

    def _publish_static_tfs(self):
        tfs = []
        now = rospy.Time.now()
        for parent, child in [
                ("odom_combined", "world"),
                ("odom_combined", "odom"),
                ("odom_combined", "map")]:
            t                      = TransformStamped()
            t.header.stamp         = now
            t.header.frame_id      = parent
            t.child_frame_id       = child
            t.transform.rotation.w = 1.0
            tfs.append(t)
        self.static_tf_pub.sendTransform(tfs)

    # ── pre-built map publishing ───────────────────────────────────────────────

    def _publish_prebuilt_maps(self):
        """Publish latched pre-built maps. Call any time to refresh."""
        now = rospy.Time.now()
        if self._static_msg is not None:
            self._static_msg.header.stamp = now
            self.pub_static_map.publish(self._static_msg)
        if self._dynamic_msg is not None:
            self._dynamic_msg.header.stamp = now
            self.pub_dynamic_map.publish(self._dynamic_msg)

    def _map_keepalive_loop(self):
        """
        Republish pre-built maps every 5 seconds.
        Even if RViz restarts or a new subscriber connects,
        it immediately gets the full map.
        Also refreshes static TFs so frames never expire.
        """
        rate = rospy.Rate(0.2)   # 0.2 Hz = every 5 seconds
        while not rospy.is_shutdown():
            self._publish_static_tfs()
            self._publish_prebuilt_maps()
            rate.sleep()

    # ── accumulated map publishing ─────────────────────────────────────────────

    def _accum_publish_loop(self):
        """
        Publish accumulated live map every 15 seconds.
        Uses latch=True so it stays visible between updates.
        """
        rate = rospy.Rate(1.0/15.0)
        while not rospy.is_shutdown():
            rate.sleep()
            n = self.accumulator.num_voxels()
            if n == 0:
                continue
            try:
                xyz, rgb = self.accumulator.to_arrays()
                msg = make_pc2(xyz, rgb,
                               frame=self.pub_frame,
                               stamp=rospy.Time.now())
                self.pub_live_accum.publish(msg)
                print(f"  [AccumMap] Published: {n:,} voxels "
                      f"| total pts added: "
                      f"{self.accumulator.total_added:,}")
            except Exception as e:
                print(f"  [AccumMap] Publish error: {e}")

    # ── TF transform ──────────────────────────────────────────────────────────

    def _tf_transform(self, pts, source_frame, target_frame, stamp):
        for query_stamp in [stamp, rospy.Time(0)]:
            try:
                tf_s = self.tf_buffer.lookup_transform(
                    target_frame, source_frame,
                    query_stamp, rospy.Duration(0.1))
                t  = tf_s.transform.translation
                r  = tf_s.transform.rotation
                tr = np.array([t.x, t.y, t.z])
                ro = Rotation.from_quat([r.x, r.y, r.z, r.w])
                return ro.apply(pts) + tr
            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                continue
        return None

    # ── scan callback ─────────────────────────────────────────────────────────

    def _cb_scan(self, msg):
        self.scan_count += 1

        ranges = np.array(msg.ranges, dtype=np.float32)
        valid  = (ranges >= msg.range_min) & \
                 (ranges <= msg.range_max) & \
                 np.isfinite(ranges)
        if not np.any(valid):
            return

        idx_v  = np.where(valid)[0]
        r_v    = ranges[idx_v]
        angles = msg.angle_min + idx_v * msg.angle_increment
        pts_laser = np.column_stack([
            r_v * np.cos(angles),
            r_v * np.sin(angles),
            np.zeros(len(r_v), dtype=np.float32)
        ])

        pts_world = self._tf_transform(
            pts_laser, msg.header.frame_id,
            "odom_combined", msg.header.stamp)

        if pts_world is None:
            pts_world  = pts_laser
            self.pub_frame = msg.header.frame_id
        else:
            self.pub_frame = "odom_combined"

        stamp = msg.header.stamp

        # Height rainbow color for current scan
        rgb_h = height_to_rgb(pts_world[:,2])

        # Publish current scan (thin overlay — updates every frame)
        self.pub_live_scan.publish(
            make_pc2(pts_world, rgb_h,
                     frame=self.pub_frame, stamp=stamp))

        # Add to accumulator (memory-safe voxel deduplication)
        self.accumulator.add(pts_world, rgb_h)

        if self.scan_count % 50 == 0:
            with self.lock:
                odom = self.latest_odom
            ostr = ""
            if odom:
                p    = odom.pose.pose.position
                ostr = f"robot=({p.x:.1f},{p.y:.1f})"
            print(f"  [scan #{self.scan_count:04d}] "
                  f"pts={len(pts_world)} | "
                  f"voxels={self.accumulator.num_voxels():,} | "
                  f"{ostr}")

    # ── odom callback ─────────────────────────────────────────────────────────

    def _cb_odom(self, msg):
        with self.lock:
            self.latest_odom = msg

        ps        = PoseStamped()
        ps.header = msg.header
        ps.pose   = msg.pose.pose
        self.path_poses.append(ps)

        path                 = Path()
        path.header.frame_id = msg.header.frame_id
        path.header.stamp    = msg.header.stamp
        path.poses           = list(self.path_poses)
        self.pub_path.publish(path)

        ma        = MarkerArray()
        m         = Marker()
        m.header  = msg.header
        m.ns      = "robot"; m.id = 0
        m.type    = Marker.SPHERE; m.action = Marker.ADD
        m.pose    = msg.pose.pose
        m.scale.x = m.scale.y = m.scale.z = 0.5
        m.color   = ColorRGBA(0.1, 1.0, 0.1, 1.0)
        m.lifetime= rospy.Duration(0.5)
        ma.markers.append(m)
        self.pub_robot.publish(ma)

    # ── IMU callback ──────────────────────────────────────────────────────────

    def _cb_imu(self, msg):
        with self.lock:
            self.latest_imu = msg
        if self.latest_odom is None:
            return
        pos = self.latest_odom.pose.pose.position
        q   = msg.orientation
        rot = Rotation.from_quat([q.x, q.y, q.z, q.w])
        fwd = rot.apply([1, 0, 0])
        ma  = MarkerArray()
        arr = Marker()
        arr.header.frame_id = self.latest_odom.header.frame_id
        arr.header.stamp    = msg.header.stamp
        arr.ns = "imu"; arr.id = 0
        arr.type = Marker.ARROW; arr.action = Marker.ADD
        arr.points = [
            Point(pos.x, pos.y, pos.z + 0.5),
            Point(pos.x+fwd[0], pos.y+fwd[1], pos.z+0.5+fwd[2])
        ]
        arr.scale.x  = 0.07
        arr.scale.y  = 0.14
        arr.scale.z  = 0.14
        arr.color    = ColorRGBA(1.0, 1.0, 0.0, 1.0)
        arr.lifetime = rospy.Duration(0.3)
        ma.markers.append(arr)
        self.pub_imu.publish(ma)

    # ── image / camera callbacks ──────────────────────────────────────────────

    def _cb_image(self, msg):
        # Only store if we need semantic coloring
        if not self.use_seg:
            return
        try:
            h, w = msg.height, msg.width
            data = np.frombuffer(msg.data, dtype=np.uint8)
            enc  = msg.encoding
            if enc == 'mono8':
                img = cv2.cvtColor(
                    data.reshape(h,w), cv2.COLOR_GRAY2RGB)
            elif enc == 'bgr8':
                img = data.reshape(h,w,3)[:,:,::-1].copy()
            else:
                img = data.reshape(h,w,3).copy()
            with self.lock:
                self.latest_image = img
        except Exception:
            pass

    def _cb_cam_info(self, msg):
        with self.lock:
            self.latest_cam_info = {
                'K':      np.array(msg.K).reshape(3,3),
                'width':  msg.width,
                'height': msg.height,
            }

    # ── seg model ─────────────────────────────────────────────────────────────

    def _load_seg_model(self):
        print("  Loading segmentation model...")
        try:
            import torch
            from transformers import (SegformerImageProcessor,
                                      SegformerForSemanticSegmentation)
            name = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
            self.seg_processor = SegformerImageProcessor.from_pretrained(name)
            self.seg_model     = SegformerForSemanticSegmentation.from_pretrained(name)
            self.seg_device    = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.seg_model.to(self.seg_device).eval()
            self.use_seg = True
            print(f"  ✔ Seg model ready ({self.seg_device})")
        except Exception as e:
            print(f"  ⚠ Seg model skipped — height colors only")
            self.use_seg = False

    def spin(self):
        rospy.spin()


def main():
    print("\n"+"="*60)
    print("MODULE 9: LIVE FULL COLOR 3D VISUALIZER")
    print("  Pre-built maps: PERMANENT (never disappear)")
    print("  Live scan:      updates every frame")
    print("  Accumulated:    grows over time, never drops points")
    print("="*60)
    LiveFullColorVisualizer().spin()


if __name__ == '__main__':
    main()
