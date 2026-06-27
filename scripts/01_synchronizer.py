#!/usr/bin/env python3
"""
Module 1: Multi-Sensor Time Synchronizer
Memory-safe version: streams frames directly to disk in chunks.
Never loads full bag into RAM at once.
"""

import rosbag
import numpy as np
import pickle
import os
import yaml
import gc
from collections import deque

CONFIG_PATH = os.path.expanduser(
    "~/catkin_ws/src/lidar_reconstruction/config/pipeline_config.yaml")
OUTPUT_DIR  = os.path.expanduser(
    "~/catkin_ws/src/lidar_reconstruction/output/debug")

def load_config():
    with open(CONFIG_PATH,'r') as f:
        return yaml.safe_load(f)

def scan_to_dict(msg):
    return {
        '_type':'LaserScan',
        'frame_id': msg.header.frame_id,
        'stamp': msg.header.stamp.to_sec(),
        'angle_min': float(msg.angle_min),
        'angle_max': float(msg.angle_max),
        'angle_increment': float(msg.angle_increment),
        'time_increment': float(msg.time_increment),
        'scan_time': float(msg.scan_time),
        'range_min': float(msg.range_min),
        'range_max': float(msg.range_max),
        'ranges': list(msg.ranges),
        'intensities': list(msg.intensities) if msg.intensities else [],
    }

def imu_to_dict(msg):
    return {
        '_type':'Imu',
        'frame_id': msg.header.frame_id,
        'stamp': msg.header.stamp.to_sec(),
        'orientation':{
            'x':float(msg.orientation.x),'y':float(msg.orientation.y),
            'z':float(msg.orientation.z),'w':float(msg.orientation.w)},
        'angular_velocity':{
            'x':float(msg.angular_velocity.x),
            'y':float(msg.angular_velocity.y),
            'z':float(msg.angular_velocity.z)},
        'linear_acceleration':{
            'x':float(msg.linear_acceleration.x),
            'y':float(msg.linear_acceleration.y),
            'z':float(msg.linear_acceleration.z)},
    }

def odom_to_dict(msg):
    return {
        '_type':'Odometry',
        'frame_id': msg.header.frame_id,
        'child_frame_id': msg.child_frame_id,
        'stamp': msg.header.stamp.to_sec(),
        'pose':{
            'position':{
                'x':float(msg.pose.pose.position.x),
                'y':float(msg.pose.pose.position.y),
                'z':float(msg.pose.pose.position.z)},
            'orientation':{
                'x':float(msg.pose.pose.orientation.x),
                'y':float(msg.pose.pose.orientation.y),
                'z':float(msg.pose.pose.orientation.z),
                'w':float(msg.pose.pose.orientation.w)}},
        'twist':{
            'linear':{
                'x':float(msg.twist.twist.linear.x),
                'y':float(msg.twist.twist.linear.y),
                'z':float(msg.twist.twist.linear.z)},
            'angular':{
                'x':float(msg.twist.twist.angular.x),
                'y':float(msg.twist.twist.angular.y),
                'z':float(msg.twist.twist.angular.z)}},
    }

def pose_cov_to_dict(msg):
    return {
        '_type':'PoseWithCovarianceStamped',
        'frame_id': msg.header.frame_id,
        'stamp': msg.header.stamp.to_sec(),
        'pose':{
            'position':{
                'x':float(msg.pose.pose.position.x),
                'y':float(msg.pose.pose.position.y),
                'z':float(msg.pose.pose.position.z)},
            'orientation':{
                'x':float(msg.pose.pose.orientation.x),
                'y':float(msg.pose.pose.orientation.y),
                'z':float(msg.pose.pose.orientation.z),
                'w':float(msg.pose.pose.orientation.w)}},
    }

def image_to_dict(msg, resize_factor=0.5):
    """Store image at reduced resolution to save RAM."""
    import cv2
    h, w = msg.height, msg.width
    data = np.frombuffer(msg.data, dtype=np.uint8)
    enc  = msg.encoding
    try:
        if enc == 'mono8':
            img = data.reshape(h, w)
        elif enc in ['rgb8','bgr8']:
            img = data.reshape(h, w, 3)
        else:
            ch  = max(len(data)//(h*w), 1)
            img = data.reshape(h, w, ch)

        if resize_factor < 1.0:
            new_w = int(w * resize_factor)
            new_h = int(h * resize_factor)
            img   = cv2.resize(img, (new_w, new_h),
                               interpolation=cv2.INTER_LINEAR)

        return {
            '_type':'Image',
            'frame_id': msg.header.frame_id,
            'stamp': msg.header.stamp.to_sec(),
            'height': img.shape[0],
            'width':  img.shape[1],
            'encoding': enc,
            'is_bigendian': int(msg.is_bigendian),
            'step': img.shape[1] * (img.shape[2] if img.ndim==3 else 1),
            'data': img.tobytes(),
        }
    except Exception:
        return None

def camera_info_to_dict(msg, resize_factor=0.5):
    K = list(msg.K)
    # Scale intrinsics if image was resized
    K[0] *= resize_factor; K[2] *= resize_factor  # fx, cx
    K[4] *= resize_factor; K[5] *= resize_factor  # fy, cy
    return {
        '_type':'CameraInfo',
        'frame_id': msg.header.frame_id,
        'stamp': msg.header.stamp.to_sec(),
        'height': int(msg.height * resize_factor),
        'width':  int(msg.width  * resize_factor),
        'distortion_model': str(msg.distortion_model),
        'D': list(msg.D), 'K': K,
        'R': list(msg.R), 'P': list(msg.P),
    }

def tf_to_dict(msg):
    transforms = []
    for tr in msg.transforms:
        transforms.append({
            'parent_frame': str(tr.header.frame_id),
            'child_frame':  str(tr.child_frame_id),
            'stamp': tr.header.stamp.to_sec(),
            'translation':{
                'x':float(tr.transform.translation.x),
                'y':float(tr.transform.translation.y),
                'z':float(tr.transform.translation.z)},
            'rotation':{
                'x':float(tr.transform.rotation.x),
                'y':float(tr.transform.rotation.y),
                'z':float(tr.transform.rotation.z),
                'w':float(tr.transform.rotation.w)},
        })
    return transforms

def nearest_message(buffer, target_time, tolerance):
    best = None; best_dt = float('inf')
    for msg_t, msg in buffer:
        dt = abs(msg_t - target_time)
        if dt < best_dt and dt < tolerance:
            best_dt = dt; best = (msg_t, msg)
    return best

class MultiSensorSynchronizer:
    def __init__(self, config):
        self.cfg          = config
        self.tol          = config['sync']['time_tolerance']
        self.bag_path     = config['bag']['path']
        self.start_offset = config['bag']['start_offset']
        self.duration     = config['bag']['duration']
        self.resize       = config['memory']['image_resize_factor']
        self.max_frames   = config['memory']['max_frames_in_ram']
        self.chunk_size   = config['memory']['chunk_size']

        self.buffers = {
            'tilt_scan':         deque(maxlen=20),
            'imu':               deque(maxlen=200),
            'odom':              deque(maxlen=100),
            'odom_combined':     deque(maxlen=100),
            'left_image':        deque(maxlen=5),
            'right_image':       deque(maxlen=5),
            'left_camera_info':  deque(maxlen=5),
            'right_camera_info': deque(maxlen=5),
        }
        self.topic_map = {
            config['topics']['tilt_scan']:        'tilt_scan',
            config['topics']['imu']:              'imu',
            config['topics']['odom']:             'odom',
            config['topics']['odom_combined']:    'odom_combined',
            config['topics']['left_image']:       'left_image',
            config['topics']['right_image']:      'right_image',
            config['topics']['left_camera_info']: 'left_camera_info',
            config['topics']['right_camera_info']:'right_camera_info',
        }
        self.converters = {
            'tilt_scan':         scan_to_dict,
            'imu':               imu_to_dict,
            'odom':              odom_to_dict,
            'odom_combined':     pose_cov_to_dict,
            'left_image':        lambda m: image_to_dict(m, self.resize),
            'right_image':       lambda m: image_to_dict(m, self.resize),
            'left_camera_info':  lambda m: camera_info_to_dict(m, self.resize),
            'right_camera_info': lambda m: camera_info_to_dict(m, self.resize),
        }

    def synchronize(self):
        print("\n"+"="*60)
        print("MODULE 1: SENSOR SYNCHRONIZATION (FULL BAG)")
        print("="*60)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        import rospy
        all_topics = list(self.topic_map.keys()) + ['/tf']

        with rosbag.Bag(self.bag_path,'r') as bag:
            bag_start  = bag.get_start_time()
            bag_end    = bag.get_end_time()
            target_start = bag_start + self.start_offset
            target_end   = bag_end if self.duration < 0 \
                           else target_start + self.duration
            total_dur = target_end - target_start

        print(f"  Bag duration : {total_dur:.1f}s")
        print(f"  Chunk size   : {self.chunk_size}s")
        n_chunks = int(np.ceil(total_dur / self.chunk_size))
        print(f"  Chunks       : {n_chunks}")

        all_chunk_paths = []
        tf_chunk_paths  = []
        total_frames    = 0

        for chunk_idx in range(n_chunks):
            chunk_start = target_start + chunk_idx * self.chunk_size
            chunk_end   = min(chunk_start + self.chunk_size, target_end)

            print(f"\n  --- Chunk {chunk_idx+1}/{n_chunks}: "
                  f"t=[{chunk_start-target_start:.0f}s, "
                  f"{chunk_end-target_start:.0f}s] ---")

            frames_chunk = []
            tf_chunk     = []
            frame_count  = 0
            scan_count   = 0

            for key in self.buffers:
                self.buffers[key].clear()

            with rosbag.Bag(self.bag_path,'r') as bag:
                for topic, msg, t in bag.read_messages(
                        topics=all_topics,
                        start_time=rospy.Time(chunk_start),
                        end_time=rospy.Time(chunk_end)):

                    ts = t.to_sec()

                    if topic == '/tf':
                        tf_chunk.append((ts, tf_to_dict(msg)))
                        continue

                    key = self.topic_map.get(topic)
                    if key is None:
                        continue
                    try:
                        plain = self.converters[key](msg)
                        if plain is not None:
                            self.buffers[key].append((ts, plain))
                    except Exception:
                        continue

                    if key == 'tilt_scan':
                        scan_count += 1
                        frame = self._try_sync(ts)
                        if frame is not None:
                            frame_count += 1
                            frames_chunk.append(frame)

            # Save chunk to disk immediately
            chunk_path = os.path.join(
                OUTPUT_DIR, f"sync_chunk_{chunk_idx:03d}.pkl")
            with open(chunk_path,'wb') as f:
                pickle.dump(frames_chunk, f, protocol=2)

            tf_path = os.path.join(
                OUTPUT_DIR, f"tf_chunk_{chunk_idx:03d}.pkl")
            with open(tf_path,'wb') as f:
                pickle.dump(tf_chunk, f, protocol=2)

            all_chunk_paths.append(chunk_path)
            tf_chunk_paths.append(tf_path)
            total_frames += frame_count

            print(f"    Frames: {frame_count} | Scans: {scan_count} | "
                  f"TF: {len(tf_chunk)}")

            # Free memory
            del frames_chunk, tf_chunk
            gc.collect()

        # Save index file
        index = {
            'chunk_paths':    all_chunk_paths,
            'tf_chunk_paths': tf_chunk_paths,
            'total_frames':   total_frames,
            'n_chunks':       n_chunks,
        }
        index_path = os.path.join(OUTPUT_DIR, "sync_index.pkl")
        with open(index_path,'wb') as f:
            pickle.dump(index, f, protocol=2)

        print(f"\n✔ Total frames synchronized : {total_frames}")
        print(f"✔ Chunks saved              : {n_chunks}")
        print(f"✔ Index saved               : {index_path}")
        print("\n✔ Module 1 COMPLETE\n")
        return index

    def _try_sync(self, scan_time):
        frame = {'timestamp': float(scan_time)}

        r = nearest_message(self.buffers['tilt_scan'], scan_time, self.tol)
        if r is None: return None
        frame['tilt_scan'] = {'time':r[0],'msg':r[1]}

        r = nearest_message(self.buffers['imu'], scan_time, self.tol)
        if r is None: return None
        frame['imu'] = {'time':r[0],'msg':r[1]}

        r = nearest_message(self.buffers['odom'], scan_time, self.tol)
        if r is None: return None
        frame['odom'] = {'time':r[0],'msg':r[1]}

        r = nearest_message(self.buffers['odom_combined'], scan_time, 0.1)
        frame['odom_combined'] = {'time':r[0],'msg':r[1]} if r else None

        r = nearest_message(self.buffers['left_image'], scan_time, 0.1)
        frame['left_image'] = {'time':r[0],'msg':r[1]} if r else None

        r = nearest_message(self.buffers['right_image'], scan_time, 0.1)
        frame['right_image'] = {'time':r[0],'msg':r[1]} if r else None

        r = nearest_message(self.buffers['left_camera_info'], scan_time, 0.2)
        frame['left_camera_info'] = {'time':r[0],'msg':r[1]} if r else None

        imu_win = [(t,m) for t,m in self.buffers['imu']
                   if abs(t-scan_time) < 0.06]
        frame['imu_window'] = sorted(imu_win, key=lambda x:x[0])
        return frame

def main():
    cfg    = load_config()
    syncer = MultiSensorSynchronizer(cfg)
    index  = syncer.synchronize()
    print(f"  Total frames: {index['total_frames']}")

if __name__ == '__main__':
    main()
