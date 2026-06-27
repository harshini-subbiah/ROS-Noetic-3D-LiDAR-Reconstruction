#!/usr/bin/env python3
"""
Module 0: Rosbag Inspector
Analyzes the bag file and outputs topic statistics, time ranges, and sample data.
"""

import rosbag
import sys
import os
import numpy as np
from collections import defaultdict

BAG_PATH = os.path.expanduser("~/bagfiles/2011-01-26-06-56-04.bag")
OUTPUT_DIR = os.path.expanduser("~/catkin_ws/src/lidar_reconstruction/output/debug")

def inspect_bag(bag_path):
    print("=" * 70)
    print("ROSBAG INSPECTOR")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with rosbag.Bag(bag_path, 'r') as bag:
        info = bag.get_type_and_topic_info()
        start_time = bag.get_start_time()
        end_time = bag.get_end_time()

        print(f"\nBag: {bag_path}")
        print(f"Duration: {end_time - start_time:.2f}s")
        print(f"Start: {start_time:.3f}")
        print(f"End:   {end_time:.3f}")
        print(f"\n{'Topic':<55} {'Type':<35} {'Count'}")
        print("-" * 110)

        topic_stats = defaultdict(lambda: {'count': 0, 'first_t': None, 'last_t': None, 'freq': 0})

        for topic, msg, t in bag.read_messages():
            ts = t.to_sec()
            s = topic_stats[topic]
            s['count'] += 1
            if s['first_t'] is None:
                s['first_t'] = ts
            s['last_t'] = ts

        for topic, tinfo in info.topics.items():
            s = topic_stats[topic]
            dur = (s['last_t'] - s['first_t']) if s['first_t'] and s['last_t'] else 1
            freq = s['count'] / dur if dur > 0 else 0
            print(f"{topic:<55} {tinfo.msg_type:<35} {tinfo.message_count:>6}  ~{freq:.1f}Hz")

        # Sample first tilt_scan message
        print("\n--- Tilt Scan Sample ---")
        for topic, msg, t in bag.read_messages(topics=['/tilt_scan']):
            print(f"  frame_id: {msg.header.frame_id}")
            print(f"  angle_min: {msg.angle_min:.4f} rad  angle_max: {msg.angle_max:.4f} rad")
            print(f"  angle_increment: {msg.angle_increment:.6f} rad")
            print(f"  range_min: {msg.range_min}  range_max: {msg.range_max}")
            print(f"  ranges count: {len(msg.ranges)}")
            break

        # Sample IMU
        print("\n--- IMU Sample ---")
        for topic, msg, t in bag.read_messages(topics=['/torso_lift_imu/data']):
            print(f"  frame_id: {msg.header.frame_id}")
            print(f"  linear_accel: ({msg.linear_acceleration.x:.3f}, {msg.linear_acceleration.y:.3f}, {msg.linear_acceleration.z:.3f})")
            print(f"  angular_vel:  ({msg.angular_velocity.x:.3f}, {msg.angular_velocity.y:.3f}, {msg.angular_velocity.z:.3f})")
            break

        # Sample Odometry
        print("\n--- Odometry Sample ---")
        for topic, msg, t in bag.read_messages(topics=['/base_odometry/odom']):
            print(f"  frame_id: {msg.header.frame_id}")
            print(f"  child_frame: {msg.child_frame_id}")
            print(f"  position: ({msg.pose.pose.position.x:.3f}, {msg.pose.pose.position.y:.3f})")
            break

        # Sample camera info
        print("\n--- Camera Info Sample ---")
        for topic, msg, t in bag.read_messages(topics=['/wide_stereo/left/camera_info']):
            print(f"  frame_id: {msg.header.frame_id}")
            print(f"  width: {msg.width}  height: {msg.height}")
            print(f"  K (intrinsic): {list(msg.K)}")
            break

        # TF frames
        print("\n--- TF Frames Observed ---")
        tf_frames = set()
        count = 0
        for topic, msg, t in bag.read_messages(topics=['/tf']):
            for transform in msg.transforms:
                tf_frames.add((transform.header.frame_id, transform.child_frame_id))
            count += 1
            if count > 200:
                break
        for parent, child in sorted(tf_frames):
            print(f"  {parent} -> {child}")

    # Save inspection report
    report_path = os.path.join(OUTPUT_DIR, "bag_inspection_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"Bag: {bag_path}\n")
        f.write(f"Duration: {end_time - start_time:.2f}s\n")
        for topic, s in topic_stats.items():
            f.write(f"{topic}: {s['count']} msgs\n")
    print(f"\n✔ Report saved: {report_path}")
    print("=" * 70)

if __name__ == '__main__':
    inspect_bag(BAG_PATH)
