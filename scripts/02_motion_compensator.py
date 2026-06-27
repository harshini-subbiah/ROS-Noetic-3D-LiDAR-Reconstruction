#!/usr/bin/env python3
"""
Module 2: Motion Distortion Correction — chunk-based, memory safe.
Reads sync chunks one at a time, processes, saves compensated chunks.
"""

import numpy as np
import pickle
import os
import yaml
import gc
from scipy.spatial.transform import Rotation, Slerp

CONFIG_PATH = os.path.expanduser(
    "~/catkin_ws/src/lidar_reconstruction/config/pipeline_config.yaml")
DEBUG_DIR   = os.path.expanduser(
    "~/catkin_ws/src/lidar_reconstruction/output/debug")

def load_config():
    with open(CONFIG_PATH,'r') as f:
        return yaml.safe_load(f)

def get_quat(d):
    o = d['orientation']
    return [o['x'],o['y'],o['z'],o['w']]

def get_angular_vel(d):
    v = d['angular_velocity']
    return np.array([v['x'],v['y'],v['z']])

def get_odom_position(d):
    p = d['pose']['position']
    return np.array([p['x'],p['y'],p['z']])

def get_odom_orientation(d):
    return d['pose']['orientation']

def get_odom_linear_vel(d):
    v = d['twist']['linear']
    return np.array([v['x'],v['y'],v['z']])

def interpolate_pose(t_query, t_list, pos_list, rot_list):
    if len(t_list) < 2:
        return pos_list[0], rot_list[0]
    t_arr = np.array(t_list)
    idx   = np.clip(np.searchsorted(t_arr, t_query), 1, len(t_arr)-1)
    t0, t1 = t_arr[idx-1], t_arr[idx]
    alpha  = np.clip((t_query-t0)/(t1-t0+1e-12), 0.0, 1.0)
    pos    = (1-alpha)*pos_list[idx-1] + alpha*pos_list[idx]
    rot    = Slerp([0,1], Rotation.concatenate(
                   [rot_list[idx-1], rot_list[idx]]))([alpha])[0]
    return pos, rot

def build_imu_trajectory(imu_window, odom_dict, scan_time):
    if not imu_window:
        return None
    base_pos = get_odom_position(odom_dict)
    ori_d    = get_odom_orientation(odom_dict)
    base_rot = Rotation.from_quat(
        [ori_d['x'],ori_d['y'],ori_d['z'],ori_d['w']])
    base_vel = get_odom_linear_vel(odom_dict)
    times, positions, rotations = [], [], []
    cur_pos = base_pos.copy(); cur_rot = base_rot
    times.append(imu_window[0][0])
    positions.append(cur_pos.copy())
    rotations.append(cur_rot)
    for i in range(1, len(imu_window)):
        t_prev, imu_prev = imu_window[i-1]
        t_curr, imu_curr = imu_window[i]
        dt = t_curr - t_prev
        if dt <= 0 or dt > 0.05:
            continue
        omega = get_angular_vel(imu_curr)
        angle = np.linalg.norm(omega) * dt
        if angle > 1e-10:
            cur_rot = cur_rot * Rotation.from_rotvec(
                omega/np.linalg.norm(omega)*angle)
        cur_pos = cur_pos + cur_rot.apply(base_vel) * dt
        times.append(t_curr)
        positions.append(cur_pos.copy())
        rotations.append(cur_rot)
    return times, positions, rotations

def correct_scan(scan_dict, imu_window, odom_dict, scan_time):
    angle_min  = scan_dict['angle_min']
    angle_inc  = scan_dict['angle_increment']
    ranges     = np.array(scan_dict['ranges'], dtype=np.float32)
    range_min  = scan_dict['range_min']
    range_max  = scan_dict['range_max']
    scan_rate  = 20.0
    n_beams    = len(ranges)
    scan_dur   = 1.0/scan_rate
    beam_angles= angle_min + np.arange(n_beams)*angle_inc
    beam_times = scan_time - scan_dur + np.linspace(0, scan_dur, n_beams)
    valid      = (ranges>=range_min)&(ranges<=range_max)&np.isfinite(ranges)
    traj       = build_imu_trajectory(imu_window, odom_dict, scan_time) \
                 if len(imu_window) >= 2 else None
    corrected, raw = [], []
    for i in range(n_beams):
        if not valid[i]:
            continue
        r = ranges[i]; a = beam_angles[i]
        p = np.array([r*np.cos(a), r*np.sin(a), 0.0])
        raw.append(p)
        if traj is None:
            corrected.append(p)
            continue
        t_t, t_p, t_r = traj
        pos_b, rot_b = interpolate_pose(beam_times[i],t_t,t_p,t_r)
        pos_r, rot_r = interpolate_pose(scan_time,    t_t,t_p,t_r)
        corrected.append(rot_r.inv().apply(rot_b.apply(p)+pos_b-pos_r))
    cp = np.array(corrected) if corrected else np.zeros((0,3))
    rp = np.array(raw)       if raw       else np.zeros((0,3))
    return cp, rp

def process_chunk(frames):
    out = []
    for frame in frames:
        s = frame.get('tilt_scan')
        i = frame.get('imu')
        o = frame.get('odom')
        if s is None or i is None or o is None:
            continue
        cp, rp = correct_scan(
            s['msg'], frame.get('imu_window',[]),
            o['msg'], float(frame['timestamp']))
        nf = dict(frame)
        nf['corrected_2d_points'] = cp
        nf['raw_2d_points']       = rp
        out.append(nf)
    return out

def main():
    cfg = load_config()
    print("\n"+"="*60)
    print("MODULE 2: MOTION DISTORTION CORRECTION (FULL BAG)")
    print("="*60)

    index_path = os.path.join(DEBUG_DIR, "sync_index.pkl")
    with open(index_path,'rb') as f:
        index = pickle.load(f)

    chunk_paths = index['chunk_paths']
    comp_paths  = []
    total_frames= 0
    total_corrected = 0

    for i, cpath in enumerate(chunk_paths):
        print(f"  Chunk {i+1}/{len(chunk_paths)}: {os.path.basename(cpath)}")
        with open(cpath,'rb') as f:
            frames = pickle.load(f)

        out = process_chunk(frames)
        total_frames    += len(out)
        total_corrected += sum(1 for fr in out
                               if len(fr.get('imu_window',[])) >= 2)

        out_path = os.path.join(
            DEBUG_DIR, f"comp_chunk_{i:03d}.pkl")
        with open(out_path,'wb') as f:
            pickle.dump(out, f, protocol=2)
        comp_paths.append(out_path)

        del frames, out
        gc.collect()

    # Save index
    index['comp_paths'] = comp_paths
    with open(index_path,'wb') as f:
        pickle.dump(index, f, protocol=2)

    print(f"\n✔ Frames processed  : {total_frames}")
    print(f"✔ Motion-corrected  : {total_corrected}")
    print(f"✔ Chunks saved      : {len(comp_paths)}")
    print("\n✔ Module 2 COMPLETE\n")

if __name__ == '__main__':
    main()
