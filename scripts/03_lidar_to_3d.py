#!/usr/bin/env python3
"""
Module 3: Tilt 2D LiDAR -> 3D — chunk-based, memory safe.
"""

import numpy as np
import pickle
import os
import yaml
import gc
from scipy.spatial.transform import Rotation, Slerp
from collections import defaultdict

CONFIG_PATH = os.path.expanduser(
    "~/catkin_ws/src/lidar_reconstruction/config/pipeline_config.yaml")
DEBUG_DIR   = os.path.expanduser(
    "~/catkin_ws/src/lidar_reconstruction/output/debug")

def load_config():
    with open(CONFIG_PATH,'r') as f:
        return yaml.safe_load(f)

class TFBuffer:
    def __init__(self, tf_messages):
        self.transforms = defaultdict(list)
        for ts, transform_list in tf_messages:
            for tr in transform_list:
                key   = (tr['parent_frame'], tr['child_frame'])
                t     = tr['translation']; r = tr['rotation']
                trans = np.array([t['x'],t['y'],t['z']])
                rot   = Rotation.from_quat([r['x'],r['y'],r['z'],r['w']])
                self.transforms[key].append((float(tr['stamp']),trans,rot))
        for key in self.transforms:
            self.transforms[key].sort(key=lambda x:x[0])

    def _interp(self, key, t):
        entries = self.transforms[key]
        if not entries: return None,None
        if len(entries)==1: return entries[0][1],entries[0][2]
        times = np.array([e[0] for e in entries])
        idx   = np.clip(np.searchsorted(times,t),1,len(entries)-1)
        t0ts,t0p,t0r = entries[idx-1]
        t1ts,t1p,t1r = entries[idx]
        alpha = np.clip((t-t0ts)/(t1ts-t0ts+1e-12),0,1)
        pos   = (1-alpha)*t0p + alpha*t1p
        rot   = Slerp([0,1],Rotation.concatenate([t0r,t1r]))([alpha])[0]
        return pos, rot

    def lookup(self, parent, child, t):
        key = (parent, child)
        if key in self.transforms: return self._interp(key,t)
        rev = (child, parent)
        if rev in self.transforms:
            p,r = self._interp(rev,t)
            if p is not None: return -r.inv().apply(p), r.inv()
        return None, None

    def chain(self, source, target, t):
        p,r = self.lookup(target, source, t)
        if p is not None: return p,r
        for mid in ['base_footprint','base_link','odom']:
            p1,r1 = self.lookup(mid, source, t)
            p2,r2 = self.lookup(target, mid, t)
            if p1 is not None and p2 is not None:
                return r2.apply(p1)+p2, r2*r1
        return None, None

def voxel_ds(points, vsize):
    if len(points)==0: return points
    keys = np.floor(points[:,:3]/vsize).astype(int)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[idx]

def scan_to_3d(c2d, tt, tr, wt, wr, cfg):
    if len(c2d)==0: return np.zeros((0,4))
    pts_base = (tr.apply(c2d)+tt) if tr is not None \
               else Rotation.from_euler('y',np.radians(30)).apply(c2d) \
                    + np.array([0.1,0.0,1.2])
    pts_world = (wr.apply(pts_base)+wt) if wr is not None else pts_base
    d   = np.linalg.norm(pts_world[:,:2],axis=1)
    h   = pts_world[:,2]
    v   = (d>=cfg['lidar_3d']['min_range']) & \
          (d<=cfg['lidar_3d']['max_range']) & \
          (h>-0.5) & (h<5.0)
    pts = pts_world[v]
    if len(pts)==0: return np.zeros((0,4))
    return np.hstack([pts, np.linalg.norm(pts,axis=1,keepdims=True)])

def process_chunk(frames, tf_buf, cfg):
    out = []
    for frame in frames:
        c2d = frame.get('corrected_2d_points')
        if c2d is None or len(c2d)==0:
            nf = dict(frame)
            nf['points_3d']       = np.zeros((0,4))
            nf['tilt_transform']  = (None,None)
            nf['world_transform'] = (None,None)
            out.append(nf); continue

        ts = float(frame['timestamp'])
        tt, tr = tf_buf.chain('laser_tilt_link','base_footprint',ts)
        wt, wr = tf_buf.chain('base_footprint','odom_combined',ts)

        if wt is None and frame.get('odom_combined'):
            od = frame['odom_combined']['msg']
            p  = od['pose']['position']; o = od['pose']['orientation']
            wt = np.array([p['x'],p['y'],p['z']])
            wr = Rotation.from_quat([o['x'],o['y'],o['z'],o['w']])

        pts3d = scan_to_3d(c2d,tt,tr,wt,wr,cfg)
        pts3d = voxel_ds(pts3d, cfg['lidar_3d']['voxel_size'])

        nf = dict(frame)
        nf['points_3d']       = pts3d
        nf['tilt_transform']  = (tt,tr)
        nf['world_transform'] = (wt,wr)
        out.append(nf)
    return out

def main():
    cfg = load_config()
    print("\n"+"="*60)
    print("MODULE 3: 2D LIDAR -> 3D (FULL BAG)")
    print("="*60)

    index_path = os.path.join(DEBUG_DIR,"sync_index.pkl")
    with open(index_path,'rb') as f:
        index = pickle.load(f)

    comp_paths   = index['comp_paths']
    tf_paths     = index['tf_chunk_paths']
    frames3d_paths = []
    total_pts    = 0

    for i,(cpath,tpath) in enumerate(zip(comp_paths,tf_paths)):
        print(f"  Chunk {i+1}/{len(comp_paths)}")
        with open(cpath,'rb') as f: frames = pickle.load(f)
        with open(tpath,'rb') as f: tf_raw = pickle.load(f)

        tf_buf = TFBuffer(tf_raw)
        out    = process_chunk(frames, tf_buf, cfg)
        total_pts += sum(len(fr['points_3d']) for fr in out)

        out_path = os.path.join(DEBUG_DIR, f"3d_chunk_{i:03d}.pkl")
        with open(out_path,'wb') as f:
            pickle.dump(out, f, protocol=2)
        frames3d_paths.append(out_path)

        del frames, tf_raw, tf_buf, out
        gc.collect()

    index['frames3d_paths'] = frames3d_paths
    with open(index_path,'wb') as f:
        pickle.dump(index, f, protocol=2)

    print(f"\n✔ Total 3D points : {total_pts:,}")
    print(f"✔ Chunks saved    : {len(frames3d_paths)}")
    print("\n✔ Module 3 COMPLETE\n")

if __name__ == '__main__':
    main()
