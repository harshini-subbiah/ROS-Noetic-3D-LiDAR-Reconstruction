#!/usr/bin/env python3
"""
Module 7: Dual Semantic Map Builder — streaming, memory safe.
Processes chunks one at a time, merges into final voxel maps.
Never loads all frames into RAM simultaneously.
"""

import numpy as np
import pickle
import os
import yaml
import gc
from collections import defaultdict, Counter

CONFIG_PATH  = os.path.expanduser(
    "~/catkin_ws/src/lidar_reconstruction/config/pipeline_config.yaml")
DEBUG_DIR    = os.path.expanduser(
    "~/catkin_ws/src/lidar_reconstruction/output/debug")
STATIC_DIR   = os.path.expanduser(
    "~/catkin_ws/src/lidar_reconstruction/output/static_map")
DYNAMIC_DIR  = os.path.expanduser(
    "~/catkin_ws/src/lidar_reconstruction/output/dynamic_map")

CITYSCAPES_COLORS = np.array([
    [0.502,0.251,0.502],[0.957,0.137,0.910],[0.275,0.275,0.275],
    [0.400,0.400,0.612],[0.745,0.600,0.600],[0.600,0.600,0.600],
    [0.980,0.667,0.118],[0.863,0.863,0.000],[0.420,0.557,0.137],
    [0.596,0.984,0.596],[0.275,0.510,0.706],[0.863,0.078,0.235],
    [1.000,0.000,0.000],[0.000,0.000,0.557],[0.000,0.000,0.275],
    [0.000,0.235,0.392],[0.000,0.314,0.392],[0.000,0.000,0.902],
    [0.467,0.043,0.125]
])

def load_config():
    with open(CONFIG_PATH,'r') as f:
        return yaml.safe_load(f)

class StreamingVoxelMap:
    """
    Memory-efficient voxel map.
    Accumulates from chunks, never stores raw points.
    """
    def __init__(self, voxel_size=0.1, max_per_voxel=5):
        self.voxel_size   = voxel_size
        self.max_per_voxel= max_per_voxel
        # key -> [xyz_sum(3), intensity_sum, label_list, count]
        self.voxels = {}

    def add_chunk(self, pts_xyzil):
        if len(pts_xyzil)==0: return
        xyz  = pts_xyzil[:,:3]
        intens = pts_xyzil[:,3] if pts_xyzil.shape[1]>3 \
                 else np.zeros(len(xyz))
        labs = pts_xyzil[:,4].astype(int) if pts_xyzil.shape[1]>4 \
               else np.full(len(xyz),255)

        keys = np.floor(xyz/self.voxel_size).astype(int)
        for i in range(len(xyz)):
            k = (keys[i,0], keys[i,1], keys[i,2])
            if k not in self.voxels:
                self.voxels[k] = [xyz[i].copy(), float(intens[i]),
                                  [int(labs[i])], 1]
            else:
                v = self.voxels[k]
                if v[3] < self.max_per_voxel:
                    v[0] += xyz[i]; v[1] += float(intens[i])
                    v[2].append(int(labs[i])); v[3] += 1

    def to_array(self):
        if not self.voxels: return np.zeros((0,5))
        rows = []
        for v in self.voxels.values():
            c   = v[3]
            xyz = v[0]/c
            it  = v[1]/c
            vl  = [l for l in v[2] if l!=255]
            lab = Counter(vl).most_common(1)[0][0] if vl else 255
            rows.append([xyz[0],xyz[1],xyz[2],it,float(lab)])
        return np.array(rows, dtype=np.float32)

    def num_voxels(self):
        return len(self.voxels)

def save_ply(pts, path):
    if len(pts)==0: return
    xyz  = pts[:,:3]
    labs = pts[:,4].astype(int) if pts.shape[1]>4 \
           else np.zeros(len(pts),dtype=int)
    with open(path,'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\n")
        f.write("property uchar blue\nproperty int label\n")
        f.write("end_header\n")
        for i in range(len(xyz)):
            lab = labs[i]
            if 0<=lab<len(CITYSCAPES_COLORS):
                r,g,b=(CITYSCAPES_COLORS[lab]*255).astype(int)
            else:
                r,g,b=100,180,255
            f.write(f"{xyz[i,0]:.4f} {xyz[i,1]:.4f} {xyz[i,2]:.4f} "
                    f"{r} {g} {b} {lab}\n")

def main():
    cfg = load_config()
    print("\n"+"="*60)
    print("MODULE 7: DUAL SEMANTIC MAP BUILDER (FULL BAG)")
    print("="*60)
    os.makedirs(STATIC_DIR,  exist_ok=True)
    os.makedirs(DYNAMIC_DIR, exist_ok=True)

    index_path = os.path.join(DEBUG_DIR,"sync_index.pkl")
    with open(index_path,'rb') as f:
        index = pickle.load(f)

    map_cfg     = cfg['map']
    static_map  = StreamingVoxelMap(map_cfg['voxel_size'],
                                    map_cfg['max_points_per_voxel'])
    dynamic_map = StreamingVoxelMap(map_cfg['voxel_size'],
                                    map_cfg['max_points_per_voxel'])

    frames_done = 0

    for i, cpath in enumerate(index['fused_paths']):
        print(f"  Chunk {i+1}/{len(index['fused_paths'])} | "
              f"static voxels: {static_map.num_voxels():,} | "
              f"dynamic voxels: {dynamic_map.num_voxels():,}")

        with open(cpath,'rb') as f:
            frames = pickle.load(f)

        for frame in frames:
            sp = frame.get('static_semantic',  np.zeros((0,5)))
            dp = frame.get('dynamic_semantic', np.zeros((0,5)))

            # Pad to 5 columns if needed
            if len(sp)>0 and sp.shape[1]==4:
                sp = np.hstack([sp, np.full((len(sp),1),255.0)])
            if len(dp)>0 and dp.shape[1]==4:
                dp = np.hstack([dp, np.full((len(dp),1),255.0)])

            if len(sp)>0: static_map.add_chunk(sp)
            if len(dp)>0: dynamic_map.add_chunk(dp)
            frames_done += 1

        del frames
        gc.collect()

    print(f"\n  Converting voxel maps to arrays...")
    static_arr  = static_map.to_array()
    dynamic_arr = dynamic_map.to_array()

    print(f"  Static map  : {len(static_arr):,} voxelized points")
    print(f"  Dynamic map : {len(dynamic_arr):,} voxelized points")

    # Save NPY
    np.save(os.path.join(STATIC_DIR,  "static_semantic_map.npy"),  static_arr)
    np.save(os.path.join(DYNAMIC_DIR, "dynamic_semantic_map.npy"), dynamic_arr)

    # Save PLY
    save_ply(static_arr,  os.path.join(STATIC_DIR,  "static_semantic_map.ply"))
    save_ply(dynamic_arr, os.path.join(DYNAMIC_DIR, "dynamic_semantic_map.ply"))

    # Combined
    combined = np.vstack([static_arr,dynamic_arr]) \
               if len(static_arr)>0 and len(dynamic_arr)>0 \
               else (static_arr if len(static_arr)>0 else dynamic_arr)
    np.save(os.path.join(DEBUG_DIR,"combined_map.npy"), combined)
    save_ply(combined, os.path.join(DEBUG_DIR,"combined_semantic_map.ply"))

    # Statistics
    stats_path = os.path.join(DEBUG_DIR,"map_statistics.txt")
    with open(stats_path,'w') as f:
        for name, arr in [("STATIC", static_arr),("DYNAMIC", dynamic_arr)]:
            f.write(f"=== {name} MAP ===\n")
            f.write(f"Points: {len(arr)}\n")
            if len(arr)>0:
                labs = arr[:,4].astype(int)
                for lab,cnt in Counter(labs).most_common():
                    f.write(f"  class {lab}: {cnt}\n")
            f.write("\n")

    print(f"\n✔ Frames processed : {frames_done}")
    print(f"✔ Static  → {os.path.join(STATIC_DIR,'static_semantic_map.ply')}")
    print(f"✔ Dynamic → {os.path.join(DYNAMIC_DIR,'dynamic_semantic_map.ply')}")
    print(f"✔ Stats   → {stats_path}")
    print("\n✔ Module 7 COMPLETE\n")

if __name__ == '__main__':
    main()
