#!/usr/bin/env python3
"""
Module 10: Floor Separation using Trajectory Discontinuity
Since PR2 odom_combined is 2D (Z always 0), we detect floor
transitions by finding sudden large position jumps in X,Y
that indicate the robot used the lift (odometry resets/jumps).
"""

import numpy as np
import os
import json
import pickle
import glob
import gc
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

BASE        = os.path.expanduser(
    "~/catkin_ws/src/lidar_reconstruction/output")
DEBUG       = f"{BASE}/debug"
STATIC_DIR  = f"{BASE}/static_map"
DYNAMIC_DIR = f"{BASE}/dynamic_map"
FLOOR1_DIR  = f"{BASE}/floor1"
FLOOR2_DIR  = f"{BASE}/floor2"
LIFT_DIR    = f"{BASE}/lift_transition"
PLOTS       = f"{DEBUG}/plots"

CITYSCAPES_COLORS = np.array([
    [0.502,0.251,0.502],[0.957,0.137,0.910],[0.275,0.275,0.275],
    [0.400,0.400,0.612],[0.745,0.600,0.600],[0.600,0.600,0.600],
    [0.980,0.667,0.118],[0.863,0.863,0.000],[0.420,0.557,0.137],
    [0.596,0.984,0.596],[0.275,0.510,0.706],[0.863,0.078,0.235],
    [1.000,0.000,0.000],[0.000,0.000,0.557],[0.000,0.000,0.275],
    [0.000,0.235,0.392],[0.000,0.314,0.392],[0.000,0.000,0.902],
    [0.467,0.043,0.125]
])

BG   = '#0d0d1a'
GRID = '#444444'

def style_ax(ax):
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.grid(True, alpha=0.2, color=GRID)

def find_lift_transitions(debug_dir,
                          jump_threshold_m=5.0,
                          time_threshold_s=5.0):
    """
    Find lift transitions by detecting sudden large jumps
    in robot XY position that happen quickly.
    PR2 odometry resets or jumps when using the lift.
    Returns list of transition timestamps.
    """
    print("\n  Scanning trajectory for lift transitions...")

    sync_chunks = sorted(glob.glob(f"{debug_dir}/sync_chunk_*.pkl"))
    all_times   = []
    all_x       = []
    all_y       = []

    # Collect all odometry positions
    for cp in sync_chunks:
        if not os.path.exists(cp):
            continue
        with open(cp,'rb') as f:
            frames = pickle.load(f)
        for fr in frames[::5]:  # sample every 5th frame
            od = fr.get('odom_combined') or fr.get('odom')
            if od is None:
                continue
            p = od['msg']['pose']['position']
            all_times.append(float(fr['timestamp']))
            all_x.append(float(p['x']))
            all_y.append(float(p['y']))
        del frames
        gc.collect()

    if len(all_times) < 2:
        print("  ⚠ Not enough odometry data")
        return [], all_times, all_x, all_y

    all_times = np.array(all_times)
    all_x     = np.array(all_x)
    all_y     = np.array(all_y)

    # Sort by time
    sort_idx  = np.argsort(all_times)
    all_times = all_times[sort_idx]
    all_x     = all_x[sort_idx]
    all_y     = all_y[sort_idx]

    # Compute inter-frame distances and times
    dx      = np.diff(all_x)
    dy      = np.diff(all_y)
    dt      = np.diff(all_times)
    dists   = np.sqrt(dx**2 + dy**2)
    speeds  = dists / np.maximum(dt, 0.001)

    # Detect jumps: large distance in short time
    # These indicate odometry discontinuity = lift usage
    jump_mask = (dists > jump_threshold_m) & \
                (dt     < time_threshold_s)
    jump_indices = np.where(jump_mask)[0]

    transitions = []
    for idx in jump_indices:
        t_jump   = all_times[idx + 1]
        x_before = all_x[idx]
        y_before = all_y[idx]
        x_after  = all_x[idx + 1]
        y_after  = all_y[idx + 1]
        transitions.append({
            'time':     float(t_jump),
            'dist':     float(dists[idx]),
            'dt':       float(dt[idx]),
            'x_before': float(x_before),
            'y_before': float(y_before),
            'x_after':  float(x_after),
            'y_after':  float(y_after),
        })
        print(f"  ✔ Transition at t={t_jump:.2f}s: "
              f"jump {dists[idx]:.1f}m in {dt[idx]:.2f}s")
        print(f"    Before: ({x_before:.1f}, {y_before:.1f})")
        print(f"    After : ({x_after:.1f},  {y_after:.1f})")

    if not transitions:
        print("  ⚠ No sudden jumps found with current thresholds")
        print(f"  Max single jump observed: {dists.max():.2f}m")
        print(f"  Trying lower threshold...")
        # Try with lower threshold
        lower_mask = (dists > dists.max()*0.5) & (dt < time_threshold_s*2)
        lower_idx  = np.where(lower_mask)[0]
        for idx in lower_idx[:3]:
            t_jump = all_times[idx+1]
            transitions.append({
                'time':     float(t_jump),
                'dist':     float(dists[idx]),
                'dt':       float(dt[idx]),
                'x_before': float(all_x[idx]),
                'y_before': float(all_y[idx]),
                'x_after':  float(all_x[idx+1]),
                'y_after':  float(all_y[idx+1]),
            })
            print(f"  (Lower threshold) Transition at t={t_jump:.2f}s: "
                  f"jump {dists[idx]:.1f}m")

    return transitions, all_times, all_x, all_y

def split_frames_by_transitions(debug_dir, transitions, lift_window_s=30.0):
    """
    Split all fused frames into floors based on transition timestamps.
    For N transitions, there are N+1 floors/regions.
    The lift_window_s seconds around each transition = lift zone.
    """
    if not transitions:
        print("  ⚠ No transitions — cannot split floors")
        return None

    # Sort transitions by time
    transitions = sorted(transitions, key=lambda x: x['time'])
    t_transitions = [tr['time'] for tr in transitions]

    print(f"\n  Splitting by {len(transitions)} transition(s)")
    print(f"  Lift window: ±{lift_window_s}s around each transition")

    # Collect points per floor from fused chunks
    n_floors  = len(transitions) + 1
    floor_s   = [[] for _ in range(n_floors)]  # static per floor
    floor_d   = [[] for _ in range(n_floors)]  # dynamic per floor
    lift_pts  = []                               # lift transition pts

    fused_chunks = sorted(glob.glob(f"{debug_dir}/fused_chunk_*.pkl"))
    if not fused_chunks:
        # Fall back to sep_chunks
        fused_chunks = sorted(glob.glob(f"{debug_dir}/sep_chunk_*.pkl"))

    total_frames = 0
    for cp in fused_chunks:
        if not os.path.exists(cp):
            continue
        with open(cp,'rb') as f:
            frames = pickle.load(f)

        for fr in frames:
            ts = float(fr['timestamp'])
            sp = fr.get('static_semantic',
                 fr.get('static_points', np.zeros((0,4))))
            dp = fr.get('dynamic_semantic',
                 fr.get('dynamic_points', np.zeros((0,4))))

            # Ensure 5 columns
            if len(sp)>0 and sp.shape[1]==4:
                sp = np.hstack([sp, np.full((len(sp),1), 255.0)])
            if len(dp)>0 and dp.shape[1]==4:
                dp = np.hstack([dp, np.full((len(dp),1), 255.0)])

            # Check if in lift zone
            in_lift = any(
                abs(ts - tr_t) < lift_window_s
                for tr_t in t_transitions)

            if in_lift:
                if len(sp)>0: lift_pts.append(sp)
                if len(dp)>0: lift_pts.append(dp)
                total_frames += 1
                continue

            # Determine which floor by comparing to transitions
            floor_idx = 0
            for tr_t in t_transitions:
                if ts > tr_t:
                    floor_idx += 1
            floor_idx = min(floor_idx, n_floors-1)

            if len(sp)>0: floor_s[floor_idx].append(sp)
            if len(dp)>0: floor_d[floor_idx].append(dp)
            total_frames += 1

        del frames
        gc.collect()

    def stack(lst):
        return np.vstack(lst) if lst else np.zeros((0,5))

    floors_static  = [stack(f) for f in floor_s]
    floors_dynamic = [stack(f) for f in floor_d]
    lift_combined  = stack(lift_pts)

    print(f"\n  Frames processed: {total_frames:,}")
    for i in range(n_floors):
        print(f"  Floor {i+1}: "
              f"static={len(floors_static[i]):,}  "
              f"dynamic={len(floors_dynamic[i]):,}")
    print(f"  Lift: {len(lift_combined):,}")

    return floors_static, floors_dynamic, lift_combined, n_floors

def save_ply(pts, path):
    if len(pts) == 0:
        print(f"    ⚠ Empty: {os.path.basename(path)}")
        return
    xyz  = pts[:,:3]
    labs = pts[:,4].astype(int) if pts.shape[1]>4 \
           else np.zeros(len(pts), dtype=int)
    with open(path,'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\n"
                "property float z\nproperty uchar red\n"
                "property uchar green\nproperty uchar blue\n"
                "property int label\nend_header\n")
        for i in range(len(xyz)):
            lab = labs[i]
            if 0<=lab<len(CITYSCAPES_COLORS):
                r,g,b = (CITYSCAPES_COLORS[lab]*255).astype(int)
            else:
                r,g,b = 100,180,255
            f.write(f"{xyz[i,0]:.4f} {xyz[i,1]:.4f} {xyz[i,2]:.4f} "
                    f"{r} {g} {b} {lab}\n")
    print(f"    ✔ {os.path.basename(path)} ({len(pts):,} pts)")

def generate_plots(floors_s, floors_d, lift,
                   transitions, all_t, all_x, all_y):
    print("\n  Generating plots...")
    os.makedirs(PLOTS, exist_ok=True)

    n     = len(floors_s)
    cols  = ['#64b4ff','#00e5ff','#aaffaa','#ffaaff']
    dcols = ['#ff5014','#ff9500','#ff44aa','#aa44ff']

    fig   = plt.figure(figsize=(22,16))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Floor Separation by Trajectory Jump Detection",
                 color='white', fontsize=15, fontweight='bold')

    n_cols = min(n+2, 4)
    n_rows = 3

    # Row 1: top-down per floor
    for i in range(n):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        style_ax(ax)
        arr = floors_s[i]
        if len(arr)>0:
            s = arr[::max(1,len(arr)//5000)]
            ax.scatter(s[:,0],s[:,1],c=cols[i%len(cols)],
                       s=0.8,alpha=0.7)
        ax.set_title(f"Floor {i+1} Static ({len(arr):,})",
                     fontsize=9,color='white')
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.set_aspect('equal')

    # Lift top-down
    ax_lift = fig.add_subplot(n_rows, n_cols, n+1)
    style_ax(ax_lift)
    if len(lift)>0:
        s = lift[::max(1,len(lift)//3000)]
        ax_lift.scatter(s[:,0],s[:,1],c='#ffff00',s=1.0,alpha=0.8)
    ax_lift.set_title(f"Lift Zone ({len(lift):,})",
                      fontsize=9,color='white')
    ax_lift.set_xlabel("X (m)"); ax_lift.set_ylabel("Y (m)")
    ax_lift.set_aspect('equal')

    # Row 2: combined + robot path
    ax_comb = fig.add_subplot(n_rows, n_cols, n_cols+1)
    style_ax(ax_comb)
    for i,(arr,col) in enumerate(zip(floors_s,cols)):
        if len(arr)>0:
            s=arr[::max(1,len(arr)//3000)]
            ax_comb.scatter(s[:,0],s[:,1],c=col,s=0.4,
                            alpha=0.6,label=f'F{i+1} static')
    for i,(arr,col) in enumerate(zip(floors_d,dcols)):
        if len(arr)>0:
            s=arr[::max(1,len(arr)//1000)]
            ax_comb.scatter(s[:,0],s[:,1],c=col,s=1.0,
                            alpha=0.8,label=f'F{i+1} dynamic')
    if len(lift)>0:
        s=lift[::max(1,len(lift)//1000)]
        ax_comb.scatter(s[:,0],s[:,1],c='#ffff00',s=1.0,
                        alpha=0.8,label='Lift')
    ax_comb.set_title("All Floors Combined",fontsize=9,color='white')
    ax_comb.set_xlabel("X (m)"); ax_comb.set_ylabel("Y (m)")
    ax_comb.set_aspect('equal')
    leg=ax_comb.legend(markerscale=5,facecolor=BG,
                       fontsize=7,ncol=2)
    for t in leg.get_texts(): t.set_color('white')

    # Robot trajectory colored by floor
    ax_traj = fig.add_subplot(n_rows, n_cols, n_cols+2)
    style_ax(ax_traj)
    if len(all_t)>1:
        t_transitions_list = [tr['time'] for tr in transitions]

        prev_floor = 0
        seg_x = [all_x[0]]; seg_y = [all_y[0]]

        for i in range(1,len(all_t)):
            # count transitions before this time
            fl = sum(1 for tr_t in t_transitions_list
                     if all_t[i] > tr_t)
            fl = min(fl, len(cols)-1)

            if fl == prev_floor:
                seg_x.append(all_x[i])
                seg_y.append(all_y[i])
            else:
                ax_traj.plot(seg_x, seg_y,
                             c=cols[prev_floor%len(cols)],
                             lw=1.5, alpha=0.8)
                seg_x = [all_x[i]]; seg_y = [all_y[i]]
                prev_floor = fl

        if seg_x:
            ax_traj.plot(seg_x, seg_y,
                         c=cols[prev_floor%len(cols)],
                         lw=1.5, alpha=0.8)

        # Mark transitions
        for tr in transitions:
            ax_traj.axvline(x=tr['x_after'], color='red',
                            lw=0.5, alpha=0.5)
            ax_traj.scatter(tr['x_before'],tr['y_before'],
                            c='red',s=100,marker='x',zorder=10)
            ax_traj.scatter(tr['x_after'],tr['y_after'],
                            c='lime',s=100,marker='o',zorder=10)

        ax_traj.scatter(all_x[0],all_y[0],
                        c='lime',s=150,marker='^',
                        zorder=11,label='Start')
        ax_traj.scatter(all_x[-1],all_y[-1],
                        c='red',s=150,marker='s',
                        zorder=11,label='End')

    ax_traj.set_title("Robot Trajectory by Floor",
                      fontsize=9,color='white')
    ax_traj.set_xlabel("X (m)"); ax_traj.set_ylabel("Y (m)")
    ax_traj.set_aspect('equal')
    leg2=ax_traj.legend(markerscale=2,facecolor=BG,fontsize=7)
    for t in leg2.get_texts(): t.set_color('white')

    # Timeline plot
    ax_time = fig.add_subplot(n_rows, 1, 3)
    style_ax(ax_time)
    if len(all_t)>1:
        ax_time.plot(all_t-all_t[0], all_x,
                     c='#64b4ff',lw=0.8,label='X position')
        ax_time.plot(all_t-all_t[0], all_y,
                     c='#ff9500',lw=0.8,label='Y position')
        for tr in transitions:
            t_rel = tr['time']-all_t[0]
            ax_time.axvline(t_rel,color='red',lw=2,
                            ls='--',alpha=0.8,
                            label=f"Lift at t={t_rel:.0f}s")
        ax_time.set_xlabel("Time (s from start)")
        ax_time.set_ylabel("Position (m)")
        ax_time.set_title("Robot X/Y Over Time — Red Lines = Lift Transitions",
                          fontsize=10,color='white')
        leg3=ax_time.legend(facecolor=BG,fontsize=8)
        for t in leg3.get_texts(): t.set_color('white')

    plt.tight_layout()
    out = f"{PLOTS}/floor_separation.png"
    plt.savefig(out,dpi=150,bbox_inches='tight',facecolor=BG)
    plt.close()
    print(f"  ✔ Plot: {out}")

    # Floor height analysis
    fig2, axes = plt.subplots(1, n+1, figsize=(6*(n+1), 5))
    fig2.patch.set_facecolor(BG)
    fig2.suptitle("Z Height Per Floor",color='white',fontsize=13)
    if n+1 == 1:
        axes = [axes]
    ax_list = list(axes) if hasattr(axes,'__iter__') else [axes]

    for i,(arr,col) in enumerate(zip(floors_s+[lift],
                                      cols+['#ffff00'])):
        if i >= len(ax_list): break
        ax = ax_list[i]
        style_ax(ax)
        lbl = f"Floor {i+1}" if i<n else "Lift"
        if len(arr)>0:
            ax.hist(arr[:,2],bins=80,color=col,
                    alpha=0.8,edgecolor='none',
                    orientation='horizontal')
        ax.set_title(f"{lbl} Z ({len(arr):,} pts)",
                     color='white',fontsize=10)
        ax.set_xlabel("Count"); ax.set_ylabel("Z (m)")

    plt.tight_layout()
    out2 = f"{PLOTS}/floor_height_profiles.png"
    plt.savefig(out2,dpi=150,bbox_inches='tight',facecolor=BG)
    plt.close()
    print(f"  ✔ Plot: {out2}")

def main():
    print("\n"+"="*55)
    print("MODULE 10: FLOOR SEPARATION")
    print("Method: Trajectory Jump Detection")
    print("="*55)

    for d in [FLOOR1_DIR, FLOOR2_DIR, LIFT_DIR]:
        os.makedirs(d, exist_ok=True)

    # Step 1: Find lift transitions from odometry
    transitions, all_t, all_x, all_y = find_lift_transitions(
        DEBUG,
        jump_threshold_m=5.0,
        time_threshold_s=5.0)

    if not transitions:
        print("\n  ⚠ No transitions found with threshold=5m")
        print("  Checking maximum observed jump...")
        # Find the single largest jump to use as the transition
        if len(all_x) > 1:
            dx    = np.diff(np.array(all_x))
            dy    = np.diff(np.array(all_y))
            dt    = np.diff(np.array(all_t))
            dists = np.sqrt(dx**2+dy**2)
            max_i = np.argmax(dists)
            max_d = dists[max_i]
            max_t = all_t[max_i+1]
            print(f"  Largest jump: {max_d:.2f}m at t={max_t:.2f}s")
            if max_d > 2.0:
                print(f"  Using this as the floor transition")
                transitions = [{
                    'time':     float(max_t),
                    'dist':     float(max_d),
                    'dt':       float(dt[max_i]),
                    'x_before': float(all_x[max_i]),
                    'y_before': float(all_y[max_i]),
                    'x_after':  float(all_x[max_i+1]),
                    'y_after':  float(all_y[max_i+1]),
                }]
            else:
                print(f"  ✘ Max jump {max_d:.2f}m too small")
                print(f"  Cannot detect floor transition.")
                print(f"  The bag may only contain one floor.")
                return

    print(f"\n  Found {len(transitions)} floor transition(s)")

    # Step 2: Split frames by transition timestamps
    result = split_frames_by_transitions(
        DEBUG, transitions, lift_window_s=20.0)

    if result is None:
        print("✘ Floor split failed")
        return

    floors_s, floors_d, lift, n_floors = result

    # Step 3: Save outputs
    print("\nSaving floor maps...")

    # Save Floor 1
    save_ply(floors_s[0], f"{FLOOR1_DIR}/floor1_static.ply")
    save_ply(floors_d[0], f"{FLOOR1_DIR}/floor1_dynamic.ply")
    np.save(f"{FLOOR1_DIR}/floor1_static.npy",  floors_s[0])
    np.save(f"{FLOOR1_DIR}/floor1_dynamic.npy", floors_d[0])

    # Save Floor 2 (if exists)
    if n_floors >= 2:
        save_ply(floors_s[1], f"{FLOOR2_DIR}/floor2_static.ply")
        save_ply(floors_d[1], f"{FLOOR2_DIR}/floor2_dynamic.ply")
        np.save(f"{FLOOR2_DIR}/floor2_static.npy",  floors_s[1])
        np.save(f"{FLOOR2_DIR}/floor2_dynamic.npy", floors_d[1])

    # Save lift
    save_ply(lift, f"{LIFT_DIR}/lift_points.ply")
    np.save(f"{LIFT_DIR}/lift_points.npy", lift)

    # Save extra floors if any
    for i in range(2, n_floors):
        extra_dir = f"{BASE}/floor{i+1}"
        os.makedirs(extra_dir, exist_ok=True)
        save_ply(floors_s[i],
                 f"{extra_dir}/floor{i+1}_static.ply")
        save_ply(floors_d[i],
                 f"{extra_dir}/floor{i+1}_dynamic.ply")
        np.save(f"{extra_dir}/floor{i+1}_static.npy",  floors_s[i])
        np.save(f"{extra_dir}/floor{i+1}_dynamic.npy", floors_d[i])

    # Save config
    cfg_out = {
        'method':      'trajectory_jump',
        'n_floors':    n_floors,
        'transitions': transitions,
        'floor_dirs': {
            'floor1': FLOOR1_DIR,
            'floor2': FLOOR2_DIR if n_floors>=2 else None,
            'lift':   LIFT_DIR,
        }
    }
    with open(f"{DEBUG}/floor_split_config.json",'w') as f:
        json.dump(cfg_out, f, indent=2)

    # Step 4: Generate plots
    generate_plots(floors_s, floors_d, lift,
                   transitions, all_t, all_x, all_y)

    print(f"\n{'='*55}")
    print("FLOOR SEPARATION COMPLETE")
    print(f"{'='*55}")
    for i in range(n_floors):
        print(f"  Floor {i+1}: "
              f"{len(floors_s[i]):,} static, "
              f"{len(floors_d[i]):,} dynamic")
    print(f"  Lift : {len(lift):,} pts")
    print(f"\n  Plots → {PLOTS}/floor_separation.png")
    print(f"          {PLOTS}/floor_height_profiles.png")

if __name__ == '__main__':
    main()
