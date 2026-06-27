#!/usr/bin/env python3
"""
Generate all analysis plots — compatible with all matplotlib versions.
"""

import numpy as np
import pickle
import os
import csv
import gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
from matplotlib.patches import Patch
import matplotlib as mpl

base    = os.path.expanduser("~/catkin_ws/src/lidar_reconstruction/output")
debug   = f"{base}/debug"
static  = f"{base}/static_map"
dynamic = f"{base}/dynamic_map"
plots   = f"{debug}/plots"
os.makedirs(plots, exist_ok=True)

CLASSES = [
    'road','sidewalk','building','wall','fence','pole',
    'traffic light','traffic sign','vegetation','terrain','sky',
    'person','rider','car','truck','bus','train','motorcycle','bicycle'
]
COLORS_F = np.array([
    [0.502,0.251,0.502],[0.957,0.137,0.910],[0.275,0.275,0.275],
    [0.400,0.400,0.612],[0.745,0.600,0.600],[0.600,0.600,0.600],
    [0.980,0.667,0.118],[0.863,0.863,0.000],[0.420,0.557,0.137],
    [0.596,0.984,0.596],[0.275,0.510,0.706],[0.863,0.078,0.235],
    [1.000,0.000,0.000],[0.000,0.000,0.557],[0.000,0.000,0.275],
    [0.000,0.235,0.392],[0.000,0.314,0.392],[0.000,0.000,0.902],
    [0.467,0.043,0.125]
])

BG_DARK  = '#1a1a2e'
BG_PANEL = '#16213e'
BG_BLACK = '#0d0d1a'
GRID_COL = '#444444'
SPINE_COL= '#444444'

# Check matplotlib version once
MPL_VERSION = tuple(int(x) for x in mpl.__version__.split('.')[:2])
print(f"  matplotlib version: {mpl.__version__}")

def make_legend(ax, **kwargs):
    """
    Create legend compatible with old and new matplotlib.
    Removes labelcolor if not supported, then manually sets text colors.
    """
    # Remove unsupported kwargs for old versions
    if MPL_VERSION < (3, 2):
        kwargs.pop('labelcolor', None)

    leg = ax.legend(**kwargs)

    if leg is not None:
        leg.get_frame().set_facecolor(BG_DARK)
        leg.get_frame().set_edgecolor(SPINE_COL)
        for text in leg.get_texts():
            text.set_color('white')
    return leg

def style_ax(ax):
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_COL)
    ax.grid(True, alpha=0.2, color=GRID_COL)

def dark_fig(nrows=1, ncols=1, figsize=(14,6)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor(BG_DARK)
    if hasattr(axes, 'flatten'):
        for ax in axes.flatten():
            style_ax(ax)
    else:
        style_ax(axes)
    return fig, axes

def sample(arr, n=8000):
    if len(arr) == 0:
        return arr
    step = max(1, len(arr)//n)
    return arr[::step]

def save_fig(fig, path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)

# ── Load maps ─────────────────────────────────────────────────────────────────
print("="*55)
print("GENERATING ALL ANALYSIS PLOTS")
print("="*55)

print("\nLoading maps...")
s_path = f"{static}/static_semantic_map.npy"
d_path = f"{dynamic}/dynamic_semantic_map.npy"
s_arr  = np.load(s_path)  if os.path.exists(s_path)  else np.zeros((0,5))
d_arr  = np.load(d_path)  if os.path.exists(d_path)  else np.zeros((0,5))
print(f"  Static : {len(s_arr):,} pts")
print(f"  Dynamic: {len(d_arr):,} pts")

if len(s_arr) > 0:
    ctr = s_arr[:,:3].mean(axis=0).copy()
    ctr[2] = 0
    s_arr[:,:3] -= ctr
    if len(d_arr) > 0:
        d_arr[:,:3] -= ctr

# ── PLOT 1: Top-down ──────────────────────────────────────────────────────────
print("\nPlot 1: Top-down map...")
fig, axes = dark_fig(1, 3, (21, 7))
fig.suptitle("Full Bag — Dual Semantic Map (Top-Down)",
             color='white', fontsize=15, fontweight='bold')

ss = sample(s_arr)
dd = sample(d_arr, 3000)

if len(ss) > 0:
    axes[0].scatter(ss[:,0], ss[:,1], s=0.4, c='#64b4ff', alpha=0.6)
axes[0].set_title(f"Static Map ({len(s_arr):,} pts)", fontsize=11)
axes[0].set_xlabel("X (m)"); axes[0].set_ylabel("Y (m)")
axes[0].set_aspect('equal')

if len(dd) > 0:
    axes[1].scatter(dd[:,0], dd[:,1], s=1.5, c='#ff5014', alpha=0.8)
axes[1].set_title(f"Dynamic Map ({len(d_arr):,} pts)", fontsize=11)
axes[1].set_xlabel("X (m)"); axes[1].set_ylabel("Y (m)")
axes[1].set_aspect('equal')

if len(ss) > 0:
    axes[2].scatter(ss[:,0], ss[:,1], s=0.3,
                    c='#64b4ff', alpha=0.4, label='Static')
if len(dd) > 0:
    axes[2].scatter(dd[:,0], dd[:,1], s=1.5,
                    c='#ff5014', alpha=0.8, label='Dynamic')
axes[2].set_title("Combined", fontsize=11)
axes[2].set_xlabel("X (m)"); axes[2].set_ylabel("Y (m)")
axes[2].set_aspect('equal')
make_legend(axes[2], markerscale=8, facecolor=BG_DARK)

save_fig(fig, f"{plots}/topdown_map.png")
print("  ✔ topdown_map.png")

# ── PLOT 2: 3D perspective ────────────────────────────────────────────────────
print("Plot 2: 3D perspective...")
fig = plt.figure(figsize=(18, 7), facecolor=BG_DARK)
fig.suptitle("Full Bag — 3D Perspective",
             color='white', fontsize=14, fontweight='bold')

for i, (arr, col, title) in enumerate([
        (s_arr, '#64b4ff', f"Static ({len(s_arr):,} pts)"),
        (d_arr, '#ff5014', f"Dynamic ({len(d_arr):,} pts)")]):
    ax = fig.add_subplot(1, 2, i+1, projection='3d')
    ax.set_facecolor(BG_BLACK)
    s2 = sample(arr, 3000)
    if len(s2) > 0:
        ax.scatter(s2[:,0], s2[:,1], s2[:,2], c=col, s=0.5, alpha=0.6)
    ax.set_title(title, color='white', fontsize=11)
    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.set_zlabel('Z', color='white')
    ax.tick_params(colors='white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

plt.tight_layout()
plt.savefig(f"{plots}/3d_map.png", dpi=150,
            bbox_inches='tight', facecolor=BG_DARK)
plt.close()
print("  ✔ 3d_map.png")

# ── PLOT 3: Class distribution ────────────────────────────────────────────────
print("Plot 3: Class distribution...")
fig, axes = dark_fig(1, 2, (14, 7))
fig.suptitle("Semantic Class Distribution",
             color='white', fontsize=14, fontweight='bold')

for ax, arr, title in [(axes[0], s_arr, "Static Map Classes"),
                        (axes[1], d_arr, "Dynamic Map Classes")]:
    ax.set_facecolor(BG_DARK)
    if len(arr) == 0:
        ax.text(0.5, 0.5, 'No data',
                ha='center', va='center', color='white')
        ax.set_title(title, color='white')
        continue
    labs   = arr[:,4].astype(int)
    cnt    = Counter(labs)
    top    = sorted(cnt.items(), key=lambda x: -x[1])[:8]
    names  = [CLASSES[k] if 0<=k<len(CLASSES) else 'unknown' for k,_ in top]
    vals   = [v for _, v in top]
    colors = [COLORS_F[k].tolist() if 0<=k<len(COLORS_F)
              else [0.5,0.5,0.5] for k,_ in top]
    wedges, texts, autos = ax.pie(
        vals, labels=names, colors=colors,
        autopct='%1.1f%%', startangle=90,
        textprops={'color':'white', 'fontsize': 8})
    for a in autos:
        a.set_color('white')
    ax.set_title(title, color='white', fontsize=12)

save_fig(fig, f"{plots}/class_distribution.png")
print("  ✔ class_distribution.png")

# ── PLOT 4: Static/Dynamic ratio ──────────────────────────────────────────────
print("Plot 4: Static/Dynamic ratio over time...")
csv_path = f"{debug}/static_dynamic_per_frame.csv"
if os.path.exists(csv_path):
    fidx=[]; spts=[]; dpts=[]; rats=[]
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fidx.append(int(row['frame_idx']))
            spts.append(int(row['static_pts']))
            dpts.append(int(row['dynamic_pts']))
            rats.append(float(row['dynamic_ratio']))

    fig, axes = dark_fig(2, 2, (18, 10))
    fig.suptitle("Full Bag — Static/Dynamic Analysis",
                 color='white', fontsize=14, fontweight='bold')

    axes[0,0].plot(fidx, spts, c='#64b4ff', lw=0.6, label='Static')
    axes[0,0].plot(fidx, dpts, c='#ff5014', lw=0.6, label='Dynamic')
    axes[0,0].set_title("Points per Frame")
    axes[0,0].set_xlabel("Frame"); axes[0,0].set_ylabel("Points")
    make_legend(axes[0,0])

    axes[0,1].plot(fidx, np.array(rats)*100, c='#ffa500', lw=0.6)
    axes[0,1].axhline(np.mean(rats)*100, c='red', ls='--',
                      label=f"Mean={np.mean(rats)*100:.1f}%")
    axes[0,1].set_title("Dynamic Ratio Over Time")
    axes[0,1].set_xlabel("Frame"); axes[0,1].set_ylabel("Dynamic %")
    make_legend(axes[0,1])

    axes[1,0].hist(np.array(rats)*100, bins=50,
                   color='#9b59b6', alpha=0.8,
                   edgecolor='white', linewidth=0.3)
    axes[1,0].set_title("Dynamic Ratio Distribution")
    axes[1,0].set_xlabel("Dynamic %")
    axes[1,0].set_ylabel("Frame count")

    ts = np.array(spts); td = np.array(dpts)
    axes[1,1].set_facecolor(BG_DARK)
    axes[1,1].pie(
        [ts.sum(), td.sum()],
        labels=[f'Static\n{ts.sum():,}', f'Dynamic\n{td.sum():,}'],
        colors=['#64b4ff', '#ff5014'],
        autopct='%1.1f%%', startangle=90,
        textprops={'color':'white', 'fontsize': 10})
    axes[1,1].set_title("Overall Split")

    save_fig(fig, f"{plots}/static_dynamic_analysis.png")
    print("  ✔ static_dynamic_analysis.png")
else:
    print("  ⚠ CSV not found — skipping")

# ── PLOT 5: Semantic spatial map ──────────────────────────────────────────────
print("Plot 5: Semantic spatial map...")
fig, ax = plt.subplots(figsize=(14, 12))
fig.patch.set_facecolor(BG_BLACK)
ax.set_facecolor(BG_BLACK)
ax.tick_params(colors='white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.title.set_color('white')
for sp in ax.spines.values():
    sp.set_edgecolor(SPINE_COL)
ax.grid(True, alpha=0.15, color=GRID_COL)

if len(s_arr) > 0:
    ss2  = sample(s_arr, 10000)
    labs = ss2[:,4].astype(int)
    rgb  = np.array([
        COLORS_F[l].tolist() if 0<=l<len(COLORS_F)
        else [0.4,0.7,1.0] for l in labs])
    ax.scatter(ss2[:,0], ss2[:,1], c=rgb, s=0.5, alpha=0.7)

if len(d_arr) > 0:
    dd2 = sample(d_arr, 3000)
    ax.scatter(dd2[:,0], dd2[:,1], c='#ff3300',
               s=3.0, alpha=0.9, label='Dynamic', zorder=5)

ax.set_title("Full Bag — Semantic Spatial Map",
             color='white', fontsize=14, fontweight='bold')
ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
ax.set_aspect('equal')

if len(s_arr) > 0:
    labs_all = s_arr[:,4].astype(int)
    top_cls  = Counter(labs_all).most_common(6)
    handles  = [
        Patch(color=COLORS_F[k].tolist() if 0<=k<len(COLORS_F)
              else [0.5,0.5,0.5],
              label=CLASSES[k] if 0<=k<len(CLASSES) else 'unknown')
        for k,_ in top_cls
    ]
    handles.append(Patch(color='#ff3300', label='Dynamic'))
    leg = ax.legend(handles=handles, loc='upper right',
                    facecolor=BG_DARK, fontsize=9)
    for text in leg.get_texts():
        text.set_color('white')

fig.patch.set_facecolor(BG_BLACK)
plt.tight_layout()
plt.savefig(f"{plots}/semantic_spatial_map.png", dpi=150,
            bbox_inches='tight', facecolor=BG_BLACK)
plt.close()
print("  ✔ semantic_spatial_map.png")

# ── PLOT 6: Height profile ────────────────────────────────────────────────────
print("Plot 6: Height profile...")
fig, axes = dark_fig(1, 2, (14, 5))
fig.suptitle("Height Profile — Z Distribution",
             color='white', fontsize=13, fontweight='bold')

if len(s_arr) > 0:
    axes[0].hist(s_arr[:,2], bins=80, color='#64b4ff',
                 alpha=0.8, edgecolor='none')
    axes[0].set_title("Static Map — Z Distribution")
    axes[0].set_xlabel("Height Z (m)")
    axes[0].set_ylabel("Point count")
    axes[0].axvline(0, color='white', ls='--', alpha=0.5, label='Ground')
    make_legend(axes[0])
else:
    axes[0].text(0.5,0.5,'No data',ha='center',va='center',color='white')
    axes[0].set_title("Static Map — Z Distribution")

if len(d_arr) > 0:
    axes[1].hist(d_arr[:,2], bins=60, color='#ff5014',
                 alpha=0.8, edgecolor='none')
    axes[1].set_title("Dynamic Map — Z Distribution")
    axes[1].set_xlabel("Height Z (m)")
    axes[1].set_ylabel("Point count")
else:
    axes[1].text(0.5,0.5,'No data',ha='center',va='center',color='white')
    axes[1].set_title("Dynamic Map — Z Distribution")

save_fig(fig, f"{plots}/height_profile.png")
print("  ✔ height_profile.png")

# ── PLOT 7: Robot trajectory ──────────────────────────────────────────────────
print("Plot 7: Robot trajectory...")
index_path = f"{debug}/sync_index.pkl"
if os.path.exists(index_path):
    with open(index_path,'rb') as f:
        idx_data = pickle.load(f)

    robot_x=[]; robot_y=[]
    chunk_paths = idx_data.get('comp_paths',
                  idx_data.get('chunk_paths', []))

    for cp in chunk_paths:
        if not os.path.exists(cp):
            continue
        with open(cp,'rb') as f:
            frames = pickle.load(f)
        for fr in frames[::5]:
            od = fr.get('odom_combined') or fr.get('odom')
            if od is None:
                continue
            m = od['msg']
            p = m['pose']['position']
            robot_x.append(p['x'])
            robot_y.append(p['y'])
        del frames
        gc.collect()

    if robot_x:
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor(BG_BLACK)
        ax.set_facecolor(BG_BLACK)
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for sp in ax.spines.values():
            sp.set_edgecolor(SPINE_COL)
        ax.grid(True, alpha=0.15, color=GRID_COL)

        if len(s_arr) > 0:
            ss3 = sample(s_arr, 8000)
            ax.scatter(ss3[:,0], ss3[:,1],
                       c='#334466', s=0.3, alpha=0.4, label='Static map')

        ax.plot(robot_x, robot_y,
                c='#00ff88', lw=1.5, alpha=0.8,
                label='Robot path', zorder=5)
        ax.scatter(robot_x[0],  robot_y[0],
                   c='lime', s=200, marker='^', zorder=10, label='Start')
        ax.scatter(robot_x[-1], robot_y[-1],
                   c='red',  s=200, marker='s', zorder=10, label='End')

        ax.set_title("Robot Trajectory over Map",
                     color='white', fontsize=13)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.set_aspect('equal')
        leg = ax.legend(facecolor=BG_DARK, markerscale=3)
        for text in leg.get_texts():
            text.set_color('white')

        fig.patch.set_facecolor(BG_BLACK)
        plt.tight_layout()
        plt.savefig(f"{plots}/robot_trajectory.png", dpi=150,
                    bbox_inches='tight', facecolor=BG_BLACK)
        plt.close()
        print("  ✔ robot_trajectory.png")
    else:
        print("  ⚠ No odom positions found — skipping")
else:
    print("  ⚠ sync_index.pkl not found — skipping")

# ── PLOT 8: Map statistics summary ───────────────────────────────────────────
print("Plot 8: Map statistics summary...")
fig, axes = dark_fig(1, 2, (14, 6))
fig.suptitle("Map Summary Statistics",
             color='white', fontsize=13, fontweight='bold')

for ax, arr, name, col in [
        (axes[0], s_arr, "Static Map",  '#64b4ff'),
        (axes[1], d_arr, "Dynamic Map", '#ff5014')]:
    ax.set_facecolor(BG_DARK)
    if len(arr) == 0:
        ax.text(0.5,0.5,'No data',ha='center',va='center',color='white')
        ax.set_title(name, color='white')
        continue
    xyz   = arr[:,:3]
    labels_col = arr[:,4] if arr.shape[1]>4 else np.full(len(arr),255.0)
    stats = [
        ("Total pts",    len(arr)),
        ("X span (m)",   round(float(xyz[:,0].max()-xyz[:,0].min()),1)),
        ("Y span (m)",   round(float(xyz[:,1].max()-xyz[:,1].min()),1)),
        ("Z span (m)",   round(float(xyz[:,2].max()-xyz[:,2].min()),1)),
        ("Z min (m)",    round(float(xyz[:,2].min()),2)),
        ("Z max (m)",    round(float(xyz[:,2].max()),2)),
        ("Labeled pts",  int(np.sum(labels_col != 255))),
    ]
    keys   = [s[0] for s in stats]
    vals   = [s[1] for s in stats]
    y_pos  = np.arange(len(stats))
    bars   = ax.barh(y_pos, vals, color=col, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(keys, color='white', fontsize=9)
    for j, v in enumerate(vals):
        ax.text(max(vals)*0.02, j, f"  {v:,}" if isinstance(v,int)
                else f"  {v}", va='center', color='white', fontsize=9)
    ax.set_title(name, color='white', fontsize=12)
    ax.set_xlabel("Value", color='white')

save_fig(fig, f"{plots}/map_statistics_summary.png")
print("  ✔ map_statistics_summary.png")

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"✔ All plots saved → {plots}/")
print(f"{'='*55}")
for fname in sorted(os.listdir(plots)):
    fp = os.path.join(plots, fname)
    print(f"  {fname:<42} {os.path.getsize(fp)/1e3:.0f} KB")
