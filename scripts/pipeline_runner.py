#!/usr/bin/env python3
"""Master pipeline runner — full bag, chunk-based, memory safe."""

import os, sys, time, subprocess

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

MODULES = [
    ("00_bag_inspector.py",        "Bag Inspection"),
    ("01_synchronizer.py",         "Sensor Synchronization"),
    ("02_motion_compensator.py",   "Motion Distortion Correction"),
    ("03_lidar_to_3d.py",          "2D LiDAR -> 3D"),
    ("04_dynamic_detector.py",     "Static/Dynamic Separation"),
    ("05_semantic_segmentor.py",   "Semantic Segmentation"),
    ("06_lidar_semantic_fusion.py","LiDAR-Semantic Fusion"),
    ("07_map_builder.py",          "Dual Map Building"),
    ("08_visualizer.py",           "Visualization"),
]

def run(script, name, idx, start_from):
    if idx < start_from:
        print(f"  [SKIP] {idx}: {name}"); return True
    print(f"\n{'='*60}\nMODULE {idx}: {name}\n{'='*60}")
    t = time.time()
    r = subprocess.run([sys.executable,
                        os.path.join(SCRIPTS_DIR,script)])
    elapsed = time.time()-t
    ok = r.returncode==0
    print(f"\n{'✔' if ok else '✘'} Module {idx} "
          f"{'DONE' if ok else 'FAILED'} in {elapsed:.1f}s")
    return ok

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-from', type=int, default=0)
    parser.add_argument('--only',       type=int, default=-1)
    parser.add_argument('--stop-on-error', action='store_true')
    args = parser.parse_args()

    print("\n"+"="*60)
    print("FULL BAG SEMANTIC 3D RECONSTRUCTION PIPELINE")
    print("="*60)

    t0 = time.time()
    results = []
    for idx,(script,name) in enumerate(MODULES):
        if args.only >= 0 and idx != args.only:
            continue
        ok = run(script,name,idx,args.start_from)
        results.append((idx,name,ok))
        if not ok and args.stop_on_error:
            print("\n⚠ Stopped on error."); break

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for idx,name,ok in results:
        print(f"  {'✔' if ok else '✘'} {idx}: {name}")
    passed = sum(1 for _,_,s in results if s)
    total  = time.time()-t0
    print(f"\n  {passed}/{len(results)} passed | "
          f"Total: {total:.0f}s ({total/60:.1f}min)")

if __name__ == '__main__':
    main()
