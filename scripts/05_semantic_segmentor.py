#!/usr/bin/env python3
"""
Module 5: Semantic Segmentation — chunk-based, memory safe.
Only loads one chunk at a time. Skips every Nth frame for speed.
"""

import numpy as np
import pickle
import os
import yaml
import gc
import cv2
from PIL import Image as PILImage

CONFIG_PATH = os.path.expanduser(
    "~/catkin_ws/src/lidar_reconstruction/config/pipeline_config.yaml")
DEBUG_DIR   = os.path.expanduser(
    "~/catkin_ws/src/lidar_reconstruction/output/debug")
SEG_DBG_DIR = os.path.join(DEBUG_DIR,"segmentation_samples")

CITYSCAPES_COLORS = np.array([
    [128,64,128],[244,35,232],[70,70,70],[102,102,156],
    [190,153,153],[153,153,153],[250,170,30],[220,220,0],
    [107,142,35],[152,251,152],[70,130,180],[220,20,60],
    [255,0,0],[0,0,142],[0,0,70],[0,60,100],
    [0,80,100],[0,0,230],[119,11,32]
], dtype=np.uint8)

def load_config():
    with open(CONFIG_PATH,'r') as f:
        return yaml.safe_load(f)

def img_from_dict(d):
    if d is None: return None
    h,w = d['height'], d['width']
    data= np.frombuffer(d['data'], dtype=np.uint8)
    enc = d['encoding']
    try:
        if enc=='mono8':
            img = cv2.cvtColor(data.reshape(h,w), cv2.COLOR_GRAY2RGB)
        elif enc=='bgr8':
            img = data.reshape(h,w,3)[:,:,::-1].copy()
        else:
            ch  = max(len(data)//(h*w),1)
            raw = data.reshape(h,w,ch)
            img = raw[:,:,:3] if ch>=3 \
                  else cv2.cvtColor(raw[:,:,0], cv2.COLOR_GRAY2RGB)
        return img
    except Exception:
        return None

class SemanticSegmentor:
    def __init__(self, cfg):
        self.skip_n   = cfg['memory']['skip_image_every_n']
        self.model    = None
        self.proc     = None
        self.use_model= False
        self.device   = 'cpu'
        self._load_model(cfg)
        self.call_count = 0

    def _load_model(self, cfg):
        print("  Loading segmentation model...")
        try:
            import torch
            from transformers import (SegformerImageProcessor,
                                      SegformerForSemanticSegmentation)
            name = cfg['semantic']['model']
            self.proc   = SegformerImageProcessor.from_pretrained(name)
            self.model  = SegformerForSemanticSegmentation.from_pretrained(name)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(self.device).eval()
            self._torch = torch
            self.use_model = True
            print(f"  ✔ Model ready ({self.device})")
        except Exception as e:
            print(f"  ⚠ Model skipped ({e}) — using fast fallback")
            self.use_model = False

    def segment(self, img_rgb):
        self.call_count += 1
        if self.call_count % self.skip_n != 0:
            return None  # skip this frame

        if self.use_model:
            try:
                import torch
                pil = PILImage.fromarray(img_rgb)
                inp = self.proc(images=pil, return_tensors="pt")
                inp = {k:v.to(self.device) for k,v in inp.items()}
                with torch.no_grad():
                    out = self.model(**inp)
                logits = torch.nn.functional.interpolate(
                    out.logits, size=img_rgb.shape[:2],
                    mode='bilinear', align_corners=False)
                lmap = logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.int32)
                # Free GPU memory immediately
                del out, logits, inp
                if self.device=='cuda':
                    torch.cuda.empty_cache()
                return lmap
            except Exception:
                pass

        return self._fast_seg(img_rgb)

    def _fast_seg(self, img):
        h,w = img.shape[:2]
        lm  = np.full((h,w),2,dtype=np.int32)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hue = hsv[:,:,0]; sat=hsv[:,:,1]; val=hsv[:,:,2]
        lm[(val>150)&(sat<80)&(hue>90)&(hue<140)]=10
        lm[(hue>35)&(hue<80)&(sat>40)]=8
        lm[2*h//3:][val[2*h//3:]<80]=0
        lm[(val>60)&(val<200)&(sat>50)]=13
        return lm

    def colorize(self, lmap):
        h,w = lmap.shape
        out = np.zeros((h,w,3),dtype=np.uint8)
        for i in range(len(CITYSCAPES_COLORS)):
            out[lmap==i] = CITYSCAPES_COLORS[i]
        return out

def process_chunk(frames, segmentor, debug_dir, chunk_idx,
                  max_debug=3, saved_so_far=0):
    out = []
    dbg_saved = 0

    for frame in frames:
        nf = dict(frame)
        nf['semantic_label_map'] = None
        nf['image_rgb']          = None

        img_data = frame.get('left_image')
        if img_data is None:
            out.append(nf); continue

        img_rgb = img_from_dict(img_data['msg'])
        if img_rgb is None:
            out.append(nf); continue

        lmap = segmentor.segment(img_rgb)

        nf['semantic_label_map'] = lmap
        nf['image_rgb']          = img_rgb if lmap is not None else None

        # Save a few debug images per chunk
        if lmap is not None and \
                dbg_saved < max_debug and \
                (saved_so_far + dbg_saved) < 15:
            try:
                colored = segmentor.colorize(lmap)
                blend   = cv2.addWeighted(img_rgb,0.6,colored,0.4,0)
                dbg_img = np.hstack([img_rgb,colored,blend])
                fname   = os.path.join(
                    debug_dir,
                    f"seg_c{chunk_idx:02d}_f{dbg_saved:02d}.jpg")
                cv2.imwrite(fname,
                            cv2.cvtColor(dbg_img, cv2.COLOR_RGB2BGR))
                dbg_saved += 1
            except Exception:
                pass

        out.append(nf)

    return out, dbg_saved

def main():
    cfg = load_config()
    print("\n"+"="*60)
    print("MODULE 5: SEMANTIC SEGMENTATION (FULL BAG)")
    print("="*60)
    os.makedirs(SEG_DBG_DIR, exist_ok=True)

    index_path = os.path.join(DEBUG_DIR,"sync_index.pkl")
    with open(index_path,'rb') as f:
        index = pickle.load(f)

    segmentor   = SemanticSegmentor(cfg)
    seg_paths   = []
    total_seg   = 0
    debug_saved = 0

    for i, cpath in enumerate(index['sep_paths']):
        print(f"  Chunk {i+1}/{len(index['sep_paths'])}")
        with open(cpath,'rb') as f:
            frames = pickle.load(f)

        out, ds = process_chunk(
            frames, segmentor, SEG_DBG_DIR, i,
            max_debug=3, saved_so_far=debug_saved)
        debug_saved += ds
        total_seg   += sum(1 for fr in out
                           if fr.get('semantic_label_map') is not None)

        out_path = os.path.join(DEBUG_DIR, f"seg_chunk_{i:03d}.pkl")
        with open(out_path,'wb') as f:
            pickle.dump(out, f, protocol=2)
        seg_paths.append(out_path)

        del frames, out
        gc.collect()

    index['seg_paths'] = seg_paths
    with open(index_path,'wb') as f:
        pickle.dump(index, f, protocol=2)

    print(f"\n✔ Frames segmented : {total_seg}")
    print(f"✔ Debug images     : {debug_saved} → {SEG_DBG_DIR}")
    print(f"✔ Chunks saved     : {len(seg_paths)}")
    print("\n✔ Module 5 COMPLETE\n")

if __name__ == '__main__':
    main()
