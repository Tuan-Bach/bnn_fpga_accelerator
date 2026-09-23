#!/usr/bin/env python3
"""
Prepare the PCB-AOI defect dataset for the BNN (patch classification).

Input:  /tmp/bnn_datasets/pcb_aoi   (klshitij1507/PCB-AOI, YOLO format, 640x640)
Output: /tmp/bnn_datasets/pcb/{train_x,train_y,test_x,test_y}.npy + meta.json

Steps:
  1. Read images as grayscale; parse YOLO boxes (6 defect classes).
  2. Re-split the 1824 train images (repo valid/test don't cover all classes)
     into train/test stratified by image (all classes kept in test).
  3. For every defect box: crop a square around it (context ~3.2x the defect),
     resize to PxP, binarize with the IMAGE-level Otsu threshold
     (lighting-adaptive, deterministic -> matches training/export exactly).
  4. Sample background patches (class 6) from non-defect regions.

Class index order (classic PCB defect benchmark):
  0 missing_hole, 1 mouse_bite, 2 open_circuit, 3 short, 4 spur, 5 spurious_copper,
  6 background
"""

import json
import glob
import os
import random

import cv2
import numpy as np
from PIL import Image

AOI = "/tmp/bnn_datasets/pcb_aoi"
OUT = "/tmp/bnn_datasets/pcb"
P = 40                      # patch size after resize (40x40 = 1600 features)
BG_SIDE = 64                # background square side in image pixels
CTX = 3.2                   # crop side = CTX * max(w, h) of the defect box
MAX_SIDE = 96
MIN_SIDE = 32
N_BG = 0                    # background patches per image (0 = pure defect classification)
TEST_FRAC = 0.15
CLASS_NAMES = ["missing_hole", "mouse_bite", "open_circuit", "short",
               "spur", "spurious_copper"]
SEED = 7


def parse_labels(lbl_path, W, H):
    """Return list of (cls, x0, y0, x1, y1) in pixel coords (clamped)."""
    boxes = []
    if not os.path.exists(lbl_path):
        return boxes
    for line in open(lbl_path):
        p = line.split()
        if len(p) != 5:
            continue
        c, cx, cy, w, h = int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
        x0 = max(0.0, (cx - w / 2.0) * W)
        y0 = max(0.0, (cy - h / 2.0) * H)
        x1 = min(W, (cx + w / 2.0) * W)
        y1 = min(H, (cy + h / 2.0) * H)
        if x1 - x0 >= 4 and y1 - y0 >= 4:
            boxes.append((c, x0, y0, x1, y1))
    return boxes


def binarize_patch(patch_u8):
    """Binarize a single (already resized) patch with its OWN Otsu threshold.

    Per-image Otsu is unstable here: the whole-image foreground fraction varies
    (0.17 vs 0.35 on similar crops), so the same copper tone flips polarity
    between images. Normalizing each patch against itself keeps features
    polarity-consistent. Near-uniform patches (blank board) -> all zeros.
    """
    if patch_u8.max() == patch_u8.min() or patch_u8.std() < 8.0:
        return np.zeros(patch_u8.shape, dtype=np.float32)
    t, _ = cv2.threshold(patch_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (patch_u8 > t).astype(np.float32)


def square_crop(img, cx, cy, side):
    """Crop a square of `side` px centered at (cx, cy), clamped to image."""
    W, H = img.size
    side = min(side, W, H)
    half = side // 2
    left = int(np.clip(cx - half, 0, W - side))
    top = int(np.clip(cy - half, 0, H - side))
    return img.crop((left, top, left + side, top + side))


def overlaps(px, py, ps, boxes, pad=6):
    """True if a square center (px,py) size ps collides with any defect box."""
    half = ps / 2.0
    for _, x0, y0, x1, y1 in boxes:
        if (px + half > x0 - pad and px - half < x1 + pad and
                py + half > y0 - pad and py - half < y1 + pad):
            return True
    return False


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUT, exist_ok=True)

    # ---- Load repo train images + labels ----
    entries = []               # (img_path, boxes, gray_pil, otsu_th)
    files = sorted(glob.glob(f"{AOI}/train/images/*"))
    print(f"train images: {len(files)}")
    for i, img_path in enumerate(files):
        name = os.path.basename(img_path)
        lbl_path = f"{AOI}/train/labels/{os.path.splitext(name)[0]}.txt"
        try:
            rgb = Image.open(img_path).convert("RGB")
        except Exception as e:
            print("skip", name, e)
            continue
        W, H = rgb.size
        boxes = parse_labels(lbl_path, W, H)
        if not boxes:
            continue
        gray = np.array(rgb.convert("L"), dtype=np.uint8)
        entries.append((img_path, boxes, gray, W, H))
    print(f"usable images: {len(entries)}")

    # ---- Stratified image-level split ----
    random.shuffle(entries)
    by_cls = {}
    for e in entries:
        cls = e[1][0][0]                      # first box class as tiebreak
        by_cls.setdefault(cls, []).append(e)
    trn, tst = [], []
    for cls, es in sorted(by_cls.items()):
        n_test = max(1, int(len(es) * TEST_FRAC))
        tst += es[:n_test]
        trn += es[n_test:]
    random.shuffle(trn)
    random.shuffle(tst)
    # ensure all defect classes present in test (move one image if missing)
    def classes_present(entries):
        return set(b[0] for e in entries for b in e[1])
    missing = [c for c in range(6) if c not in classes_present(tst)]
    for c in missing:
        for i, e in enumerate(trn):
            if any(b[0] == c for b in e[1]):
                tst.append(trn.pop(i))
                break
    print(f"split -> train {len(trn)} imgs, test {len(tst)} imgs")
    print("train classes:", sorted(classes_present(trn)))
    print("test  classes:", sorted(classes_present(tst)))

    # ---- Patch extraction ----
    def extract(entries, want_bg):
        xs, ys = [], []
        for img_path, boxes, gray, W, H in entries:
            img_pil = Image.fromarray(np.stack([gray] * 3, axis=-1))
            for c, x0, y0, x1, y1 in boxes:
                bw, bh = x1 - x0, y1 - y0
                side = int(np.clip(CTX * max(bw, bh), MIN_SIDE, MAX_SIDE))
                crop = square_crop(img_pil, (x0 + x1) / 2, (y0 + y1) / 2, side)
                patch = crop.resize((P, P), Image.BILINEAR)
                a = np.array(patch.convert("L"), dtype=np.uint8)
                xs.append(binarize_patch(a).reshape(-1))
                ys.append(c)
            if want_bg:
                placed = 0
                tries = 0
                while placed < N_BG and tries < 60:
                    tries += 1
                    px = random.uniform(BG_SIDE / 2, W - BG_SIDE / 2)
                    py = random.uniform(BG_SIDE / 2, H - BG_SIDE / 2)
                    if overlaps(px, py, BG_SIDE, boxes):
                        continue
                    crop = square_crop(img_pil, px, py, BG_SIDE)
                    patch = crop.resize((P, P), Image.BILINEAR)
                    a = np.array(patch.convert("L"), dtype=np.uint8)
                    xs.append(binarize_patch(a).reshape(-1))
                    ys.append(6)
                    placed += 1
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int64)

    trn_x, trn_y = extract(trn, want_bg=True)
    tst_x, tst_y = extract(tst, want_bg=True)
    print(f"train patches: {trn_x.shape}, test patches: {tst_x.shape}")

    def tally(y):
        import collections
        c = collections.Counter(y.tolist())
        return {CLASS_NAMES[k]: v for k, v in sorted(c.items())}

    print("train class counts:", tally(trn_y))
    print("test  class counts:", tally(tst_y))

    np.save(f"{OUT}/train_x.npy", trn_x)
    np.save(f"{OUT}/train_y.npy", trn_y)
    # Flip augmentation (train only): defects are roughly flip-invariant,
    # and the 6.6k-patch set overfits badly without it (train->99.6% @ 43-52% test).
    txf = trn_x.reshape(-1, P, P)
    aug = np.concatenate([txf[:, :, ::-1].reshape(-1, P * P),   # horizontal flip
                          txf[:, ::-1, :].reshape(-1, P * P)],  # vertical flip
                         axis=0).astype(np.float32)
    trn_x_aug = np.concatenate([trn_x, aug], axis=0)
    trn_y_aug = np.concatenate([trn_y, np.concatenate([trn_y, trn_y])], axis=0)
    np.save(f"{OUT}/train_x.npy", trn_x_aug)
    np.save(f"{OUT}/train_y.npy", trn_y_aug)
    print(f"after flip-aug: train {trn_x_aug.shape}")
    np.save(f"{OUT}/test_x.npy", tst_x)
    np.save(f"{OUT}/test_y.npy", tst_y)
    meta = {"P": P, "n_classes": len(CLASS_NAMES), "class_names": CLASS_NAMES,
            "bin_threshold": 0.0, "invert": False,
            "split": {"train_imgs": len(trn), "test_imgs": len(tst)}}
    with open(f"{OUT}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved to", OUT)


if __name__ == "__main__":
    main()