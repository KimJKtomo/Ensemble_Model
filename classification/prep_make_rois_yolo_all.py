# prep_make_rois_yolo_all_fixedpaths.py
import os, glob, csv, random
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image

# =========================
# Fixed paths & params
# =========================
IM_ROOT = "/mnt/data/KimJG/Yolo_test/YOLOv9-Fracture-Detection/GRAZPEDWRI-DX/data/images"
LB_ROOT = "/mnt/data/KimJG/Yolo_test/YOLOv9-Fracture-Detection/GRAZPEDWRI-DX/data/labels"
OUT_DIR = "/mnt/data/KimJG/SwinT/mediaiOA_swinT_cls-main_original/20250623_Single/WristFX_0730/classification/roi_out"

FRACTURE_CLASSES = {3}     # 필요시 {3,4} 등으로 변경
CROP_SIZE = 128
POS_PER_BOX = 2
NEG_PER_IMAGE = 2
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# =========================
# Helpers
# =========================
def _to_rgb_uint8(img: Image.Image) -> Image.Image:
    import numpy as np
    if img.mode in ("I;16", "I"):
        arr = np.array(img, dtype=np.uint16)
        lo, hi = np.percentile(arr, (1, 99))
        hi = max(hi, lo + 1)
        arr = np.clip((arr - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")

    if img.mode in ("LA", "P"):
        img = img.convert("RGBA")
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return img

def _try_open_image(img_path: str):
    if os.path.isfile(img_path):
        try:
            return Image.open(img_path)
        except Exception:
            return None
    stem = os.path.splitext(img_path)[0]
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        p = stem + ext
        if os.path.isfile(p):
            try:
                return Image.open(p)
            except Exception:
                continue
    return None

def _xywhn_to_xyxy(xc, yc, w, h, W, H):
    x1 = int((xc - w/2) * W)
    y1 = int((yc - h/2) * H)
    x2 = int((xc + w/2) * W)
    y2 = int((yc + h/2) * H)
    return max(0, x1), max(0, y1), min(W, x2), min(H, y2)

def _crop_safe(W, H, cx, cy, size):
    x1 = max(0, int(cx - size//2)); y1 = max(0, int(cy - size//2))
    x2 = min(W, x1 + size);         y2 = min(H, y1 + size)
    if x2 - x1 < size: x1, x2 = max(0, W - size), W
    if y2 - y1 < size: y1, y2 = max(0, H - size), H
    return x1, y1, x2, y2

def _bone_rich(img_np: np.ndarray, edge_thr=0.06) -> bool:
    g = img_np.mean(axis=2)
    gx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
    gy = np.abs(np.diff(g, axis=0, prepend=g[:1, :]))
    grad = np.hypot(gx, gy)
    return (grad > 5).mean() > edge_thr

def _mine_negative_rois(pil: Image.Image, k=2, size=128, tries=20):
    W, H = pil.size; boxes = []
    for _ in range(tries):
        cx = random.randint(size//2, max(size//2, W - size//2))
        cy = random.randint(size//2, max(size//2, H - size//2))
        x1, y1, x2, y2 = _crop_safe(W, H, cx, cy, size)
        crop = pil.crop((x1, y1, x2, y2))
        if _bone_rich(np.asarray(crop)):
            boxes.append((x1, y1, x2, y2))
            if len(boxes) >= k: break
    if not boxes:
        boxes.append(_crop_safe(W, H, W // 2, H // 2, size))
    return boxes

def _jitter_center(x1, y1, x2, y2, max_j=16):
    cx = (x1 + x2) / 2 + random.randint(-max_j, max_j)
    cy = (y1 + y2) / 2 + random.randint(-max_j, max_j)
    return cx, cy

# =========================
# Core
# =========================
def process_split(split: str, im_root: str, lb_root: str, out_root: str,
                  fracture_classes: set, size: int, pos_per_box: int, neg_per_image: int):
    out_split = Path(out_root) / split
    pos_dir = out_split / "roi_pos"; pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir = out_split / "roi_neg"; neg_dir.mkdir(parents=True, exist_ok=True)
    idx_csv = out_split / "roi_index.csv"

    rows = []
    label_dir = Path(lb_root) / split
    image_dir = Path(im_root) / split
    txt_list = sorted(glob.glob(str(label_dir / "*.txt")))
    print(f"[{split}] label files: {len(txt_list)}")

    for tpath in txt_list:
        stem = Path(tpath).stem
        img_guess = str(image_dir / f"{stem}.png")
        pil = _try_open_image(img_guess)
        if pil is None:
            # fallback: try same stem with other extensions in image_dir
            pil = _try_open_image(str(image_dir / stem))
            if pil is None:
                # final fallback: replace 'labels' -> 'images'
                mirror = str((Path(lb_root).parent / "images" / split / f"{stem}.png"))
                pil = _try_open_image(mirror)
                if pil is None:
                    continue
                img_guess = mirror

        pil = _to_rgb_uint8(pil)
        W, H = pil.size

        with open(tpath, "r") as f:
            lines = [l.strip().split() for l in f if l.strip()]

        pos_boxes = []
        for parts in lines:
            if len(parts) != 5: continue
            cid, xc, yc, w, h = parts
            cid = int(float(cid))
            if cid in fracture_classes:
                xc, yc, w, h = map(float, (xc, yc, w, h))
                pos_boxes.append(_xywhn_to_xyxy(xc, yc, w, h, W, H))

        if pos_boxes:
            for (x1, y1, x2, y2) in pos_boxes:
                for _ in range(pos_per_box):
                    cx, cy = _jitter_center(x1, y1, x2, y2)
                    bx1, by1, bx2, by2 = _crop_safe(W, H, cx, cy, size)
                    roi = pil.crop((bx1, by1, bx2, by2))
                    out_p = pos_dir / f"{stem}_{bx1}_{by1}_{bx2}_{by2}.png"
                    roi.save(out_p)
                    rows.append([str(img_guess), str(out_p), 1])
        else:
            for (bx1, by1, bx2, by2) in _mine_negative_rois(pil, k=neg_per_image, size=size):
                roi = pil.crop((bx1, by1, bx2, by2))
                out_p = neg_dir / f"{stem}_{bx1}_{by1}_{bx2}_{by2}.png"
                roi.save(out_p)
                rows.append([str(img_guess), str(out_p), 0])

    with open(idx_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path_full", "path_roi", "label"])
        w.writerows(rows)
    pos_cnt = sum(1 for r in rows if r[2] == 1)
    neg_cnt = sum(1 for r in rows if r[2] == 0)
    print(f"[{split}] saved {len(rows)} crops (pos={pos_cnt}, neg={neg_cnt}) -> {idx_csv}")

def run_all():
    for split in ("train", "valid", "test"):
        process_split(
            split=split,
            im_root=IM_ROOT,
            lb_root=LB_ROOT,
            out_root=OUT_DIR,
            fracture_classes=FRACTURE_CLASSES,
            size=CROP_SIZE,
            pos_per_box=POS_PER_BOX,
            neg_per_image=NEG_PER_IMAGE
        )

if __name__ == "__main__":
    run_all()
