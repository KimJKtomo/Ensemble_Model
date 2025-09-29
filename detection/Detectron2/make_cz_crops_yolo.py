import argparse
import random
from pathlib import Path
import cv2
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

class YoloBox:
    def __init__(self, cls, cx, cy, w, h):
        self.cls = cls
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h

    def to_xyxy(self, W, H):
        x = self.cx * W
        y = self.cy * H
        w = self.w * W
        h = self.h * H
        return int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)

    def to_line(self):
        return f"{self.cls} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"

    @staticmethod
    def from_line(line):
        p = line.strip().split()
        if len(p) != 5:
            return None
        return YoloBox(int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4]))

def clamp(v, a, b): return max(a, min(b, v))

def load_labels(txt_path: Path):
    if not txt_path.exists():
        return []
    out = []
    for ln in txt_path.read_text().splitlines():
        if not ln.strip(): continue
        b = YoloBox.from_line(ln)
        if b: out.append(b)
    return out

def save_labels(txt_path: Path, boxes):
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w") as f:
        for b in boxes:
            f.write(b.to_line() + "\n")

def compute_crop_xyxy(bx1, by1, bx2, by2, scale, W, H, min_side):
    cx = (bx1 + bx2) / 2
    cy = (by1 + by2) / 2
    bw = (bx2 - bx1) * scale
    bh = (by2 - by1) * scale
    bw = max(bw, min_side)
    bh = max(bh, min_side)
    side = int(round(max(bw, bh)))
    sx1 = int(round(cx - side / 2))
    sy1 = int(round(cy - side / 2))
    sx2 = sx1 + side
    sy2 = sy1 + side
    sx1 = clamp(sx1, 0, W); sy1 = clamp(sy1, 0, H)
    sx2 = clamp(sx2, 0, W); sy2 = clamp(sy2, 0, H)
    return sx1, sy1, sx2, sy2

def remap_box_to_crop(bx1, by1, bx2, by2, crop, Wc, Hc, cls_id):
    cx1, cy1, cx2, cy2 = crop
    nx1 = clamp(bx1 - cx1, 0, Wc)
    ny1 = clamp(by1 - cy1, 0, Hc)
    nx2 = clamp(bx2 - cx1, 0, Wc)
    ny2 = clamp(by2 - cy1, 0, Hc)
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    w = (nx2 - nx1) / Wc
    h = (ny2 - ny1) / Hc
    cx = (nx1 + nx2) / 2 / Wc
    cy = (ny1 + ny2) / 2 / Hc
    return YoloBox(cls_id, cx, cy, w, h)

def process_split(split, args):
    img_dir = args.data / "images" / split
    lbl_dir = args.data / "labels" / split
    out_img_dir = args.data / "images" / f"{split}_crop"
    out_lbl_dir = args.data / "labels" / f"{split}_crop"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(list(img_dir.rglob("*")), desc=f"{split}"):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        stem = img_path.stem
        lbl_path = lbl_dir / f"{stem}.txt"
        boxes = load_labels(lbl_path)
        if not boxes:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        for bi, box in enumerate(boxes):
            if args.include_classes and box.cls not in args.include_classes:
                continue
            bx1,by1,bx2,by2 = box.to_xyxy(W,H)
            for s in args.scales:
                cx1,cy1,cx2,cy2 = compute_crop_xyxy(bx1,by1,bx2,by2,s,W,H,args.min_crop)
                crop = img[cy1:cy2, cx1:cx2]
                Hc, Wc = crop.shape[:2]
                remap = remap_box_to_crop(bx1,by1,bx2,by2,(cx1,cy1,cx2,cy2),Wc,Hc,box.cls)
                if not remap:
                    continue
                out_name = f"{stem}_c{box.cls}_b{bi}_s{str(s).replace('.','p')}"
                out_img_path = out_img_dir / f"{out_name}.png"  # PNG 유지
                out_lbl_path = out_lbl_dir / f"{out_name}.txt"
                cv2.imwrite(str(out_img_path), crop)
                save_labels(out_lbl_path,[remap])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="데이터 루트 (/mnt/.../data)")
    ap.add_argument("--scales", type=float, nargs="+", default=[1.6])
    ap.add_argument("--min-crop", type=int, default=256)
    ap.add_argument("--include-classes", type=int, nargs="+", default=None)
    args = ap.parse_args()

    for split in ["train","valid","test"]:
        process_split(split,args)

if __name__=="__main__":
    main()
