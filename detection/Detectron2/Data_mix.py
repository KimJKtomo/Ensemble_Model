from pathlib import Path

DATA = Path("/mnt/data/KimJG/Yolo_test/YOLOv9-Fracture-Detection/GRAZPEDWRI-DX/data")
IMG_SPLITS = ["train","valid","test"]

def relabel_fracture_to_zero(src_lbl_dir: Path, dst_lbl_dir: Path):
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    for p in src_lbl_dir.glob("*.txt"):
        lines_out=[]
        for ln in p.read_text().splitlines():
            t=ln.split()
            if len(t)==5 and t[0]=="3":  # 3=fracture → 0
                lines_out.append("0 "+" ".join(t[1:]))
        (dst_lbl_dir/p.name).write_text(("\n".join(lines_out)+"\n") if lines_out else "")

# 1) 원본 라벨: labels/{split} → labels_fracture_only/{split}
for sp in IMG_SPLITS:
    src = DATA/"labels"/sp
    dst = DATA/"labels_fracture_only"/sp
    dst.mkdir(parents=True, exist_ok=True)
    relabel_fracture_to_zero(src, dst)

# 2) 크롭 라벨: labels/{split}_crop → labels_crop_fracture_only/{split}_crop
for sp in IMG_SPLITS:
    src = DATA/"labels"/f"{sp}_crop"
    if not src.exists(): continue
    dst = DATA/"labels_crop_fracture_only"/f"{sp}_crop"
    dst.mkdir(parents=True, exist_ok=True)
    relabel_fracture_to_zero(src, dst)

# 3) MIX 폴더 생성: images/{split}_mix, labels/{split}_mix
def symlink_safe(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists() or dst.is_symlink(): dst.unlink()
    except FileNotFoundError:
        pass
    dst.symlink_to(src)

IMG_EXTS = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}

for sp in IMG_SPLITS:
    mix_img = DATA/"images"/f"{sp}_mix"
    mix_lbl = DATA/"labels"/f"{sp}_mix"
    mix_img.mkdir(parents=True, exist_ok=True)
    mix_lbl.mkdir(parents=True, exist_ok=True)

    # 3-1) 원본 이미지 + 재라벨(Fracture→0)
    src_img_dir = DATA/"images"/sp
    src_lbl_dir = DATA/"labels_fracture_only"/sp
    for img in src_img_dir.iterdir():
        if img.suffix.lower() not in IMG_EXTS: continue
        lbl = src_lbl_dir/(img.stem + ".txt")
        if not lbl.exists(): lbl.write_text("")  # 빈 라벨 허용
        symlink_safe(img, mix_img/img.name)
        symlink_safe(lbl, mix_lbl/(img.stem + ".txt"))

    # 3-2) 크롭 이미지 + 재라벨(Fracture→0)
    crop_img_dir = DATA/"images"/f"{sp}_crop"
    crop_lbl_dir = DATA/"labels_crop_fracture_only"/f"{sp}_crop"
    if crop_img_dir.exists():
        for img in crop_img_dir.iterdir():
            if img.suffix.lower() not in IMG_EXTS: continue
            lbl = crop_lbl_dir/(img.stem + ".txt")
            if not lbl.exists(): lbl.write_text("")
            symlink_safe(img, mix_img/img.name)
            symlink_safe(lbl, mix_lbl/(img.stem + ".txt"))

print("DONE:")
print(" images/train_mix  →", DATA/"images"/"train_mix")
print(" images/valid_mix  →", DATA/"images"/"valid_mix")
print(" images/test_mix   →", DATA/"images"/"test_mix")
print(" labels/train_mix  →", DATA/"labels"/"train_mix")
print(" labels/valid_mix  →", DATA/"labels"/"valid_mix")
print(" labels/test_mix   →", DATA/"labels"/"test_mix")

