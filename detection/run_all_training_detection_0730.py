    # WristFX_0730/detection/run_all_training_detection_0730.py
import os
import shutil
import subprocess
from pathlib import Path

# ✅ 설정 경로
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
IMAGE_DIR = DATASET_DIR / "images"
LABEL_DIR = DATASET_DIR / "labels"
YOLO_CONFIG = BASE_DIR / "models" / "detect" / "yolov9-c.yaml"  # ← 수정됨
SAVE_DIR = BASE_DIR / "runs" / "train" / "yolov9-c_0730"

# ✅ 정확한 원본 이미지 및 라벨 경로
RAW_IMAGE_DIR = Path("GRAZPEDWRI-DX/data/images/train_aug")
RAW_LABEL_DIR = Path("GRAZPEDWRI-DX/data/labels/train_aug")



def split_dataset():
    print("📁 Splitting dataset into train/val (8:2)...")

    image_files = sorted(list(RAW_IMAGE_DIR.glob("*.png")))
    total = len(image_files)
    split_idx = int(total * 0.8)
    train_imgs, val_imgs = image_files[:split_idx], image_files[split_idx:]

    for subset, imgs in zip(["train", "val"], [train_imgs, val_imgs]):
        img_out_dir = IMAGE_DIR / subset
        lbl_out_dir = LABEL_DIR / subset
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in imgs:
            shutil.copy(img_path, img_out_dir / img_path.name)
            label_name = img_path.stem + ".txt"
            lbl_path = RAW_LABEL_DIR / label_name
            if lbl_path.exists():
                shutil.copy(lbl_path, lbl_out_dir / label_name)

    print(f"✅ Train: {len(train_imgs)}, Val: {len(val_imgs)}")

def write_data_yaml():
    print("📝 Writing data.yaml for YOLOv9...")
    data_yaml = f"""\
path: {DATASET_DIR}
train: images/train
val: images/val
nc: 1
names: ['fracture']
"""
    with open(BASE_DIR / "data.yaml", "w") as f:
        f.write(data_yaml)
    print("✅ data.yaml saved.")

def train_yolo():
    print("🚀 Starting YOLOv9 training...")

    command = f"""
    python train_dual_ddp.py --img 640 --batch 16 --epochs 100 \
        --data data.yaml --cfg {YOLO_CONFIG} \
        --weights yolov9-c.pt --device 0,1 \
        --project runs/train --name yolov9-c_0730 --exist-ok
    """
    subprocess.run(command, shell=True, check=True)

if __name__ == "__main__":
    print("🧠 [Detection Pipeline] run_all_training_detection_0730.py")
    split_dataset()
    write_data_yaml()
    train_yolo()
    print("✅ Training complete.")
