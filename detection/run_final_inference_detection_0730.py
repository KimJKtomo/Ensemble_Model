# WristFX_0730/detection/run_final_inference_detection_0730.py

import os
import torch
from pathlib import Path
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox
import cv2
import pandas as pd
from tqdm import tqdm
from models.common import DetectMultiBackend
from sklearn.metrics import precision_score, recall_score

# ✅ 경로 설정
BASE_DIR = Path(__file__).resolve().parent
WEIGHT_PATH = BASE_DIR / "runs" / "train_1020" / "yolov9-e" / "weights" / "best.pt"
IMG_DIR = BASE_DIR / "dataset" / "images" / "val"
LABEL_DIR = BASE_DIR / "dataset" / "labels" / "val"
SAVE_DIR = BASE_DIR / "inference_results"
YOLO_CFG = BASE_DIR / "models" / "detect" / "yolov9-e.yaml"  # ← 수정됨

SAVE_DIR.mkdir(exist_ok=True, parents=True)

model = DetectMultiBackend(
    weights=WEIGHT_PATH,
    device="cuda:0",
    dnn=False,
    data=BASE_DIR / "data.yaml"
)
model.eval()

image_files = list(IMG_DIR.glob("*.png"))
results = []

for img_path in tqdm(image_files, desc="🔍 Inference"):
    # ✅ 이미지 전처리
    orig = cv2.imread(str(img_path))
    img = letterbox(orig, new_shape=640)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR → RGB
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0).to("cuda:0")

    # ✅ 추론
    with torch.no_grad():
        pred = model(img)
        pred = non_max_suppression(pred)[0]

    filename = img_path.name
    pred_class = 1 if pred is not None and len(pred) else 0
    gt_label_file = LABEL_DIR / (img_path.stem + ".txt")
    gt_class = 1 if gt_label_file.exists() else 0

    results.append({
        "filename": filename,
        "gt_label": gt_class,
        "pred_label": pred_class
    })

    # ✅ 결과 이미지 저장
    vis = orig.copy()
    if pred_class == 1:
        for *xyxy, conf, cls in pred:
            xyxy = [int(x.item()) for x in xyxy]
            cv2.rectangle(vis, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(vis, f"{conf:.2f}", (xyxy[0], xyxy[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    save_path = SAVE_DIR / f"{img_path.stem}.png"
    cv2.imwrite(str(save_path), vis)

# ✅ 평가 지표 출력
df = pd.DataFrame(results)
df.to_csv(SAVE_DIR / "detection_results.csv", index=False)

y_true = df["gt_label"].tolist()
y_pred = df["pred_label"].tolist()
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
tp = sum((df["gt_label"] == 1) & (df["pred_label"] == 1))
fp = sum((df["gt_label"] == 0) & (df["pred_label"] == 1))
fn = sum((df["gt_label"] == 1) & (df["pred_label"] == 0))
mAP = tp / (tp + fp + fn + 1e-6)

print(f"\n📊 Detection Evaluation:")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"mAP-like (TP / (TP+FP+FN)): {mAP:.4f}")
