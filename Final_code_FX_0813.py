import os
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from timm import create_model
from torch.nn import Sigmoid
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from unified_dataset_0704 import UnifiedDataset

# YOLO 관련 경로 추가
import sys
sys.path.append("/mnt/data/KimJG/Yolo_test/YOLOv9-Fracture-Detection")
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox
from models.common import DetectMultiBackend

# 설정 하이퍼파라미터 적용완료
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sigmoid = Sigmoid()
CONF_THRES = 0.25
IOU_THRES = 0.3
ALPHA = 0.5
CLASS_NAMES = {3: "fracture"}
final_thresh =0.45

# 경로
MODEL_BASE_PATH = "/mnt/data/KimJG/SwinT/mediaiOA_swinT_cls-main_original/20250623_Single/0723_best_ddp_convnextv2_agegroup{}.pt"
YOLO_WEIGHT = "/mnt/data/KimJG/Yolo_test/YOLOv9-Fracture-Detection/runs/train/yolov9-c3/weights/best.pt"
GT_LABEL_DIR = "/mnt/data/KimJG/ELBOW_test/Kaggle_dataset/folder_structure/yolov5/labels"
CSV_PATH = "/mnt/data/KimJG/SwinT/mediaiOA_swinT_cls-main_original/20250623_Single/test_set_0704.csv"
OUTPUT_DIR = "0728_final_inference_ensemble_convnextv2_yolo_2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 전처리
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def AGE_GROUP_FN(age):
    age = float(age)
    if age < 5: return 0
    elif age < 10: return 1
    elif age < 15: return 2
    else: return 3

# 데이터 로딩
df_test = pd.read_csv(CSV_PATH)
df_test["age_group_label"] = df_test["age"].astype(float).apply(AGE_GROUP_FN)
df_test["label"] = df_test["fracture_visible"].apply(lambda x: 1.0 if x == 1 else 0.0)

# YOLO 로딩
yolo_model = DetectMultiBackend(YOLO_WEIGHT, device=DEVICE)
yolo_model.eval()

all_results = []

# 나이 그룹별 실행
for AGE_GROUP in [0, 1, 2, 3]:
    print(f"\n🔍 Running Age Group {AGE_GROUP}...")
    model = create_model("convnextv2_large.fcmae_ft_in22k_in1k_384", pretrained=False, num_classes=1)
    model.load_state_dict(torch.load(MODEL_BASE_PATH.format(AGE_GROUP), map_location=DEVICE))
    model.to(DEVICE).eval()

    cam = GradCAMPlusPlus(model=model, target_layers=[model.stages[-1].blocks[-1].conv_dw])
    df_group = df_test[df_test["age_group_label"] == AGE_GROUP].reset_index(drop=True)
    dataset = UnifiedDataset(df_group, transform=transform, task="fracture_only")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    save_dir = os.path.join(OUTPUT_DIR, f"agegroup{AGE_GROUP}")
    orig_dir = os.path.join(save_dir, "original")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(orig_dir, exist_ok=True)

    for i, (img_tensor, label) in enumerate(tqdm(loader)):
        img_tensor = img_tensor.to(DEVICE)
        img_path = df_group.iloc[i]["image_path"]
        orig = cv2.imread(img_path)
        h, w = orig.shape[:2]

        with torch.no_grad():
            cls_out = model(img_tensor).squeeze()
            cls_prob = sigmoid(cls_out).item()
            cls_pred = int(cls_prob > 0.5)

        grayscale_cam = cam(input_tensor=img_tensor)[0]
        grayscale_cam = cv2.resize(grayscale_cam, (w, h))
        rgb_img = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB) / 255.0
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        yolo_input = letterbox(orig, new_shape=640, stride=32, auto=True)[0]
        yolo_input = yolo_input[:, :, ::-1].transpose(2, 0, 1)
        yolo_input = np.ascontiguousarray(yolo_input)
        im = torch.from_numpy(yolo_input).to(DEVICE).float() / 255.0
        im = im.unsqueeze(0)

        with torch.no_grad():
            yolo_pred = yolo_model(im)[0][1]
            yolo_pred = non_max_suppression(yolo_pred, CONF_THRES, IOU_THRES, classes=[3])[0]

        yolo_score = 0.0
        overlay = cam_image.copy()

        if yolo_pred is not None and len(yolo_pred) > 0:
            yolo_pred[:, :4] = scale_boxes(im.shape[2:], yolo_pred[:, :4], orig.shape).round()
            yolo_score = yolo_pred[:, 4].max().item()
            for *xyxy, conf, cls_id in yolo_pred:
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(overlay, f"YOLO {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        filestem = df_group.iloc[i]["filestem"]
        gt_path = os.path.join(GT_LABEL_DIR, f"{filestem}.txt")
        if os.path.exists(gt_path):
            with open(gt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5 or int(parts[0]) != 3:
                        continue
                    cls_id, xc, yc, bw, bh = map(float, parts)
                    x1 = int((xc - bw / 2) * w)
                    y1 = int((yc - bh / 2) * h)
                    x2 = int((xc + bw / 2) * w)
                    y2 = int((yc + bh / 2) * h)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(overlay, "GT", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        final_score = ALPHA * yolo_score + (1 - ALPHA) * cls_prob
        final_pred = int(final_score > final_thresh)

        filename = os.path.basename(img_path)
        save_name = filename.replace(".png", f"_pred{final_pred}_label{int(label.item())}.png")
        cv2.imwrite(os.path.join(save_dir, save_name), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(orig_dir, filename), orig)  # ✅ 원본 저장

        all_results.append({
            "filename": filename,
            "true_label": int(label.item()),
            "cls_prob": round(cls_prob, 4),
            "cls_pred": cls_pred,
            "yolo_score": round(yolo_score, 4),
            "final_score": round(final_score, 4),
            "final_pred": final_pred,
            "age_group": AGE_GROUP
        })

# 결과 저장
df_result = pd.DataFrame(all_results)
df_result.to_csv(os.path.join(OUTPUT_DIR, "final_ensemble_results.csv"), index=False)

# Confusion Matrix
cm = confusion_matrix(df_result["true_label"], df_result["final_pred"])
print("\n📊 Final Confusion Matrix:\n", cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Fracture"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Soft Voting (ConvNeXt + YOLO)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "final_confusion_matrix.png"))
plt.close()
