import sys
sys.path.append("/mnt/data/KimJG/Yolo_test/YOLOv9-Fracture-Detection")
import os
os.environ["PYTHONPATH"] += os.pathsep + "/mnt/data/KimJG/Yolo_test/YOLOv9-Fracture-Detection"
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from timm import create_model
from torch.nn import Sigmoid
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from ray import tune
from ray.air import session

# ✅ YOLO 관련

from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from unified_dataset_0704 import UnifiedDataset

# ✅ 기본 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sigmoid = Sigmoid()

# ✅ 전처리 정의
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ✅ Age Group Mapping
def AGE_GROUP_FN(age):
    age = float(age)
    if age < 5: return 0
    elif age < 10: return 1
    elif age < 15: return 2
    else: return 3

# ✅ Objective 함수 정의
def objective(config):
    MODEL_BASE_PATH = "/mnt/data/KimJG/SwinT/mediaiOA_swinT_cls-main_original/20250623_Single/0723_best_ddp_convnextv2_agegroup{}.pt"
    YOLO_WEIGHT = "/mnt/data/KimJG/Yolo_test/YOLOv9-Fracture-Detection/runs/train/yolov9-c3/weights/best.pt"
    CSV_PATH = "/mnt/data/KimJG/SwinT/mediaiOA_swinT_cls-main_original/20250623_Single/test_set_0704.csv"

    df = pd.read_csv(CSV_PATH)
    df["age_group_label"] = df["age"].astype(float).apply(AGE_GROUP_FN)
    df["label"] = df["fracture_visible"].apply(lambda x: 1.0 if x == 1 else 0.0)

    yolo_model = DetectMultiBackend(YOLO_WEIGHT, device=DEVICE)
    yolo_model.eval()

    all_preds, all_labels = [], []

    for AGE_GROUP in [0, 1, 2, 3]:
        model = create_model("convnextv2_large.fcmae_ft_in22k_in1k_384", pretrained=False, num_classes=1)
        model.load_state_dict(torch.load(MODEL_BASE_PATH.format(AGE_GROUP), map_location=DEVICE))
        model.to(DEVICE).eval()

        df_group = df[df["age_group_label"] == AGE_GROUP].reset_index(drop=True)
        dataset = UnifiedDataset(df_group, transform=transform, task="fracture_only")
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        for i, (img_tensor, label) in enumerate(loader):
            img_tensor = img_tensor.to(DEVICE)
            img_path = df_group.iloc[i]["image_path"]
            orig = cv2.imread(img_path)
            h, w = orig.shape[:2]

            with torch.no_grad():
                cls_out = model(img_tensor).squeeze()
                cls_prob = sigmoid(cls_out).item()

            # ✅ YOLO 처리
            yolo_input = letterbox(orig, new_shape=640, stride=32, auto=True)[0]
            yolo_input = yolo_input[:, :, ::-1].transpose(2, 0, 1)
            yolo_input = np.ascontiguousarray(yolo_input)
            im = torch.from_numpy(yolo_input).to(DEVICE).float() / 255.0
            im = im.unsqueeze(0)

            with torch.no_grad():
                yolo_pred = yolo_model(im)[0][1]
                yolo_pred = non_max_suppression(yolo_pred, config["conf_thres"], config["iou_thres"], classes=[3])[0]

            yolo_score = config["default_bbox_score"]
            if yolo_pred is not None and len(yolo_pred) > 0:
                yolo_score = yolo_pred[:, 4].max().item()

            final_score = config["alpha"] * yolo_score + (1 - config["alpha"]) * cls_prob
            final_pred = int(final_score > config["final_thresh"])

            all_preds.append(final_pred)
            all_labels.append(int(label.item()))

    f1 = f1_score(all_labels, all_preds)
    session.report({
        "f1": f1
    })


# ✅ Search Space 정의
search_space = {
    "conf_thres": tune.grid_search([0.25, 0.3, 0.35, 0.4, 0.45]),
    "iou_thres": tune.grid_search([0.3, 0.35, 0.4]),
    "alpha": tune.grid_search([0.4, 0.45 ,0.5, 0.55 ,0.6]),
    "final_thresh": tune.grid_search([0.25,0.30, 0.35, 0.40, 0.45]),
    "default_bbox_score": tune.grid_search([0.0, 0.05, 0.1, 0.15, 0.2])
}

if __name__ == "__main__":
    tune.run(
        objective,
        config=search_space,
        resources_per_trial={"cpu": 4, "gpu": 0.5},
        local_dir="ray_results/0725_tune",
        metric="f1",
        mode="max"
    )
