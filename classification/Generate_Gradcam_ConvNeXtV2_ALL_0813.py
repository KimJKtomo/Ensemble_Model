# cam_wrist_eigencam.py
import os
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from timm import create_model
from torch.nn import Sigmoid
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from unified_dataset_0704 import UnifiedDataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ===== 기본 설정 =====
BASE_DIR = os.path.dirname(__file__)
MODEL_BASE_DIR = "0805_Model"
MODEL_BASE_PATH = "best_ddp_convnextv2_{}_{}.pt"  # e.g., best_ddp_convnextv2_AP_0.pt
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "eigencam_results_rgb")
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

sigmoid = Sigmoid()
all_results = []

def AGE_GROUP_FN(age):
    age = float(age)
    if age < 5: return 0
    elif age < 10: return 1
    elif age < 15: return 2
    else: return 3

torch.backends.cudnn.benchmark = True

# ===== AP / Lat 별 루프 =====
for proj in ["AP", "Lat"]:
    test_csv = os.path.join(BASE_DIR, f"test_set_0730_{proj}.csv")
    df_full = pd.read_csv(test_csv)
    df_full["age_group"] = df_full["age"].astype(float).apply(AGE_GROUP_FN)
    df_full["label"] = df_full["fracture_visible"].apply(lambda x: 1.0 if x == 1 else 0.0)

    for age_group in [0, 1, 2, 3]:
        print(f"\n🔍 EigenCAM for {proj}_{age_group}...")

        # 모델 로드
        model_path = os.path.join(BASE_DIR, MODEL_BASE_DIR, MODEL_BASE_PATH.format(proj, age_group))
        output_dir = os.path.join(OUTPUT_BASE_DIR, proj, f"agegroup{age_group}")
        os.makedirs(output_dir, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_model(
            "convnextv2_large.fcmae_ft_in22k_in1k_384",
            pretrained=False,
            num_classes=1,
            drop_rate=0.1,
            drop_path_rate=0.2
        )
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        # ConvNeXtV2 target layer (권장: 마지막 블록의 depthwise conv)
        target_layers = [model.stages[-1].blocks[-1].conv_dw]

        # 데이터 로더
        df_group = df_full[df_full["age_group"] == age_group].reset_index(drop=True)
        dataset = UnifiedDataset(df_group, transform=transform, task="fracture_only")
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

        results = []
        with EigenCAM(model=model, target_layers=target_layers) as cam:
            for i, (img_tensor, label) in enumerate(tqdm(loader)):
                img_tensor = img_tensor.to(device, non_blocking=True)

                # 예측
                with torch.no_grad():
                    output = model(img_tensor).squeeze()
                    prob = sigmoid(output).item()
                    pred = int(prob > 0.5)

                # 원본 이미지 로드 (BGR)
                img_path = df_group.iloc[i]["image_path"]
                orig_bgr = cv2.imread(img_path)
                if orig_bgr is None:
                    print(f"❌ 이미지 로딩 실패: {img_path}")
                    continue
                h, w = orig_bgr.shape[:2]

                # RGB float(0~1) 유지
                rgb_img_float = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

                # EigenCAM 생성
                grayscale_cam = cam(input_tensor=img_tensor)[0]           # [H, W] in [0,1]
                grayscale_cam = cv2.resize(grayscale_cam, (w, h))
                cam_rgb_uint8 = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)  # RGB uint8
                cam_rgb_bgr = cv2.cvtColor(cam_rgb_uint8, cv2.COLOR_RGB2BGR)

                # 저장
                save_name = os.path.basename(img_path).replace(".png", f"_eigen_pred{pred}_label{int(label.item())}.png")
                cv2.imwrite(os.path.join(output_dir, save_name), cam_rgb_bgr)

                results.append({
                    "filename": os.path.basename(img_path),
                    "true_label": int(label.item()),
                    "pred_label": pred,
                    "probability": prob,
                    "age_group": age_group,
                    "projection": proj
                })

        # Confusion Matrix & CSV
        if len(results) > 0:
            df_result = pd.DataFrame(results)
            cm = confusion_matrix(df_result["true_label"], df_result["pred_label"])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Fracture"])
            disp.plot(cmap="Blues", values_format="d")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"confmat_{proj}_{age_group}.png"))
            plt.close()

            df_result.to_csv(os.path.join(output_dir, "eigencam_summary.csv"), index=False)
            all_results.extend(results)

# 전체 결과 저장
if len(all_results) > 0:
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(os.path.join(OUTPUT_BASE_DIR, "eigencam_results_all.csv"), index=False)

print(f"\n✅ Done. Saved under: {OUTPUT_BASE_DIR}")
