# Generate_Gradcam_ConvNeXtV2_ALL_0919_fixed.py
import os
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from timm import create_model
from torch import nn
from torch.nn import Sigmoid
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from unified_dataset_0704 import UnifiedDataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ===== 기본 설정 =====
BASE_DIR = os.path.dirname(__file__)

# 학습 산출물 네이밍 규칙과 일치 (Late-Fusion 학습 스크립트 기준)
MODEL_DIR = os.path.join(BASE_DIR, "0917_Model")
MODEL_NAME_TMPL = "0917_ddp_convnextv2_{}_{}.pt"  # {proj} {age_group}

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Late-Fusion state_dict → 단일 백본으로 로드 =====
def build_backbone_and_head():
    # 분류 헤드 제거된 백본
    backbone = create_model(
        "convnextv2_large.fcmae_ft_in22k_in1k_384",
        pretrained=False,
        num_classes=0,
        global_pool='avg',
        drop_rate=0.0,
        drop_path_rate=0.0
    )
    # 얕은 로짓 헤드
    classifier = nn.Linear(backbone.num_features, 1)
    # 시퀀셜 래핑
    model = nn.Sequential(backbone, classifier)
    return model, backbone

def load_state_into_backbone(backbone, model_path):
    # 안전 로드. PyTorch>=2.4는 weights_only 지원. 하위 버전도 동작하도록 try/except.
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)

    # Late-Fusion 키 프리픽스 추출: g.* 우선, 없으면 c.* 시도, 둘 다 없으면 단일 백본 가정
    has_g = any(k.startswith("g.") for k in state.keys())
    has_c = any(k.startswith("c.") for k in state.keys())

    if has_g:
        state_backbone = {k.replace("g.", ""): v for k, v in state.items() if k.startswith("g.")}
    elif has_c:
        state_backbone = {k.replace("c.", ""): v for k, v in state.items() if k.startswith("c.")}
    else:
        # 단일 백본 저장본이라면 그대로 사용 (헤드 키는 무시될 수 있도록 strict=False)
        # head.* 또는 head.0/2.* 같은 키가 있어도 백본에는 적재되지 않음
        state_backbone = {k: v for k, v in state.items() if not k.startswith("head")}

    missing, unexpected = backbone.load_state_dict(state_backbone, strict=False)
    # strict=False: 버퍼/일부 키 불일치 허용

    return missing, unexpected

# ===== AP / Lat 루프 =====
for proj in [ "Lat"]:
    test_csv = os.path.join(BASE_DIR, f"test_set_0730_{proj}.csv")
    if not os.path.isfile(test_csv):
        print(f"❌ CSV 없음: {test_csv} → 건너뜀")
        continue

    df_full = pd.read_csv(test_csv)
    # 컬럼 정규화
    df_full["age_group"] = df_full["age"].astype(float).apply(AGE_GROUP_FN)
    df_full["label"] = df_full["fracture_visible"].apply(lambda x: 1.0 if x == 1 else 0.0)

    for age_group in [0, 1, 2, 3]:
        print(f"\n🔍 EigenCAM for {proj}_{age_group}...")

        # 출력 폴더
        out_dir = os.path.join(OUTPUT_BASE_DIR, proj, f"agegroup{age_group}")
        os.makedirs(out_dir, exist_ok=True)

        # 모델 경로
        model_path = os.path.join(MODEL_DIR, MODEL_NAME_TMPL.format(proj, age_group))
        if not os.path.isfile(model_path):
            print(f"❌ 가중치 없음: {model_path} → 건너뜀")
            continue

        # 모델 구성 및 가중치 로드
        model, backbone = build_backbone_and_head()
        load_state_into_backbone(backbone, model_path)
        model.to(device).eval()

        # 타깃 레이어: 마지막 스테이지 마지막 블록 depthwise conv
        target_layers = [backbone.stages[-1].blocks[-1].conv_dw]

        # 데이터 로더
        df_group = df_full[df_full["age_group"] == age_group].reset_index(drop=True)
        if len(df_group) == 0:
            print(f"⚠️ 샘플 없음: {proj}_{age_group}")
            continue

        dataset = UnifiedDataset(df_group, transform=transform, task="fracture_only")
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

        results = []
        with EigenCAM(model=model, target_layers=target_layers) as cam:
            for i, (img_tensor, label) in enumerate(tqdm(loader)):
                img_tensor = img_tensor.to(device, non_blocking=True)

                # 예측
                with torch.no_grad():
                    logit = model(img_tensor).squeeze()
                    prob = sigmoid(logit).item()
                    pred = int(prob > 0.5)

                # 원본 이미지 읽기
                img_path = df_group.iloc[i]["image_path"]
                orig_bgr = cv2.imread(img_path)
                if orig_bgr is None:
                    print(f"❌ 이미지 로딩 실패: {img_path}")
                    continue
                h, w = orig_bgr.shape[:2]
                rgb_float = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

                # CAM
                grayscale_cam = cam(input_tensor=img_tensor)[0]
                grayscale_cam = cv2.resize(grayscale_cam, (w, h))
                cam_rgb = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)  # RGB uint8
                cam_bgr = cv2.cvtColor(cam_rgb, cv2.COLOR_RGB2BGR)

                # 저장
                save_name = os.path.basename(img_path).rsplit(".", 1)[0]
                save_name = f"{save_name}_eigen_pred{pred}_label{int(label.item())}.png"
                cv2.imwrite(os.path.join(out_dir, save_name), cam_bgr)

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
            plt.savefig(os.path.join(out_dir, f"confmat_{proj}_{age_group}.png"))
            plt.close()

            df_result.to_csv(os.path.join(out_dir, "eigencam_summary.csv"), index=False)
            all_results.extend(results)

# 전체 결과 저장
if len(all_results) > 0:
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(os.path.join(OUTPUT_BASE_DIR, "eigencam_results_all.csv"), index=False)

print(f"\n✅ Done. Saved under: {OUTPUT_BASE_DIR}")
