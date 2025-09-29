# Final_inference_APLAT_ensemble_from_uploaded_paths_0818.py
# - AP/LAT × 연령그룹 분류(ConvNeXtV2) + YOLO 탐지 점수 앙상블
# - 저장: raw + composite(DET/GT/점수) + composite_cam(EigenCAM+DET/GT)
# - 혼동행렬/메트릭/오분류(FP/FN)/요약 + 실행 매니페스트
# - CAM: 분류 결과 Fracture(=1)일 때만 생성(기본), FP/FN은 항상 CAM 생성 & 분류 저장
import os, sys, json, importlib
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from timm import create_model
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torchvision.transforms as transforms

# =========================
# 0) 경로/설정  (여기만 교체)
# =========================
BASE_DIR = os.path.abspath(os.path.dirname(__file__) or ".")

# AP / LAT 별 테스트 CSV (새 경로)
CSV_PATHS = {
    "AP":  "/mnt/data/KimJG/SwinT/mediaiOA_swinT_cls-main_original/20250623_Single/WristFX_0730/classification/test_set_0730_AP.csv",
    "Lat": "/mnt/data/KimJG/SwinT/mediaiOA_swinT_cls-main_original/20250623_Single/WristFX_0730/classification/test_set_0730_Lat.csv",
}

OUT_DIR  = os.path.join(BASE_DIR, "0820_final_inference_ensemble_convnextv2_yolo_v2")
os.makedirs(OUT_DIR, exist_ok=True)

# AP/LAT 입력 CSV 분리 사용
TEST_CSVS = {"AP": CSV_PATHS["AP"], "Lat": CSV_PATHS["Lat"]}

# GT 라벨(그대로 사용 중이면 수정 불필요)
GT_LABEL_DIR = "/mnt/data/KimJG/ELBOW_test/Kaggle_dataset/folder_structure/yolov5/labels"

# YOLOv9 (그대로 유지)
sys.path.append("/mnt/data/KimJG/Yolo_test/YOLOv9-Fracture-Detection")
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

YOLO_WEIGHT = "/mnt/data/KimJG/Yolo_test/YOLOv9-Fracture-Detection/runs/train/yolov9-c3/weights/best.pt"
YOLO_CONF_THRES = 0.25
YOLO_IOU_THRES  = 0.30
YOLO_TARGET_CLASSES = [3]

# =========================
# 1) 업로드 코드에서 모델 경로/기본 설정 흡수(있으면)
# =========================
def try_import(module_name):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None

cfg_gradcam = try_import("Generate_Gradcam_ConvNeXtV2_ALL_0813")
cfg_final   = try_import("Final_code_FX_0813")

MODEL_BASE_DIR = None
MODEL_BASENAME = None
if cfg_gradcam and hasattr(cfg_gradcam, "MODEL_BASE_DIR"): MODEL_BASE_DIR = cfg_gradcam.MODEL_BASE_DIR
if cfg_gradcam and hasattr(cfg_gradcam, "MODEL_BASENAME"): MODEL_BASENAME = cfg_gradcam.MODEL_BASENAME
if cfg_final   and hasattr(cfg_final,   "MODEL_BASE_DIR"): MODEL_BASE_DIR = cfg_final.MODEL_BASE_DIR or MODEL_BASE_DIR
if cfg_final   and hasattr(cfg_final,   "MODEL_BASENAME"): MODEL_BASENAME = cfg_final.MODEL_BASENAME or MODEL_BASENAME
if MODEL_BASE_DIR is None: MODEL_BASE_DIR = os.path.join(BASE_DIR, "0805_Model")
if MODEL_BASENAME is None: MODEL_BASENAME = "best_ddp_convnextv2_{proj}_{age}.pt"

# =========================
# 2) 하이퍼/환경
# =========================
IMG_SIZE   = 384
BATCH_SIZE = 12
ALPHA        = {"AP": 0.6, "Lat": 0.6}   # det 가중치
FINAL_THRESH = {"AP": 0.3, "Lat": 0.3}   # 앙상블 임계

# CAM은 '분류 결과가 Fracture일 때만' 생성 (기본: cls_pred 기준)
CAM_FOR_FRACTURE_ONLY = True        # True면 cls_pred==1일 때만 CAM 저장
CAM_TRIGGER_FIELD     = "cls_pred"  # "cls_pred" 또는 "final_pred"

# GT 클래스 필터 (None이면 전체, 정수면 해당 class만 표시)
GT_CLASS_FILTER = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from unified_dataset_0704 import UnifiedDataset
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
])

def age_group_fn(age: float) -> int:
    age = float(age)
    if   age < 5:  return 0
    elif age < 10: return 1
    elif age < 15: return 2
    else:          return 3

# 색상
COLOR_DET = (0, 255, 255)  # 노랑(Det)
COLOR_GT  = (0,   0, 255)  # 빨강(GT)

# =========================
# 3) 유틸
# =========================
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True); return path

def resolve_model_path(model_dir: str, proj: str, age_group: int, basename_tpl: str) -> str:
    cands = [
        os.path.join(model_dir, basename_tpl.format(proj=proj, age=age_group)),
        os.path.join(model_dir, basename_tpl.format(proj=proj.lower(), age=age_group)),
        os.path.join(model_dir, basename_tpl.format(proj=proj.upper(), age=age_group)),
        os.path.join(model_dir, basename_tpl.format(proj=("LAT" if proj.lower()=="lat" else proj), age=age_group)),
        os.path.join(model_dir, basename_tpl.format(proj=("AP"  if proj.lower()=="ap"  else proj), age=age_group)),
    ]
    for p in cands:
        if os.path.exists(p): return p
    raise FileNotFoundError(f"[{proj}_{age_group}] 모델 가중치를 찾을 수 없습니다.\n- " + "\n- ".join(cands))

def load_state_dict_flex(model: torch.nn.Module, state_obj):
    try:
        state = torch.load(state_obj, map_location=DEVICE, weights_only=True) if isinstance(state_obj, str) else state_obj
    except Exception:
        state = torch.load(state_obj, map_location=DEVICE) if isinstance(state_obj, str) else state_obj
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        sd = state["state_dict"]
    else:
        sd = state
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model

def save_confmat_and_metrics(df: pd.DataFrame, save_prefix: str):
    y_true = df["true_label"].to_numpy().astype(int)
    y_pred = df["final_pred"].to_numpy().astype(int)
    cm  = confusion_matrix(y_true, y_pred, labels=[0,1])
    acc = accuracy_score(y_true, y_pred)
    try: f1 = f1_score(y_true, y_pred) if (y_pred.sum()>0 or y_true.sum()>0) else 0.0
    except Exception: f1 = 0.0
    pd.DataFrame(cm, index=["True_0(Normal)","True_1(Fracture)"],
                 columns=["Pred_0(Normal)","Pred_1(Fracture)"]).to_csv(f"{save_prefix}_confusion_matrix.csv")
    pd.DataFrame([{"acc": acc, "f1": f1}]).to_csv(f"{save_prefix}_metrics.csv", index=False)

def save_misclassified(df: pd.DataFrame, save_path: str):
    fp = df[(df["true_label"]==0) & (df["final_pred"]==1)]
    fn = df[(df["true_label"]==1) & (df["final_pred"]==0)]
    pd.concat([fp.assign(case="FP"), fn.assign(case="FN")], ignore_index=True).to_csv(save_path, index=False)

def write_run_manifest(save_dir: str, manifest: dict):
    manifest["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(save_dir, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

# =========================
# 4) GT/그리기 + CAM
# =========================
def _read_gt_txt_boxes(stem: str, img_w: int, img_h: int):
    boxes = []
    if not GT_LABEL_DIR or not os.path.isdir(GT_LABEL_DIR): return boxes
    txt_path = os.path.join(GT_LABEL_DIR, f"{stem}.txt")
    if not os.path.exists(txt_path): return boxes
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: continue
            cls = int(float(parts[0]))
            if GT_CLASS_FILTER is not None and cls != int(GT_CLASS_FILTER):
                continue
            cx  = float(parts[1]) * img_w
            cy  = float(parts[2]) * img_h
            w   = float(parts[3]) * img_w
            h   = float(parts[4]) * img_h
            x1  = int(max(0, cx - w/2)); y1 = int(max(0, cy - h/2))
            x2  = int(min(img_w-1, cx + w/2)); y2 = int(min(img_h-1, cy + h/2))
            boxes.append((x1,y1,x2,y2,1.0,cls))
    return boxes

def _draw_boxes(image_bgr, boxes, color, label_prefix=""):
    for (x1,y1,x2,y2,conf,cls) in boxes or []:
        if 0.0 <= x1 <= 1.001 and 0.0 <= x2 <= 1.001 and 0.0 <= y1 <= 1.001 and 0.0 <= y2 <= 1.001:
            H, W = image_bgr.shape[:2]
            _x1, _y1, _x2, _y2 = int(x1*W), int(y1*H), int(x2*W), int(y2*H)
        else:
            _x1, _y1, _x2, _y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image_bgr, (_x1, _y1), (_x2, _y2), color, 2)
        txt = f"{label_prefix}{cls}:{float(conf):.2f}" if label_prefix else f"{cls}:{float(conf):.2f}"
        cv2.putText(image_bgr, txt, (_x1, max(12, _y1-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return image_bgr

def _draw_overlay_and_save(img_path: str,
                           det_prob: float, det_pred: int,
                           cls_prob: float, cls_pred: int,
                           ens_score: float, final_pred: int,
                           det_boxes=None, gt_boxes=None,
                           raw_save_path: str=None, overlay_save_path: str=None):
    img = cv2.imread(img_path)
    if img is None: return
    if raw_save_path: cv2.imwrite(raw_save_path, img)
    canvas = img.copy()
    canvas = _draw_boxes(canvas, det_boxes, COLOR_DET, "DET ")
    canvas = _draw_boxes(canvas, gt_boxes, COLOR_GT,  "GT ")
    panel = [f"DET prob:{det_prob:.3f} pred:{det_pred}",
             f"CLS prob:{cls_prob:.3f} pred:{cls_pred}",
             f"ENS score:{ens_score:.3f} pred:{final_pred}"]
    y = 18
    for line in panel:
        cv2.putText(canvas, line, (8,y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(canvas, line, (8,y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)
        y += 20
    if overlay_save_path: cv2.imwrite(overlay_save_path, canvas)

# ---- EigenCAM (레퍼런스 스타일) ----
def _try_build_cam():
    try:
        from pytorch_grad_cam import EigenCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        return EigenCAM, show_cam_on_image
    except Exception:
        return None, None

def _find_target_layers_for_convnextv2(model):
    """마지막 stage의 마지막 block의 depthwise conv를 타깃 (없으면 마지막 block)."""
    try:
        return [model.stages[-1].blocks[-1].conv_dw]
    except Exception:
        try:
            return [model.stages[-1].blocks[-1]]
        except Exception:
            return []

def _draw_overlay_cam_and_save(model, img_path, det_boxes, gt_boxes, save_path):
    EigenCAM, show_cam_on_image = _try_build_cam()
    if EigenCAM is None:
        print("[CAM][WARN] pytorch-grad-cam import 실패. CAM 생략.")
        return

    orig_bgr = cv2.imread(img_path)
    if orig_bgr is None: return
    H, W = orig_bgr.shape[:2]

    # RGB float [0,1] for show_cam_on_image
    rgb_float = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # 분류와 동일 전처리로 텐서 생성
    from torchvision.transforms import ToPILImage
    pil_img = ToPILImage()(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB))
    img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    target_layers = _find_target_layers_for_convnextv2(model)
    if not target_layers:
        print("[CAM][WARN] target layer 없음. CAM 생략.")
        return

    model.eval()
    with EigenCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=img_tensor)[0]  # [Hc,Wc] in [0,1]

    cam_resized = cv2.resize(grayscale_cam, (W, H), interpolation=cv2.INTER_LINEAR)
    cam_rgb_uint8 = show_cam_on_image(rgb_float, cam_resized, use_rgb=True)  # RGB uint8
    cam_bgr = cv2.cvtColor(cam_rgb_uint8, cv2.COLOR_RGB2BGR)

    cam_bgr = _draw_boxes(cam_bgr, det_boxes, COLOR_DET, "DET ")
    cam_bgr = _draw_boxes(cam_bgr, gt_boxes,  COLOR_GT,  "GT ")
    cv2.imwrite(save_path, cam_bgr)

# =========================
# 5) YOLO 런타임
# =========================
def yolo_setup(yolo_weight: str, device: torch.device):
    model = DetectMultiBackend(yolo_weight, device=device)
    model.eval()
    return model

def _extract_pred_tensor(raw):
    """다양한 리턴 형태 대비: 텐서를 찾아 반환."""
    if torch.is_tensor(raw):
        return raw
    if isinstance(raw, (list, tuple)):
        cand = [x for x in raw if torch.is_tensor(x) and x.ndim >= 2 and x.shape[-1] >= 4]
        if len(cand) > 0:
            cand.sort(key=lambda t: t.shape[-1], reverse=True)
            return cand[0]
        for x in raw:
            t = _extract_pred_tensor(x)
            if t is not None:
                return t
    return None

@torch.no_grad()
def yolo_infer_image(yolo_model, bgr_img: np.ndarray,
                     conf_thres=0.25, iou_thres=0.30, classes=None):
    """
    bgr_img: HxWx3 (BGR)
    return: (best_score: float, boxes: List[(x1,y1,x2,y2,conf,cls)])
    """
    if bgr_img is None:
        return 0.0, []
    img_lb = letterbox(bgr_img, new_shape=640, stride=32, auto=True)[0]
    img_rgb = img_lb[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    img_rgb = np.ascontiguousarray(img_rgb)
    im = torch.from_numpy(img_rgb).to(DEVICE).float() / 255.0
    im = im.unsqueeze(0)

    raw = yolo_model(im)  # 다양한 형태 가능
    pred_logits = _extract_pred_tensor(raw)
    if pred_logits is None:
        try:
            det = raw[0] if isinstance(raw, (list, tuple)) else raw
            if det is None or len(det) == 0:
                return 0.0, []
            boxes, best_score = [], 0.0
            for *xyxy, conf, cls_id in det:
                x1,y1,x2,y2 = [int(v) for v in xyxy]
                c = float(conf)
                best_score = max(best_score, c)
                boxes.append((x1,y1,x2,y2,c,int(cls_id)))
            return best_score, boxes
        except Exception:
            return 0.0, []

    det = non_max_suppression(pred_logits, conf_thres, iou_thres, classes=classes, agnostic=False)[0]
    boxes, best_score = [], 0.0
    if det is not None and len(det) > 0:
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], bgr_img.shape).round()
        for *xyxy, conf, cls_id in det:
            x1, y1, x2, y2 = map(int, xyxy)
            c = float(conf)
            best_score = max(best_score, c)
            boxes.append((x1, y1, x2, y2, c, int(cls_id.item()) if hasattr(cls_id, "item") else int(cls_id)))
    return best_score, boxes

# =========================
# 6) 분류 추론(ConvNeXt)
# =========================
@torch.no_grad()
def infer_one_split(proj: str, age_group: int, df_split: pd.DataFrame):
    model_path = resolve_model_path(MODEL_BASE_DIR, proj, age_group, MODEL_BASENAME)
    model = create_model("convnextv2_large.fcmae_ft_in22k_in1k_384", pretrained=False, num_classes=1).to(DEVICE)
    model = load_state_dict_flex(model, model_path)
    model.eval()

    dataset = UnifiedDataset(df_split, transform=transform, task="fracture_only")
    loader  = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0 if os.environ.get("DL_NUM_WORKERS_ZERO", "0") == "1" else 4,
        pin_memory=(DEVICE.type == "cuda"), persistent_workers=False
    )

    sigmoid = nn.Sigmoid()
    rows, idx = [], 0
    for images, targets in loader:
        images = images.to(DEVICE)
        logits = model(images).squeeze(-1)
        probs  = sigmoid(logits).cpu().numpy()
        preds  = (probs > 0.5).astype(int)
        for k in range(len(preds)):
            meta = df_split.iloc[idx + k]
            rows.append({
                "projection": proj,
                "age_group":  age_group,
                "filename":   os.path.basename(str(meta["image_path"])),
                "cls_prob":   float(probs[k]),
                "cls_pred":   int(preds[k]),
                "true_label": int(meta["fracture_visible"]),
            })
        idx += len(preds)
    return pd.DataFrame(rows)

# =========================
# 7) 메인
# =========================
def main():
    print("[DBG] BASE_DIR =", BASE_DIR)
    print("[DBG] OUT_DIR  =", os.path.abspath(OUT_DIR))
    for k in ["AP","Lat"]:
        print(f"[DBG] {k} CSV exists? ", os.path.exists(TEST_CSVS[k]))

    all_proj_rows = []
    manifest = {
        "MODEL_BASE_DIR": MODEL_BASE_DIR,
        "MODEL_BASENAME": MODEL_BASENAME,
        "TEST_CSVS": TEST_CSVS,
        "IMG_SIZE": IMG_SIZE, "BATCH_SIZE": BATCH_SIZE,
        "ALPHA": ALPHA, "FINAL_THRESH": FINAL_THRESH,
        "YOLO_WEIGHT": YOLO_WEIGHT,
        "YOLO_CONF_THRES": YOLO_CONF_THRES, "YOLO_IOU_THRES": YOLO_IOU_THRES,
        "YOLO_TARGET_CLASSES": YOLO_TARGET_CLASSES,
        "CAM_FOR_FRACTURE_ONLY": CAM_FOR_FRACTURE_ONLY,
        "CAM_TRIGGER_FIELD": CAM_TRIGGER_FIELD,
        "GT_CLASS_FILTER": GT_CLASS_FILTER,
    }

    for proj, csv_path in TEST_CSVS.items():
        if not os.path.exists(csv_path):
            print(f"[경고] {proj} 테스트 CSV가 없습니다: {csv_path}")
            continue

        proj_dir = ensure_dir(os.path.join(OUT_DIR, proj))
        df_full = pd.read_csv(csv_path)

        print(f"[DBG] {proj} CSV columns(top12):", list(df_full.columns)[:12])
        if "projection" in df_full.columns:
            pv = (df_full["projection"].astype(str).str.strip().str.lower().head(10).tolist())
            print(f"[DBG] {proj} projection values(head):", pv)

        # 필수 컬럼
        if "image_path" not in df_full.columns:
            raise ValueError(f"[{proj}] 'image_path' 컬럼 필요: {csv_path}")
        if "age" not in df_full.columns:
            raise ValueError(f"[{proj}] 'age' 컬럼 필요: {csv_path}")
        if "fracture_visible" not in df_full.columns:
            if "label" in df_full.columns: df_full["fracture_visible"] = df_full["label"]
            else: raise ValueError(f"[{proj}] 'fracture_visible' 컬럼 필요(또는 'label'로 매핑).")
        df_full["fracture_visible"] = df_full["fracture_visible"].fillna(0).astype(int)

        # projection 정규화 필터
        if "projection" in df_full.columns:
            proj_norm = (df_full["projection"].astype(str).str.strip().str.lower()
                         .replace({"ap view":"ap","apview":"ap","a-p":"ap",
                                   "lat view":"lat","lateral":"lat","lat.":"lat"}))
            before = len(df_full)
            df_full = df_full[proj_norm == proj.lower()].reset_index(drop=True)
            print(f"[INFO] {proj}: projection filter -> {len(df_full)}/{before}")

        # age_group
        df_full["age_group"] = df_full["age"].astype(float).apply(age_group_fn)

        # UnifiedDataset label
        if "label" not in df_full.columns:
            df_full["label"] = df_full["fracture_visible"].astype(int)

        # 이미지 존재율 + 누락 기록
        exists_mask = df_full["image_path"].astype(str).apply(os.path.exists)
        if (~exists_mask).any():
            df_full.loc[~exists_mask, ["image_path"]].to_csv(os.path.join(proj_dir, f"missing_images_{proj}.csv"), index=False)
            print(f"[WARN] Missing images for {proj}: {(~exists_mask).sum()} rows -> missing_images_{proj}.csv")
        print(f"[INFO] Existing images for {proj}: {exists_mask.sum()} / {len(df_full)}")

        # stem 매핑
        stem2path = {}
        _dupes = set()
        for pth in df_full["image_path"].astype(str).tolist():
            st = os.path.splitext(os.path.basename(pth))[0]
            if st in stem2path and stem2path[st] != pth: _dupes.add(st)
            stem2path[st] = pth
        if _dupes:
            print(f"[WARN] Duplicate filename stems detected ({len(_dupes)}): e.g., {list(_dupes)[:5]}")

        # 분류 추론
        proj_rows = []
        for age_group in [0,1,2,3]:
            df_g = df_full[(df_full["age_group"] == age_group) & (df_full["image_path"].astype(str).apply(os.path.exists))].reset_index(drop=True)
            if len(df_g) == 0:
                print(f"[INFO] {proj} age{age_group}: no samples.")
                continue
            df_cls = infer_one_split(proj, age_group, df_g)
            proj_rows.append(df_cls)
        if len(proj_rows) == 0:
            print(f"[{proj}] 추론 대상 없음")
            continue
        df_proj = pd.concat(proj_rows, ignore_index=True)

        # ===== YOLO 직접 추론 =====
        print(f"[INFO][{proj}] Running YOLO inference ...")
        yolo_model = yolo_setup(YOLO_WEIGHT, DEVICE)
        det_probs, det_preds = [], []
        yolo_boxes_map = {}  # stem -> boxes

        DET_BIN_THRESH = 0.5  # det_prob -> det_pred
        for _, r in df_proj.iterrows():
            stem = os.path.splitext(os.path.basename(r["filename"]))[0]
            img_path = stem2path.get(stem, None)
            if img_path is None or not os.path.exists(img_path):
                det_probs.append(0.0); det_preds.append(0); yolo_boxes_map[stem] = []
                continue
            img0 = cv2.imread(img_path)
            best_score, y_boxes = yolo_infer_image(
                yolo_model, img0,
                conf_thres=YOLO_CONF_THRES, iou_thres=YOLO_IOU_THRES,
                classes=YOLO_TARGET_CLASSES
            )
            det_probs.append(float(best_score))
            det_preds.append(int(best_score > DET_BIN_THRESH))
            yolo_boxes_map[stem] = y_boxes

        df_proj["det_prob"] = det_probs
        df_proj["det_pred"] = det_preds

        # ===== 앙상블 =====
        a  = ALPHA.get(proj, 0.6)
        th = FINAL_THRESH.get(proj, 0.5)
        df_proj["ens_score"]  = a * df_proj["det_prob"].astype(float) + (1.0 - a) * df_proj["cls_prob"].astype(float)
        df_proj["final_pred"] = (df_proj["ens_score"] > th).astype(int)
        df_proj["alpha"] = a; df_proj["final_thresh"] = th

        # ===== 저장 =====
        pred_csv = os.path.join(proj_dir, f"{proj}_predictions_ensemble.csv")
        df_proj.to_csv(pred_csv, index=False)
        save_confmat_and_metrics(df_proj, os.path.join(proj_dir, f"{proj}"))
        save_misclassified(df_proj, os.path.join(proj_dir, f"misclassified_{proj}.csv"))

        # 시각화 폴더
        overlay_dir   = ensure_dir(os.path.join(proj_dir, "overlays"))
        raw_dir       = ensure_dir(os.path.join(overlay_dir, "raw"))
        comp_dir      = ensure_dir(os.path.join(overlay_dir, "composite"))
        comp_cam_dir  = ensure_dir(os.path.join(overlay_dir, "composite_cam"))
        mis_fp_dir    = ensure_dir(os.path.join(comp_dir, "FP"))
        mis_fn_dir    = ensure_dir(os.path.join(comp_dir, "FN"))
        # ★ CAM 결과도 FP/FN로 따로 저장
        mis_fp_cam_dir = ensure_dir(os.path.join(comp_cam_dir, "FP"))
        mis_fn_cam_dir = ensure_dir(os.path.join(comp_cam_dir, "FN"))

        # CAM용 모델 1회 로드(같은 proj)
        def _load_model_for_cam(proj_):
            for ag in [3,2,1,0]:
                try:
                    mp = resolve_model_path(MODEL_BASE_DIR, proj_, ag, MODEL_BASENAME)
                    m  = create_model("convnextv2_large.fcmae_ft_in22k_in1k_384", pretrained=False, num_classes=1).to(DEVICE)
                    load_state_dict_flex(m, mp); m.eval()
                    print(f"[CAM] Using weights: {proj_}_age{ag} -> {os.path.basename(mp)}")
                    return m
                except Exception:
                    continue
            print(f"[CAM][WARN] {proj_}: 가중치 로드 실패 (CAM 생략)")
            return None
        model_for_cam = _load_model_for_cam(proj)

        # 이미지 저장 루프
        for _, r in df_proj.iterrows():
            stem = os.path.splitext(os.path.basename(r["filename"]))[0]
            img_path = stem2path.get(stem, None)
            if img_path is None or not os.path.exists(img_path):
                continue

            raw_name      = f"{stem}.jpg"
            comp_name     = f"{stem}_ens{int(r['final_pred'])}_cls{int(r['cls_pred'])}_det{int(r['det_pred'])}.jpg"
            comp_cam_name = f"{stem}_CAM.jpg"
            raw_save_path  = os.path.join(raw_dir,  raw_name)
            comp_save_path = os.path.join(comp_dir,  comp_name)
            comp_cam_path  = os.path.join(comp_cam_dir, comp_cam_name)

            det_boxes = yolo_boxes_map.get(stem, [])
            img0 = cv2.imread(img_path)
            gt_boxes = _read_gt_txt_boxes(stem, img0.shape[1], img0.shape[0]) if img0 is not None else []

            # composite 저장
            _draw_overlay_and_save(
                img_path=img_path,
                det_prob=float(r["det_prob"]), det_pred=int(r["det_pred"]),
                cls_prob=float(r["cls_prob"]),   cls_pred=int(r["cls_pred"]),
                ens_score=float(r["ens_score"]), final_pred=int(r["final_pred"]),
                det_boxes=det_boxes, gt_boxes=gt_boxes,
                raw_save_path=raw_save_path, overlay_save_path=comp_save_path
            )

            # CAM 생성 조건 (FP/FN은 강제 생성)
            is_fp = (int(r["true_label"]) == 0 and int(r["final_pred"]) == 1)
            is_fn = (int(r["true_label"]) == 1 and int(r["final_pred"]) == 0)

            make_cam = True
            if CAM_FOR_FRACTURE_ONLY:
                trigger_val = int(r.get(CAM_TRIGGER_FIELD, 0))  # "cls_pred" 또는 "final_pred"
                make_cam = (trigger_val == 1)
            if is_fp or is_fn:
                make_cam = True

            if make_cam and model_for_cam is not None:
                try:
                    _draw_overlay_cam_and_save(
                        model_for_cam, img_path,
                        det_boxes=det_boxes, gt_boxes=gt_boxes,
                        save_path=comp_cam_path
                    )
                    # CAM 결과도 FP/FN 폴더로 분류 저장
                    if is_fp:
                        cv2.imwrite(os.path.join(mis_fp_cam_dir, comp_cam_name), cv2.imread(comp_cam_path))
                    elif is_fn:
                        cv2.imwrite(os.path.join(mis_fn_cam_dir, comp_cam_name), cv2.imread(comp_cam_path))
                except Exception:
                    pass

            # composite 오분류 복사
            if is_fp:
                cv2.imwrite(os.path.join(mis_fp_dir, comp_name), cv2.imread(comp_save_path))
            elif is_fn:
                cv2.imwrite(os.path.join(mis_fn_dir, comp_name), cv2.imread(comp_save_path))

        all_proj_rows.append(df_proj)

    # ===== 전체(AP+LAT) 요약 =====
    if len(all_proj_rows) > 0:
        df_all = pd.concat(all_proj_rows, ignore_index=True)
        df_all.to_csv(os.path.join(OUT_DIR, "ALL_predictions_ensemble.csv"), index=False)
        save_confmat_and_metrics(df_all, os.path.join(OUT_DIR, "ALL"))

        def safe_f1(g):
            try: return f1_score(g["true_label"], g["final_pred"]) if (g["final_pred"].sum()>0 or g["true_label"].sum()>0) else 0.0
            except Exception: return 0.0

        summary = (df_all.groupby(["projection", "age_group"])
                   .apply(lambda g: pd.Series({
                       "n": len(g),
                       "acc": accuracy_score(g["true_label"], g["final_pred"]),
                       "f1":  safe_f1(g),
                       "cls_acc": accuracy_score(g["true_label"], (g["cls_prob"]>0.5).astype(int)),
                       "det_acc": accuracy_score(g["true_label"], g["det_pred"]),
                       "alpha": g["alpha"].iloc[0] if "alpha" in g.columns else np.nan,
                       "final_thresh": g["final_thresh"].iloc[0] if "final_thresh" in g.columns else np.nan,
                   }))
                   .reset_index())
        summary.to_csv(os.path.join(OUT_DIR, "group_summary_ensemble.csv"), index=False)

    # 실행 매니페스트
    write_run_manifest(OUT_DIR, manifest)

if __name__ == "__main__":
    main()
