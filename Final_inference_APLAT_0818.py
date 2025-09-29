# Final_inference_APLAT_ensemble_from_uploaded_paths_0818.py
# - AP/LAT × 연령그룹별 분류(ConvNeXtV2, 0811 산출물) + YOLO 탐지 점수 앙상블
# - 업로드한 코드들에서 경로/설정값을 최대한 import로 흡수 (없으면 합리적 기본값으로 동작)
# - 결과 저장:
#   * 프로젝션별 폴더(AP/Lat) 아래 CSV/지표/혼동행렬/오분류 목록
#   * 이미지 오버레이(원본 + YOLO bbox + DET/CLS/ENS 텍스트) + FP/FN 하위 폴더
#   * 전체 통합(ALL_*.csv / group_summary_ensemble.csv)
#   * 실행 매니페스트(run_manifest.json)

import os
import glob
import json
import importlib
from datetime import datetime

import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from timm import create_model
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torchvision.transforms as transforms

# =========================
# 1) 업로드 코드에서 설정 흡수
# =========================
BASE_DIR = os.path.dirname(__file__)

def try_import(module_name):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None

cfg_gradcam = try_import("Generate_Gradcam_ConvNeXtV2_ALL_0813")
cfg_final   = try_import("Final_code_FX_0813")

# ----- 분류 모델 경로/이름 -----
MODEL_BASE_DIR = None
MODEL_BASENAME = None

if cfg_gradcam and hasattr(cfg_gradcam, "MODEL_BASE_DIR"):
    MODEL_BASE_DIR = getattr(cfg_gradcam, "MODEL_BASE_DIR")
if cfg_gradcam and hasattr(cfg_gradcam, "MODEL_BASENAME"):
    MODEL_BASENAME = getattr(cfg_gradcam, "MODEL_BASENAME")

if cfg_final and hasattr(cfg_final, "MODEL_BASE_DIR"):
    MODEL_BASE_DIR = getattr(cfg_final, "MODEL_BASE_DIR")
if cfg_final and hasattr(cfg_final, "MODEL_BASENAME"):
    MODEL_BASENAME = getattr(cfg_final, "MODEL_BASENAME")

if MODEL_BASE_DIR is None:
    MODEL_BASE_DIR = os.path.join(BASE_DIR, "0805_Model")
if MODEL_BASENAME is None:
    MODEL_BASENAME = "best_ddp_convnextv2_{proj}_{age}.pt"  # ex) AP/LAT/Lat 대소문자 혼용 대응은 아래에서 처리

# ----- 테스트 CSV 경로 -----
TEST_CSVS = {}
if cfg_final and hasattr(cfg_final, "TEST_CSVS"):
    TEST_CSVS = getattr(cfg_final, "TEST_CSVS")
elif cfg_gradcam and hasattr(cfg_gradcam, "TEST_CSVS"):
    TEST_CSVS = getattr(cfg_gradcam, "TEST_CSVS")

if not TEST_CSVS:
    ap_guess  = os.path.join(BASE_DIR, "test_set_0730_AP.csv")
    lat_guess = os.path.join(BASE_DIR, "test_set_0730_Lat.csv")
    if os.path.exists(ap_guess) and os.path.exists(lat_guess):
        TEST_CSVS = {"AP": ap_guess, "Lat": lat_guess}
    else:
        one_guess = os.path.join(BASE_DIR, "test_set_0704.csv")
        if os.path.exists(one_guess):
            TEST_CSVS = {"AP": one_guess}
        else:
            candidates = sorted(glob.glob(os.path.join(BASE_DIR, "test_set_*.csv")))
            if candidates:
                TEST_CSVS = {"AP": candidates[0]}
                if len(candidates) > 1:
                    TEST_CSVS["Lat"] = candidates[1]

# ----- YOLO 입력 (CSV 우선, 없으면 labels/txt) -----
YOLO_CSVS = {}
if cfg_final and hasattr(cfg_final, "YOLO_CSVS"):
    YOLO_CSVS = getattr(cfg_final, "YOLO_CSVS")
elif cfg_gradcam and hasattr(cfg_gradcam, "YOLO_CSVS"):
    YOLO_CSVS = getattr(cfg_gradcam, "YOLO_CSVS")
if not YOLO_CSVS:
    YOLO_CSVS = {"AP": None, "Lat": None}

YOLO_TXT_LABEL_DIRS = {}
if cfg_final and hasattr(cfg_final, "YOLO_TXT_LABEL_DIRS"):
    YOLO_TXT_LABEL_DIRS = getattr(cfg_final, "YOLO_TXT_LABEL_DIRS")
elif cfg_gradcam and hasattr(cfg_gradcam, "YOLO_TXT_LABEL_DIRS"):
    YOLO_TXT_LABEL_DIRS = getattr(cfg_gradcam, "YOLO_TXT_LABEL_DIRS")
if not YOLO_TXT_LABEL_DIRS:
    YOLO_TXT_LABEL_DIRS = {
        "AP":  os.path.join(BASE_DIR, "runs", "detect_AP", "labels"),
        "Lat": os.path.join(BASE_DIR, "runs", "detect_Lat", "labels"),
    }

# ----- 이미지 크기/배치 -----
IMG_SIZE = 384
if cfg_gradcam and hasattr(cfg_gradcam, "IMG_SIZE"):
    try:
        IMG_SIZE = int(getattr(cfg_gradcam, "IMG_SIZE"))
    except Exception:
        pass

BATCH_SIZE = 12

# ----- 앙상블 하이퍼파라미터 (프로젝션별 지원) -----
ALPHA = {"AP": 0.6, "Lat": 0.6}         # det 비중
FINAL_THRESH = {"AP": 0.5, "Lat": 0.5}   # 최종 임계값

# ----- 출력 폴더 -----
OUT_DIR = os.path.join(BASE_DIR, "final_results_0818_ensemble")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 2) Dataset / Transform
# =========================
from unified_dataset_0704 import UnifiedDataset

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def age_group_fn(age: float) -> int:
    age = float(age)
    if age < 5:   return 0
    elif age < 10:return 1
    elif age < 15:return 2
    else:         return 3

# =========================
# 3) 안전/저장 유틸
# =========================
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def resolve_model_path(model_dir: str, proj: str, age_group: int, basename_tpl: str) -> str:
    candidates = [
        os.path.join(model_dir, basename_tpl.format(proj=proj, age=age_group)),
        os.path.join(model_dir, basename_tpl.format(proj=proj.lower(), age=age_group)),
        os.path.join(model_dir, basename_tpl.format(proj=proj.upper(), age=age_group)),
        os.path.join(model_dir, basename_tpl.format(proj=("LAT" if proj.lower()=="lat" else proj), age=age_group)),
        os.path.join(model_dir, basename_tpl.format(proj=("AP"  if proj.lower()=="ap"  else proj), age=age_group)),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"[{proj}_{age_group}] 모델 파일을 찾을 수 없습니다.\nTried:\n- " + "\n- ".join(candidates))

def load_state_dict_flex(model: torch.nn.Module, state_obj):
    if isinstance(state_obj, dict) and "state_dict" in state_obj and isinstance(state_obj["state_dict"], dict):
        state = state_obj["state_dict"]
    else:
        state = state_obj
    needs_strip = any(k.startswith("module.") for k in state.keys())
    if needs_strip:
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model

def save_confmat_and_metrics(df: pd.DataFrame, save_prefix: str):
    y_true = df["true_label"].to_numpy().astype(int)
    y_pred = df["final_pred"].to_numpy().astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    acc = accuracy_score(y_true, y_pred)
    try:
        f1  = f1_score(y_true, y_pred) if (y_pred.sum()>0 or y_true.sum()>0) else 0.0
    except Exception:
        f1 = 0.0

    cm_df = pd.DataFrame(cm, index=["True_0(Normal)", "True_1(Fracture)"],
                            columns=["Pred_0(Normal)", "Pred_1(Fracture)"])
    cm_df.to_csv(f"{save_prefix}_confusion_matrix.csv")
    pd.DataFrame([{"acc": acc, "f1": f1}]).to_csv(f"{save_prefix}_metrics.csv", index=False)

def save_misclassified(df: pd.DataFrame, save_path: str):
    fp = df[(df["true_label"]==0) & (df["final_pred"]==1)]
    fn = df[(df["true_label"]==1) & (df["final_pred"]==0)]
    out = pd.concat([fp.assign(case="FP"), fn.assign(case="FN")], ignore_index=True)
    out.to_csv(save_path, index=False)

def write_run_manifest(save_dir: str, manifest: dict):
    manifest["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(save_dir, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

# =========================
# 4) YOLO 결합/박스 로딩 유틸
# =========================
def load_yolo_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "filename" not in df.columns or "det_prob" not in df.columns:
        raise ValueError(f"[YOLO CSV] 'filename' 또는 'det_prob' 없음: {csv_path}")
    if "det_pred" not in df.columns:
        df["det_pred"] = (df["det_prob"].astype(float) > 0.5).astype(int)
    df["filename"] = df["filename"].apply(lambda p: os.path.basename(str(p)))
    return df

def _read_yolo_txt_boxes(txt_path: str, img_w: int, img_h: int):
    """labels/*.txt 한 파일에서 박스 리스트 반환: [(x1,y1,x2,y2,conf,cls), ...]"""
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx  = float(parts[1]) * img_w
            cy  = float(parts[2]) * img_h
            w   = float(parts[3]) * img_w
            h   = float(parts[4]) * img_h
            x1  = int(max(0, cx - w/2))
            y1  = int(max(0, cy - h/2))
            x2  = int(min(img_w-1, cx + w/2))
            y2  = int(min(img_h-1, cy + h/2))
            conf = float(parts[5]) if len(parts) >= 6 else 0.0
            boxes.append((x1,y1,x2,y2,conf,cls))
    return boxes

def _read_yolo_csv_boxes(df_csv, fname: str):
    """
    CSV에서 filename 매칭되는 박스 목록: [(x1,y1,x2,y2,conf,cls), ...]
    CSV 컬럼 예: filename,x1,y1,x2,y2,conf,cls
    """
    if df_csv is None or df_csv.empty:
        return []
    stem = os.path.splitext(os.path.basename(fname))[0]
    sel = df_csv[df_csv["filename"].apply(lambda p: os.path.splitext(os.path.basename(str(p)))[0]) == stem]
    needed = {"x1","y1","x2","y2"}
    out = []
    if len(sel)>0 and needed.issubset(set(sel.columns)):
        for _, r in sel.iterrows():
            conf = float(r.get("conf", 0.0))
            cls  = int(r.get("cls", -1))
            out.append((int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"]), conf, cls))
    return out

def _collect_yolo_boxes_for_image(proj: str, img_path: str, df_yolo_csv: pd.DataFrame):
    """해당 이미지의 YOLO 박스를 CSV(우선) 또는 TXT에서 수집."""
    boxes = []
    # CSV 우선
    boxes.extend(_read_yolo_csv_boxes(df_yolo_csv, img_path))
    # TXT 병합
    labels_dir = YOLO_TXT_LABEL_DIRS.get(proj)
    if labels_dir and os.path.isdir(labels_dir):
        txt_path = os.path.join(labels_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
        img = cv2.imread(img_path)
        if img is not None:
            H, W = img.shape[:2]
            boxes.extend(_read_yolo_txt_boxes(txt_path, W, H))
    return boxes

def attach_yolo(proj: str, df_pred: pd.DataFrame):
    """분류 결과 df_pred에 det_prob/det_pred를 머지, 누락 filename 리스트도 반환."""
    df_pred = df_pred.copy()
    df_pred["filename"] = df_pred["filename"].apply(lambda p: os.path.basename(str(p)))

    # CSV 우선
    df_yolo = None
    if YOLO_CSVS.get(proj):
        csv_path = YOLO_CSVS[proj]
        if csv_path and os.path.exists(csv_path):
            df_yolo = load_yolo_csv(csv_path)

    if df_yolo is None:
        # TXT만 있는 경우: det_prob만 최대 conf로 계산하기 위해 파일별 프록시 df 생성
        rows = []
        labels_dir = YOLO_TXT_LABEL_DIRS.get(proj)
        if labels_dir and os.path.isdir(labels_dir):
            for fp in glob.glob(os.path.join(labels_dir, "*.txt")):
                stem = os.path.splitext(os.path.basename(fp))[0]
                best_conf = 0.0
                with open(fp, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 6:
                            best_conf = max(best_conf, float(parts[5]))
                rows.append({"filename": stem, "det_prob": best_conf, "det_pred": int(best_conf > 0.5)})
            df_yolo = pd.DataFrame(rows)
        else:
            df_yolo = pd.DataFrame(columns=["filename", "det_prob", "det_pred"])

    # 키 맞추기 (확장자 제거)
    df_pred["key"] = df_pred["filename"].apply(lambda n: os.path.splitext(n)[0])
    df_yolo["key"] = df_yolo["filename"].apply(lambda n: os.path.splitext(os.path.basename(str(n)))[0])

    merged = pd.merge(df_pred, df_yolo[["key", "det_prob", "det_pred"]], on="key", how="left")
    missing = merged.loc[merged["det_prob"].isna(), "filename"].tolist()

    merged["det_prob"] = merged["det_prob"].fillna(0.0).astype(float)
    merged["det_pred"] = merged["det_pred"].fillna(0).astype(int)

    merged.drop(columns=["key"], inplace=True)
    return merged, missing, (df_yolo if "x1" in df_yolo.columns else None)

# =========================
# 5) 오버레이 유틸
# =========================
def _draw_overlay(img_path: str,
                  det_prob: float,
                  det_pred: int,
                  cls_prob: float,
                  cls_pred: int,
                  ens_score: float,
                  final_pred: int,
                  yolo_boxes=None,
                  save_path: str=None):
    """
    원본 이미지 위에 YOLO 박스와 점수(DET/CLS/ENS)를 그려 저장.
    yolo_boxes: [(x1,y1,x2,y2,conf,cls), ...] 또는 None
    """
    img = cv2.imread(img_path)
    if img is None:
        return

    H, W = img.shape[:2]
    # 박스
    if yolo_boxes:
        for (x1,y1,x2,y2,conf,cls) in yolo_boxes:
            # 정규화 좌표 대비
            if 0 <= x1 <= 1 and 0 <= x2 <= 1 and 0 <= y1 <= 1 and 0 <= y2 <= 1:
                x1i, y1i, x2i, y2i = int(x1*W), int(y1*H), int(x2*W), int(y2*H)
            else:
                x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
            cv2.putText(img, f"{cls}:{conf:.2f}", (x1i, max(12, y1i - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    # 좌상단 점수판
    panel = [
        f"DET prob:{det_prob:.3f} pred:{det_pred}",
        f"CLS prob:{cls_prob:.3f} pred:{cls_pred}",
        f"ENS score:{ens_score:.3f} pred:{final_pred}",
    ]
    y = 18
    for line in panel:
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)
        y += 20

    if save_path:
        cv2.imwrite(save_path, img)

# =========================
# 6) 분류 추론
# =========================
@torch.no_grad()
def infer_one_split(proj: str, age_group: int, df_split: pd.DataFrame) -> pd.DataFrame:
    model_path = resolve_model_path(MODEL_BASE_DIR, proj, age_group, MODEL_BASENAME)

    model = create_model(
        "convnextv2_large.fcmae_ft_in22k_in1k_384",
        pretrained=False, num_classes=1
    ).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model = load_state_dict_flex(model, state)
    model.eval()

    dataset = UnifiedDataset(df_split, transform=transform, task="fracture_only")
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    sigmoid = nn.Sigmoid()
    rows = []
    idx = 0
    for images, targets in loader:
        images = images.to(DEVICE)
        logits = model(images).squeeze(dim=-1)
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
    all_proj_rows = []
    manifest = {
        "MODEL_BASE_DIR": MODEL_BASE_DIR,
        "MODEL_BASENAME": MODEL_BASENAME,
        "TEST_CSVS": TEST_CSVS,
        "YOLO_CSVS": YOLO_CSVS,
        "YOLO_TXT_LABEL_DIRS": YOLO_TXT_LABEL_DIRS,
        "IMG_SIZE": IMG_SIZE,
        "BATCH_SIZE": BATCH_SIZE,
        "ALPHA": ALPHA,
        "FINAL_THRESH": FINAL_THRESH,
    }

    for proj, csv_path in TEST_CSVS.items():
        if not os.path.exists(csv_path):
            print(f"[경고] {proj} 테스트 CSV가 없습니다: {csv_path}")
            continue

        # 프로젝션별 하위 폴더
        proj_dir = ensure_dir(os.path.join(OUT_DIR, proj))
        df_full = pd.read_csv(csv_path)

        # 필수 컬럼
        if "image_path" not in df_full.columns:
            raise ValueError(f"[{proj}] 'image_path' 컬럼이 필요합니다: {csv_path}")
        if "age" not in df_full.columns:
            raise ValueError(f"[{proj}] 'age' 컬럼이 필요합니다: {csv_path}")

        if "fracture_visible" not in df_full.columns:
            if "label" in df_full.columns:
                df_full["fracture_visible"] = df_full["label"]
            else:
                raise ValueError(f"[{proj}] 'fracture_visible' 컬럼이 필요합니다(없으면 'label'로 매핑).")

        df_full["fracture_visible"] = df_full["fracture_visible"].fillna(0).astype(int)

        # projection 컬럼이 있으면 proj로 필터
        if "projection" in df_full.columns:
            df_full = df_full[df_full["projection"].astype(str).str.lower().isin([proj.lower()])].reset_index(drop=True)

        # age → age_group
        df_full["age_group"] = df_full["age"].astype(float).apply(age_group_fn)

        # UnifiedDataset용 label 생성
        df_full = df_full.copy()
        if "label" not in df_full.columns:
            df_full["label"] = df_full["fracture_visible"].astype(int)

        # 분류 추론 (연령그룹별)
        proj_rows = []
        for age_group in [0,1,2,3]:
            df_g = df_full[df_full["age_group"] == age_group].reset_index(drop=True)
            if len(df_g) == 0:
                continue
            df_cls = infer_one_split(proj, age_group, df_g)
            proj_rows.append(df_cls)

        if len(proj_rows) == 0:
            print(f"[{proj}] 추론 대상 없음")
            continue

        df_proj = pd.concat(proj_rows, ignore_index=True)

        # YOLO 결합
        df_proj, yolo_missing, df_yolo_csv = attach_yolo(proj, df_proj)

        # 앙상블
        a  = ALPHA.get(proj, 0.6)
        th = FINAL_THRESH.get(proj, 0.5)
        df_proj["ens_score"]  = a * df_proj["det_prob"].astype(float) + (1.0 - a) * df_proj["cls_prob"].astype(float)
        df_proj["final_pred"] = (df_proj["ens_score"] > th).astype(int)
        df_proj["alpha"] = a
        df_proj["final_thresh"] = th

        # 저장(프로젝션별)
        pred_csv = os.path.join(proj_dir, f"{proj}_predictions_ensemble.csv")
        df_proj.to_csv(pred_csv, index=False)

        save_confmat_and_metrics(df_proj, os.path.join(proj_dir, f"{proj}"))
        save_misclassified(df_proj, os.path.join(proj_dir, f"misclassified_{proj}.csv"))
        if len(yolo_missing) > 0:
            pd.DataFrame({"filename": sorted(set(yolo_missing))}).to_csv(
                os.path.join(proj_dir, f"yolo_missing_{proj}.csv"), index=False
            )

        # ---- 오버레이 이미지 저장 ----
        overlay_dir = ensure_dir(os.path.join(proj_dir, "overlays"))
        mis_fp_dir  = ensure_dir(os.path.join(overlay_dir, "FP"))
        mis_fn_dir  = ensure_dir(os.path.join(overlay_dir, "FN"))

        # filename → 원본 경로 매핑
        stem2path = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in df_full["image_path"].astype(str).tolist()
        }

        for _, r in df_proj.iterrows():
            stem = os.path.splitext(os.path.basename(r["filename"]))[0]
            img_path = stem2path.get(stem, None)
            if img_path is None or not os.path.exists(img_path):
                continue

            save_name = f"{stem}_ens{int(r['final_pred'])}_cls{int(r['cls_pred'])}_det{int(r['det_pred'])}.jpg"
            save_path = os.path.join(overlay_dir, save_name)

            # 박스 수집(CSV 우선 + TXT 병합)
            yolo_boxes = _collect_yolo_boxes_for_image(proj, img_path, df_yolo_csv)

            _draw_overlay(
                img_path=img_path,
                det_prob=float(r["det_prob"]),
                det_pred=int(r["det_pred"]),
                cls_prob=float(r["cls_prob"]),
                cls_pred=int(r["cls_pred"]),
                ens_score=float(r["ens_score"]),
                final_pred=int(r["final_pred"]),
                yolo_boxes=yolo_boxes,
                save_path=save_path
            )

            # 오분류 복사 저장
            if int(r["true_label"]) == 0 and int(r["final_pred"]) == 1:
                cv2.imwrite(os.path.join(mis_fp_dir, save_name), cv2.imread(save_path))
            elif int(r["true_label"]) == 1 and int(r["final_pred"]) == 0:
                cv2.imwrite(os.path.join(mis_fn_dir, save_name), cv2.imread(save_path))

        all_proj_rows.append(df_proj)

    # 전체(AP+LAT) 통합 저장/요약
    if len(all_proj_rows) > 0:
        df_all = pd.concat(all_proj_rows, ignore_index=True)
        df_all.to_csv(os.path.join(OUT_DIR, "ALL_predictions_ensemble.csv"), index=False)
        save_confmat_and_metrics(df_all, os.path.join(OUT_DIR, "ALL"))

        def safe_f1(g):
            try:
                return f1_score(g["true_label"], g["final_pred"]) if (g["final_pred"].sum()>0 or g["true_label"].sum()>0) else 0.0
            except Exception:
                return 0.0

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
