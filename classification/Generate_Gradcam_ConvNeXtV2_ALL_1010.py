# Generate_Gradcam_ConvNeXtV2_ALL_0919.py  (OpenCV-only, image_path 지원)
import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

# =========================
# Args
# =========================
ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True, help="cols: image_path(or img_path), projection(AP|Lat), age, fracture_visible or label")
ap.add_argument("--model_root", required=True, help=".../classification (그 안에 Model_Fold/ 존재)")
ap.add_argument("--outdir", required=True, help="결과 저장 폴더")
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--no_fp16", action="store_true")
ap.add_argument("--save_cam", action="store_true", help="Grad-CAM 이미지 저장")
ap.add_argument("--cam_layer", default="", help="특정 레이어 경로. 공백이면 자동 선택")
ap.add_argument("--prob_th", type=float, default=0.5)
args = ap.parse_args()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
AMP = (DEVICE.type == "cuda") and (not args.no_fp16)

MODEL_NAME = "convnextv2_large.fcmae_ft_in22k_in1k_384"
IMG_SIZE = 384

# OpenCV → Tensor 전처리
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_cv2(path: str) -> (torch.Tensor, np.ndarray):
    """returns: (tensor[1,3,H,W], orig_bgr)"""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_res = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x = rgb_res.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return x, bgr

def age_group_fn(a: float):
    a = float(a)
    if a < 5: return 0
    if a < 10: return 1
    if a < 15: return 2
    return 3

def weight_path_of(model_root: Path, proj: str, age_group: int) -> Path:
    return model_root / "Model_Fold_1110" / f"{proj}_{age_group}_ensemble_avg.pt"

def build_model():
    m = create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=1,
        global_pool='avg',
        drop_rate=0.0,
        drop_path_rate=0.0
    )
    return m

def load_state(model, weight_path: Path, device: torch.device):
    sd = torch.load(str(weight_path), map_location="cpu")
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()

# ---------------- CAM 유틸 ----------------
def find_last_conv2d(module: nn.Module) -> nn.Module:
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last

def get_target_layer_for_convnext(model: nn.Module) -> nn.Module:
    layer = None
    try:
        layer = find_last_conv2d(model.stages[-1])
    except Exception:
        pass
    if layer is None:
        layer = find_last_conv2d(model)
    if layer is None:
        raise RuntimeError("Conv2d target layer를 찾지 못했습니다.")
    return layer

def make_cam(model, input_tensor, target_layer, prob):
    # 1) pytorch-grad-cam 경로 (설치된 경우)
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
        cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(DEVICE.type=="cuda"))
        grayscale_cam = cam(input_tensor=input_tensor, targets=[BinaryClassifierOutputTarget(0)])[0]
        grayscale_cam = np.clip(grayscale_cam, 0, 1)
        return grayscale_cam
    except Exception:
        pass
    # 2) Fallback 간이 CAM
    activations, gradients = {}, {}
    def fwd_hook(module, inp, out):
        activations["feat"] = out.detach()
    def bwd_hook(module, grad_in, grad_out):
        gradients["grad"] = grad_out[0].detach()
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    model.zero_grad(set_to_none=True)
    out = model(input_tensor)              # [B,1]
    loss = out.squeeze(1).sum()
    loss.backward()
    h1.remove(); h2.remove()
    feat = activations.get("feat")
    grad = gradients.get("grad")
    if feat is None or grad is None or feat.dim()!=4 or grad.dim()!=4:
        return None
    weights = grad.mean(dim=(2,3), keepdim=True)
    cam = (weights * feat).sum(dim=1)
    cam = F.relu(cam)[0].cpu().numpy()
    cam = (cam - cam.min() + 1e-8) / (cam.max() - cam.min() + 1e-8)
    return cam

def overlay_cam_on_image(bgr_img, cam_map, alpha=0.35):
    h, w = bgr_img.shape[:2]
    cam_resized = cv2.resize(cam_map, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = np.float32(heatmap) * alpha + np.float32(bgr_img)
    overlay = overlay / np.max(overlay)
    return np.uint8(255 * overlay)

# =========================
# Main
# =========================
def main():
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    cam_dir = outdir / "cams"
    if args.save_cam:
        cam_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    # ---- image_path/img_path 정규화 ----
    if "img_path" not in df.columns:
        if "image_path" in df.columns:
            df = df.rename(columns={"image_path": "img_path"})
        else:
            raise ValueError("CSV에 img_path 또는 image_path 컬럼이 필요합니다.")

    # ---- 나머지 컬럼 정규화 ----
    if "projection" not in df.columns:
        df["projection"] = df["img_path"].apply(lambda p: "AP" if "AP" in str(p) or "ap" in str(p).lower() else "Lat")
    if "age" not in df.columns:
        df["age"] = 10
    if "label" not in df.columns:
        if "fracture_visible" in df.columns:
            df["label"] = df["fracture_visible"].fillna(0).astype(int)
        else:
            df["label"] = -1

    model_root = Path(args.model_root)

    preds = []
    groups = df.groupby(["projection", df["age"].apply(age_group_fn)], as_index=False)
    for (proj, _), sub in groups:
        ageg_series = sub["age"].apply(age_group_fn)
        ageg = int(ageg_series.iloc[0])
        wpath = weight_path_of(model_root, proj, ageg)
        if not wpath.exists():
            print(f"[SKIP] 가중치 없음: {wpath}")
            for _, r in sub.iterrows():
                preds.append({**r.to_dict(), "age_group": ageg, "prob": -1.0, "pred": -1})
            continue

        model = build_model()
        model = load_state(model, wpath, DEVICE)

        target_layer = None
        if args.save_cam:
            if args.cam_layer.strip():
                try:
                    obj = model
                    for tok in args.cam_layer.strip().split("."):
                        obj = obj[int(tok)] if tok.isdigit() else getattr(obj, tok)
                    target_layer = obj if isinstance(obj, nn.Conv2d) else get_target_layer_for_convnext(model)
                except Exception:
                    target_layer = get_target_layer_for_convnext(model)
            else:
                target_layer = get_target_layer_for_convnext(model)

        for _, r in sub.iterrows():
            img_path = r["img_path"]
            try:
                x, orig_bgr = preprocess_cv2(img_path)
            except Exception as e:
                print(f"[READ_FAIL] {img_path}: {e}")
                preds.append({**r.to_dict(), "age_group": ageg, "prob": -1.0, "pred": -1})
                continue

            x = x.to(DEVICE, memory_format=torch.channels_last, non_blocking=True)

            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
                    logit = model(x).squeeze(1)
                    prob = torch.sigmoid(logit).item()
            pred = int(prob >= args.prob_th)
            row = {**r.to_dict(), "age_group": ageg, "prob": float(prob), "pred": int(pred)}
            preds.append(row)

            if args.save_cam:
                try:
                    cam_map = make_cam(model, x, target_layer, prob)
                    if cam_map is not None:
                        bn = Path(img_path).stem
                        out_p = cam_dir / f"{proj}_age{ageg}_{bn}_p{prob:.3f}.jpg"
                        overlay = overlay_cam_on_image(orig_bgr, cam_map, alpha=0.35)
                        cv2.imwrite(str(out_p), overlay)
                except Exception as e:
                    print(f"[CAM_FAIL] {img_path}: {e}")

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    df_out = pd.DataFrame(preds)
    out_pred = Path(args.outdir) / "predictions.csv"
    df_out.to_csv(out_pred, index=False)
    print(f"[SAVE] {out_pred}")

    if "label" in df_out.columns and (df_out["label"] >= 0).any():
        valid = df_out[df_out["label"] >= 0]
        if len(valid) > 0:
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
            y = valid["label"].astype(int).values
            p = valid["prob"].values
            yhat = (p >= args.prob_th).astype(int)
            acc = accuracy_score(y, yhat)
            f1 = f1_score(y, yhat, zero_division=0)
            try:
                auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
            except Exception:
                auc = np.nan
            tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0,1]).ravel()
            sen = tp / (tp + fn) if (tp+fn)>0 else np.nan
            spe = tn / (tn + fp) if (tn+fp)>0 else np.nan
            print(f"[SUMMARY] N={len(valid)} Acc={acc:.4f} F1={f1:.4f} AUROC={auc:.4f} Sens={sen:.4f} Spec={spe:.4f}")

if __name__ == "__main__":
    main()
