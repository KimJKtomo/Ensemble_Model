# WristFX_0730: Pediatric Wrist & Elbow Fracture AI Pipeline

소아 손목/팔꿈치 방사선 영상에서 **골절 및 fat pad sign 검출**을 지원하는 AI 파이프라인.  
YOLOv9 기반 **Detection**, Swin Transformer  / ConvNeXt_v2 기반 **Classification**, 그리고 **Ensemble** 전략으로 성능을 향상. Grad-CAM 기반 시각화 지원.

---

## 📦 Features
- **Detection**: YOLOv9 (Fracture, Fat pad)
- **Classification**: Swin Transformer (ROI별 binary classification)
- **Ensemble**: Cls × 0.6 + Det × 0.4
- **Explainability**: Grad-CAM, CAM overlays
- **MLOps**: MLflow 기반 로그 관리, reproducibility

---

## 📂 Project Structure
```
WristFX_0730/
├─ classification/    # SwinT 학습 및 테스트
├─ detection/         # YOLOv9 학습 및 테스트
├─ Ensemble/          # Ensemble inference (AP/LAT, fracture + fat pad)
│  ├─ Result/         # inference output (images, CAM)
│  └─ Final_inference_APLAT_ensemble_from_uploaded_paths_0820.py
├─ Cropping/          # ROI crop logic
├─ Labeling/          # labeling utils
├─ PediatricFracture_Wrist/ # dataset scripts
├─ setup_pediatric_wrist_oneclick.py
├─ requirements.txt
└─ README.md
```

---

## ⚙️ Installation
```bash
conda create -n wristfx python=3.10 -y
conda activate wristfx
pip install -r requirements.txt
```

---

## 🚀 Usage

### 원클릭 실행
```bash
python setup_pediatric_wrist_oneclick.py
```

### 개별 학습
```bash
# Classification
python classification/train_swinT_parts_final.py --config config.yaml

# Detection
python detection/train_crop.py --cfg models/yolov9-c.yaml --data data/wrist.yaml
```

### Ensemble Inference
```bash
python Ensemble/Final_inference_APLAT_ensemble_from_uploaded_paths_0820.py
```

---

## 📊 Results (Ensemble)

- Ensemble 계산식: `Final = Cls*0.6 + Det*0.4`
- Threshold = `0.25`

| Date | Task | View | Samples (pos / neg) | Sensitivity | AUROC | Notes |
|---|---|---|---|---:|---:|---|
| 2025-09-02 | FX | AP  | 311 (217 / 94) | **0.9333** | **0.9789** | Miss 6, Over 2 |
| 2025-09-02 | FX | LAT | 320 (219 / 101) | **0.9149** | **0.9783** | Miss 8, Over 3 |
| 2025-09-04 | FX+Fat Pad | AP  | 311 (217 / 94) | **0.9255** | — | Miss 7, Over 16 |
| 2025-09-04 | FX+Fat Pad | LAT | 320 (219 / 101) | **0.9307** | — | Miss 8, Over 3 |

---

## 🖼️ Example Visualizations

원본 X-ray vs Grad-CAM Overlay (Detection + Classification Ensemble).  
**모든 이미지는 비식별화된 샘플 데이터.**

| Original | Grad-CAM |
|---|---|
| ![](./Ensemble/Result/01029957HBD_CR16860.1.4.jpg) | ![](./Ensemble/Result/01029957HBD_CR16860.1.4_CAM.jpg) |
| ![](./Ensemble/Result/02016545HBD_CR08599.1.3.jpg) | ![](./Ensemble/Result/02016545HBD_CR08599.1.3_CAM.jpg) |
| ![](./Ensemble/Result/02025808HBD_CR14258.1.3.jpg) | ![](./Ensemble/Result/02025808HBD_CR14258.1.3_CAM.jpg) |
| ![](./Ensemble/Result/02030557HBD_CR17256.1.4.jpg) | ![](./Ensemble/Result/02030557HBD_CR17256.1.4_CAM.jpg) |

---

## 🤝 Acknowledgements
- **Seoul Asan Medical Center** Pediatric Emergency/Trauma Team  
- **MURA**, **GRAZPEDWRI-DX** dataset  
- **Ultralytics YOLO**, **timm (Swin Transformer)**, **ConvNeXt_v2**
- **MLflow**, **PyTorch**, **OpenMMLab**
