#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PediatricFracture_Wrist One-Click 환경 구축 스크립트
- 원본(Cls/Det) 폴더를 새 폴더로 복사
- 절대경로/가중치/데이터 경로/DDP nproc 자동 패치
- 주석을 강화해 '복사 즉시 실행' 가능 상태로 조정

사용법:
  python setup_pediatric_wrist_oneclick.py \
    --src "/mnt/data/KimJG/SwinT/mediaiOA_swinT_cls-main_original/20250623_Single/WristFX_0730" \
    --dst "./PediatricFracture_Wrist" \
    [--dry-run]
"""
import argparse
import os
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---------- helpers ----------
def copy_tree(src: Path, dst: Path):
    if dst.exists():
        print(f"⚠️  대상 폴더가 이미 존재합니다: {dst}")
    else:
        print(f"📁 복사: {src} → {dst}")
        shutil.copytree(src, dst)

def patch_file(path: Path, subs: list[tuple[re.Pattern, str]]):
    text = path.read_text(encoding="utf-8")
    orig = text
    for pat, repl in subs:
        text = pat.sub(repl, text, count=0)
    if text != orig:
        path.write_text(text, encoding="utf-8")
        print(f"✏️  패치됨: {path.relative_to(project_root)}")
    else:
        print(f"➖ 변경 없음: {path.relative_to(project_root)}")

def safe_exists(p: Path):
    if not p.exists():
        print(f"⚠️  경고: 대상 없음 {p}")

# ---------- args ----------
ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="원본 WristFX_0730 절대경로")
ap.add_argument("--dst", default=str(HERE / "PediatricFracture_Wrist"), help="복사될 새 프로젝트 경로")
ap.add_argument("--dry-run", action="store_true", help="복사/패치 없이 무엇이 변경되는지 로그만")
args = ap.parse_args()

src_root = Path(args.src).resolve()
project_root = Path(args.dst).resolve()

if not src_root.exists():
    raise SystemExit(f"❌ 원본 경로가 존재하지 않습니다: {src_root}")

# ---------- copy ----------
if not args.dry_run:
    copy_tree(src_root, project_root)

# ---------- patches ----------
# 1) classification/train_ddp_fracture_per_agegroup_convnextv2_0811.py 의 CONFIG_PATH 절대경로 제거
cls_ddp = project_root / "classification" / "train_ddp_fracture_per_agegroup_convnextv2_0811.py"
safe_exists(cls_ddp)
if cls_ddp.exists() and not args.dry_run:
    patch_file(
        cls_ddp,
        [
            # CONFIG_PATH = "/home/.../best_summary.json" -> env 또는 로컬 파일로
            (re.compile(r'CONFIG_PATH\s*=\s*["\']/.+?best_summary\.json["\']'),
             'CONFIG_PATH = os.environ.get("BEST_SUMMARY_JSON", os.path.join(os.path.dirname(__file__), "best_summary.json"))'),
            # 파일 부재 시 기존 코드가 바로 예외를 내므로, 안전가드 추가: try/except 또는 존재 검사 후 기본값 사용
            (re.compile(r'with open\(CONFIG_PATH, "r"\) as f:\s*best_cfg_json = json\.load\(f\)\s*'
                        r'(.+?)'
                        r'else:\s*\n\s*raise ValueError\("best_summary\.json 포맷을 알 수 없습니다\."\)',
                        re.DOTALL),
             'try:\n    with open(CONFIG_PATH, "r") as f:\n        best_cfg_json = json.load(f)\nexcept FileNotFoundError:\n    if int(os.environ.get("LOCAL_RANK", "0")) == 0:\n        print(f"⚠️ best_summary.json 없음: {CONFIG_PATH} → 내부 기본 설정으로 진행")\n    best_cfg_json = {}\n\\1else:\n    pass')
        ]
    )

# 2) classification/run_all_training_0813.py 의 --nproc_per_node=2 고정 → GPU 자동감지
cls_runner = project_root / "classification" / "run_all_training_0813.py"
safe_exists(cls_runner)
if cls_runner.exists() and not args.dry_run:
    patch_file(
        cls_runner,
        [
            # BASE: import torch 추가 및 nproc 자동결정
            (re.compile(r'(^import os\s*\nimport subprocess)', re.MULTILINE),
             'import os\nimport subprocess\nimport torch  # GPU 개수 자동 감지용'),
            # nproc 문자열 교체
            (re.compile(r'(\[\s*"torchrun",\s*"--nproc_per_node=)2(")', re.MULTILINE),
             r'\1"+str(max(torch.cuda.device_count(), 1))+"\2'),
        ]
    )

# 3) detection/run_all_training_detection_0730.py
det_train = project_root / "detection" / "run_all_training_detection_0730.py"
safe_exists(det_train)
if det_train.exists() and not args.dry_run:
    patch_file(
        det_train,
        [
            # RAW_IMAGE_DIR/RAW_LABEL_DIR → env 로 오버라이드 가능 + 상대 기본값 명확화
            (re.compile(r'RAW_IMAGE_DIR\s*=\s*Path\("GRAZPEDWRI-DX/data/images/train_aug"\)'),
             'RAW_IMAGE_DIR = Path(os.environ.get("DET_RAW_IMAGE_DIR", str((BASE_DIR/ "GRAZPEDWRI-DX" / "data" / "images" / "train_aug"))))'),
            (re.compile(r'RAW_LABEL_DIR\s*=\s*Path\("GRAZPEDWRI-DX/data/labels/train_aug"\)'),
             'RAW_LABEL_DIR = Path(os.environ.get("DET_RAW_LABEL_DIR", str((BASE_DIR/ "GRAZPEDWRI-DX" / "data" / "labels" / "train_aug"))))'),
            # weights 이름 고정 → 내부 weights/yolov9-c-640.pt 기본값 + DET_DEVICE 환경변수
            (re.compile(r'--weights\s+yolov9-c\.pt'),
             r'--weights "+os.environ.get("DET_PRETRAIN", str(BASE_DIR / "weights" / "yolov9-c-640.pt"))+"'),
            (re.compile(r'--device\s+0,1'),
             r'--device "+os.environ.get("DET_DEVICE", ("0,"*(max(1, os.cpu_count()//32))).strip(",") )+"')
        ]
    )

# 4) detection/run_final_inference_detection_0730.py 는 경로가 상대라 기본 OK

# 5) 최상위에 one_click.py 생성 (GPU 풀로드로 전체 파이프라인 실행)
one_click = project_root / "one_click.py"
if not args.dry_run:
    one_click.write_text(
        """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
        + r'''
"""
One-Click Runner
- GPU 개수 자동 감지(DDP 풀로드)
- 분류 학습 → GradCAM → 탐지 학습 → 탐지 추론 → (옵션) 앙상블까지 연속 실행
환경변수:
  BEST_SUMMARY_JSON : (선택) 분류 최적 하이퍼파라미터 JSON
  DET_RAW_IMAGE_DIR / DET_RAW_LABEL_DIR : (선택) 탐지 원본 이미지/라벨 경로
  DET_PRETRAIN : (선택) YOLO 사전가중치 경로(기본: detection/weights/yolov9-c-640.pt)
"""
import os
import subprocess
import sys
from pathlib import Path

import torch

PROJ = Path(__file__).resolve().parent
CLS = PROJ / "classification"
DET = PROJ / "detection"
ENS = PROJ / "Ensemble"

def run(desc, args):
    print(f"\n📌 {desc}")
    print("   $ " + " ".join(args))
    subprocess.run(args, check=True)

def main():
    # 1) 분류 파이프라인(DDP 풀로드)
    nproc = max(torch.cuda.device_count(), 1)
    run("Cls: test set 생성", ["python", str(CLS / "age_split_testset_0730.py")])
    run("Cls: train/val 분할", ["python", str(CLS / "generate_age_trainval_split_0730.py")])
    run("Cls: 8모델 DDP 학습", ["torchrun", f"--nproc_per_node={nproc}", str(CLS / "train_ddp_fracture_per_agegroup_convnextv2_0811.py")])
    run("Cls: Grad-CAM", ["python", str(CLS / "Generate_Gradcam_ConvNeXtV2_ALL_0813.py")])

    # 2) 탐지 파이프라인(학습+추론)
    run("Det: 데이터셋 split & data.yaml", ["python", str(DET / "run_all_training_detection_0730.py")])
    run("Det: 최종 추론/평가", ["python", str(DET / "run_final_inference_detection_0730.py")])

    # 3) (옵션) 앙상블 추론 스텝이 있으면 실행
    ens_py = ENS / "Final_inference_APLAT_ensemble_from_uploaded_paths_0820.py"
    if ens_py.exists():
        print("\nℹ️ Ensemble 스크립트를 발견했습니다. 필요 시 인자에 맞춰 직접 실행하세요:")
        print(f"   python {ens_py}")

    print("\n🎉 One-Click 파이프라인 완료")

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(f"❌ 실행 실패: {e}")
''',
        encoding="utf-8",
    )
    os.chmod(one_click, 0o755)
    print(f"✅ 생성: {one_click.relative_to(project_root)}")

print("\n✅ 준비 완료. 다음 명령으로 실행하세요:")
print(f"   python {one_click}")
print("\n(참고) DDP 풀로드는 자동으로 현재 장착된 GPU 개수를 감지해 사용합니다.")
