# make_fold_ensemble_best.py
import argparse, glob
from pathlib import Path
import torch
from timm import create_model

MODEL_NAME = "convnextv2_large.fcmae_ft_in22k_in1k_384"  # 학습과 동일

def list_groups(model_fold_dir: Path):
    # 예: AP_0_fold3/best.pt → 그룹키 "AP_0"
    items = glob.glob(str(model_fold_dir / "*_fold*" / "best.pt"))
    return sorted({ Path(p).parts[-2].rsplit("_", 1)[0] for p in items })  # "AP_0", "Lat_2" 등

def average_state_dict(ckpt_paths, weight=None):
    avg, n = None, 0
    for i, p in enumerate(sorted(ckpt_paths)):
        sd = torch.load(p, map_location="cpu")
        w = 1.0 if weight is None else float(weight[i])
        if avg is None:
            avg = {k: sd[k].float() * w for k in sd.keys()}
        else:
            for k in avg:
                avg[k] += sd[k].float() * w
        n += w
    if avg is None:
        return None
    for k in avg:
        avg[k] /= max(n, 1e-8)
    return avg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_root", required=True, help="classification 디렉토리 (그 안에 Model_Fold/ 존재)")
    ap.add_argument("--write_best", action="store_true", help="각 그룹 폴더에 best.pt로 저장")
    ap.add_argument("--outfile_suffix", default="_ensemble_avg.pt", help="파일명 접미사")
    ap.add_argument("--weighted", action="store_true", help="폴드 수 가중치 대신 사용자 가중치 사용(동일 가중 권장)")
    args = ap.parse_args()

    fold_dir = Path(args.model_root) / "Model_Fold"
    groups = list_groups(fold_dir)
    if not groups:
        print("폴드 체크포인트 없음")
        return

    for g in groups:
        ckpts = sorted(glob.glob(str(fold_dir / f"{g}_fold*" / "best.pt")))
        if not ckpts:
            print(f"[SKIP] {g}: 폴드 없음")
            continue

        # 가중치 벡터: 기본 동일가중
        weight = None
        if args.weighted:
            # 필요 시 사용자 로직으로 교체 (예: 검증 F1 기반)
            weight = [1.0] * len(ckpts)

        avg_sd = average_state_dict(ckpts, weight)
        if avg_sd is None:
            print(f"[SKIP] {g}: 평균 실패")
            continue

        # 모델 템플릿에 로드해 shape 검증
        m = create_model(MODEL_NAME, pretrained=False, num_classes=1)
        m.load_state_dict(avg_sd, strict=True)

        # 저장 경로
        out_file = fold_dir / f"{g}{args.outfile_suffix}"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(m.state_dict(), str(out_file))
        print(f"[SAVE] {out_file}")

        if args.write_best:
            # 각 그룹 폴더에도 best.pt로 복사 저장
            g_dir = fold_dir / f"{g}_fold0"
            g_dir = g_dir if g_dir.exists() else Path(str(fold_dir / f"{g}_fold")).parent / f"{g}_fold0"
            # 표준 위치: Model_Fold/{g}/best.pt
            target_dir = fold_dir / g
            target_dir.mkdir(parents=True, exist_ok=True)
            best_path = target_dir / "best.pt"
            torch.save(m.state_dict(), str(best_path))
            print(f"[SAVE] {best_path}")

if __name__ == "__main__":
    main()
