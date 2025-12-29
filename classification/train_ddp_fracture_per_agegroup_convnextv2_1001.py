# train_ddp_fracture_per_agegroup_convnextv2_1001.py
import os
import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
import torchvision.transforms as transforms
from timm import create_model
import mlflow

from unified_dataset_0704 import UnifiedDataset


# -------------------------
# Args
# -------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--fold", type=int, default=0)   # 학습할 폴드 index
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--outdir", type=str, default=".")  # 저장 루트
args = ap.parse_args()


# -------------------------
# DDP init
# -------------------------
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")
DEVICE = torch.device("cuda", local_rank)
torch.backends.cudnn.benchmark = True


def is_main() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


# -------------------------
# Paths / Configs
# -------------------------
BASE_DIR = os.path.dirname(__file__)
SAVE_ROOT = Path(args.outdir).resolve() / "Model_Fold"
IMG_SIZE = 384
MODEL_NAME = "convnextv2_large.fcmae_ft_in22k_in1k_384"

# Age Group 함수
def AGE_GROUP_FN(age):
    age = float(age)
    if age < 5: return 0
    elif age < 10: return 1
    elif age < 15: return 2
    else: return 3

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce)
        fl = self.alpha * (1 - pt) ** self.gamma * bce
        return fl.mean() if self.reduction == "mean" else fl.sum()

# Transform
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 투영×연령대 설정
CONFIG = {
    ("AP", 0): {"lr": 1e-4, "drop_path": 0.3, "epochs": 50, "loss": "focal", "gamma": 2.0},
    ("AP", 1): {"lr": 1e-5, "drop_path": 0.2, "epochs": 40, "loss": "bce"},
    ("AP", 2): {"lr": 1e-5, "drop_path": 0.2, "epochs": 50, "loss": "bce"},
    ("AP", 3): {"lr": 5e-6, "drop_path": 0.1, "epochs": 40, "loss": "bce"},
    ("Lat", 0): {"lr": 1e-4, "drop_path": 0.3, "epochs": 60, "loss": "focal", "gamma": 3.0},
    ("Lat", 1): {"lr": 1e-4, "drop_path": 0.2, "epochs": 50, "loss": "focal", "gamma": 2.0},
    ("Lat", 2): {"lr": 1e-5, "drop_path": 0.2, "epochs": 60, "loss": "bce"},
    ("Lat", 3): {"lr": 5e-6, "drop_path": 0.1, "epochs": 40, "loss": "bce"},
}


def build_sampler_weights(labels_tensor: torch.Tensor) -> WeightedRandomSampler:
    # labels_tensor: shape [N], values in {0.,1.}
    y = labels_tensor.float().cpu().numpy()
    n1 = (y == 1).sum()
    n0 = (y == 0).sum()
    # inverse-frequency weights
    w0 = 1.0 / max(n0, 1)
    w1 = 1.0 / max(n1, 1)
    weights = torch.tensor([w1 if yy == 1 else w0 for yy in y], dtype=torch.float)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    return sampler


try:
    for proj in ["AP", "Lat"]:
        if is_main():
            print(f"\n🚀 Start Training Projection: {proj}")

        # 입력 CSV
        train_csv = os.path.join(BASE_DIR, f"age_train_tmp_{proj}.csv")
        val_csv   = os.path.join(BASE_DIR, f"age_val_tmp_{proj}.csv")
        df_train  = pd.read_csv(train_csv)
        df_val    = pd.read_csv(val_csv)
        df_all    = pd.concat([df_train, df_val], ignore_index=True)

        # 라벨/연령대
        df_all["label"] = df_all["fracture_visible"].fillna(0).astype(float)
        df_all["age_group_label"] = df_all["age"].astype(float).apply(AGE_GROUP_FN)

        for age_group in [0, 1, 2, 3]:
            if is_main():
                print(f"\n🔹 Training {proj}_{age_group} (fold {args.fold}/{args.k})")

            cfg    = CONFIG[(proj, age_group)]
            lr     = cfg["lr"]
            dpr    = cfg["drop_path"]
            epochs = cfg["epochs"]

            df_g = df_all[df_all["age_group_label"] == age_group].reset_index(drop=True)
            if len(df_g) == 0:
                if is_main():
                    print(f"[SKIP] {proj}_{age_group}: 데이터 없음")
                continue

            # 유효 K 계산
            c0 = int((df_g["label"] == 0).sum())
            c1 = int((df_g["label"] == 1).sum())
            k_eff = min(args.k, c0, c1)
            if k_eff < 2:
                if is_main():
                    print(f"[SKIP] {proj}_{age_group}: 표본 부족(c0={c0}, c1={c1})")
                continue

            # Group K-Fold 우선
            grp_col = "patient_id" if "patient_id" in df_g.columns else ("case_id" if "case_id" in df_g.columns else None)
            if grp_col:
                splitter = StratifiedGroupKFold(n_splits=k_eff, shuffle=True, random_state=args.seed)
                splits = list(splitter.split(df_g, df_g["label"], groups=df_g[grp_col].astype(str)))
            else:
                splitter = StratifiedKFold(n_splits=k_eff, shuffle=True, random_state=args.seed)
                splits = list(splitter.split(df_g, df_g["label"]))

            if args.fold >= len(splits):
                if is_main():
                    print(f"[SKIP] {proj}_{age_group}: fold {args.fold} >= {len(splits)}")
                continue

            tr_idx, va_idx = splits[args.fold]
            df_train_g = df_g.iloc[tr_idx].reset_index(drop=True)
            df_val_g   = df_g.iloc[va_idx].reset_index(drop=True)

            # Dataset
            train_dataset = UnifiedDataset(df_train_g, transform=transform, task="fracture_only")
            val_dataset   = UnifiedDataset(df_val_g,   transform=transform, task="fracture_only")

            # Sampler (class imbalance 대응)
            y_train = torch.tensor(df_train_g["label"].values, dtype=torch.float32)
            weighted_sampler = build_sampler_weights(y_train)

            # Loader
            train_loader = DataLoader(train_dataset, batch_size=12, sampler=weighted_sampler,
                                      num_workers=4, pin_memory=True, drop_last=False)
            val_loader   = DataLoader(val_dataset,   batch_size=12, shuffle=False,
                                      num_workers=4, pin_memory=True, drop_last=False)

            # Model
            model = create_model(MODEL_NAME, pretrained=True, num_classes=1, drop_path_rate=dpr)
            model = model.to(DEVICE, memory_format=torch.channels_last)

            # 파라미터 contiguous 보장
            for p in model.parameters():
                if not p.is_sparse:
                    p.data = p.data.contiguous()

            # DDP
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                gradient_as_bucket_view=True,
                static_graph=True,  # PyTorch 2.1+
            )

            # Loss
            if cfg["loss"] == "focal":
                criterion = FocalLoss(alpha=0.25, gamma=cfg.get("gamma", 2.0))
            else:
                n1 = float((df_train_g["label"] == 1).sum())
                n0 = float((df_train_g["label"] == 0).sum())
                pos_weight = torch.tensor([n0 / (n1 + 1e-6)], device=DEVICE)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            # Optim
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

            # MLflow
            if is_main():
                run_name = f"ConvNeXtV2_{proj}_Age{age_group}_fold{args.fold}"
                mlflow.start_run(run_name=run_name)
                best_f1 = 0.0

            # Train
            for epoch in range(epochs):
                model.train()
                total_loss, preds, labels = 0.0, [], []
                for images, targets in train_loader:
                    images  = images.to(DEVICE, memory_format=torch.channels_last, non_blocking=True)
                    targets = targets.to(DEVICE).float().view(-1)

                    optimizer.zero_grad(set_to_none=True)
                    outputs = model(images).squeeze(dim=-1)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

                    total_loss += float(loss.detach())
                    probs = torch.sigmoid(outputs).detach().cpu().numpy()
                    preds += (probs > 0.5).astype(int).tolist()
                    labels += targets.detach().cpu().numpy().astype(int).tolist()

                if is_main():
                    acc = accuracy_score(labels, preds)
                    f1  = f1_score(labels, preds)
                    print(f"[Train] {proj}_{age_group} fold{args.fold} Ep {epoch} | Loss {total_loss:.4f} | Acc {acc:.4f} | F1 {f1:.4f}")
                    mlflow.log_metric("train_acc", acc, step=epoch)
                    mlflow.log_metric("train_f1",  f1,  step=epoch)

                    # Val
                    model.eval()
                    v_preds, v_labels = [], []
                    with torch.no_grad():
                        for images, targets in val_loader:
                            images  = images.to(DEVICE, memory_format=torch.channels_last, non_blocking=True)
                            targets = targets.to(DEVICE).float().view(-1)
                            outputs = model(images).squeeze(dim=-1)
                            probs = torch.sigmoid(outputs).cpu().numpy()
                            v_preds  += (probs > 0.5).astype(int).tolist()
                            v_labels += targets.cpu().numpy().astype(int).tolist()

                    v_acc = accuracy_score(v_labels, v_preds)
                    v_f1  = f1_score(v_labels, v_preds)
                    print(f"[Val]   {proj}_{age_group} fold{args.fold} Ep {epoch} | Acc {v_acc:.4f} | F1 {v_f1:.4f}")
                    mlflow.log_metric("val_acc", v_acc, step=epoch)
                    mlflow.log_metric("val_f1",  v_f1,  step=epoch)

                    # Save best
                    if v_f1 > best_f1:
                        best_f1 = v_f1
                        save_dir = SAVE_ROOT / f"{proj}_{age_group}_fold{args.fold}"
                        save_dir.mkdir(parents=True, exist_ok=True)
                        model_path = save_dir / "best.pt"
                        state = (model.module if isinstance(model, DDP) else model).state_dict()
                        torch.save(state, str(model_path))
                        mlflow.log_artifact(str(model_path))
                        print(f"✅ Saved: {model_path}")

            if is_main() and mlflow.active_run():
                mlflow.end_run()

    # 모든 학습 종료 동기화
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

finally:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
