#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_ddp_fracture_per_agegroup_convnextv2_1111_kfold5.py

기존 train_ddp_fracture_per_agegroup_convnextv2_1111.py 기반 수정본.

핵심 변경:
1) AgeGroup(0~3)별 StratifiedKFold(기본 5-fold) Cross-Validation 수행
2) DDP 환경에서 WeightedRandomSampler 제거 -> DistributedSampler 사용
3) fold별 best ckpt 저장 + fold summary CSV 저장

실행 예)
torchrun --standalone --nproc_per_node=2 train_ddp_fracture_per_agegroup_convnextv2_1111_kfold5.py

Env override (기존 호환):
- BASE_DIR, SAVE_DIR, BATCH_SIZE, EPOCHS, IMG_SIZE, LR, MODEL_NAME
- USE_AMP(0/1), USE_GRAD_CKPT(0/1)
- ES_PATIENCE, ES_MIN_DELTA
- K_FOLDS (default=5)
- ONLY_AGE_GROUP (optional, e.g. "0")
- ONLY_FOLD (optional, e.g. "2")  # 0-index
"""
import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
import re
import math
import csv
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
import torchvision.transforms as transforms
from timm import create_model
from torch.cuda.amp import autocast, GradScaler
import mlflow

from unified_dataset_0704 import UnifiedDataset


# =========================
# DDP 초기화
# =========================
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")
DEVICE = torch.device("cuda", local_rank)


# =========================
# 경로 및 설정 (env 기반 override 가능)
# =========================
BASE_DIR = os.getenv("BASE_DIR", "/mnt/data/downloads/Wrist/One_Click_Wrist/0115/wrist_dataset_0115")
SAVE_DIR = os.getenv("SAVE_DIR", "/mnt/data/downloads/Wrist/One_Click_Wrist/0115/Train_model_0115")
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "12"))
EPOCHS = int(os.getenv("EPOCHS", "50"))
IMG_SIZE = int(os.getenv("IMG_SIZE", "512"))
LR = float(os.getenv("LR", "1e-4"))
MODEL_NAME = os.getenv("MODEL_NAME", "convnextv2_large.fcmae_ft_in22k_in1k_384")

# =========================
# Stage-wise fine-tuning / LR schedule
# =========================
USE_STAGEWISE = os.getenv("USE_STAGEWISE", "1") == "1"
# Format: "head:3,last:7,full:10"  (epochs per segment). "full" segment can be omitted.
STAGE_PLAN_STR = os.getenv("STAGE_PLAN", "head:3,last3:7,last2:30,full:10")
WARMUP_EPOCHS = int(os.getenv("WARMUP_EPOCHS", "1"))  # per segment
MIN_LR_RATIO = float(os.getenv("MIN_LR_RATIO", "0.05"))
# Discriminative LR multipliers (base = LR)
LR_HEAD_MULT = float(os.getenv("LR_HEAD_MULT", "1.0"))
LR_LAST_MULT = float(os.getenv("LR_LAST_MULT", "0.5"))
LR_BACKBONE_MULT = float(os.getenv("LR_BACKBONE_MULT", "0.1"))

USE_AMP = os.getenv("USE_AMP", "1") == "1"
USE_MLFLOW = os.getenv("USE_MLFLOW", "1") == "1"
USE_GRAD_CKPT = os.getenv("USE_GRAD_CKPT", "1") == "1"

ES_PATIENCE = int(os.getenv("ES_PATIENCE", "50"))
ES_MIN_DELTA = float(os.getenv("ES_MIN_DELTA", "1e-4"))

K_FOLDS = int(os.getenv("K_FOLDS", "5"))
ONLY_AGE_GROUP = os.getenv("ONLY_AGE_GROUP", "").strip()
ONLY_FOLD = os.getenv("ONLY_FOLD", "").strip()


def AGE_GROUP_FN(age):
    age = float(age)
    if age < 5:
        return 0
    elif age < 10:
        return 1
    elif age < 15:
        return 2
    else:
        return 3


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def parse_stage_plan(plan_str: str):
    """Parse stage plan string like 'head:3,last:7,full:10'."""
    plan = []
    for chunk in [c.strip() for c in plan_str.split(',') if c.strip()]:
        if ':' not in chunk:
            raise ValueError(f"Invalid STAGE_PLAN token: {chunk}")
        name, n = chunk.split(':', 1)
        name = name.strip().lower()
        n = int(n.strip())
        if n <= 0:
            continue
        if name not in ("head", "last", "last3", "last2", "last1", "full"):
            raise ValueError(f"Unknown stage name: {name}")
        plan.append((name, n))
    # basic sanity: must start with head
    if len(plan) == 0:
        plan = [("full", EPOCHS)]
    return plan


def _param_stage_id(param_name: str):
    """Return stage index if name contains 'stages.{i}.' else None."""
    m = re.search(r"\bstages\.(\d+)\.", param_name)
    if not m:
        return None
    return int(m.group(1))


def set_trainable_stagewise(ddp_model: DDP, mode: str):
    """Freeze/unfreeze parameters by backbone stage.

    Supported modes:
      - head : head/classifier only
      - last3: stages.(max) + head
      - last2: stages.(max-1 .. max) + head
      - last1: stages.(max-2 .. max) + head
      - last : alias of last3 (backward-compatible)
      - full : all params trainable
    """
    base = ddp_model.module if isinstance(ddp_model, DDP) else ddp_model
    mode = mode.lower().strip()

    # find max stage id present (ConvNeXtV2 in timm typically stages.0~3)
    stage_ids = []
    for n, _p in base.named_parameters():
        sid = _param_stage_id(n)
        if sid is not None:
            stage_ids.append(sid)
    max_stage = max(stage_ids) if stage_ids else 3

    def stage_threshold(m: str):
        if m in ("head",):
            return None
        if m in ("last", "last3"):
            return max_stage
        if m == "last2":
            return max_stage - 1
        if m == "last1":
            return max_stage - 2
        if m == "full":
            return 0
        raise ValueError(f"Unknown stage-wise mode: {m}")

    th = stage_threshold(mode)

    for n, p in base.named_parameters():
        if not p.is_floating_point():
            p.requires_grad = False
            continue

        is_head = ("head" in n) or ("classifier" in n)
        sid = _param_stage_id(n)

        if mode == "head":
            p.requires_grad = bool(is_head)
            continue

        # always train head for non-head modes
        if is_head:
            p.requires_grad = True
            continue

        # stage params
        if sid is not None and th is not None and sid >= th:
            p.requires_grad = True
        else:
            p.requires_grad = False

def build_param_groups(ddp_model: DDP, base_lr: float, mode: str):
    """Create optimizer param groups with discriminative LR."""
    base = ddp_model.module if isinstance(ddp_model, DDP) else ddp_model
    mode = mode.lower()
    stage_ids = []
    for n, _p in base.named_parameters():
        sid = _param_stage_id(n)
        if sid is not None:
            stage_ids.append(sid)
    last_stage = max(stage_ids) if stage_ids else 3

    head_params, last_params, backbone_params = [], [], []
    for n, p in base.named_parameters():
        if not p.requires_grad:
            continue
        is_head = ("head" in n) or ("classifier" in n)
        sid = _param_stage_id(n)
        if is_head:
            head_params.append(p)
        elif sid == last_stage:
            last_params.append(p)
        else:
            backbone_params.append(p)

    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": base_lr * LR_HEAD_MULT})
    if mode in ("last", "last3", "last2", "last1", "full") and last_params:
        groups.append({"params": last_params, "lr": base_lr * LR_LAST_MULT})
    if mode in ("last2", "last1", "full") and backbone_params:
        groups.append({"params": backbone_params, "lr": base_lr * LR_BACKBONE_MULT})
    return groups


class WarmupCosinePerSegment:
    """Per-segment LR scheduler: linear warmup then cosine decay to min_lr_ratio."""
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, min_lr_ratio: float):
        self.optimizer = optimizer
        self.warmup_epochs = max(int(warmup_epochs), 0)
        self.total_epochs = max(int(total_epochs), 1)
        self.min_lr_ratio = float(min_lr_ratio)
        self.base_lrs = [g.get("lr", 1e-4) for g in optimizer.param_groups]

    def step(self, epoch_in_seg: int):
        e = int(epoch_in_seg)
        for i, g in enumerate(self.optimizer.param_groups):
            base = self.base_lrs[i]
            if self.warmup_epochs > 0 and e < self.warmup_epochs:
                lr = base * float(e + 1) / float(self.warmup_epochs)
            else:
                # cosine over remaining epochs
                t = e - self.warmup_epochs
                T = max(self.total_epochs - self.warmup_epochs, 1)
                # clamp
                t = min(max(t, 0), T)
                cos = 0.5 * (1.0 + math.cos(math.pi * t / T))
                lr = base * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cos)
            g["lr"] = float(lr)


def build_base_model():
    """Create timm ConvNeXtV2 backbone on DEVICE (NOT wrapped in DDP)."""
    base_model = create_model(MODEL_NAME, pretrained=True, num_classes=1)
    if USE_GRAD_CKPT:
        if hasattr(base_model, "set_grad_checkpointing"):
            try:
                base_model.set_grad_checkpointing(True)
            except TypeError:
                base_model.set_grad_checkpointing()
        else:
            for m in base_model.modules():
                if hasattr(m, "gradient_checkpointing"):
                    try:
                        m.gradient_checkpointing(True)
                    except TypeError:
                        m.gradient_checkpointing = True
    base_model = base_model.to(DEVICE)
    return base_model


def wrap_ddp(base_model: nn.Module) -> DDP:
    """(Re)wrap a module with DDP. Required when trainable params set changes (stage-wise)."""
    ddp_model = DDP(
        base_model,
        device_ids=[local_rank],
        output_device=local_rank,
        # Stage-wise toggles trainability across segments; allow unused param detection.
        find_unused_parameters=False,
    )
    return ddp_model

def get_criterion(train_labels):
    # train_labels: list[float] in {0.0, 1.0}
    count_0 = train_labels.count(0.0)
    count_1 = train_labels.count(1.0)

    if count_1 == 0:
        return nn.BCEWithLogitsLoss(), (count_0, count_1)

    imbalance_ratio = max(count_0, count_1) / (min(count_0, count_1) + 1e-6)
    if imbalance_ratio > 5.0 and min(count_0, count_1) >= 50:
        return FocalLoss(alpha=0.25, gamma=2.0), (count_0, count_1)

    pos_weight = torch.tensor([count_0 / (count_1 + 1e-6)], device=DEVICE, dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight), (count_0, count_1)


def ddp_broadcast_stopflag(stop_flag: torch.Tensor):
    # rank0에서 stop_flag 세팅 후 전 rank로 broadcast
    dist.broadcast(stop_flag, src=0)
    return stop_flag.item() == 1


def ddp_allreduce_confusion(tp: torch.Tensor, tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor):
    """All-reduce confusion counts across ranks (SUM)."""
    for t in (tp, tn, fp, fn):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return tp, tn, fp, fn


def compute_metrics_from_confusion(tp: int, tn: int, fp: int, fn: int):
    """Return (acc, f1) from confusion counts. Safe for empty/degenerate cases."""
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    denom = (2 * tp + fp + fn)
    f1 = (2 * tp) / denom if denom > 0 else 0.0
    return float(acc), float(f1)


def ddp_allreduce_loss(loss_sum: torch.Tensor, n_samples: torch.Tensor):
    """All-reduce loss sum and sample count (SUM)."""
    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(n_samples, op=dist.ReduceOp.SUM)
    return loss_sum, n_samples



def setup_mlflow(tracking_root: str):
    """Ensure MLflow file-store exists and experiment is created.

    Avoids MissingConfigException when mlruns/0/meta.yaml is missing.
    """
    if not USE_MLFLOW:
        return
    try:
        root = Path(tracking_root).resolve()
        mlruns_dir = root / "mlruns"
        mlruns_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"file:{mlruns_dir.as_posix()}")
        exp_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "default")
        mlflow.set_experiment(exp_name)  # creates experiment if missing
    except Exception as e:
        # Fallback: disable mlflow to keep training alive.
        if local_rank == 0:
            print(f"[WARN] MLflow setup failed -> disabling mlflow. err={e}")
        globals()["USE_MLFLOW"] = 0

def main():
    if local_rank == 0:
        print("\n🚀 Start Training (AP+LAT 통합, K-Fold CV)")
        print(f"   BASE_DIR={BASE_DIR}")
        print(f"   SAVE_DIR={SAVE_DIR}")
        print(f"   K_FOLDS={K_FOLDS}")
        print(f"   USE_AMP={USE_AMP}, USE_GRAD_CKPT={USE_GRAD_CKPT}")

    train_csv = os.path.join(BASE_DIR, "train.csv")
    val_csv = os.path.join(BASE_DIR, "val.csv")
    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        raise FileNotFoundError(f"train.csv 또는 val.csv가 {BASE_DIR}에 없습니다.")

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_all = pd.concat([df_train, df_val]).reset_index(drop=True)

    # label 컬럼
    if "fracture_visible" in df_all.columns:
        df_all["label"] = df_all["fracture_visible"].fillna(0).astype(float)
    elif "label" in df_all.columns:
        df_all["label"] = df_all["label"].fillna(0).astype(float)
    else:
        raise KeyError("CSV에 fracture_visible 또는 label 컬럼이 없습니다.")

    if "age" not in df_all.columns:
        raise KeyError("CSV에 age 컬럼이 없습니다.")
    df_all["age_group_label"] = df_all["age"].astype(float).apply(AGE_GROUP_FN)

    age_groups = [0, 1, 2, 3]
    if ONLY_AGE_GROUP != "":
        age_groups = [int(ONLY_AGE_GROUP)]

    for age_group in age_groups:
        if local_rank == 0:
            print(f"\n🔹 AgeGroup {age_group} | {K_FOLDS}-Fold CV")

        df_g = df_all[df_all["age_group_label"] == age_group].copy().reset_index(drop=True)
        if len(df_g) < 10 or df_g["label"].nunique() < 2:
            if local_rank == 0:
                print(f"⚠️ AgeGroup {age_group}: 데이터 부족, 스킵 (rows={len(df_g)})")
            continue

        skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

        # rank0에서만 fold summary 기록
        fold_rows = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df_g, df_g["label"])):
            if ONLY_FOLD != "" and fold_idx != int(ONLY_FOLD):
                continue

            if local_rank == 0:
                print(f"\n   [Fold {fold_idx}/{K_FOLDS-1}] start")

            df_train_g = df_g.iloc[train_idx].reset_index(drop=True)
            df_val_g = df_g.iloc[val_idx].reset_index(drop=True)

            train_dataset = UnifiedDataset(df_train_g, transform=transform, task="fracture_only")
            val_dataset = UnifiedDataset(df_val_g, transform=transform, task="fracture_only")

            # DistributedSampler (DDP)
            train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
            val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)

            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                sampler=train_sampler,
                num_workers=4,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                sampler=val_sampler,
                num_workers=4,
                pin_memory=True,
            )

            base_model = build_base_model()
            model = None  # DDP wrapper (recreated per stage segment)
            criterion, (count_0, count_1) = get_criterion(df_train_g["label"].tolist())
            scaler = GradScaler(enabled=USE_AMP)

            # stage plan (global setting)
            try:
                stage_plan = parse_stage_plan(STAGE_PLAN_STR) if USE_STAGEWISE else [("full", EPOCHS)]
            except Exception as e:
                if local_rank == 0:
                    print(f"[WARN] invalid STAGE_PLAN='{STAGE_PLAN_STR}', fallback to full:{EPOCHS}. err={e}")
                stage_plan = [("full", EPOCHS)]

            # total epochs for this fold (for logging only)
            total_epochs_fold = sum([n for _, n in stage_plan])
            global_epoch = 0

            # Build optimizer/scheduler per segment (created lazily per segment)
            optimizer = None
            seg_scheduler = None


            # MLflow run (rank0 only)
            best_f1 = 0.0
            patience_counter = 0
            if local_rank == 0 and USE_MLFLOW:
                setup_mlflow(SAVE_DIR)
                run_name = f"ConvNeXtV2_CV_Age{age_group}_Fold{fold_idx}"
                mlflow.start_run(run_name=run_name)
                mlflow.log_params({
                    "age_group": age_group,
                    "fold": fold_idx,
                    "k_folds": K_FOLDS,
                    "batch_size": BATCH_SIZE,
                    "epochs": EPOCHS,
                    "img_size": IMG_SIZE,
                    "lr": LR,
                    "model_name": MODEL_NAME,
                    "count_0": int(count_0),
                    "count_1": int(count_1),
                    "use_amp": int(USE_AMP),
                    "use_grad_ckpt": int(USE_GRAD_CKPT),
                    "use_stagewise": int(USE_STAGEWISE),
                    "stage_plan": STAGE_PLAN_STR,
                    "warmup_epochs": WARMUP_EPOCHS,
                    "min_lr_ratio": MIN_LR_RATIO,
                    "lr_head_mult": LR_HEAD_MULT,
                    "lr_last_mult": LR_LAST_MULT,
                    "lr_backbone_mult": LR_BACKBONE_MULT,
                })

            stop_flag = torch.zeros(1, device=DEVICE)

            stop_flag = torch.zeros(1, device=DEVICE)

            # =========================
            # Stage-wise fine-tuning loop
            # =========================
            for seg_name, seg_epochs in stage_plan:
                if stop_flag.item() == 1:
                    break

                # Re-wrap DDP at each segment because stage-wise changes the set of trainable parameters.
                if model is not None:
                    dist.barrier()
                    try:
                        del model
                    except Exception:
                        pass
                    torch.cuda.empty_cache()

                # set trainable params for this segment (apply on the underlying module)
                if USE_STAGEWISE:
                    set_trainable_stagewise(base_model, seg_name)
                    mode_for_groups = seg_name
                else:
                    set_trainable_stagewise(base_model, "full")
                    mode_for_groups = "full"

                model = wrap_ddp(base_model)

                # rebuild optimizer + per-segment scheduler
                param_groups = build_param_groups(model, LR, mode_for_groups)
                optimizer = torch.optim.AdamW(param_groups)
                seg_scheduler = WarmupCosinePerSegment(
                    optimizer=optimizer,
                    warmup_epochs=WARMUP_EPOCHS,
                    total_epochs=seg_epochs,
                    min_lr_ratio=MIN_LR_RATIO,
                )

                if local_rank == 0:
                    lrs = [g.get("lr", None) for g in optimizer.param_groups]
                    print(f"\n[SEGMENT] Age{age_group} Fold{fold_idx} seg='{seg_name}' epochs={seg_epochs} lrs={lrs}")

                for epoch_in_seg in range(seg_epochs):
                    if stop_flag.item() == 1:
                        break

                    # sampler epoch seed (must be global epoch)
                    train_sampler.set_epoch(global_epoch)

                    # LR schedule step (per segment)
                    seg_scheduler.step(epoch_in_seg)

                    model.train()
                    # aggregate confusion counts across all ranks
                    loss_sum = torch.zeros(1, device=DEVICE)
                    n_samples = torch.zeros(1, device=DEVICE, dtype=torch.long)
                    tp = torch.zeros(1, device=DEVICE, dtype=torch.long)
                    tn = torch.zeros(1, device=DEVICE, dtype=torch.long)
                    fp = torch.zeros(1, device=DEVICE, dtype=torch.long)
                    fn = torch.zeros(1, device=DEVICE, dtype=torch.long)

                    for images, targets in train_loader:
                        images = images.to(DEVICE, non_blocking=True)
                        targets = targets.to(DEVICE, non_blocking=True).float().view(-1)

                        optimizer.zero_grad(set_to_none=True)

                        with autocast(enabled=USE_AMP):
                            outputs = model(images).squeeze(dim=-1)
                            loss = criterion(outputs, targets)

                        scaler.scale(loss).backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        scaler.step(optimizer)
                        scaler.update()

                        bs = targets.numel()
                        loss_sum += loss.detach() * bs
                        n_samples += bs

                        pred = (torch.sigmoid(outputs.detach()) > 0.5).long().view(-1)
                        lab = targets.detach().long().view(-1)
                        tp += ((pred == 1) & (lab == 1)).sum()
                        tn += ((pred == 0) & (lab == 0)).sum()
                        fp += ((pred == 1) & (lab == 0)).sum()
                        fn += ((pred == 0) & (lab == 1)).sum()

                    ddp_allreduce_loss(loss_sum, n_samples)
                    ddp_allreduce_confusion(tp, tn, fp, fn)

                    mean_loss = (loss_sum.item() / max(int(n_samples.item()), 1))
                    acc, f1 = compute_metrics_from_confusion(
                        int(tp.item()), int(tn.item()), int(fp.item()), int(fn.item())
                    )

                    if local_rank == 0:
                        cur_lrs = [g.get("lr", None) for g in optimizer.param_groups]
                        print(
                            f"[Train] Age{age_group} Fold{fold_idx} G{global_epoch:03d} seg={seg_name} e={epoch_in_seg:03d} "
                            f"| Loss:{mean_loss:.6f} | Acc:{acc:.4f} | F1:{f1:.4f} | lrs={cur_lrs}"
                        )
                        mlflow.log_metric("train_loss", mean_loss, step=global_epoch)
                        mlflow.log_metric("train_acc", acc, step=global_epoch)
                        mlflow.log_metric("train_f1", f1, step=global_epoch)
                        mlflow.log_metric("seg_id", {"head": 0, "last": 1, "last3": 1, "last2": 2, "last1": 3, "full": 4}.get(seg_name, -1), step=global_epoch)

                    # Validation (all ranks -> reduce)
                    model.eval()
                    v_tp = torch.zeros(1, device=DEVICE, dtype=torch.long)
                    v_tn = torch.zeros(1, device=DEVICE, dtype=torch.long)
                    v_fp = torch.zeros(1, device=DEVICE, dtype=torch.long)
                    v_fn = torch.zeros(1, device=DEVICE, dtype=torch.long)
                    with torch.no_grad():
                        for images, targets in val_loader:
                            images = images.to(DEVICE, non_blocking=True)
                            targets = targets.to(DEVICE, non_blocking=True).float().view(-1)
                            with autocast(enabled=USE_AMP):
                                outputs = model(images).squeeze(dim=-1)
                                probs = torch.sigmoid(outputs)

                            pred = (probs > 0.5).long().view(-1)
                            lab = targets.detach().long().view(-1)
                            v_tp += ((pred == 1) & (lab == 1)).sum()
                            v_tn += ((pred == 0) & (lab == 0)).sum()
                            v_fp += ((pred == 1) & (lab == 0)).sum()
                            v_fn += ((pred == 0) & (lab == 1)).sum()

                    ddp_allreduce_confusion(v_tp, v_tn, v_fp, v_fn)
                    val_acc, val_f1 = compute_metrics_from_confusion(
                        int(v_tp.item()), int(v_tn.item()), int(v_fp.item()), int(v_fn.item())
                    )

                    if local_rank == 0:
                        print(
                            f"[Val]   Age{age_group} Fold{fold_idx} G{global_epoch:03d} seg={seg_name} e={epoch_in_seg:03d} "
                            f"| Acc:{val_acc:.4f} | F1:{val_f1:.4f}"
                        )
                        mlflow.log_metric("val_acc", val_acc, step=global_epoch)
                        mlflow.log_metric("val_f1", val_f1, step=global_epoch)

                        # Best save / Early stop (rank0 decides)
                        if val_f1 > best_f1 + ES_MIN_DELTA:
                            best_f1 = val_f1
                            patience_counter = 0

                            best_model_path = os.path.join(
                                SAVE_DIR,
                                f"best_ddp_convnextv2_AP_{age_group}_fold{fold_idx}.pt"
                            )
                            torch.save(model.module.state_dict(), best_model_path)
                            mlflow.log_artifact(best_model_path)
                            print(f"✅ Saved BEST model: {best_model_path}")
                        else:
                            patience_counter += 1
                            print(f"⏳ EarlyStopping patience: {patience_counter}/{ES_PATIENCE}")

                            if patience_counter >= ES_PATIENCE:
                                latest_model_path = os.path.join(
                                    SAVE_DIR,
                                    f"latest_ddp_convnextv2_AP_{age_group}_fold{fold_idx}.pt"
                                )
                                torch.save(model.module.state_dict(), latest_model_path)
                                mlflow.log_artifact(latest_model_path)
                                print(f"⛔ Early stop at global_epoch {global_epoch} | saved latest: {latest_model_path}")
                                stop_flag.fill_(1)

                    # sync stop_flag
                    if ddp_broadcast_stopflag(stop_flag):
                        break

                    global_epoch += 1

                # end segment

            # end fold (segments done)
            # fold 종료 처리
            if local_rank == 0:
                fold_rows.append({
                    "age_group": age_group,
                    "fold": fold_idx,
                    "best_val_f1": float(best_f1),
                    "epochs_ran": int(global_epoch),
                })
                if USE_MLFLOW:
                    mlflow.log_metric("best_val_f1", float(best_f1))
                    mlflow.end_run()
            # 메모리 정리 (DDP fold 반복)
            del model
            torch.cuda.empty_cache()
            dist.barrier()

        # age_group별 summary 저장 (rank0)
        if local_rank == 0 and fold_rows:
            out_csv = os.path.join(SAVE_DIR, f"cv_summary_age{age_group}.csv")
            with open(out_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["age_group", "fold", "best_val_f1", "epochs_ran"])
                writer.writeheader()
                writer.writerows(fold_rows)

            mean_f1 = sum(r["best_val_f1"] for r in fold_rows) / max(len(fold_rows), 1)
            print(f"\n✅ AgeGroup {age_group} CV summary saved: {out_csv}")
            print(f"   mean(best_val_f1)={mean_f1:.4f}")



if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure clean shutdown for NCCL even on exceptions
        try:
            dist.destroy_process_group()
        except Exception:
            pass