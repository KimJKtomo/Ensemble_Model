# train_ddp_fracture_per_agegroup_convnextv2_0917_latefusion.py
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
import torchvision.transforms as transforms
from timm import create_model
import mlflow
from unified_dataset_0704 import UnifiedDataset

# =========================
# DDP init
# =========================
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")
DEVICE = torch.device("cuda", local_rank)
torch.backends.cudnn.benchmark = True

# =========================
# Paths / Config
# =========================
BASE_DIR = os.path.dirname(__file__)
BATCH_SIZE = 6
EPOCHS = 50
IMG_G, IMG_C = 448, 224  # Global / Crop
LR = 1e-4
MODEL_NAME = "convnextv2_large.fcmae_ft_in22k_in1k_384"

def AGE_GROUP_FN(age):
    age = float(age)
    if age < 5: return 0
    elif age < 10: return 1
    elif age < 15: return 2
    else: return 3

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

# =========================
# Transforms
# =========================
tfm_g = transforms.Compose([
    transforms.Resize((IMG_G, IMG_G)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
tfm_c = transforms.Compose([
    # TODO: ROI 좌표 보유 시 → 좌표 crop 후 Resize로 교체
    transforms.Resize((IMG_C, IMG_C)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# =========================
# Dataset wrapper: (global, crop, label)
# =========================
class DualInputDataset(torch.utils.data.Dataset):
    def __init__(self, df, task):
        self.base = UnifiedDataset(df, transform=None, task=task)
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        img, y = self.base[i]  # PIL or Tensor
        pil = img if hasattr(img, "size") else transforms.ToPILImage()(img)
        g = tfm_g(pil)
        c = tfm_c(pil)  # 임시: 원본에서 축소. ROI 있으면 교체
        return g, c, float(y)

# =========================
# Late-fusion Model: ConvNeXt 두 백본 → 임베딩 concat → FC
# =========================
class ConvNeXtLateFusion(nn.Module):
    def __init__(self, backbone=MODEL_NAME, num_classes=1, shared=False, drop_p=0.2):
        super().__init__()
        self.g = create_model(backbone, pretrained=True, num_classes=0, global_pool='avg')
        self.c = self.g if shared else create_model(backbone, pretrained=True, num_classes=0, global_pool='avg')
        F = self.g.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(F*2),
            nn.Dropout(drop_p),
            nn.Linear(F*2, num_classes)
        )
    def forward(self, xg, xc):
        fg = self.g(xg)  # [B,F]
        fc = self.c(xc)  # [B,F]
        z = torch.cat([fg, fc], dim=1)
        return self.head(z).squeeze(-1)

try:
    for proj in ["AP", "Lat"]:
        if local_rank == 0:
            print(f"\n🚀 Start Training Projection: {proj}")

        train_csv = os.path.join(BASE_DIR, f"age_train_tmp_{proj}.csv")
        val_csv   = os.path.join(BASE_DIR, f"age_val_tmp_{proj}.csv")

        df_train = pd.read_csv(train_csv)
        df_val   = pd.read_csv(val_csv)
        df_all   = pd.concat([df_train, df_val]).reset_index(drop=True)

        df_all["label"] = df_all["fracture_visible"].fillna(0).astype(float)
        df_all["age_group_label"] = df_all["age"].astype(float).apply(AGE_GROUP_FN)

        for age_group in [0, 1, 2, 3]:
            if local_rank == 0:
                print(f"\n🔹 Training {proj}_{age_group}...")

            df_g = df_all[df_all["age_group_label"] == age_group].copy()

            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
            for train_idx, val_idx in splitter.split(df_g, df_g["label"]):
                df_train_g = df_g.iloc[train_idx]
                df_val_g   = df_g.iloc[val_idx]

            # Datasets / Loaders
            train_dataset = DualInputDataset(df_train_g, task="fracture_only")
            val_dataset   = DualInputDataset(df_val_g,   task="fracture_only")

            train_labels = df_train_g["label"].tolist()
            count_0, count_1 = train_labels.count(0.0), train_labels.count(1.0)
            weight_0, weight_1 = 1.0 / (count_0 + 1e-6), 1.0 / (count_1 + 1e-6)
            sample_weights = [weight_0 if l == 0.0 else weight_1 for l in train_labels]
            weighted_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=weighted_sampler,
                                      num_workers=4, pin_memory=True)
            val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                                      num_workers=4, pin_memory=True)

            # Model / Loss / Opt
            model = ConvNeXtLateFusion(backbone=MODEL_NAME, num_classes=1, shared=False).to(DEVICE)
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)

            if count_1 == 0:
                criterion = nn.BCEWithLogitsLoss()
            else:
                imbalance_ratio = max(count_0, count_1) / (min(count_0, count_1) + 1e-6)
                if imbalance_ratio > 5.0 and min(count_0, count_1) >= 50:
                    criterion = FocalLoss(alpha=0.25, gamma=2.0)
                else:
                    pos_weight = torch.tensor([count_0 / (count_1 + 1e-6)]).to(DEVICE)
                    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

            # MLflow
            if local_rank == 0:
                run_name = f"ConvNeXtV2_LateFusion_{proj}_Age{age_group}"
                mlflow.start_run(run_name=run_name)
                best_f1 = 0.0

            # Train
            for epoch in range(EPOCHS):
                model.train()
                total_loss, preds, labels = 0.0, [], []

                for g, c, targets in train_loader:
                    g = g.to(DEVICE, non_blocking=True)
                    c = c.to(DEVICE, non_blocking=True)
                    targets = targets.to(DEVICE).float().view(-1)

                    optimizer.zero_grad(set_to_none=True)
                    outputs = model(g, c)  # [B]
                    loss = criterion(outputs, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

                    total_loss += loss.item()
                    probs = torch.sigmoid(outputs).detach().cpu().numpy()
                    preds += (probs > 0.5).astype(int).tolist()
                    labels += targets.cpu().numpy().astype(int).tolist()

                if local_rank == 0:
                    acc = accuracy_score(labels, preds)
                    f1  = f1_score(labels, preds)
                    print(f"[Train] {proj}_{age_group} Epoch {epoch} | Loss: {total_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")
                    mlflow.log_metric("train_acc", acc, step=epoch)
                    mlflow.log_metric("train_f1",  f1,  step=epoch)

                    # Validation
                    model.eval()
                    val_preds, val_labels = [], []
                    with torch.no_grad():
                        for g, c, targets in val_loader:
                            g = g.to(DEVICE, non_blocking=True)
                            c = c.to(DEVICE, non_blocking=True)
                            targets = targets.to(DEVICE).float().view(-1)
                            outputs = model(g, c)
                            probs = torch.sigmoid(outputs).cpu().numpy()
                            val_preds  += (probs > 0.5).astype(int).tolist()
                            val_labels += targets.cpu().numpy().astype(int).tolist()

                    val_acc = accuracy_score(val_labels, val_preds)
                    val_f1  = f1_score(val_labels, val_preds)
                    print(f"[Val] {proj}_{age_group} Epoch {epoch} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
                    mlflow.log_metric("val_acc", val_acc, step=epoch)
                    mlflow.log_metric("val_f1",  val_f1,  step=epoch)

                    if val_f1 > best_f1:
                        best_f1 = val_f1
                        model_path = os.path.join(BASE_DIR, f"0917_ddp_convnextv2_{proj}_{age_group}.pt")
                        torch.save(model.module.state_dict(), model_path)
                        mlflow.log_artifact(model_path)
                        print(f"✅ Saved: {model_path}")

            if local_rank == 0:
                mlflow.end_run()

finally:
    if dist.is_initialized():
        dist.destroy_process_group()
