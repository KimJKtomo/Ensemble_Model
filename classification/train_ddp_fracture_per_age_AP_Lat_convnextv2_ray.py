# train_ddp_fracture_per_age_AP_Lat_convnextv2_ray.py

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from ray import tune
from ray.tune import CLIReporter
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from timm import create_model
import torchvision.transforms as transforms
from unified_dataset_0704 import UnifiedDataset
from ray.air import session




# ✅ 기본 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 384
MODEL_NAME = "convnextv2_large.fcmae_ft_in22k_in1k_384"
EPOCHS = 300
PATIENCE = 15

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
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean() if self.reduction == "mean" else focal_loss.sum()

def train_ray_objective(config, proj, age_group, df_group):
    df_group = df_group.copy()
    df_group["label"] = df_group["fracture_visible"].fillna(0).astype(float)

    # ✅ Stratified train/val split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for train_idx, val_idx in splitter.split(df_group, df_group["label"]):
        df_train = df_group.iloc[train_idx]
        df_val = df_group.iloc[val_idx]

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = UnifiedDataset(df_train, transform=transform, task="fracture_only")
    val_dataset = UnifiedDataset(df_val, transform=transform, task="fracture_only")

    # ✅ Sampler with imbalance handling
    train_labels = df_train["label"].tolist()
    count_0 = train_labels.count(0.0)
    count_1 = train_labels.count(1.0)
    weight_0 = 1.0 / (count_0 + 1e-6)
    weight_1 = 1.0 / (count_1 + 1e-6)
    sample_weights = [weight_0 if l == 0.0 else weight_1 for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=12, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False, num_workers=4)

    # ✅ 모델 로딩
    model = create_model(MODEL_NAME, pretrained=True, num_classes=1, drop_path_rate=config["drop_path"]).to(DEVICE)

    # ✅ 손실 함수 선택
    if config["loss_type"] == "focal":
        criterion = FocalLoss(gamma=config["gamma"])
    else:
        pos_weight = torch.tensor([count_0 / (count_1 + 1e-6)]).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    best_f1 = 0.0
    wait = 0

    # ✅ Epoch 루프
    for epoch in range(EPOCHS):
        model.train()
        preds, targets_all = [], []
        total_loss = 0.0

        for images, targets in train_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE).view(-1)
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            preds += (torch.sigmoid(outputs) > 0.5).int().cpu().tolist()
            targets_all += targets.int().cpu().tolist()

        # ✅ Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(DEVICE)
                outputs = model(images).squeeze()
                probs = torch.sigmoid(outputs).cpu().numpy()
                val_preds += (probs > 0.5).astype(int).tolist()
                val_targets += targets.numpy().astype(int).tolist()

        val_f1 = f1_score(val_targets, val_preds)
        val_acc = accuracy_score(val_targets, val_preds)

        session.report({"f1": val_f1, "acc": val_acc})

        if val_f1 > best_f1:
            best_f1 = val_f1
            wait = 0
            torch.save(model.state_dict(), f"best_model_{proj}_{age_group}.pt")
        else:
            wait += 1
            if wait >= PATIENCE:
                break

def run_all():
    # ✅ 데이터 로드
    df_ap = pd.read_csv("age_train_tmp_AP.csv")
    df_lat = pd.read_csv("age_train_tmp_Lat.csv")
    df_ap["projection"] = "AP"
    df_lat["projection"] = "Lat"
    df = pd.concat([df_ap, df_lat])
    df["age_group_label"] = df["age"].apply(AGE_GROUP_FN)

    for proj in ["AP", "Lat"]:
        for age_group in [0, 1, 2, 3]:
            df_g = df[(df["projection"] == proj) & (df["age_group_label"] == age_group)]

            config = {
                "lr": tune.grid_search([1e-4, 1e-5, 5e-6]),
                "drop_path": tune.grid_search([0.1, 0.2, 0.3]),
                "loss_type": tune.grid_search(["bce", "focal"]),
                "gamma": tune.grid_search([2.0, 3.0])
            }

            tune.run(
                tune.with_parameters(train_ray_objective, proj=proj, age_group=age_group, df_group=df_g),
                config=config,
                metric="f1",
                mode="max",
                name=f"Tune_{proj}_{age_group}",
                local_dir="./ray_results",
                resources_per_trial={"cpu": 4, "gpu": 1},
                num_samples=1,
                progress_reporter=CLIReporter(metric_columns=["f1", "acc"])
            )

if __name__ == "__main__":
    run_all()
