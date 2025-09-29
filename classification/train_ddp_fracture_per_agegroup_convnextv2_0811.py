import os
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedShuffleSplit
import torchvision.transforms as transforms
from timm import create_model
import mlflow
import matplotlib.pyplot as plt
from unified_dataset_0704 import UnifiedDataset

# ✅ DDP 초기화
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")
DEVICE = torch.device("cuda", local_rank)

# ✅ 경로 설정
BASE_DIR = os.path.dirname(__file__)
IMG_SIZE = 384
MODEL_NAME = "convnextv2_large.fcmae_ft_in22k_in1k_384"

MODEL_SAVE_BASE = os.path.join(BASE_DIR, "0805_Model")
CM_SAVE_BASE = os.path.join(BASE_DIR, "0805_ConfusionMatrix")
os.makedirs(MODEL_SAVE_BASE, exist_ok=True)
os.makedirs(CM_SAVE_BASE, exist_ok=True)

# ✅ Age Group 함수
def AGE_GROUP_FN(age):
    age = float(age)
    if age < 5: return 0
    elif age < 10: return 1
    elif age < 15: return 2
    else: return 3

# ✅ Focal Loss 정의
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean() if self.reduction == "mean" else focal_loss.sum()

# ✅ Transform 정의
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ✅ JSON에서 Best Config 로드
CONFIG_PATH = "/home/kimgk3793/ray_results/best_summary.json"
with open(CONFIG_PATH, "r") as f:
    best_cfg_json = json.load(f)

CONFIG = {}
if isinstance(best_cfg_json, dict):
    for key, params in best_cfg_json.items():
        proj, age_group = key.split("_")
        CONFIG[(proj, int(age_group))] = {
            "lr": float(params["lr"]),
            "drop_path": float(params["drop_path"]),
            "loss": params["loss_type"],
            "gamma": float(params.get("gamma", 2.0))
        }
elif isinstance(best_cfg_json, list):
    for entry in best_cfg_json:
        if len(entry) == 3:
            proj, age_group, params = entry
        elif len(entry) == 2 and isinstance(entry[0], (list, tuple)):
            proj, age_group = entry[0]
            params = entry[1]
        else:
            raise ValueError(f"지원하지 않는 포맷: {entry}")
        CONFIG[(proj, int(age_group))] = {
            "lr": float(params["lr"]),
            "drop_path": float(params["drop_path"]),
            "loss": params["loss_type"],
            "gamma": float(params.get("gamma", 2.0))
        }
else:
    raise ValueError("best_summary.json 포맷을 알 수 없습니다.")

if local_rank == 0:
    print("✅ Loaded Best Config:")
    for k, v in CONFIG.items():
        print(f"{k}: {v}")

# ✅ 학습 루프
for proj in ["AP", "Lat"]:
    if local_rank == 0:
        print(f"\n🚀 Start Training Projection: {proj}")

    train_csv = os.path.join(BASE_DIR, f"age_train_tmp_{proj}.csv")
    val_csv = os.path.join(BASE_DIR, f"age_val_tmp_{proj}.csv")

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_all = pd.concat([df_train, df_val]).reset_index(drop=True)

    df_all["label"] = df_all["fracture_visible"].fillna(0).astype(float)
    df_all["age_group_label"] = df_all["age"].astype(float).apply(AGE_GROUP_FN)

    for age_group in [0, 1, 2, 3]:
        if local_rank == 0:
            print(f"\n🔹 Training {proj}_{age_group}...")

        cfg = CONFIG[(proj, age_group)]
        lr = cfg["lr"]
        drop_path = cfg["drop_path"]

        df_g = df_all[df_all["age_group_label"] == age_group].copy()
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        for train_idx, val_idx in splitter.split(df_g, df_g["label"]):
            df_train_g = df_g.iloc[train_idx]
            df_val_g = df_g.iloc[val_idx]

        train_dataset = UnifiedDataset(df_train_g, transform=transform, task="fracture_only")
        val_dataset = UnifiedDataset(df_val_g, transform=transform, task="fracture_only")

        train_labels = df_train_g["label"].tolist()
        count_0, count_1 = train_labels.count(0.0), train_labels.count(1.0)
        weight_0, weight_1 = 1.0 / (count_0 + 1e-6), 1.0 / (count_1 + 1e-6)
        sample_weights = [weight_0 if l == 0.0 else weight_1 for l in train_labels]
        weighted_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        train_loader = DataLoader(train_dataset, batch_size=12, sampler=weighted_sampler, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False, num_workers=4, pin_memory=True)

        model = create_model(MODEL_NAME, pretrained=True, num_classes=1, drop_path_rate=drop_path).to(DEVICE)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        # ✅ Loss 설정
        if cfg["loss"] == "focal":
            criterion = FocalLoss(alpha=0.25, gamma=cfg["gamma"])
        else:
            pos_weight = torch.tensor([count_0 / (count_1 + 1e-6)]).to(DEVICE)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        if local_rank == 0:
            run_name = f"ConvNeXtV2_{proj}_Age{age_group}"
            mlflow.start_run(run_name=run_name)
            best_f1 = 0.0

        for epoch in range(50):  # Epoch 고정
            model.train()
            total_loss, preds, labels = 0, [], []

            for images, targets in train_loader:
                images, targets = images.to(DEVICE), targets.to(DEVICE).float().view(-1)
                optimizer.zero_grad()
                outputs = model(images).squeeze(dim=-1)
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
                f1 = f1_score(labels, preds)
                mlflow.log_metric("train_acc", acc, step=epoch)
                mlflow.log_metric("train_f1", f1, step=epoch)

                # Confusion Matrix 저장 (Train)
                cm_train = confusion_matrix(labels, preds)
                cm_dir = os.path.join(CM_SAVE_BASE, f"{proj}_{age_group}", f"epoch_{epoch}")
                os.makedirs(cm_dir, exist_ok=True)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm_train)
                disp.plot(cmap=plt.cm.Blues)
                plt.title(f"Train Confusion Matrix - {proj} Age {age_group} Epoch {epoch}")
                plt.savefig(os.path.join(cm_dir, "train_cm.png"))
                plt.close()

                # ✅ Validation
                model.eval()
                val_preds, val_labels = [], []
                with torch.no_grad():
                    for images, targets in val_loader:
                        images = images.to(DEVICE)
                        targets = targets.to(DEVICE).float().view(-1)
                        outputs = model(images).squeeze(dim=-1)
                        probs = torch.sigmoid(outputs).cpu().numpy()
                        val_preds += (probs > 0.5).astype(int).tolist()
                        val_labels += targets.cpu().numpy().astype(int).tolist()

                val_acc = accuracy_score(val_labels, val_preds)
                val_f1 = f1_score(val_labels, val_preds)
                mlflow.log_metric("val_acc", val_acc, step=epoch)
                mlflow.log_metric("val_f1", val_f1, step=epoch)

                # Confusion Matrix 저장 (Val)
                cm_val = confusion_matrix(val_labels, val_preds)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm_val)
                disp.plot(cmap=plt.cm.Blues)
                plt.title(f"Val Confusion Matrix - {proj} Age {age_group} Epoch {epoch}")
                plt.savefig(os.path.join(cm_dir, "val_cm.png"))
                plt.close()

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    model_dir = os.path.join(MODEL_SAVE_BASE, f"{proj}_{age_group}")
                    os.makedirs(model_dir, exist_ok=True)
                    model_path = os.path.join(model_dir, f"best_epoch{epoch}.pt")
                    torch.save(model.module.state_dict(), model_path)
                    mlflow.log_artifact(model_path)
                    print(f"✅ Saved Best Model: {model_path}")

        if local_rank == 0:
            mlflow.end_run()

# ✅ 종료
dist.destroy_process_group()
