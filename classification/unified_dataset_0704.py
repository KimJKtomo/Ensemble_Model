import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage

class UnifiedFractureDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"❌ File not found: {img_path}")

        # ✅ OpenCV 이미지 로딩
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"❌ Unable to load image: {img_path}")
        h, w, _ = image.shape

        # ✅ 마스크 제거
        txt_path = img_path.replace(".png", ".txt").replace(".jpg", ".txt")
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, xc, yc, bw, bh = map(float, parts)
                    if int(cls) == 8:  # Mask 클래스
                        x1 = int((xc - bw / 2) * w)
                        y1 = int((yc - bh / 2) * h)
                        x2 = int((xc + bw / 2) * w)
                        y2 = int((yc + bh / 2) * h)
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)

        # ✅ RGB로 변환 후 PIL 변환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = ToPILImage()(image)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor([row["label"]]).float()
        return image, label


class UnifiedDataset(Dataset):
    def __init__(self, dataframe, transform=None, task="fracture"):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.task = task  # fracture_only, age_group, fracture 등

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        if not isinstance(img_path, str):
            img_path = str(img_path)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"❌ 이미지 경로 없음: {img_path}")

        # ✅ OpenCV 이미지 로딩 + 마스크 제거
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"❌ 이미지 로딩 실패: {img_path}")
        h, w, _ = image.shape

        txt_path = img_path.replace(".png", ".txt").replace(".jpg", ".txt")
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, xc, yc, bw, bh = map(float, parts)
                    if int(cls) == 8:
                        x1 = int((xc - bw / 2) * w)
                        y1 = int((yc - bh / 2) * h)
                        x2 = int((xc + bw / 2) * w)
                        y2 = int((yc + bh / 2) * h)
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = ToPILImage()(image)

        if self.transform:
            image = self.transform(image)

        if self.task == "fracture_only":
            label = float(row["label"])
        elif self.task == "age_group":
            label = float(row["age_group_label"])
        elif self.task == "fracture":
            label = float(row["label"])
        else:
            raise ValueError(f"❌ Unknown task mode: {self.task}")

        return image, torch.tensor(label).float()
