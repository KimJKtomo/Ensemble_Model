import os
import pandas as pd
from sklearn.model_selection import train_test_split
from load_new_dxmodule_0730 import get_combined_dataset_ao

def age_group_label(age):
    age = float(age)
    if age < 5: return 0
    elif age < 10: return 1
    elif age < 15: return 2
    else: return 3


SAVE_DIR = "./classification"
os.makedirs(SAVE_DIR, exist_ok=True)

print("📌 Step 2: Generating train/val splits...")

for proj in [1, 2]:  # 1: AP, 2: Lat
    print(f"📌 Step 2: Generating random train/val split by projection {proj}...")
    df = get_combined_dataset_ao(projection=proj)

    # ✅ 명시적으로 복사
    df_proj = df.copy()

    # ✅ 안전하게 컬럼 할당
    df_proj["label"] = df_proj["fracture_visible"].apply(lambda x: 1 if x == 1 else 0)
    df_proj["age_group_label"] = df_proj["age"].astype(float).apply(age_group_label)

    for group in range(4):  # age groups 0~3
        df_group = df_proj[df_proj["age_group_label"] == group]
        train_df, val_df = train_test_split(
            df_group, test_size=0.2, stratify=df_group["label"], random_state=42
        )
        train_path = os.path.join(SAVE_DIR, f"train_split_0730_proj{proj}_group{group}.csv")
        val_path = os.path.join(SAVE_DIR, f"val_split_0730_proj{proj}_group{group}.csv")
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        print(f"✅ Saved Train/Val splits for projection {proj} / age group {group}")
