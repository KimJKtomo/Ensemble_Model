# ddp_train_latefusion_from_roiindex.py
import os, re, warnings
from pathlib import Path

import pandas as pd
import torch, torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms as T
from PIL import Image
from timm import create_model
from sklearn.metrics import accuracy_score, f1_score

# -----------------------------
# Config
# -----------------------------
ROOT = "/mnt/data/KimJG/SwinT/mediaiOA_swinT_cls-main_original/20250623_Single/WristFX_0730/classification/roi_out"
CSV = {
    "train": f"{ROOT}/train/roi_index.csv",
    "valid": f"{ROOT}/valid/roi_index.csv",
    "test":  f"{ROOT}/test/roi_index.csv",
}
SAVE_PATH = f"{ROOT}/convnextv2_latefusion_ddp_best.pt"

BACKBONE = "convnextv2_base.fcmae_ft_in22k_in1k_384"
BATCH = 1
ACCUM_STEPS = 2
EPOCHS = 20
LR = 3e-4
WD = 0.05

STRICT_IMAGE_PAIR = int(os.environ.get("STRICT_IMAGE_PAIR", "1"))
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# -----------------------------
# Utils
# -----------------------------
_to_tensor = T.Compose([
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float32),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def _open_rgb(path):
    img = Image.open(path)
    if img.mode in ("I;16","I"):
        import numpy as np
        arr = np.array(img, dtype="uint16")
        lo, hi = np.percentile(arr, (1, 99)); hi = max(hi, lo + 1)
        arr = ((arr - lo) * 255.0 / (hi - lo)).clip(0, 255).astype("uint8")
        img = Image.fromarray(arr, mode="L")
    if img.mode in ("LA","P"):
        img = img.convert("RGBA")
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (0,0,0,255))
        img = Image.alpha_composite(bg, img).convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return img

def _canon_stem(p: str) -> str:
    s = Path(p).stem.split('.')[0]
    while re.search(r'(_\d+){4}$', s):  # _x1_y1_x2_y2 제거
        s = re.sub(r'(_\d+){4}$', '', s)
    for tag in ("_crop","-crop","_roi","-roi","_det","-det"):
        if s.endswith(tag):
            s = s[: -len(tag)]
    return s

def _pad(x, H, W):
    ph, pw = H - x.shape[-2], W - x.shape[-1]
    return F.pad(x, (0, pw, 0, ph))

# -----------------------------
# Dataset
# -----------------------------
class DualInputCSV(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        miss = [c for c in ("path_full","path_roi","label") if c not in self.df.columns]
        if miss:
            raise ValueError(f"CSV columns missing: {miss}")

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        gf, cf = str(r.path_full), str(r.path_roi)
        same = _canon_stem(gf) == _canon_stem(cf)
        if STRICT_IMAGE_PAIR:
            assert same, f"[PAIR MISMATCH] base({_canon_stem(gf)} vs {_canon_stem(cf)}) | full({gf}) | roi({cf})"
        elif not same:
            warnings.warn(f"[PAIR WARN] base({_canon_stem(gf)} vs {_canon_stem(cf)}) | full({gf}) | roi({cf})")
        g, c = _open_rgb(gf), _open_rgb(cf)
        return _to_tensor(g), _to_tensor(c), float(r.label)

def collate_dual_pad(batch):
    gs, cs, ys = zip(*batch)
    Hg = max(t.shape[-2] for t in gs); Wg = max(t.shape[-1] for t in gs)
    Hc = max(t.shape[-2] for t in cs); Wc = max(t.shape[-1] for t in cs)
    gs = torch.stack([_pad(t, Hg, Wg) for t in gs], 0)
    cs = torch.stack([_pad(t, Hc, Wc) for t in cs], 0)
    ys = torch.tensor(ys, dtype=torch.float32)
    return (gs, cs), ys

# -----------------------------
# Model
# -----------------------------
class ConvNeXtV2LateFusion(nn.Module):
    def __init__(self, backbone=BACKBONE):
        super().__init__()
        self.g = create_model(backbone, pretrained=True, num_classes=0, global_pool='avg')
        self.c = create_model(backbone, pretrained=True, num_classes=0, global_pool='avg')
        fdim = self.g.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(fdim * 2),
            nn.Dropout(0.2),
            nn.Linear(fdim * 2, 1)
        )
    def forward(self, xg, xc):
        fg = self.g(xg)
        fc = self.c(xc)
        z = torch.cat([fg, fc], 1)
        return self.head(z).squeeze(1)

# -----------------------------
# Eval (DDP-safe gather)
# -----------------------------
def evaluate_ddp(model, loader, device):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for (xg, xc), y in loader:
            xg = xg.to(device, non_blocking=True)
            xc = xc.to(device, non_blocking=True)
            y  = y.to(device, non_blocking=True)
            p = torch.sigmoid(model(xg, xc))
            preds += (p > 0.5).int().cpu().tolist()
            gts   += y.int().cpu().tolist()
    obj = {"p": preds, "g": gts}
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, obj)
    if dist.get_rank() == 0:
        import numpy as np
        P = np.concatenate([o["p"] for o in gathered])
        G = np.concatenate([o["g"] for o in gathered])
        return accuracy_score(G, P), f1_score(G, P)
    return None, None

# -----------------------------
# Train
# -----------------------------
def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    ds_train = DualInputCSV(CSV["train"])
    ds_val   = DualInputCSV(CSV["valid"])
    ds_test  = DualInputCSV(CSV["test"])

    train_sampler = DistributedSampler(ds_train, shuffle=True, drop_last=False)
    val_sampler   = DistributedSampler(ds_val,   shuffle=False, drop_last=False)
    test_sampler  = DistributedSampler(ds_test,  shuffle=False, drop_last=False)

    dl_kwargs = dict(num_workers=2, pin_memory=False, persistent_workers=True, collate_fn=collate_dual_pad)
    train_loader = DataLoader(ds_train, batch_size=BATCH, sampler=train_sampler, **dl_kwargs)
    val_loader   = DataLoader(ds_val,   batch_size=BATCH, sampler=val_sampler,   **dl_kwargs)
    test_loader  = DataLoader(ds_test,  batch_size=BATCH, sampler=test_sampler,  **dl_kwargs)

    y_train = pd.read_csv(CSV["train"])["label"].astype(int).tolist()
    pos_weight = torch.tensor([(y_train.count(0)+1e-6)/(y_train.count(1)+1e-6)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = ConvNeXtV2LateFusion().to(device)
    model = model.to(memory_format=torch.channels_last)  # 핵심

    # grad contiguous 훅
    def _make_grad_contiguous(module, grad_input, grad_output):
        for p in module.parameters():
            if p.grad is not None and not p.grad.is_contiguous():
                p.grad = p.grad.contiguous()
    model.register_full_backward_hook(_make_grad_contiguous)

    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        gradient_as_bucket_view=False,  # 핵심
    )

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scaler = torch.amp.GradScaler('cuda')

    is_main = dist.get_rank() == 0
    best_f1 = -1.0

    for epoch in range(EPOCHS):
        model.train(); train_sampler.set_epoch(epoch)
        total = 0.0
        opt.zero_grad(set_to_none=True)

        for step, ((xg, xc), y) in enumerate(train_loader, 1):
            xg = xg.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            xc = xc.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y  = y.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                logits = model(xg, xc)
                loss = criterion(logits, y) / ACCUM_STEPS

            scaler.scale(loss).backward()
            if is_main and step % 50 == 0:
                print(f"epoch {epoch} step {step} loss {(loss.item()*ACCUM_STEPS):.4f}", flush=True)

            if step % ACCUM_STEPS == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)

            total += loss.item()

        va, vf = evaluate_ddp(model, val_loader, device)
        if is_main:
            print(f"Epoch {epoch:03d} | train_loss {total:.3f} | val_acc {va} | val_f1 {vf}", flush=True)
            if vf is not None and vf > best_f1:
                best_f1 = vf
                torch.save(model.module.state_dict(), SAVE_PATH)
                print(f"saved: {SAVE_PATH}", flush=True)
        dist.barrier()

    if is_main:
        state = torch.load(SAVE_PATH, map_location=device)
        model.module.load_state_dict(state)
    dist.barrier()

    ta, tf = evaluate_ddp(model, test_loader, device)
    if is_main:
        print(f"[TEST] acc {ta:.4f} | f1 {tf:.4f}", flush=True)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
