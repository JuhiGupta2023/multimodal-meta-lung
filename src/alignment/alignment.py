# src/alignment/alignment_train.py
"""
Alignment pretraining script (2D <-> 3D contrastive alignment).
Drop-in replacement built from the user's original file; robust checkpoints,
config support, and stable InfoNCE implementation.

Usage examples:
    python alignment_train.py --cfg configs/alignment_config.yaml
    python alignment_train.py --data_root "C:/path/to/dataset" --out "./checkpoints/alignment/final_aligned.pt"

Expectations:
 - dataset folder structure matches the repo (BMP_2D, MHD_3D, etc.)
 - requirements: torch, torchvision, simpleitk, opencv-python, tqdm, pyyaml
"""
import os, glob, math, random, xml.etree.ElementTree as ET, argparse, time, yaml
import numpy as np
import cv2
import SimpleITK as sitk
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

ROOT = os.environ.get("DATA_ROOT", r"C:\Users\juhi\Desktop\2ndTechnical\dataset")

# ------------------- DEFAULT PATHS (can be overridden by cfg / CLI) -------------------
ROOT_DEFAULT = r"C:\Users\juhi\Desktop\2ndTechnical\dataset"
BMP2D_IMG_DIR_DEFAULT = os.path.join(ROOT_DEFAULT, "BMP_2D", "BMP_2D", "Image")
BMP2D_ANN_DIR_DEFAULT = os.path.join(ROOT_DEFAULT, "BMP_2D", "BMP_2D", "Annotations")
MHD_DIR_DEFAULT       = os.path.join(ROOT_DEFAULT, "MHD_3D", "MHD_3D")
SSL_BACKBONE_DEFAULT  = os.path.join(ROOT_DEFAULT, "ssl_backbone.pt")
CHECKPOINT_DIR_DEFAULT = os.path.join(".", "checkpoints", "alignment")
OUT_ALIGNED_DEFAULT = os.path.join(CHECKPOINT_DIR_DEFAULT, "final_aligned.pt")

# ------------------------ small helpers and I/O ------------------------
def parse_xml_box(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        obj = root.find('object')
        if obj is None:
            return None
        bb = obj.find('bndbox')
        xmin = int(bb.find('xmin').text)
        ymin = int(bb.find('ymin').text)
        xmax = int(bb.find('xmax').text)
        ymax = int(bb.find('ymax').text)
        return xmin, ymin, xmax, ymax
    except Exception:
        return None

def load_mhd(path):
    img = sitk.ReadImage(path)
    vol = sitk.GetArrayFromImage(img).astype(np.float32)  # [z,y,x]
    v = (vol - np.mean(vol)) / (np.std(vol) + 1e-6)
    return v

def safe_crop_2d(img, cx, cy, sz=160):
    h, w = img.shape[:2]
    x1 = max(0, cx - sz//2); x2 = min(w, cx + sz//2)
    y1 = max(0, cy - sz//2); y2 = min(h, cy + sz//2)
    crop = img[y1:y2, x1:x2]
    top = max(0, sz - crop.shape[0]); left = max(0, sz - crop.shape[1])
    if top>0 or left>0:
        crop = cv2.copyMakeBorder(crop, 0, top, 0, left, borderType=cv2.BORDER_REPLICATE)
    crop = cv2.resize(crop, (224,224), interpolation=cv2.INTER_LINEAR)
    return crop

def safe_cube_3d(vol, cz, cy, cx, d=64, h=64, w=64):
    Z,Y,X = vol.shape
    z1 = max(0, cz - d//2); z2 = min(Z, cz + d//2)
    y1 = max(0, cy - h//2); y2 = min(Y, cy + h//2)
    x1 = max(0, cx - w//2); x2 = min(X, cx + w//2)
    cube = vol[z1:z2, y1:y2, x1:x2]
    pad = [(0, d - cube.shape[0]), (0, h - cube.shape[1]), (0, w - cube.shape[2])]
    pad = [(max(0,a), max(0,b)) for a,b in pad]
    cube = np.pad(cube, pad, mode='edge')
    return cube  # [D,H,W]

# ------------------------ Dataset ------------------------
class Paired2D3D(Dataset):
    def __init__(self, img_dir, ann_dir, mhd_dir, split_files=None, depth=64):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.mhd_dir = mhd_dir
        self.depth = depth

        all_bmps = sorted(glob.glob(os.path.join(img_dir, "*.bmp")))
        if split_files is not None:
            split_set = set(split_files)
            all_bmps = [p for p in all_bmps if os.path.splitext(os.path.basename(p))[0] in split_set]

        self.items = []
        for bmp_path in all_bmps:
            stem = os.path.splitext(os.path.basename(bmp_path))[0]
            xml_path = os.path.join(ann_dir, stem + ".xml")
            if not os.path.exists(xml_path): 
                continue
            box = parse_xml_box(xml_path)
            if box is None: 
                continue
            pid = int(stem.split("_")[0])
            z_idx = int(stem.split("_")[1])
            mhd_path = os.path.join(mhd_dir, f"{pid}.mhd")
            if not os.path.exists(mhd_path):
                continue
            self.items.append((bmp_path, xml_path, mhd_path, z_idx))

        self.t2d = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        bmp_path, xml_path, mhd_path, z_idx = self.items[i]
        img = cv2.imread(bmp_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        xmin, ymin, xmax, ymax = parse_xml_box(xml_path)
        cx = (xmin + xmax)//2; cy = (ymin + ymax)//2
        crop2d = safe_crop_2d(img, cx, cy, sz=160)
        x2d = self.t2d(crop2d)  # [3,224,224]

        vol = load_mhd(mhd_path)
        cz = np.clip(z_idx, 0, vol.shape[0]-1)
        cube = safe_cube_3d(vol, cz, cy, cx, d=self.depth, h=64, w=64)
        x3d = torch.from_numpy(cube).unsqueeze(0).float()  # [1,D,H,W]

        # random 3d flips (inplace)
        if random.random() < 0.5: x3d = torch.flip(x3d, dims=[2])
        if random.random() < 0.5: x3d = torch.flip(x3d, dims=[3])
        if random.random() < 0.5 and x3d.shape[1] > 1: x3d = torch.flip(x3d, dims=[1])

        return x2d, x3d

# ------------------------ Encoders ------------------------
class Encoder2D(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        try:
            # prefer explicit API to avoid backward incompat
            m = models.resnet18(weights=None)
        except Exception:
            m = models.resnet18(pretrained=False)
        m.fc = nn.Identity()
        self.backbone = m
        self.proj = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, proj_dim)
        )
    def forward(self, x):
        h = self.backbone(x)
        z = F.normalize(self.proj(h), dim=1)
        return h, z

class BasicBlock3D(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(c_out)
        self.conv2 = nn.Conv3d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(c_out)
        self.down = None
        if c_in!=c_out or stride!=1:
            self.down = nn.Sequential(nn.Conv3d(c_in, c_out, 1, stride=stride, bias=False),
                                      nn.BatchNorm3d(c_out))
    def forward(self,x):
        idn = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.down is not None: idn = self.down(idn)
        return F.relu(x + idn)

class Backbone3D(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1,32,7,stride=2,padding=3,bias=False),
                                   nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d(3,stride=2,padding=1))
        self.l1 = BasicBlock3D(32,64, stride=2)
        self.l2 = BasicBlock3D(64,128, stride=2)
        self.l3 = BasicBlock3D(128,256, stride=2)
        self.pool = nn.AdaptiveAvgPool3d(1)
    def forward(self,x):
        x = self.stem(x); x = self.l1(x); x = self.l2(x); x = self.l3(x)
        return self.pool(x).flatten(1)  # [B,256]

class Encoder3D(nn.Module):
    def __init__(self, ssl_ckpt=None, proj_dim=128, feat_dim=256):
        super().__init__()
        self.backbone = Backbone3D()
        self.feat_dim = feat_dim
        # try load checkpoint safely
        if ssl_ckpt is not None and os.path.exists(ssl_ckpt):
            try:
                ck = torch.load(ssl_ckpt, map_location="cpu")
                if isinstance(ck, dict):
                    # flex keys
                    cand_keys = ["enc3d_state", "state_dict", "model_state_dict"]
                    state = None
                    for k in cand_keys:
                        if k in ck:
                            state = ck[k]; break
                    if state is None: state = ck
                else:
                    state = ck
                # try strict then lenient
                try:
                    self.backbone.load_state_dict(state)
                    print("Loaded 3D backbone state (strict).")
                except Exception:
                    new_state = {k.replace("module.", ""):v for k,v in state.items()}
                    self.backbone.load_state_dict(new_state, strict=False)
                    print("Loaded 3D backbone state (lenient).")
            except Exception as e:
                print("Warning: failed to load SSL ckpt:", e)
        else:
            if ssl_ckpt is not None:
                print("Warning: ssl_ckpt specified but file not found:", ssl_ckpt)
            else:
                print("No ssl_ckpt provided — backbone initialized randomly.")

        self.proj = nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim),
                                  nn.ReLU(inplace=True), nn.Linear(self.feat_dim, proj_dim))

    def forward(self, x):
        out = self.backbone(x)
        if isinstance(out, (list,tuple)): out = out[0]
        if out.dim() > 2:
            spatial_dims = tuple(range(2, out.dim()))
            feat = out.mean(dim=spatial_dims)
        else:
            feat = out
        if feat.size(1) != self.feat_dim:
            if not hasattr(self, "_feat_adaptor"):
                self._feat_adaptor = nn.Linear(feat.size(1), self.feat_dim).to(feat.device)
            feat = self._feat_adaptor(feat)
        proj = self.proj(feat)
        return feat, proj

# ------------------------ InfoNCE (stable cross-modal) ------------------------
def info_nce(z2d, z3d, temp=0.2):
    # z2d: [B, D], z3d: [B, D] (same proj_dim)
    # normalize
    z2 = F.normalize(z2d, dim=1)
    z3 = F.normalize(z3d, dim=1)
    B = z2.size(0)
    z = torch.cat([z2, z3], dim=0)  # [2B, D]
    sim = torch.matmul(z, z.t()) / temp
    # mask out self-sim
    mask = torch.eye(2*B, device=sim.device).bool()
    sim.masked_fill_(mask, -1e9)
    # positive pairs: i <-> i+B and i+B <-> i
    labels = torch.arange(2*B, device=sim.device)
    pos_idx = (labels + B) % (2*B)  # positive index for each row
    # cross-entropy: treat pos_idx as target by converting sim-> logits where target index is pos_idx
    loss = F.cross_entropy(sim, pos_idx)
    return loss

# ------------------------ main training script ------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, default="configs/alignment_config.yaml")
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def load_cfg(path):
    if not os.path.exists(path): return {}
    with open(path, "r") as f:
        return yaml.safe_load(f)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()
    cfg = load_cfg(args.cfg)
    data_root = args.data_root or cfg.get("data_root", ROOT_DEFAULT)
    bmp_dir = cfg.get("bmp_dir", BMP2D_IMG_DIR_DEFAULT)
    ann_dir = cfg.get("ann_dir", BMP2D_ANN_DIR_DEFAULT)
    mhd_dir = cfg.get("mhd_dir", MHD_DIR_DEFAULT)
    ssl_ckpt = cfg.get("ssl_backbone", SSL_BACKBONE_DEFAULT)
    out_path = args.out or cfg.get("out", OUT_ALIGNED_DEFAULT)
    epochs = int(cfg.get("epochs", 10))
    batch_size = int(cfg.get("batch_size", 6))
    lr = float(cfg.get("lr", 1e-3))
    proj_dim = int(cfg.get("proj_dim", 128))
    depth = int(cfg.get("depth", 64))
    freeze_2d = bool(cfg.get("freeze_2d", True))
    workers = int(args.workers)
    temp = float(cfg.get("temp", 0.2))

    set_seed(int(args.seed))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # read optional split file list
    def read_list(txt_path):
        if txt_path is None: return None
        if not os.path.exists(txt_path): return None
        with open(txt_path,"r") as f:
            return [line.strip() for line in f if line.strip()]
    train_list = read_list(os.path.join(data_root, "BMP_2D", "ImageSets", "train.txt"))

    ds = Paired2D3D(os.path.join(data_root, bmp_dir) if not os.path.isabs(bmp_dir) else bmp_dir,
                    os.path.join(data_root, ann_dir) if not os.path.isabs(ann_dir) else ann_dir,
                    os.path.join(data_root, mhd_dir) if not os.path.isabs(mhd_dir) else mhd_dir,
                    split_files=train_list, depth=depth)
    n = len(ds)
    print("Paired samples:", n)
    if n == 0:
        raise RuntimeError("Paired2D3D returned zero samples. Check dataset paths and split file.")

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=torch.cuda.is_available())

    enc2d = Encoder2D(proj_dim=proj_dim).to(device)
    enc3d = Encoder3D(ssl_ckpt=ssl_ckpt, proj_dim=proj_dim).to(device)

    # freezing logic
    if ssl_ckpt is not None and os.path.exists(ssl_ckpt):
        print("SSL checkpoint found; freezing 3D backbone for pure alignment.")
        for p in enc3d.backbone.parameters(): p.requires_grad = False
    else:
        print("No SSL 3D ckpt -> 3D backbone will be trainable.")
        for p in enc3d.backbone.parameters(): p.requires_grad = True

    for p in enc2d.backbone.parameters():
        p.requires_grad = not freeze_2d if hasattr(enc2d.backbone, 'parameters') else False
    if freeze_2d:
        print("Freezing 2D backbone (config).")

    # optimizer: projection heads always trainable; include enc3d params if trainable
    params = list(enc2d.proj.parameters()) + list(enc3d.proj.parameters())
    for p in enc3d.backbone.parameters():
        if p.requires_grad: params.append(p)

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    latest_ckpt = os.path.join(os.path.dirname(out_path), "latest.pt")
    best_ckpt = os.path.join(os.path.dirname(out_path), "best.pt")
    best_loss = float("inf")
    start_epoch = 1

    if os.path.exists(latest_ckpt):
        print("Resuming from latest checkpoint:", latest_ckpt)
        ck = torch.load(latest_ckpt, map_location=device)
        try:
            enc2d.load_state_dict(ck["enc2d_state"])
            enc3d.load_state_dict(ck["enc3d_state"])
            opt.load_state_dict(ck["optimizer_state"])
            start_epoch = ck.get("epoch", 0) + 1
            best_loss = ck.get("best_loss", best_loss)
            print(f"Resumed epoch {ck.get('epoch')} best_loss={best_loss:.4f}")
        except Exception as e:
            print("Warning: resume failed to load all keys:", e)

    # training loop
    for ep in range(start_epoch, epochs + 1):
        enc2d.train(); enc3d.train()
        running = 0.0
        loop = tqdm(dl, desc=f"Align ep {ep}/{epochs}")
        for x2d, x3d in loop:
            x2d = x2d.to(device, non_blocking=True)
            x3d = x3d.to(device, non_blocking=True)
            _, z2d = enc2d(x2d)
            _, z3d = enc3d(x3d)
            # normalize
            z2d = F.normalize(z2d, dim=1)
            z3d = F.normalize(z3d, dim=1)
            loss = info_nce(z2d, z3d, temp=temp)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * x2d.size(0)
            loop.set_postfix({"loss": float(loss.item())})

        avg_loss = running / max(1, n)
        scheduler.step()
        print(f"Epoch {ep}: avg alignment loss = {avg_loss:.4f}")

        torch.save({
            "epoch": ep,
            "enc2d_state": enc2d.state_dict(),
            "enc3d_state": enc3d.state_dict(),
            "optimizer_state": opt.state_dict(),
            "best_loss": best_loss,
        }, latest_ckpt)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": ep,
                "enc2d_state": enc2d.state_dict(),
                "enc3d_state": enc3d.state_dict(),
                "optimizer_state": opt.state_dict(),
                "best_loss": best_loss,
            }, best_ckpt)
            print(f"✅ New best loss: {best_loss:.4f} — saved best checkpoint")

    # final save
    torch.save({
        "enc2d_state": enc2d.state_dict(),
        "enc3d_state": enc3d.state_dict(),
    }, out_path)
    print("✅ Training complete. Final aligned encoders saved to:", out_path)

if __name__ == "__main__":
    main()
