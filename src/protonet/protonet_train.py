# src/protonet/protonet_train.py
"""
Prototypical Network training (episodic) for multimodal fusion.
Saves: best head checkpoint, embeddings CSVs, failure_ranking.csv

Usage:
  python protonet_train.py --data-root /path/to/dataset --aligned-ckpt ./checkpoints/alignment/final_aligned.pt
"""
import os, sys, glob, random, math, yaml, argparse, time
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------- utils / dataset indexing -----------------
def parse_xml_box(xml_path):
    try:
        root = ET.parse(xml_path).getroot()
        bb = root.find("object").find("bndbox")
        xmin = int(bb.find("xmin").text); ymin = int(bb.find("ymin").text)
        xmax = int(bb.find("xmax").text); ymax = int(bb.find("ymax").text)
        return (xmin+xmax)//2, (ymin+ymax)//2
    except Exception:
        return None

def load_bmp_as_tensor(bmp_path, resize=(224,224)):
    img = Image.open(bmp_path).convert("L")
    if resize:
        img = img.resize(resize)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - x.mean()) / (x.std() + 1e-6)
    x = torch.from_numpy(x).unsqueeze(0)    # [1,H,W]
    return x

import SimpleITK as sitk
def load_mhd(path):
    img = sitk.ReadImage(path)
    vol = sitk.GetArrayFromImage(img).astype(np.float32)  # [Z,Y,X]
    vol = (vol - vol.mean()) / (vol.std() + 1e-6)
    return vol

def safe_crop3d(vol, z, y, x, d=64, h=64, w=64):
    Z, Y, X = vol.shape
    z = int(np.clip(int(z), 0, Z-1)); y = int(np.clip(int(y), 0, Y-1)); x = int(np.clip(int(x), 0, X-1))
    z0, y0, x0 = z - d//2, y - h//2, x - w//2
    z1, y1, x1 = z0 + d, y0 + h, x0 + w
    pad = [(0,0),(0,0),(0,0)]
    if z0 < 0: pad[0] = (-z0, pad[0][1]); z0 = 0
    if y0 < 0: pad[1] = (-y0, pad[1][1]); y0 = 0
    if x0 < 0: pad[2] = (-x0, pad[2][1]); x0 = 0
    if z1 > Z: pad[0] = (pad[0][0], z1 - Z); z1 = Z
    if y1 > Y: pad[1] = (pad[1][0], y1 - Y); y1 = Y
    if x1 > X: pad[2] = (pad[2][0], x1 - X); x1 = X
    cube = vol[z0:z1, y0:y1, x0:x1]
    if any(p != (0,0) for p in pad):
        cube = np.pad(cube, pad, mode='edge')
    return cube.astype(np.float32)

def collect_index_from_bmpcls(root):
    """
    Expect structure: BMP_classification/BMP_classification/{train,test}/{class_id}/*.bmp
    Returns dict: class_id -> list of items (dict) with keys: stem,bmp,mhd,z,cx,cy,label
    """
    base = os.path.join(root, "BMP_classification", "BMP_classification")
    ann_dir = os.path.join(root, "BMP_2D", "BMP_2D", "Annotations")
    mhd_dir = os.path.join(root, "MHD_3D", "MHD_3D")
    idx = defaultdict(list)
    for split in ("train","test"):
        split_dir = os.path.join(base, split)
        if not os.path.isdir(split_dir): continue
        for cls in sorted(os.listdir(split_dir)):
            cdir = os.path.join(split_dir, cls)
            if not os.path.isdir(cdir): continue
            for bmp in glob.glob(os.path.join(cdir, "*.bmp")):
                stem = os.path.splitext(os.path.basename(bmp))[0]
                xml = os.path.join(ann_dir, stem + ".xml")
                if not os.path.exists(xml):
                    # skip if annotation missing
                    continue
                center = parse_xml_box(xml)
                if center is None:
                    continue
                cx, cy = center
                # find mhd by patient id prefix (like 0006_33 -> pid 6)
                m = stem.split("_")[0]
                try:
                    pid = str(int(m))
                except:
                    pid = m
                mhd = os.path.join(mhd_dir, pid + ".mhd")
                if not os.path.exists(mhd):
                    continue
                z_idx = int(stem.split("_")[1])
                item = {"stem": stem, "bmp": bmp, "mhd": mhd, "z": z_idx, "cx": cx, "cy": cy, "label": int(cls)}
                idx[int(cls)].append(item)
    return idx

# ----------------- encoder loader (flexible) -----------------
class ResNet18_2D_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18
        net = resnet18(weights=None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        if x.size(1) == 1: x = x.repeat(1,3,1,1)
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.pool(x).flatten(1)  # 512

class BasicBlock3D(nn.Module):
    def __init__(self,c_in,c_out,stride=1):
        super().__init__()
        self.conv1=nn.Conv3d(c_in,c_out,3,stride=stride,padding=1,bias=False); self.bn1=nn.BatchNorm3d(c_out)
        self.conv2=nn.Conv3d(c_out,c_out,3,padding=1,bias=False); self.bn2=nn.BatchNorm3d(c_out)
        self.down=None
        if c_in!=c_out or stride!=1:
            self.down=nn.Sequential(nn.Conv3d(c_in,c_out,1,stride=stride,bias=False), nn.BatchNorm3d(c_out))
    def forward(self,x):
        idn=x; x=F.relu(self.bn1(self.conv1(x))); x=self.bn2(self.conv2(x))
        if self.down is not None: idn=self.down(idn)
        return F.relu(x+idn)

class Backbone3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv3d(1,32,7,stride=2,padding=3,bias=False), nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d(3,stride=2,padding=1))
        self.l1=BasicBlock3D(32,64,2); self.l2=BasicBlock3D(64,128,2); self.l3=BasicBlock3D(128,256,2)
        self.pool=nn.AdaptiveAvgPool3d(1)
    def forward(self,x):
        x=self.stem(x); x=self.l1(x); x=self.l2(x); x=self.l3(x)
        return self.pool(x).flatten(1)  # 256

def load_aligned_encoders(ckpt_path, device):
    sd = torch.load(ckpt_path, map_location=device)
    enc2d = ResNet18_2D_Encoder().to(device)
    enc3d = Backbone3D().to(device)
    # try common key prefixes
    def try_load(mod, keys):
        for k in keys:
            sub = {kk[len(k):]:vv for kk,vv in sd.items() if kk.startswith(k)}
            if sub:
                try:
                    mod.load_state_dict(sub, strict=False)
                    return True
                except Exception:
                    pass
        # try load entire dict leniently
        try:
            mod.load_state_dict(sd, strict=False); return True
        except Exception:
            return False
    _ = try_load(enc2d, ["enc2d.","encoder2d.","model2d.","resnet2d."])
    _ = try_load(enc3d, ["enc3d.","encoder3d.","model3d.","backbone3d."])
    enc2d.eval(); enc3d.eval()
    return enc2d, enc3d

# ----------------- Head -----------------
class FusionProtoHead(nn.Module):
    def __init__(self,in_dim=768,hid=512,out_dim=256):
        super().__init__()
        self.proj=nn.Sequential(nn.Linear(in_dim,hid), nn.ReLU(inplace=True), nn.Linear(hid,out_dim))
        self.scale=nn.Parameter(torch.tensor(10.0))
    def forward(self,e): return F.normalize(self.proj(e), dim=1)
    def logits(self,q,p):
        # q: [Nq, d], p: [n_way, d]
        dmat = torch.cdist(q, p, p=2.0)**2
        return -self.scale * dmat

# ----------------- Episodic builders (deterministic-ish) -----------------
def episode_from_indices(cls_index, n_way, k_shot, q_query, device, enc2d, enc3d, head, minority_cls=None, rng=None, vol_cache=None, stats=None):
    if rng is None: rng = random
    if vol_cache is None: vol_cache = {}
    classes = sorted([c for c in cls_index if len(cls_index[c])>0])
    assert len(classes) >= n_way, "Not enough classes"
    sel = rng.sample(classes, n_way)

    support, query, y_s, y_q = [], [], [], []
    used_replacement = False; used_replacement_minority = False

    for i,c in enumerate(sel):
        pool = cls_index[c]
        need = k_shot + q_query
        if len(pool) < need:
            picks = [rng.choice(pool) for _ in range(need)]
            used_replacement = True
            if stats is not None: stats["per_class_replacement"][c] += 1
        else:
            picks = rng.sample(pool, need)
        for j,it in enumerate(picks):
            try:
                x2d = load_bmp_as_tensor(it["bmp"])   # [1,H,W]
                if it["mhd"] not in vol_cache: vol_cache[it["mhd"]] = load_mhd(it["mhd"])
                vol = vol_cache[it["mhd"]]
                cube = safe_crop3d(vol, z=it["z"], y=it["cy"], x=it["cx"], d=64, h=64, w=64)
                x3d = torch.from_numpy(cube).unsqueeze(0)  # [1,D,H,W]
            except Exception:
                if stats is not None: stats["skipped_samples"] += 1
                # fallback -> sample random
                it2 = rng.choice(pool)
                x2d = load_bmp_as_tensor(it2["bmp"])
                if it2["mhd"] not in vol_cache: vol_cache[it2["mhd"]] = load_mhd(it2["mhd"])
                vol = vol_cache[it2["mhd"]]
                cube = safe_crop3d(vol, z=it2["z"], y=it2["cy"], x=it2["cx"], d=64, h=64, w=64)
                x3d = torch.from_numpy(cube).unsqueeze(0)
            # augment lightly
            if random.random() < 0.3: x2d = torch.flip(x2d, dims=[2])
            if random.random() < 0.3: x3d = torch.flip(x3d, dims=[2])
            dct = {"x2d": x2d, "x3d": x3d}
            if j < k_shot:
                support.append(dct); y_s.append(i)
            else:
                query.append(dct); y_q.append(i)

    # embed
    with torch.no_grad():
        x2d_s = torch.stack([b["x2d"] for b in support]).to(device)
        x3d_s = torch.stack([b["x3d"] for b in support]).unsqueeze(1).to(device) \
                if x3d_s_shape := False else torch.stack([b["x3d"] for b in support]).to(device)
        # note: enc3d expects shape [B,1,D,H,W], ensure correct dim
        if x3d_s.dim() == 4: x3d_s = x3d_s.unsqueeze(1)
        x2d_q = torch.stack([b["x2d"] for b in query]).to(device)
        x3d_q = torch.stack([b["x3d"] for b in query]).to(device)
        if x3d_q.dim() == 4: x3d_q = x3d_q.unsqueeze(1)

        e2d_s = enc2d(x2d_s) if not isinstance(enc2d, tuple) else enc2d(x2d_s)
        e3d_s = enc3d(x3d_s)
        if isinstance(e2d_s, (tuple,list)): e2d_s = e2d_s[0]
        if isinstance(e3d_s, (tuple,list)): e3d_s = e3d_s[0]
        e2d_q = enc2d(x2d_q); e3d_q = enc3d(x3d_q)
        if isinstance(e2d_q, (tuple,list)): e2d_q = e2d_q[0]
        if isinstance(e3d_q, (tuple,list)): e3d_q = e3d_q[0]

        if torch.cuda.is_available():
            pass

        # fusion
        if head is None:
            raise RuntimeError("head is required")
        if head is not None:
            # determine modality by head input dim (heuristic)
            # assume enc2d:512, enc3d:256
            e_s = torch.cat([e2d_s, e3d_s], dim=1)
            e_q = torch.cat([e2d_q, e3d_q], dim=1)
            emb_s = head(e_s)
            emb_q = head(e_q)

    y_q = torch.tensor(y_q, device=device)
    return emb_s, emb_q, y_q

def protos_from_support(emb_s, n_way, k_shot):
    protos=[]; pos=0
    for _ in range(n_way):
        protos.append(emb_s[pos:pos+k_shot].mean(0, keepdim=True))
        pos += k_shot
    return torch.cat(protos, dim=0)

# ----------------- evaluation helpers -----------------
def eval_episode_acc(index, enc2d, enc3d, head, device, episodes=200, n_way=2, k_shot=3, q_query=2):
    accs=[]
    for _ in range(episodes):
        emb_s, emb_q, y_q = episode_from_indices(index, n_way, k_shot, q_query, device, enc2d, enc3d, head)
        P = protos_from_support(emb_s, n_way, k_shot)
        pred = head.logits(emb_q, P).argmax(1)
        accs.append((pred == y_q).float().mean().item())
    return float(np.mean(accs)), float(np.std(accs, ddof=1)/math.sqrt(len(accs)))

# ----------------- embeddings extraction -----------------
def extract_val_embeddings(index, enc2d, enc3d, head, device, out_dir, max_per_class=200):
    os.makedirs(out_dir, exist_ok=True)
    rows_2d, rows_3d, rows_f = [], [], []
    enc2d.eval(); enc3d.eval(); head.eval()
    with torch.no_grad():
        for cls, items in index.items():
            cnt = 0
            for it in items:
                if cnt >= max_per_class: break
                try:
                    e2 = enc2d(load_bmp_as_tensor(it["bmp"]).unsqueeze(0).to(device))
                    e3_in = torch.from_numpy(safe_crop3d(load_mhd(it["mhd"]), it["z"], it["cy"], it["cx"], d=64, h=64, w=64)).unsqueeze(0).to(device)
                    if e3_in.dim() == 4: e3_in = e3_in.unsqueeze(1)
                    e3 = enc3d(e3_in)
                    if isinstance(e2, (tuple,list)): e2 = e2[0]
                    if isinstance(e3, (tuple,list)): e3 = e3[0]
                    fused = torch.cat([e2, e3], dim=1)
                    emb = head(fused)
                    # to cpu numpy
                    rows_2d.append({"stem": it["stem"], "label": it["label"], **{f"f2_{i}": float(x) for i,x in enumerate(e2.flatten().cpu().numpy())}})
                    rows_3d.append({"stem": it["stem"], "label": it["label"], **{f"f3_{i}": float(x) for i,x in enumerate(e3.flatten().cpu().numpy())}})
                    rows_f.append({"stem": it["stem"], "label": it["label"], **{f"f_{i}": float(x) for i,x in enumerate(emb.flatten().cpu().numpy())}})
                    cnt += 1
                except Exception as e:
                    # skip bad IO
                    continue
    pd.DataFrame(rows_2d).to_csv(os.path.join(out_dir, "val_2d_embeddings.csv"), index=False)
    pd.DataFrame(rows_3d).to_csv(os.path.join(out_dir, "val_3d_embeddings.csv"), index=False)
    pd.DataFrame(rows_f).to_csv(os.path.join(out_dir, "val_fusion_embeddings.csv"), index=False)
    return os.path.join(out_dir, "val_2d_embeddings.csv")

# ----------------- failure ranking (fusion margin & disagreement) -----------------
def compute_failure_ranking(val_index, enc2d, enc3d, head, device, out_csv):
    enc2d.eval(); enc3d.eval(); head.eval()
    records = []
    with torch.no_grad():
        for cls, items in val_index.items():
            for it in items:
                try:
                    e2 = enc2d(load_bmp_as_tensor(it["bmp"]).unsqueeze(0).to(device))
                    e3_in = torch.from_numpy(safe_crop3d(load_mhd(it["mhd"]), it["z"], it["cy"], it["cx"])).unsqueeze(0).to(device)
                    if e3_in.dim() == 4: e3_in = e3_in.unsqueeze(1)
                    e3 = enc3d(e3_in)
                    if isinstance(e2, (tuple,list)): e2 = e2[0]
                    if isinstance(e3, (tuple,list)): e3 = e3[0]
                    fused = torch.cat([e2, e3], dim=1)
                    emb = head(fused)
                    # compute distance to prototypes built from train (lazy: use same-class mean from train set)
                    records.append({"stem": it["stem"], "class": cls, "emb": emb.cpu().numpy().flatten()})
                except Exception:
                    continue
    # quick ranking based on nearest neighbor margins (placeholder: save raw embs)
    rows = []
    for r in records:
        rows.append({"stem": r["stem"], "class": int(r["class"])})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return out_csv

# ----------------- main train loop -----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=".", help="Dataset root")
    p.add_argument("--aligned-ckpt", type=str, required=True, help="Aligned encoders checkpoint")
    p.add_argument("--cfg", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="./checkpoints/protonet")
    return p.parse_args()

def load_cfg(path):
    if path is None or not os.path.exists(path): return {}
    with open(path, "r") as f:
        return yaml.safe_load(f)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def main():
    args = parse_args()
    cfg = load_cfg(args.cfg) if args.cfg else {}
    data_root = args.data_root or cfg.get("data_root", ".")
    aligned_ckpt = args.aligned_ckpt
    out_dir = args.out_dir or cfg.get("out_dir", "./checkpoints/protonet")
    os.makedirs(out_dir, exist_ok=True)
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    epochs = args.epochs or int(cfg.get("epochs", 200))
    n_way = int(cfg.get("n_way", 2)); k_shot = int(cfg.get("k_shot", 3)); q_query = int(cfg.get("q_query", 2))
    balanced_train = bool(cfg.get("balanced_train", True))
    eval_episodes = int(cfg.get("val_episodes", 200))
    seed = int(cfg.get("seed", 42)); set_seed(seed)

    print("Collecting train/val index...")
    idx = collect_index_from_bmpcls(data_root)
    # simple split: use existing train/test in BMP_classification; we already read both.
    # Create train_index, val_index by sampling 80/20 within class if needed
    train_index, val_index = {}, {}
    for c, items in idx.items():
        if len(items) < 2:
            train_index[c] = items.copy()
            val_index[c] = []
            continue
        # try to keep mapping: if items from 'train'/'test' were both loaded, we cannot know split
        # So do an internal split 0.8/0.2 deterministic
        rng = random.Random(seed)
        rng.shuffle(items)
        split = int(0.8 * len(items))
        train_index[c] = items[:split] if split>0 else items
        val_index[c] = items[split:] if split>0 else []

    device = torch.device(device)
    print("Device:", device)
    print("Loading aligned encoders from:", aligned_ckpt)
    enc2d, enc3d = load_aligned_encoders(aligned_ckpt, device)
    for p in enc2d.parameters(): p.requires_grad_(False)
    for p in enc3d.parameters(): p.requires_grad_(False)

    IN_DIM = int(cfg.get("in_dim", 768))
    head = FusionProtoHead(in_dim=IN_DIM).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=float(cfg.get("lr", 5e-4)), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_acc = -1.0; best_state = None; bad = 0; patience = int(cfg.get("patience", 40))

    # training loop
    for ep in range(1, epochs+1):
        head.train()
        if balanced_train:
            emb_s, emb_q, yq = episode_from_indices(train_index, n_way, k_shot, q_query, device, enc2d, enc3d, head)
        else:
            emb_s, emb_q, yq = episode_from_indices(train_index, n_way, k_shot, q_query, device, enc2d, enc3d, head)
        P = protos_from_support(emb_s, n_way, k_shot)
        logits = head.logits(emb_q, P)
        loss = F.cross_entropy(logits, yq)
        opt.zero_grad(); loss.backward(); opt.step(); scheduler.step()

        if ep % int(max(1, epochs//20)) == 0 or ep == 1:
            # validate
            acc, se = eval_episode_acc(val_index, enc2d, enc3d, head, device, episodes=eval_episodes, n_way=n_way, k_shot=k_shot, q_query=q_query)
            print(f"ep {ep:03d} | train loss {loss.item():.4f} | val episodic acc {acc:.3f} +/- {1.96*se:.3f}")
            # save best
            if acc > best_acc:
                best_acc = acc; bad = 0
                best_state = {"head": head.state_dict(), "epoch": ep, "val_acc": acc}
                torch.save(best_state, os.path.join(out_dir, "protonet_best.pt"))
                print("-> New best prototypical head saved.")
            else:
                bad += 1
                if bad >= patience:
                    print(f"Early stopping at ep {ep} (no improvement for {patience} evals).")
                    break

    # final save
    torch.save({"head": head.state_dict(), "best_acc": best_acc}, os.path.join(out_dir, "protonet_final.pt"))
    print("Training finished. Best episodic val acc:", best_acc)

    # Extract embeddings for val set
    emb_out = os.path.join(out_dir, "embeddings")
    print("Extracting embeddings to:", emb_out)
    extract_val_embeddings(val_index, enc2d, enc3d, head, device, out_dir=emb_out, max_per_class=200)

    # compute failure ranking (simple)
    fr = os.path.join(out_dir, "failure_ranking.csv")
    compute_failure_ranking(val_index, enc2d, enc3d, head, device, fr)
    print("Saved failure ranking to:", fr)
    print("All done.")

if __name__ == "__main__":
    main()
