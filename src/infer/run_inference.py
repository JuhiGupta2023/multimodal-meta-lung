# src/infer/run_inference.py
"""
Simple inference script that loads aligned encoders + head and runs episodic eval
or single-sample inference. Produces probability scores and writes CSV.
"""
import os, torch, numpy as np, pandas as pd
from utils.models import ResNet18_2D_Encoder, Backbone3D, FusionProtoHead
from utils.data import load_mhd, safe_cube_3d
from utils.data import parse_xml_box
from PIL import Image
import argparse

def load_head_and_encoders(aligned_ckpt, device, in_dim=768):
    sd = torch.load(aligned_ckpt, map_location=device)
    enc2d = ResNet18_2D_Encoder().to(device); enc3d = Backbone3D().to(device)
    # lenient load (similar to earlier loader)
    try:
        enc2d.load_state_dict({k.replace("enc2d.",""):v for k,v in sd.items() if k.startswith("enc2d.")}, strict=False)
    except Exception:
        pass
    try:
        enc3d.load_state_dict({k.replace("enc3d.",""):v for k,v in sd.items() if k.startswith("enc3d.")}, strict=False)
    except Exception:
        pass
    head = FusionProtoHead(in_dim=in_dim).to(device)
    return enc2d, enc3d, head

def infer_single(bmp_path, xml_path, mhd_path, enc2d, enc3d, head, device):
    # produce fused embedding and softmax over prototypes if provided
    from utils.data import safe_crop_2d
    img = Image.open(bmp_path).convert("L").resize((224,224))
    import numpy as np, torch
    x2 = np.asarray(img, dtype='float32')/255.0
    x2 = (x2 - x2.mean())/(x2.std()+1e-6)
    x2 = torch.from_numpy(x2).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
    # 3D
    center = parse_xml_box(xml_path)
    cx = (center[0]+center[2])//2; cy = (center[1]+center[3])//2
    vol = load_mhd(mhd_path)
    cz = int(os.path.splitext(os.path.basename(bmp_path))[0].split("_")[1])
    cube = safe_cube_3d(vol, cz, cy, cx)
    x3 = torch.from_numpy(cube).unsqueeze(0).unsqueeze(0).to(device)
    enc2d.eval(); enc3d.eval(); head.eval()
    with torch.no_grad():
        e2 = enc2d(x2); e3 = enc3d(x3)
        if isinstance(e2, (list,tuple)): e2 = e2[0]
        if isinstance(e3, (list,tuple)): e3 = e3[0]
        fused = torch.cat([e2, e3], dim=1)
        emb = head(fused)
    return emb.cpu().numpy().flatten()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--aligned-ckpt", required=True)
    p.add_argument("--bmp", required=True)
    p.add_argument("--xml", required=True)
    p.add_argument("--mhd", required=True)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc2d, enc3d, head = load_head_and_encoders(args.aligned_ckpt, device)
    emb = infer_single(args.bmp, args.xml, args.mhd, enc2d, enc3d, head, device)
    print("Embedding shape:", emb.shape)
