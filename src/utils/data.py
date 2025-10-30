# src/utils/data.py
"""
Dataset helpers reused across alignment / protonet / inference.
"""
import os, glob, random
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset

def parse_xml_box(xml_path):
    try:
        root = ET.parse(xml_path).getroot()
        obj = root.find("object")
        if obj is None: return None
        bb = obj.find("bndbox")
        xmin = int(bb.find("xmin").text); ymin = int(bb.find("ymin").text)
        xmax = int(bb.find("xmax").text); ymax = int(bb.find("ymax").text)
        return xmin, ymin, xmax, ymax
    except Exception:
        return None

def load_mhd(path):
    import SimpleITK as sitk
    img = sitk.ReadImage(path)
    vol = sitk.GetArrayFromImage(img).astype('float32')  # [Z,Y,X]
    vol = (vol - vol.mean()) / (vol.std() + 1e-6)
    return vol

def safe_crop_2d(img, cx, cy, sz=160):
    h,w = img.shape[:2]
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
    return cube.astype('float32')

class Paired2D3D(Dataset):
    """
    Paired dataset that yields (2D_tensor, 3D_tensor, meta_dict)
    Used for alignment pretraining.
    """
    def __init__(self, img_dir, ann_dir, mhd_dir, split_files=None, depth=64):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.mhd_dir = mhd_dir
        all_bmps = sorted(glob.glob(os.path.join(img_dir, "*.bmp")))
        if split_files is not None:
            split_set = set(split_files)
            all_bmps = [p for p in all_bmps if os.path.splitext(os.path.basename(p))[0] in split_set]
        self.items = []
        for bmp in all_bmps:
            stem = os.path.splitext(os.path.basename(bmp))[0]
            xml = os.path.join(ann_dir, stem + ".xml")
            if not os.path.exists(xml): continue
            bb = parse_xml_box(xml)
            if bb is None: continue
            pid = int(stem.split("_")[0])
            z_idx = int(stem.split("_")[1])
            mhd = os.path.join(mhd_dir, f"{pid}.mhd")
            if not os.path.exists(mhd): continue
            self.items.append((bmp, xml, mhd, z_idx))
        self.depth = depth

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        bmp, xml, mhd, z_idx = self.items[idx]
        img = cv2.imread(bmp, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        bb = parse_xml_box(xml)
        cx = (bb[0] + bb[2])//2; cy = (bb[1] + bb[3])//2
        crop2d = safe_crop_2d(img, cx, cy, sz=160)
        # to tensor normalized [-1,1]
        x2d = torch.from_numpy(crop2d.transpose(2,0,1).astype('float32')/255.0)
        x2d = (x2d - 0.5) / 0.5
        vol = load_mhd(mhd)
        cube = safe_cube_3d(vol, z_idx, cy, cx, d=self.depth, h=64, w=64)
        x3d = torch.from_numpy(cube).unsqueeze(0)  # [1,D,H,W]
        return x2d, x3d, {"bmp": bmp, "xml": xml, "mhd": mhd, "z": z_idx}
