#!/usr/bin/env python3
"""
Synthetic "realistic-ish" toy dataset generator for smoke tests.

Creates:
  ./sample_data/toy_dataset/
    BMP_2D/BMP_2D/Image/*.bmp
    BMP_2D/BMP_2D/Annotations/*.xml
    BMP_2D/ImageSets/train.txt
    MHD_3D/MHD_3D/*.mhd

Usage:
  python sample_data/generate_sample_data.py --out ./sample_data/toy_dataset --n_pat 6
"""
import os, sys, argparse, math, random
import numpy as np
import cv2
import SimpleITK as sitk
import xml.etree.ElementTree as ET

def write_xml_bbox(xml_path, stem, xmin, ymin, xmax, ymax):
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = stem + ".bmp"
    obj = ET.SubElement(root, "object")
    ET.SubElement(obj, "name").text = "nodule"
    bnd = ET.SubElement(obj, "bndbox")
    ET.SubElement(bnd, "xmin").text = str(int(xmin))
    ET.SubElement(bnd, "ymin").text = str(int(ymin))
    ET.SubElement(bnd, "xmax").text = str(int(xmax))
    ET.SubElement(bnd, "ymax").text = str(int(ymax))
    tree = ET.ElementTree(root)
    tree.write(xml_path)

def make_volume(shape=(64, 128, 128), nodule_center=(32, 64, 64), nodule_r=6):
    """Create simple CT-like volume:
       - background gaussian noise
       - two ellipses as lungs (lower intensity)
       - bright spherical nodule at given center
    """
    D, H, W = shape
    vol = np.random.normal(loc=-800, scale=30, size=shape).astype(np.float32)  # CT-like HU around -800
    # add lung ellipses per-slice (same mask for all z)
    yy, xx = np.mgrid[0:H, 0:W]
    left_center = (int(H*0.5), int(W*0.35))
    right_center = (int(H*0.5), int(W*0.65))
    left_mask = ((yy - left_center[0])**2 / ( (H*0.35)**2 ) + (xx - left_center[1])**2 / ((W*0.25)**2)) < 1.0
    right_mask = ((yy - right_center[0])**2 / ( (H*0.35)**2 ) + (xx - right_center[1])**2 / ((W*0.25)**2)) < 1.0
    lung_mask = (left_mask | right_mask)
    for z in range(D):
        # lungs have higher intensity than background (but still negative HU)
        vol[z][lung_mask] = vol[z][lung_mask] + 200 + np.random.normal(0,10,size=lung_mask.sum())
    # insert spherical nodule (bright)
    cz, cy, cx = nodule_center
    zz, yy, xx = np.mgrid[0:D, 0:H, 0:W]
    sphere = (zz - cz)**2 + (yy - cy)**2 + (xx - cx)**2 <= (nodule_r**2)
    vol[sphere] = 50 + np.random.normal(0,5,size=vol[sphere].shape)  # near soft-tissue HU
    # simple smoothing to look a bit more organic
    vol = cv2.GaussianBlur(vol, (1,1), 0) if hasattr(cv2, "GaussianBlur") else vol
    return vol

def save_mhd(volume, out_path):
    img = sitk.GetImageFromArray(volume)  # SITK expects z,y,x
    # set some metadata (spacing) to plausible CT values
    img.SetSpacing((1.0, 0.7, 0.7))  # z, y, x spacing
    sitk.WriteImage(img, out_path)

def save_bmp_slice(img_slice, out_path):
    # img_slice: numpy 2D float (HU-like). We map to 0-255 for visualization
    # simple window/level for CT-look: window [-1200, 400]
    wmin, wmax = -1200.0, 400.0
    v = np.clip((img_slice - wmin) / (wmax - wmin), 0.0, 1.0)
    v = (v * 255.0).astype(np.uint8)
    # ensure 3-channel
    v3 = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(out_path, v3)

def generate_toy(out_root, n_patients=6, seed=42):
    random.seed(seed); np.random.seed(seed)
    # folders
    bmp_img_dir = os.path.join(out_root, "BMP_2D", "BMP_2D", "Image")
    bmp_ann_dir = os.path.join(out_root, "BMP_2D", "BMP_2D", "Annotations")
    bmp_imagesets = os.path.join(out_root, "BMP_2D", "ImageSets")
    mhd_dir = os.path.join(out_root, "MHD_3D", "MHD_3D")
    os.makedirs(bmp_img_dir, exist_ok=True)
    os.makedirs(bmp_ann_dir, exist_ok=True)
    os.makedirs(bmp_imagesets, exist_ok=True)
    os.makedirs(mhd_dir, exist_ok=True)

    train_list = []
    # for each patient: create 3D volume and several 2D annotated slices
    for pid in range(1, n_patients+1):
        # random volume dims (keep manageable)
        D, H, W = 64, 128, 128
        # pick nodule location inside lungs roughly
        cz = random.randint(12, D-12)
        cy = int(H * (0.4 + 0.2 * random.random()))  # y near center
        cx = int(W * (0.35 + 0.3 * random.random()))
        r = random.randint(4, 8)
        vol = make_volume(shape=(D,H,W), nodule_center=(cz, cy, cx), nodule_r=r)
        mhd_name = f"{pid}.mhd"
        mhd_path = os.path.join(mhd_dir, mhd_name)
        save_mhd(vol, mhd_path)
        # create 1..3 annotated slices (vary z offset)
        n_slices = random.choice([1,2,3])
        for s in range(n_slices):
            z = np.clip(cz + random.randint(-2,2), 0, D-1)
            slice_img = vol[z]
            # create bbox roughly around sphere projection
            radius_px = int(round(r * (1.0 + 0.2*random.random())))
            xmin = max(0, cx - radius_px - random.randint(0,3))
            ymin = max(0, cy - radius_px - random.randint(0,3))
            xmax = min(W-1, cx + radius_px + random.randint(0,3))
            ymax = min(H-1, cy + radius_px + random.randint(0,3))
            # stem format: '0001_13' where second num is slice index
            stem = f"{pid:04d}_{z}"
            bmp_path = os.path.join(bmp_img_dir, stem + ".bmp")
            xml_path = os.path.join(bmp_ann_dir, stem + ".xml")
            save_bmp_slice(slice_img, bmp_path)
            write_xml_bbox(xml_path, stem, xmin, ymin, xmax, ymax)
            train_list.append(stem)
    # save train.txt
    train_txt = os.path.join(bmp_imagesets, "train.txt")
    with open(train_txt, "w") as f:
        for s in train_list:
            f.write(s + "\n")

    # write a small csv that lists volumes (optional)
    print(f"[INFO] Generated toy dataset at: {out_root}")
    print(f"- Patients: {n_patients}; slices: {len(train_list)}")
    return out_root

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="./sample_data/toy_dataset", help="output folder")
    parser.add_argument("--n_pat", type=int, default=6, help="number of synthetic patients/volumes")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_toy(args.out, n_patients=args.n_pat, seed=args.seed)
