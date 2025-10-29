#!/usr/bin/env python3
"""
generate_toy_data.py

Create a tiny synthetic dataset compatible with the alignment/protonet scripts.

Produces the following structure under --out (default ./toy_dataset):

toy_dataset/
  BMP_2D/
    BMP_2D/
      Image/
        0001_30.bmp
        ...
      Annotations/
        0001_30.xml
        ...
    ImageSets/
      train.txt
  MHD_3D/
    MHD_3D/
      1.mhd  (and 1.raw)
      ...
  all_anno_3D.csv

Usage:
    python generate_toy_data.py --out ./toy_dataset --n_patients 5 --slices_per_patient 3
"""
import os
import argparse
import numpy as np
from PIL import Image, ImageDraw
import SimpleITK as sitk
import csv
import random
import xml.etree.ElementTree as ET

def make_dirs(root):
    bmp_image_dir = os.path.join(root, "BMP_2D", "BMP_2D", "Image")
    ann_dir = os.path.join(root, "BMP_2D", "BMP_2D", "Annotations")
    imagesets_dir = os.path.join(root, "BMP_2D", "ImageSets")
    mhd_dir = os.path.join(root, "MHD_3D", "MHD_3D")
    os.makedirs(bmp_image_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    os.makedirs(imagesets_dir, exist_ok=True)
    os.makedirs(mhd_dir, exist_ok=True)
    return bmp_image_dir, ann_dir, imagesets_dir, mhd_dir

def write_pascal_voc_xml(xml_path, filename, width, height, xmin, ymin, xmax, ymax, label="nodule"):
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "filename").text = filename
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "1"

    obj = ET.SubElement(annotation, "object")
    ET.SubElement(obj, "name").text = label
    bnd = ET.SubElement(obj, "bndbox")
    ET.SubElement(bnd, "xmin").text = str(int(xmin))
    ET.SubElement(bnd, "ymin").text = str(int(ymin))
    ET.SubElement(bnd, "xmax").text = str(int(xmax))
    ET.SubElement(bnd, "ymax").text = str(int(ymax))

    tree = ET.ElementTree(annotation)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)

def create_sphere_volume(Z=80, Y=128, X=128, center=None, radius=8, intensity=200):
    vol = np.zeros((Z, Y, X), dtype=np.float32)
    if center is None:
        cz = random.randint(radius+2, Z-radius-3)
        cy = random.randint(radius+2, Y-radius-3)
        cx = random.randint(radius+2, X-radius-3)
    else:
        cz, cy, cx = center
    zz, yy, xx = np.ogrid[:Z, :Y, :X]
    mask = (zz - cz)**2 + (yy - cy)**2 + (xx - cx)**2 <= radius**2
    vol[mask] = intensity
    # add gaussian background and slight smoothing
    vol = vol + np.random.randn(*vol.shape).astype(np.float32) * 5.0
    vol = vol.astype(np.float32)
    return vol, (cz, cy, cx)

def save_mhd(volume, out_path, spacing=(1.0,1.0,1.0)):
    """
    Save volume as .mhd + raw using SimpleITK.
    volume: numpy array shape (Z, Y, X)
    """
    img = sitk.GetImageFromArray(volume)  # SimpleITK expects (z,y,x)
    img.SetSpacing(spacing)
    sitk.WriteImage(img, out_path)

def save_bmp_slice(bmp_array, out_path):
    # bmp_array expected as 2D numpy float (0..255 or arbitrary). We'll clip and convert uint8.
    arr = np.clip(bmp_array, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr)
    im.save(out_path, format="BMP")

def main(out_root, n_patients=5, slices_per_patient=3, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    bmp_image_dir, ann_dir, imagesets_dir, mhd_dir = make_dirs(out_root)
    train_list = []

    csv_rows = []  # for all_anno_3D.csv with columns: image,x_center,y_center,z_center

    print(f"[INFO] Generating toy dataset at: {out_root}")
    for pid in range(1, n_patients+1):
        # make synthetic volume
        Z, Y, X = 80, 128, 128
        radius = random.randint(6, 12)
        vol, (cz, cy, cx) = create_sphere_volume(Z=Z, Y=Y, X=X, radius=radius, intensity=180 + random.randint(-30,30))
        mhd_name = f"{pid}.mhd"
        mhd_path = os.path.join(mhd_dir, mhd_name)
        save_mhd(vol, mhd_path)
        print(f"  - Patient {pid}: saved volume {mhd_path} (nodule at z={cz}, y={cy}, x={cx}, r={radius})")

        # create several slices around the center (z offsets)
        z_offsets = list(range(cz - slices_per_patient//2, cz + (slices_per_patient+1)//2))
        # clamp
        z_offsets = [max(0, min(Z-1, z)) for z in z_offsets]
        for z in z_offsets:
            stem = f"{pid:04d}_{z}"
            bmp_name = stem + ".bmp"
            xml_name = stem + ".xml"
            bmp_path = os.path.join(bmp_image_dir, bmp_name)
            xml_path = os.path.join(ann_dir, xml_name)

            # create slice image: use vol[z,:,:], add contrast and normalize to 0..255
            slice_img = vol[z].copy()
            # enhance contrast slightly
            slice_img = (slice_img - slice_img.min())
            if slice_img.max() > 0:
                slice_img = slice_img / slice_img.max()
            slice_img = (slice_img * 255.0).astype(np.uint8)

            # create a bounding box around (cx,cy) projected to 2D with size approx 2*radius
            box_half = int(radius * 1.8)
            xmin = max(0, int(cx - box_half))
            xmax = min(X-1, int(cx + box_half))
            ymin = max(0, int(cy - box_half))
            ymax = min(Y-1, int(cy + box_half))

            # To make the image look more CT-like, draw a subtle circular blob on top (already sphere)
            # Save bmp
            save_bmp_slice(slice_img, bmp_path)

            # Save xml
            write_pascal_voc_xml(xml_path, bmp_name, width=X, height=Y, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

            # add to CSV rows
            csv_rows.append({
                "image": bmp_name,
                "x_center": int(cx),
                "y_center": int(cy),
                "z_center": int(z)
            })

            # include stem in train list (we'll use all created stems as train for toy)
            train_list.append(stem)

    # write ImageSets/train.txt (unique stems)
    train_txt_path = os.path.join(imagesets_dir, "train.txt")
    with open(train_txt_path, "w") as f:
        for s in sorted(set(train_list)):
            f.write(s + "\n")

    # write all_anno_3D.csv
    csv_path = os.path.join(out_root, "all_anno_3D.csv")
    with open(csv_path, "w", newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=["image", "x_center", "y_center", "z_center"])
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)

    print(f"[INFO] Wrote {len(csv_rows)} slice annotations to {csv_path}")
    print(f"[INFO] Wrote train list to {train_txt_path}")
    print(f"[INFO] Done. Dataset root: {out_root}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="./toy_dataset", help="output dataset root")
    p.add_argument("--n_patients", type=int, default=5, help="number of synthetic patients/volumes")
    p.add_argument("--slices_per_patient", type=int, default=3, help="slices per patient to create")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args.out, n_patients=args.n_patients, slices_per_patient=args.slices_per_patient, seed=args.seed)
