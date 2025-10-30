# src/eval/evaluate.py
"""
Evaluation utilities: episodic evaluation, embedding export, failure ranking (fusion margin).
"""
import os, numpy as np, pandas as pd, torch
from sklearn.metrics.pairwise import cosine_similarity

def extract_embeddings(index, enc2d, enc3d, head, device, out_dir, max_per_class=200):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    enc2d.eval(); enc3d.eval(); head.eval()
    with torch.no_grad():
        for cls, items in index.items():
            cnt = 0
            for it in items:
                if cnt >= max_per_class: break
                try:
                    x2 = ImageToTensor(it["bmp"]).to(device)
                except Exception:
                    continue
                vol = load_mhd(it["mhd"])
                cube = safe_cube_3d(vol, it["z"], it["cy"], it["cx"])
                x3 = torch.from_numpy(cube).unsqueeze(0).unsqueeze(0).to(device)
                e2 = enc2d(x2.unsqueeze(0))
                e3 = enc3d(x3)
                if isinstance(e2, (list,tuple)): e2 = e2[0]
                if isinstance(e3, (list,tuple)): e3 = e3[0]
                fused = torch.cat([e2, e3], dim=1)
                emb = head(fused)
                rows.append({"stem": it["stem"], "label": int(it["label"]), **{f"f{i}": float(v) for i,v in enumerate(emb.cpu().numpy().flatten())}})
                cnt += 1
    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "val_fusion_embeddings.csv")
    df.to_csv(out_csv, index=False)
    return out_csv

# small helper used above (kept local to avoid circular imports)
from PIL import Image
import numpy as np, torch
def ImageToTensor(path):
    img = Image.open(path).convert("L").resize((224,224))
    x = np.asarray(img, dtype='float32') / 255.0
    x = (x - x.mean()) / (x.std() + 1e-6)
    x = torch.from_numpy(x).unsqueeze(0)  # [1,H,W]
    return x
# reuse safe_cube_3d, load_mhd if available in package scope; otherwise import from utils.data in calling code.
