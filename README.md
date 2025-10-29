# \# Multimodal Meta-Learning for Few-Shot Lung Nodule Classification

# 

# This repository provides the official implementation of the paper  

# \*\*“Multimodal Meta-Learning and Calibration for Few-Shot Lung Nodule Classification.”\*\*

# 

# It integrates \*\*cross-modal representation alignment\*\* between 2D and 3D CT data with \*\*Prototypical Meta-Learning\*\*,  

# followed by \*\*calibration-based trustworthiness analysis\*\*.  

# The framework is reproducible and modular for research and educational use.

# 

# ---

# 

# \## 📘 Repository Overview

# 

# multimodal-meta-lung/

# │

# ├── src/

# │ ├── alignment/ # 2D–3D alignment training (contrastive InfoNCE)

# │ ├── protonet/ # Few-shot Prototypical Network training

# │ └── utils/ # Metrics, visualization, etc.

# │

# ├── configs/

# │ ├── alignment\_config.yaml # Config for cross-modal alignment

# │ └── protonet\_config.yaml # Config for meta-learning

# │

# ├── notebooks/

# │ ├── Alignment\_training.ipynb

# │ └── Protonet\_training.ipynb

# │

# ├── sample\_data/ # Tiny placeholder data (no PHI)

# │ └── README\_DATA.md

# │

# ├── requirements.txt

# ├── LICENSE

# ├── .gitignore

# └── README.md



\## ⚙️ Environment Setup



\### 1️⃣ Using Conda

```bash

conda create -n meta-lung python=3.10

conda activate meta-lung

pip install -r requirements.txt



