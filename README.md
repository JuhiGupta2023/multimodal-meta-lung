# Multimodal Meta-Learning for Few-Shot Lung Nodule Classification

**Repo:** `multimodal-meta-lung`  
**Author / Contact:** Juhi Gupta — (add email)  
**Status:** Code + toy-data for reproducibility. Full dataset used in experiments is not included (private / restricted). See *Dataset* below for instructions.

---

## TL;DR
This repository contains code to reproduce the experiments in *Multimodal Meta-Learning for Few-Shot Lung Nodule Classification* (alignment pretraining of 2D & 3D encoders, Prototypical network few-shot training, calibration and interpretability analyses). The project is organized so reviewers can run a quick smoke test using the provided toy data and run full experiments locally when the dataset is available.

---

## Repo structure
├── .github/workflows/ # CI smoke workflow (toy-data)
├── checkpoints/ # saved checkpoints (alignment, protonet)
├── configs/ # yaml configs for alignment/protonet
├── sample_data/ # toy dataset used by CI and smoke runs
├── src/
│ ├── alignment/ # alignment pretraining script
│ │ └── alignment_train.py
│ ├── protonet/ # prototypical meta-learning training
│ │ └── protonet_train.py
│ ├── utils/ # helper modules (data/models/loss/metrics/io)
│ ├── eval/ # evaluation & embedding export
│ └── infer/ # simple inference utility
├── run_smoke.sh # convenience script that runs toy-data tests
├── requirements.txt
└── README.md


---

## Dataset (what you need to run full experiments)
**Note:**The original dataset is "Lung Nodule Dataset with Histopathology-based Cancer Type Annotation" by Jian et al., which provides computed tomography (CT) scans of biopsy-confirmed lung nodules together with histopathology-derived subtype labels. The repo contains a `sample_data/` toy generator for CI and smoke runs. This dataset can be downloaded from the following GitHub repository: 
https://github.com/chycxyzd/LDFC.



