# ProtFrag - Protein Fragment Prediction from pLM Embeddings

<p align="center">
  <img src="logo.png" width="180" alt="logo" />
</p>

## Overview

This project implements a multi-task deep learning model to predict protein fragments from ProtT5 embeddings.

The model performs two related tasks:

1. **Binary Classification**: Predicts if a sequence is Complete vs. Fragment
2. **Multilabel Classification**: Predicts the type of fragment (N-terminal, C-terminal, Internal gaps)

This repository provides a complete pipeline — from raw UniProt data parsing and redundancy reduction to embedding generation, model training, and evaluation.

---

## 🚀 Repository Structure
```
.
├── configs/
│   └── default.yaml              # Hyperparameters for data, model, training
│
├── data/
│   ├── embeddings/               # Stores [entry].pt embedding files
│   ├── processed/
│   │   ├── clustered/            # Output of MMseqs2
│   │   ├── metadata_raw.csv      # Output of step 1 (parsing)
│   │   └── metadata.csv          # Output of step 3 (splits)
│   └── raw/
│       ├── fragments.fasta       # Raw UniProt downloads
│       ├── complete.fasta
│       └── fragment_annotations.tsv
│
├── scripts/
│   ├── 01_parse_uniprot_data.py          # Parses FASTA/TSV → metadata_raw.csv
│   ├── 02_run_mmseqs.sh                  # Clusters sequences for redundancy
│   ├── 03_create_train_val_test_splits.py # Creates final metadata.csv
│   └── 04_precompute_embeddings.py       # Generates embeddings
│
├── src/                          # All Python source code
│   ├── __init__.py
│   ├── data.py                   # PyTorch Dataset and DataModule
│   ├── metrics.py                # Custom MCC and Multilabel metrics
│   ├── model.py                  # The FragmentDetector LightningModule
│   └── utils/
│       └── fragment_parser.py    # Core logic for parsing NON_TER/NON_CONS
│
├── checkpoints/                  # Saved model .ckpt files
├── lightning_logs/               # TensorBoard logs
├── results/                      # Evaluation outputs (plots, predictions.csv)
│
├── train.py                      # Main training script
├── evaluate.py                   # Main evaluation script
├── requirements.txt              # Project dependencies
├── QUICKSTART.md                 # Step-by-step tutorial
└── README.md                     # This file
```

---

## 🏗️ Model Architecture

The model is a multi-task classifier with a shared backbone:
```
Input: ProtT5 Embedding (1024-dim)
    ↓
Shared Encoder:
    Linear(1024 → 512) + BatchNorm + ReLU + Dropout
    Linear(512 → 256) + BatchNorm + ReLU + Dropout
    ↓                     ↓
Binary Head           Multilabel Head
(1 neuron)            (3 neurons)
    ↓                     ↓
Complete/Fragment    [N-term, C-term, Internal]
```

### Loss Function

The total loss is a weighted sum of the two task losses. Class weights are used to handle data imbalance:

$$L_{total} = w_b \cdot L_{BCE}(binary) + w_m \cdot L_{BCE}(multilabel)$$

---

## 💡 Key Design Decisions

- **Multi-task Learning**: A shared encoder learns common fragment features, while separate heads specialize
- **Redundancy Reduction**: `scripts/02_run_mmseqs.sh` is used to cluster sequences and ensure the test set is not "contaminated" with sequences highly similar to the training set
- **Correct C-Terminal Parsing**: `src/utils/fragment_parser.py` correctly uses sequence length to differentiate N-terminal, C-terminal, and internal NON_TER annotations
- **Multilabel (Not Multiclass)**: The fragment type head is multilabel (sigmoid on 3 neurons), as fragments can have multiple incompleteness types simultaneously
- **Stratified Splitting**: `scripts/03_...` creates reproducible splits from the non-redundant set, stratified by both fragment status and sequence length bins
- **Robust Evaluation**: The primary metric is Matthews Correlation Coefficient (MCC), suitable for imbalanced datasets
- **Config-Driven**: All hyperparameters, paths, and training settings are controlled via `configs/default.yaml` for easy experimentation

---

## ⚡ Usage

For a complete step-by-step guide, see **QUICKSTART.md**.

### General Workflow
```bash
# 1. Download Data
# (Run the wget commands in the quickstart to populate data/raw/)

# 2. Parse Data
python scripts/01_parse_uniprot_data.py

# 3. Reduce Redundancy
bash scripts/02_run_mmseqs.sh

# 4. Create Splits
# (This script automatically finds the output from step 3)
python scripts/03_create_train_val_test_splits.py

# 5. Generate Embeddings (requires GPU)
# (This script reads the final metadata.csv from step 4)
python scripts/04_precompute_embeddings.py

# 6. Train Model
python train.py --config configs/default.yaml

# 7. Evaluate Model
python evaluate.py --checkpoint [path_to_checkpoint.ckpt]
```

---

## 🩺 Troubleshooting

### 🧠 OutOfMemoryError (OOM)

- Reduce `data.batch_size` in `configs/default.yaml`
- Set `training.precision: 16` for mixed-precision

### 📂 Embeddings Not Found

- Ensure `data/embeddings/` contains a `.pt` file for every entry in `data/processed/metadata.csv`
- Re-run `scripts/04_precompute_embeddings.py` if the data changed

### 📉 Poor Convergence (Low val/binary_mcc)

- Try decreasing `model.learning_rate` (e.g., to 0.0001)
- Increase `model.dropout` if overfitting occurs (train loss << val loss)