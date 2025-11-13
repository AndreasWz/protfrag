<p align="center">
  <img src="logo.png" width="180" alt="logo" />
</p>

# ProtFrag - Protein Fragment Prediction from pLM Embeddings

This project implements a multi-task deep learning model to predict protein fragments from ProtT5 embeddings.

The model performs two related tasks:

1. **Binary Classification**: Predicts if a sequence is Complete vs. Fragment.
2. **Multilabel Classification**: Predicts the type of fragment (N-terminal, C-terminal, Internal gaps).

This repository provides a complete pipeline — from raw UniProt data parsing and embedding preparation to model training, hyperparameter tuning, and a comprehensive evaluation suite.

---

## 🚀 Repository Structure

```
.
├── configs/
│   └── default.yaml              # Hyperparameters for data, model, training
│
├── data/
│   ├── embeddings/               # Stores individual [entry].pt files
│   ├── processed/
│   │   ├── metadata_raw.csv      # Output of 01_parse...
│   │   ├── clustered/            # Output of 02_run_mmseqs...
│   │   └── metadata.csv          # Output of 04_create_splits... (FINAL)
│   └── uniprot/
│       ├── bulk_embeddings/      # (Your downloaded HDF5 files)
│       ├── fragments.fasta
│       ├── complete.fasta
│       └── fragment_annotations.tsv
│
├── scripts/
│   ├── 01_parse_uniprot_data.py           # Parses FASTA/TSV -> metadata_raw.csv
│   ├── 02_run_mmseqs.sh                   # Creates representative_ids.txt
│   ├── 03_unpack_embeddings.py            # (NEW) Converts bulk H5 -> individual .pt files
│   ├── 04_create_train_val_test_splits.py # (Formerly 03) Creates final metadata.csv
│   └── (05_... synthetic data scripts)
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
├── lightning_logs/               # Local CSV/W&B logs
├── results/                      # Evaluation outputs (plots, .json, .txt)
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

- **Multi-task Learning**: A shared encoder learns common fragment features, while separate heads specialize.

- **Redundancy Reduction**: MMseqs2 is used to cluster the dataset and remove redundant sequences, preventing data leakage between train and test sets and ensuring the model learns generalizable features.

- **Correct C-Terminal Parsing**: `src/utils/fragment_parser.py` correctly uses sequence length to differentiate N-terminal, C-terminal, and internal NON_TER annotations.

- **Multilabel (Not Multiclass)**: The fragment type head is multilabel (sigmoid on 3 neurons), as fragments can have multiple incompleteness types simultaneously.

- **Stratified Splitting**: The `scripts/04_...` script creates reproducible splits stratified by both fragment status and sequence length bins to prevent the model from learning trivial length-based heuristics.

- **Robust Evaluation**: The primary metric is Matthews Correlation Coefficient (MCC), which is ideal for imbalanced datasets. We also monitor `val/loss_total` with EarlyStopping to prevent severe overfitting.

- **Config-Driven**: All hyperparameters, paths, and training settings are controlled via `configs/default.yaml` and can be overridden via the command line.

---

## ⚡ Usage

For a complete step-by-step guide, see **QUICKSTART.md**.

### General Workflow

```bash
# 1. Download UniProt raw data (FASTA, TSV)
# 2. Download UniProt bulk embeddings (HDF5)
# (See QUICKSTART for details)

# 3. Run the 4-step data processing pipeline
python scripts/01_parse_uniprot_data.py
bash scripts/02_run_mmseqs.sh
python scripts/03_unpack_embeddings.py
python scripts/04_create_train_val_test_splits.py

# 4. Train the model (and monitor on W&B)
python train.py --config configs/default.yaml

# 5. (Optional) Run Hyperparameter Experiments
python train.py --config configs/default.yaml --override model.learning_rate=0.0001

# 6. Evaluate your best model from W&B
python evaluate.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/fragment-detector-BEST_MCC-....ckpt \
  --output-dir results/evaluation_final
```

---

## 🩺 Troubleshooting

### 🧠 OutOfMemoryError (OOM)

- Reduce `data.batch_size` in `configs/default.yaml`.
- Set `training.precision: 16` for mixed-precision.

### 📂 Embeddings Not Found

- **During `04_create_splits...`**: Your `03_unpack_embeddings.py` script may have been interrupted or failed. Re-run it.
- **During `train.py`**: Your `data/processed/metadata.csv` is out of sync with your `data/embeddings/` folder. Re-run `scripts/04_create_train_val_test_splits.py` to re-scan the folder and create a clean `metadata.csv`.

### 📉 Poor Convergence (Low `val/binary_mcc`)

- Your `learning_rate` might be too high (e.g., 0.001). As we found, 0.0001 is much more stable.
- Try increasing `model.weight_decay` (e.g., to 0.01) to fight overfitting.

### 🌐 W&B Error 401: User Not Logged In

- Your W&B API key is invalid or expired.
- Run `wandb login --relogin` in your terminal and paste a new API key.

---

© 2025 PROTFRAG-TEAM — Protein Prediction II — TUM WS2025/26