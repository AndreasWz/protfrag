# 🚀 Quick Start Guide

This guide provides the 5 steps needed to download the data, process it, and train the fragment prediction model.

---

## Step 1: Setup & Dependencies

### Clone the repository and navigate into it

```bash
git clone [your-repo-url]
cd protfrag
```

### Install all Python dependencies (using uv)

```bash
uv venv
source .venv/bin/activate
uv sync
```

Or if you want to use pip:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Install mmseqs2

This is often easiest with conda:

```bash
conda install -c bioconda mmseqs2
```

If you dont use conda:

```bash
wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz
tar xvf mmseqs-linux-avx2.tar.gz
sudo mv mmseqs/bin/mmseqs /usr/local/bin/
```

---

## Step 2: Download UniProt Data

Create the directories and download all raw data.

```bash
mkdir -p data/uniprot
mkdir -p data/uniprot/bulk_embeddings
```

### A) Download Raw Sequence/Annotation Data (for Parsing)

```bash
# Fragment sequences (FASTA)
wget "https://rest.uniprot.org/uniprotkb/stream?query=(reviewed:true)%20AND%20(fragment:true)&format=fasta" \
  -O data/uniprot/fragments.fasta

# Fragment annotations (TSV)
wget "https://rest.uniprot.org/uniprotkb/stream?fields=accession,ft_non_cons,ft_non_ter,fragment&query=(reviewed:true)%20AND%20(fragment:true)&format=tsv" \
  -O data/uniprot/fragment_annotations.tsv

# Complete sequences (FASTA)
wget "https://rest.uniprot.org/uniprotkb/stream?query=(reviewed:true)%20AND%20(fragment:false)&format=fasta" \
  -O data/uniprot/complete.fasta
```

### B) Download Bulk Embeddings (for Training)

Go to the UniProt Embeddings download page (or use the link you found) and download the HDF5 files for Swiss-Prot. Save them into `data/uniprot/bulk_embeddings/`. You should have:

- `data/uniprot/bulk_embeddings/swiss-prot_fragments.h5` (or similar)
- `data/uniprot/bulk_embeddings/swiss-prot_complete.h5` (or similar)

---

## Step 3: Data Preparation Pipeline

Run these 4 scripts in order. This is the core data processing workflow.

### 1. Parse Data

Reads the `.fasta` and `.tsv` files, combines them, and labels fragment types.

```bash
python scripts/01_parse_uniprot_data.py
```

**Output**: `data/processed/metadata_raw.csv` (All ~570k sequences)

### 2. Reduce Redundancy

Clusters all sequences and creates a list of non-redundant "representatives".

```bash
bash scripts/02_run_mmseqs.sh
```

**Output**: `data/processed/clustered/representative_ids.txt` (All ~380k IDs)

### 3. Unpack Embeddings

Reads your `representative_ids.txt` and extracts only those ~380k embeddings from your big H5 files, saving them as individual `.pt` files.

```bash
# IMPORTANT: Adjust the paths to your H5 files in the script 03_unpack_embeddings.py if necessary!
python scripts/03_unpack_embeddings.py
```

**Output**: `data/embeddings/` folder (filled with ~380k `.pt` files)

### 4. Create Final Splits

Reads the `metadata_raw.csv`, filters it to the ~380k embeddings that exist in `data/embeddings/` (ignores the 11 missing ones), and creates the final Train/Val/Test splits.

```bash
python scripts/04_create_train_val_test_splits.py
```

**Output**: `data/processed/metadata.csv` (The final file for training)

---

## Step 4: Train the Model

Now you are ready to train on the full, clean, non-redundant dataset.

### 1. Start Training (Default)

```bash
python train.py --config configs/default.yaml
```

The script loads the `metadata.csv` (~380k) and begins training.

### 2. Monitor Training

```bash
wandb login --relogin
```

(Paste your API key from [wandb.ai/settings](https://wandb.ai/settings))

- Open the link that wandb outputs in your terminal (e.g. `https://wandb.ai/...`).
- Monitor `val/binary_mcc` (performance) and `val/loss_total` (stability).

### 3. (Optional) Start Hyperparameter Experiments

Run additional experiments with different parameters to find the best model.

```bash
# Experiment: Slower learning rate (highly recommended!)
python train.py --config configs/default.yaml \
  --override model.learning_rate=0.0001

# Experiment: Stronger regularization
python train.py --config configs/default.yaml \
  --override model.learning_rate=0.0001 model.weight_decay=0.01
```

---

## Step 5: Evaluation

Evaluate your best trained model on the test set.

### Find Your Best Model

1. Go to your W&B dashboard and find the run with the best `val/binary_mcc` while keeping overfitting in mind `val/binary_loss`
2. Go to your `checkpoints/` folder and find the `.ckpt` file that belongs to this run (e.g. `fragment-detector-BEST_MCC-epoch=...-val/binary_mcc=0.765.ckpt`).

### Run the Evaluation Script

```bash
python evaluate.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/[NAME_OF_YOUR_BEST_CHECKPOINT].ckpt \
  --output-dir results/evaluation_FINAL
```

### Check Results

Look in `results/evaluation_FINAL/` for:

- **`baseline_comparison.png`** (The most important plot!)
- **`test_metrics.json`** (The final scores)
- **`error_analysis.txt`** (The "Top 10" errors)
- **`binary_roc_curve.png`** and **`binary_pr_curve.png`**
- **`predictions.csv`** (All raw data)

---

🎉 **Congratulations!** You've successfully trained and evaluated your protein fragment prediction model!