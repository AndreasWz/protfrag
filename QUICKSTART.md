# 🚀 Quick Start Guide

This guide provides the full 6-step pipeline to download data, process it, and train the fragment prediction model.

---

## Step 1: Setup & Dependencies

Clone the repository and navigate into it:
```bash
git clone [your-repo-url]
cd protein-fragment-prediction
```

Install all Python dependencies (using uv):
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Install mmseqs2. This is often easiest with conda:
```bash
conda install -c bioconda mmseqs2
```

---

## Step 2: Download UniProt Data

Create the raw data directory and download the necessary files from UniProt.
```bash
mkdir -p data/raw

# Fragment sequences (FASTA)
wget "https://rest.uniprot.org/uniprotkb/stream?query=(reviewed:true)%20AND%20(fragment:true)&format=fasta" \
  -O data/raw/fragments.fasta

# Fragment annotations (TSV)
wget "https://rest.uniprot.org/uniprotkb/stream?fields=accession,ft_non_cons,ft_non_ter,fragment&query=(reviewed:true)%20AND%20(fragment:true)&format=tsv" \
  -O data/raw/fragment_annotations.tsv

# Complete sequences (FASTA)
wget "https://rest.uniprot.org/uniprotkb/stream?query=(reviewed:true)%20AND%20(fragment:false)&format=fasta" \
  -O data/raw/complete.fasta
```

---

## Step 3: Data Preparation Pipeline

These scripts will turn the raw files into a single, non-redundant metadata.csv file ready for training.

### 1. Parse Data

This script reads the .fasta and .tsv files, combines them, and uses the FragmentAnnotationParser to correctly label all fragment types.
```bash
python scripts/01_parse_uniprot_data.py
```

**Output**: `data/processed/metadata_raw.csv`

### 2. Reduce Redundancy

This script runs mmseqs2 on all sequences from the previous step and generates a list of representative (cluster head) sequence IDs.
```bash
bash scripts/02_run_mmseqs.sh
```

**Output**: `data/processed/clustered/representative_ids.txt`

### 3. Create Splits

This script reads metadata_raw.csv and filters it using representative_ids.txt. It then creates stratified train/validation/test splits from this non-redundant set.
```bash
python scripts/03_create_train_val_test_splits.py
```

**Output**: `data/processed/metadata.csv` (this is the final file used by the DataModule)

---

## Step 4: Generate Embeddings

This script iterates through the final, non-redundant metadata.csv, generates ProtT5 embeddings, and saves them. A GPU is strongly recommended.
```bash
python scripts/04_precompute_embeddings.py \
  --metadata data/processed/metadata.csv \
  --output-dir data/embeddings \
  --batch-size 8 \
  --device cuda
```

**Output**: `data/embeddings/[entry_id].pt` for each representative protein.

**Note on Compute**: If you only have limited compute, you can stop this script early. Then, re-run `scripts/03_create_train_val_test_splits.py`. It will find the embeddings you do have and create a smaller metadata.csv just for those, allowing you to train on a subset.

---

## Step 5: Train the Model

Now you are ready to train on your non-redundant dataset.

### 1. Start Training
```bash
python train.py --config configs/default.yaml
```

This will load the data, calculate class weights, and start training.

### 2. Monitor Training
```bash
# In a new terminal
tensorboard --logdir lightning_logs/
```

Navigate to http://localhost:6006/ to see `val/binary_mcc` and other metrics.

### 3. Override Configs (Optional)
```bash
# Train with a different batch size and dropout
python train.py --config configs/default.yaml \
  --override data.batch_size=64 model.dropout=0.5
```

---

## Step 6: Evaluate the Model

Evaluate your trained model on the non-redundant test set.

Find your best checkpoint (e.g., `checkpoints/fragment-detector-epoch=...-val/binary_mcc=...ckpt`).

Run the evaluation script:
```bash
python evaluate.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/[your-best-checkpoint-name].ckpt \
  --output-dir results/evaluation
```

**Check Results**: Look in `results/evaluation/` for:

- `predictions.csv`
- `binary_confusion_matrix.png`
- `probability_distributions.png`
- `test_metrics.txt`