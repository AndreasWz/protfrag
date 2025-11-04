# 🚀 Quick Start Guide

This guide provides the 5 steps needed to download the data, process it, and train the fragment prediction model.

## Step 1: Setup & Dependencies
🧰 Environment Setup (with uv)

uv
 is a modern, fast Python package and environment manager — it replaces pip, venv, and poetry with a single tool.

1. Install uv
### Using pipx (recommended)
pipx install uv

### Or via pip
pip install uv

2. Create and activate a virtual environment
uv venv
source .venv/bin/activate

3. Sync dependencies

uv sync

## install mmseqs

Install `mmseqs2`. This is often easiest with conda:
    ```bash
    conda install -c bioconda mmseqs2
    ```

## Step 2: Download UniProt Data
Create the raw data directory and download the necessary files from UniProt.

Bash

mkdir -p data/raw

#Fragment sequences (FASTA)
wget "https://rest.uniprot.org/uniprotkb/stream?query=(reviewed:true)%20AND%20(fragment:true)&format=fasta" \
  -O data/raw/fragments.fasta

#Fragment annotations (TSV)
wget "https://rest.uniprot.org/uniprotkb/stream?fields=accession,ft_non_cons,ft_non_ter,fragment&query=(reviewed:true)%20AND%20(fragment:true)&format=tsv" \
  -O data/raw/fragment_annotations.tsv

#Complete sequences (FASTA)
wget "https://rest.uniprot.org/uniprotkb/stream?query=(reviewed:true)%20AND%20(fragment:false)&format=fasta" \
  -O data/raw/complete.fasta

## Step 3: Data Preparation Pipeline
These scripts will turn the raw files into a single, non-redundant metadata.csv file ready for training.

Parse Data: This script reads the .fasta and .tsv files, combines them, and uses the FragmentAnnotationParser to correctly label all fragment types.

Basho

python scripts/01_parse_uniprot_data.py
Output: data/processed/metadata_raw.csv

Reduce Redundancy (NEW): This script runs mmseqs2 on all sequences from the previous step and generates a list of representative sequence IDs.

Bash

bash scripts/02_run_mmseqs.sh
Output: data/processed/clustered/representative_ids.txt

Create Splits: This script reads metadata_raw.csv and filters it using representative_ids.txt. It then creates stratified train/validation/test splits from this non-redundant set.

Bash

python scripts/03_create_train_val_test_splits.py
Output: data/processed/metadata.csv (this is the final file used by the DataModule)

## Step 4: Train the Model
Now you are ready to train. All settings are controlled by the config file.

Start Training:

Bash

python train.py --config configs/default.yaml
This will:

Load the FragmentDataModule and FragmentDetector.

Calculate class weights to handle imbalance.

Start training, logging to lightning_logs/ and checkpoints/.

Automatically use the best checkpoint to run a final test.

Monitor Training:

Bash

#In a new terminal
tensorboard --logdir lightning_logs/
Navigate to http://localhost:6006/ to see val/binary_mcc and other metrics.

Override Configs (Optional): You can easily change hyperparameters from the command line:

Bash

#Train with a different batch size and dropout
python train.py --config configs/default.yaml \
  --override data.batch_size=64 model.dropout=0.5

## Step 5: Evaluate the Model
After training, a .ckpt file is saved in checkpoints/. You can use this file to re-run evaluation and generate plots.

Find your best checkpoint (e.g., checkpoints/fragment-detector-epoch=...-val/binary_mcc=...ckpt).

Run the evaluation script:

Bash

python evaluate.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/[your-best-checkpoint-name].ckpt \
  --output-dir results/evaluation
Check Results: Look in results/evaluation/ for:

predictions.csv: Detailed predictions for every sequence in the test set.

binary_confusion_matrix.png: A plot of the confusion matrix.

probability_distributions.png: Histograms of predicted probabilities.

test_metrics.txt: The final test metrics (MCC, AUROC, F1, etc.).