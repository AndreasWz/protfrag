#!/bin/bash
set -euo pipefail

# --- Configuration ---
# Input CSV file from 01_parse_uniprot_data.py
INPUT_CSV="data/processed/metadata_raw.csv"

# Output directory for all clustering data
OUTPUT_DIR="data/processed/clustered"

# Final list of representative (cluster head) IDs
FINAL_ID_LIST="${OUTPUT_DIR}/representative_ids.txt"

# MMseqs parameters
SEQ_ID_THRESH=0.95 # Sequence identity threshold (as in A1)
COV_THRESH=0.8   # Coverage threshold (as in A1)

# Temporary files
FASTA_FILE="${OUTPUT_DIR}/all_sequences.fasta"
DB_NAME="${OUTPUT_DIR}/mmseqs_db"
CLUSTER_DB_NAME="${OUTPUT_DIR}/mmseqs_cluster"
CLUSTER_TSV="${OUTPUT_DIR}/mmseqs_cluster.tsv"
TMP_DIR="${OUTPUT_DIR}/tmp"

# --- Script ---
echo "Starting redundancy reduction..."
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TMP_DIR"

# --- Step 1: Convert metadata_raw.csv to FASTA ---
# Uses awk to grab 'entry' (col 1) and 'sequence' (col 7)
# Assumes comma-separated, no headers (NR>1), and no internal quotes.
echo "Converting CSV to FASTA format..."
awk -F, 'NR>1 {print ">"$1"\n"$7}' "$INPUT_CSV" > "$FASTA_FILE"

if [ ! -s "$FASTA_FILE" ]; then
    echo "Error: Failed to create FASTA file from $INPUT_CSV"
    exit 1
fi
echo "FASTA file created at $FASTA_FILE"

# --- Step 2: Run MMseqs2 Clustering ---
echo "Creating MMseqs database..."
mmseqs createdb "$FASTA_FILE" "$DB_NAME"

echo "Running MMseqs cluster (this may take a while)..."
mmseqs cluster "$DB_NAME" "$CLUSTER_DB_NAME" "$TMP_DIR" \
    --min-seq-id "$SEQ_ID_THRESH" \
    -c "$COV_THRESH" \
    --cov-mode 0 # Coverage of query AND target

echo "Creating cluster TSV..."
mmseqs createtsv "$DB_NAME" "$DB_NAME" "$CLUSTER_DB_NAME" "$CLUSTER_TSV"

# --- Step 3: Extract Representative (Cluster Head) IDs ---
# The cluster TSV format is [cluster_head_id, member_id]
# We just need the unique IDs from the first column.
echo "Extracting representative sequence IDs..."
awk '{print $1}' "$CLUSTER_TSV" | sort -u > "$FINAL_ID_LIST"

# --- Cleanup ---
rm -rf "$TMP_DIR"
echo "Cleanup complete."

echo "---"
echo "✅ Redundancy reduction complete."
echo "Representative ID list saved to: $FINAL_ID_LIST"
echo "Found $(wc -l < "$FINAL_ID_LIST") representative sequences."