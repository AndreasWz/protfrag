#!/usr/bin/env bash
set -e
DB=mmseqs_db
OUT=cluster_res
seqs_fasta=$1  # input FASTA
mkdir -p mmseqs_tmp
mmseqs createdb "$seqs_fasta" $DB mmseqs_tmp
mmseqs cluster $DB ${DB}_cluster mmseqs_tmp tmp --min-seq-id 0.95 -c 0.8
mmseqs createtsv $DB ${DB}_cluster ${OUT}.tsv
