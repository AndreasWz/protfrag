#!/usr/bin/env python3
"""
scripts/make_balanced_splits.py

Create cluster-aware train/val/test splits and ensure 50/50 fragment/non-fragment by length-matched sampling.

Inputs:
 - dataset.csv (id,sequence,is_fragment,...)
 - clusters.tsv (mmseqs `createtsv` like: query_id \t representative_id) or a mapping with cluster ids
    This script expects a two-column TSV mapping seq_id -> cluster_id. If mmseqs outputs different format,
    adapt the load_clusters() function.

Outputs:
 - data/splits/train.csv, val.csv, test.csv (balanced 50/50 in each split)

Usage:
  python scripts/make_balanced_splits.py --dataset dataset.csv --clusters mmseqs_clusters.tsv --out-dir data/splits --seed 42
"""
import argparse
import os
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

def load_clusters(cluster_tsv):
    # Expects two columns: seq_id \t cluster_id (or seq_id \t rep_id). Adapt if your format differs.
    df = pd.read_csv(cluster_tsv, sep="\t", header=None, usecols=[0,1], names=["seq","cluster"], dtype=str)
    return dict(zip(df["seq"], df["cluster"]))

def attach_clusters(df, seq2cluster):
    df = df.copy()
    df["cluster"] = df["id"].map(lambda x: seq2cluster.get(str(x), str(x)))
    return df

def cluster_stratified_split(df, seed=42, val_frac=0.1, test_frac=0.1):
    # Build cluster-level strata using frag ratio and length bin
    df["length"] = df["sequence"].str.len()
    # Create length bin
    df["len_bin"] = pd.qcut(df["length"].rank(method="first"), q=10, labels=False, duplicates="drop")
    # aggregate per-cluster
    cg = df.groupby("cluster").agg({"is_fragment":"sum", "id":"count", "len_bin": lambda x: x.mode().iloc[0]})
    cg.columns = ["frag_sum","count","len_bin_mode"]
    cg["frag_ratio"] = cg["frag_sum"] / cg["count"]
    # build strata label
    cg = cg.reset_index()
    cg["frag_bin"] = pd.qcut(cg["frag_ratio"].rank(method="first"), q=5, labels=False, duplicates="drop")
    cg["stratum"] = cg["frag_bin"].astype(str) + "_" + cg["len_bin_mode"].astype(str)
    # split clusters
    clusters = cg["cluster"].tolist()
    strata = cg["stratum"].tolist()
    train_c, temp_c, train_s, temp_s = train_test_split(clusters, strata, test_size=(val_frac + test_frac), random_state=seed, stratify=strata)
    # split temp into val/test
    rel = test_frac / (val_frac + test_frac)
    val_c, test_c = train_test_split(temp_c, test_size=rel, random_state=seed, stratify=[cg.set_index("cluster").loc[c]["stratum"] for c in temp_c])
    return set(train_c), set(val_c), set(test_c)

def length_matched_negative_sampling(part_df, seed=42):
    # For a given partition (train/val/test), sample negatives to match positive length distribution
    pos = part_df[part_df["is_fragment"]==1].copy()
    neg = part_df[part_df["is_fragment"]==0].copy()
    if len(pos)==0:
        # nothing to match
        return part_df.sample(frac=1, random_state=seed)
    pos["len_bin"] = pd.qcut(pos["sequence"].str.len(), q=20, duplicates="drop")
    # Use pos bins to cut negatives
    try:
        neg["len_bin"] = pd.cut(neg["sequence"].str.len(), bins=pos["len_bin"].cat.categories)
    except Exception:
        # fallback coarse binning
        neg["len_bin"] = pd.cut(neg["sequence"].str.len(), bins=20)
    sampled_negs = []
    for bin_cat, g in pos.groupby("len_bin"):
        n = len(g)
        candidates = neg[neg["len_bin"] == bin_cat]
        if len(candidates) >= n and n>0:
            sampled_negs.append(candidates.sample(n=n, random_state=seed))
        elif n>0:
            sampled_negs.append(candidates.sample(n=n, replace=True, random_state=seed))
    if sampled_negs:
        neg_samp = pd.concat(sampled_negs)
    else:
        # no negs in same length bins: sample uniformly
        neg_samp = neg.sample(n=len(pos), replace=True if len(neg)<len(pos) else False, random_state=seed)
    balanced = pd.concat([pos, neg_samp]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return balanced

def make_splits(dataset_csv, cluster_tsv, out_dir, seed=42):
    df = pd.read_csv(dataset_csv)
    seq2cluster = load_clusters(cluster_tsv)
    df = attach_clusters(df, seq2cluster)
    train_c, val_c, test_c = cluster_stratified_split(df, seed=seed)
    df["split"] = df["cluster"].map(lambda c: "train" if c in train_c else ("val" if c in val_c else ("test" if c in test_c else "train")))
    os.makedirs(out_dir, exist_ok=True)
    for s in ["train","val","test"]:
        part = df[df["split"]==s]
        balanced = length_matched_negative_sampling(part, seed=seed)
        balanced.to_csv(os.path.join(out_dir, f"{s}.csv"), index=False)
        print(f"Wrote {s}.csv: {len(balanced)} rows, frags={balanced['is_fragment'].sum()}")
    print("Finished splits. Each split is length-matched and balanced 50/50 fragment/non-fragment.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="dataset.csv produced by prepare_dataset_from_uniprot.py")
    p.add_argument("--clusters", required=True, help="mmseqs cluster mapping tsv (seq_id\\tcluster_id)")
    p.add_argument("--out-dir", default="data/splits")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    make_splits(args.dataset, args.clusters, args.out_dir, seed=args.seed)
