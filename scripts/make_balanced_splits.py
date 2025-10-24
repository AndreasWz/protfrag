#!/usr/bin/env python3
"""
scripts/make_balanced_splits.py

Create cluster-aware train/val/test splits and ensure 50/50 fragment/non-fragment by length-matched sampling.
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_clusters(cluster_tsv):
    # Expects two columns: seq_id \t cluster_id
    df = pd.read_csv(cluster_tsv, sep="\t", header=None, usecols=[0,1], names=["seq","cluster"], dtype=str)
    return dict(zip(df["seq"], df["cluster"]))

def attach_clusters(df, seq2cluster):
    df = df.copy()
    df["cluster"] = df["id"].map(lambda x: seq2cluster.get(str(x), str(x)))
    return df

def cluster_stratified_split(df, cluster_col="cluster", val_frac=0.1, test_frac=0.1, seed=42):
    """
    Split clusters into train/val/test, ensuring that clusters with only one sequence go to train.
    """
    np.random.seed(seed)
    cluster_counts = df[cluster_col].value_counts()
    tiny_clusters = cluster_counts[cluster_counts == 1].index.tolist()
    normal_clusters = cluster_counts[cluster_counts > 1].index.tolist()

    # Assign tiny clusters to train
    train_c = set(tiny_clusters)

    if normal_clusters:
        # For normal clusters, do a simple random split (stratified by fragment ratio if possible)
        temp_df = df[df[cluster_col].isin(normal_clusters)].copy()
        # build a simple stratum: fragment vs non-fragment ratio per cluster
        cg = temp_df.groupby(cluster_col).agg({"is_fragment":"sum","id":"count"})
        cg["frag_ratio"] = cg["is_fragment"]/cg["id"]
        # discretize into 5 bins
        cg["stratum"] = pd.qcut(cg["frag_ratio"], q=5, labels=False, duplicates="drop")
        clusters = cg.index.tolist()
        strata = cg["stratum"].tolist()
        
        # Train + temp (val+test)
        train_clusters, temp_clusters = train_test_split(
            clusters, test_size=(val_frac + test_frac), random_state=seed, stratify=strata
        )
        # Val / Test split
        rel = test_frac / (val_frac + test_frac)
        temp_strata = [cg.loc[c]["stratum"] for c in temp_clusters]
        val_clusters, test_clusters = train_test_split(
            temp_clusters, test_size=rel, random_state=seed, stratify=temp_strata
        )

        train_c.update(train_clusters)
    else:
        val_clusters, test_clusters = [], []

    return set(train_c), set(val_clusters), set(test_clusters)

def length_matched_negative_sampling(part_df, seed=42):
    """
    For a partition (train/val/test), sample negatives to match positive length distribution
    """
    pos = part_df[part_df["is_fragment"]==1].copy()
    neg = part_df[part_df["is_fragment"]==0].copy()

    if len(pos) == 0:
        return part_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    if len(neg) == 0:
        return pos.sample(frac=1, random_state=seed).reset_index(drop=True)

    pos["len_bin"] = pd.qcut(pos["sequence"].str.len(), q=min(20, len(pos)), duplicates="drop")
    try:
        neg["len_bin"] = pd.cut(neg["sequence"].str.len(), bins=pos["len_bin"].cat.categories)
    except Exception:
        neg["len_bin"] = pd.cut(neg["sequence"].str.len(), bins=min(20, len(neg)))

    sampled_negs = []
    for bin_cat, g in pos.groupby("len_bin"):
        n = len(g)
        candidates = neg[neg["len_bin"] == bin_cat]
        if len(candidates) == 0:
            continue
        elif len(candidates) >= n:
            sampled_negs.append(candidates.sample(n=n, random_state=seed))
        else:
            sampled_negs.append(candidates.sample(n=n, replace=True, random_state=seed))

    if sampled_negs:
        neg_samp = pd.concat(sampled_negs)
    else:
        neg_samp = neg.sample(n=len(pos), replace=(len(neg)<len(pos)), random_state=seed)

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
