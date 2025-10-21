#!/usr/bin/env python3
"""
scripts/evaluate_model.py

Load a Lightning checkpoint and evaluate on the test split.
Computes: MCC, ROC-AUC, PR-AUC, confusion matrix and per-type precision/recall/F1.
Saves a CSV of predictions with id, true labels and probabilities.

Usage:
  python scripts/evaluate_model.py --ckpt path/to/ckpt --split-csv data/splits/test.csv --out-dir results/eval --batch-size 64 --device cpu

Notes:
 - The model class must be available as models.fragment_detector.FragmentDetector
 - The dataset CSV should contain embedding_path (pointing to .npy files). The dataloader will load embeddings via your existing DataModule/Dataset.
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef, roc_auc_score, precision_recall_fscore_support, confusion_matrix, average_precision_score
from data.datamodule import ProtFragDataModule
from models.fragment_detector import FragmentDetector
from torch.utils.data import DataLoader
from data.dataset import EmbeddingSequenceDataset

def load_split_df(split_csv):
    return pd.read_csv(split_csv)

def make_dataloader_from_df(df, batch_size=64, pool="mean", num_workers=2):
    records = df.to_dict(orient="records")
    ds = EmbeddingSequenceDataset(records, pool=pool)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dl

def evaluate(ckpt, split_csv, out_dir, batch_size=64, device="cpu"):
    device = torch.device(device)
    os.makedirs(out_dir, exist_ok=True)
    # load model from checkpoint (Lightning will restore hyperparams)
    model = FragmentDetector.load_from_checkpoint(ckpt)
    model.eval()
    model.to(device)

    df = load_split_df(split_csv)
    dl = make_dataloader_from_df(df, batch_size=batch_size)

    ids = []
    y_true = []
    y_type_true = []
    y_probs = []
    y_type_probs = []

    with torch.no_grad():
        for batch in dl:
            emb = batch["emb"].to(device)
            det_logit, type_logits = model(emb)
            probs = torch.sigmoid(det_logit).cpu().numpy()
            types_p = torch.sigmoid(type_logits).cpu().numpy()

            ids.extend(batch["id"])
            y_true.extend(batch["is_fragment"].cpu().numpy().astype(int).tolist())
            y_type_true.extend(batch["fragment_types"].cpu().numpy().tolist())
            y_probs.extend(probs.tolist())
            y_type_probs.extend(types_p.tolist())

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_type_true = np.array(y_type_true)
    y_type_probs = np.array(y_type_probs)

    # Binary metrics
    try:
        roc = roc_auc_score(y_true, y_probs)
    except Exception:
        roc = float("nan")
    try:
        ap = average_precision_score(y_true, y_probs)
    except Exception:
        ap = float("nan")
    y_pred = (y_probs >= 0.5).astype(int)
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except Exception:
        mcc = float("nan")
    cm = confusion_matrix(y_true, y_pred)

    # Multilabel per-type metrics (precision/recall/f1 per label + macro/micro)
    per_type = {}
    if y_type_true.size > 0:
        for i in range(y_type_true.shape[1]):
            try:
                p, r, f, _ = precision_recall_fscore_support(y_type_true[:, i], (y_type_probs[:, i] >= 0.5).astype(int), average="binary", zero_division=0)
                per_type[f"type_{i}"] = {"precision": float(p), "recall": float(r), "f1": float(f)}
            except Exception:
                per_type[f"type_{i}"] = {"precision": None, "recall": None, "f1": None}
    # Save predictions CSV
    pred_df = pd.DataFrame({
        "id": ids,
        "is_fragment_true": y_true.tolist(),
        "is_fragment_prob": y_probs.tolist()
    })
    # expand type probs/true if available
    if y_type_true.size > 0:
        for i in range(y_type_true.shape[1]):
            pred_df[f"type_{i}_true"] = y_type_true[:, i].tolist()
            pred_df[f"type_{i}_prob"] = y_type_probs[:, i].tolist()

    pred_csv = os.path.join(out_dir, "predictions.csv")
    pred_df.to_csv(pred_csv, index=False)

    # Results summary
    summary = {
        "roc_auc": float(roc) if not np.isnan(roc) else None,
        "average_precision": float(ap) if not np.isnan(ap) else None,
        "mcc": float(mcc) if not np.isnan(mcc) else None,
        "confusion_matrix": cm.tolist(),
        "per_type": per_type,
        "n_samples": int(len(y_true))
    }
    summary_csv = os.path.join(out_dir, "summary.json")
    import json
    with open(summary_csv, "w") as f:
        json.dump(summary, f, indent=2)

    print("Evaluation complete. Saved:", pred_csv, summary_csv)
    print("ROC-AUC:", summary["roc_auc"], "AP:", summary["average_precision"], "MCC:", summary["mcc"])
    for k,v in per_type.items():
        print(k, v)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split-csv", required=True, help="CSV for test split (with embedding_path)")
    p.add_argument("--out-dir", default="results/eval")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    evaluate(args.ckpt, args.split_csv, args.out_dir, batch_size=args.batch_size, device=args.device)
