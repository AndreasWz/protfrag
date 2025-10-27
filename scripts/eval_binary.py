#!/usr/bin/env python3
"""
Robust eval script that:
 - adds repo root to PYTHONPATH,
 - auto-discovers the first LightningModule in models/,
 - auto-discovers a LightningDataModule in data/ if available,
 - evaluates detection logits only and writes a CSV of probs,
 - prints simple metrics if true labels are present.
"""
import sys
import pathlib
repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import os
import glob
import importlib
import inspect
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm

# user config (edit if needed)
CKPT_PATH = "lightning_logs/version_8/checkpoints/epoch=7-step=3200.ckpt"
EMB_DIR = "data/embeddings"
DATA_CSV = "dataset.csv"
BATCH_SIZE = 32
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# helper: dynamically import modules under a package path and find classes
def find_lightningmodule_in_package(pkg_dir, package_name_hint):
    """
    Look through .py files in pkg_dir, import them, and return the first class
    that subclasses pytorch_lightning.LightningModule.
    Returns (module, class) or (None, None).
    """
    try:
        import pytorch_lightning as pl
    except Exception:
        print("pytorch_lightning not importable. Make sure your env has it.")
        return None, None

    mod_candidates = []
    pyfiles = sorted(pathlib.Path(pkg_dir).glob("*.py"))
    for p in pyfiles:
        if p.name.startswith("__"):
            continue
        mod_name = f"{package_name_hint}.{p.stem}"
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            # import may fail if module expects args; skip but print minimal info
            # print(f"Skipping import {mod_name}: {e}")
            continue
        # inspect module for classes subclassing pl.LightningModule
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            try:
                if issubclass(obj, pl.LightningModule) and obj is not pl.LightningModule:
                    print(f"Found LightningModule: {mod_name}:{name}")
                    return mod, obj
            except Exception:
                continue
    return None, None

def find_datamodule_in_package(pkg_dir, package_name_hint):
    try:
        import pytorch_lightning as pl
    except Exception:
        return None, None
    pyfiles = sorted(pathlib.Path(pkg_dir).glob("*.py"))
    for p in pyfiles:
        if p.name.startswith("__"):
            continue
        mod_name = f"{package_name_hint}.{p.stem}"
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            try:
                if issubclass(obj, pl.LightningDataModule) and obj is not pl.LightningDataModule:
                    print(f"Found LightningDataModule: {mod_name}:{name}")
                    return mod, obj
            except Exception:
                continue
    return None, None

def load_model_and_datamodule():
    model_mod, ModelClass = find_lightningmodule_in_package(repo_root/"models", "models")
    if ModelClass is None:
        raise RuntimeError("Could not find a LightningModule in models/. Check that models/ contains a .py with a class subclassing pytorch_lightning.LightningModule.")
    # load checkpoint
    print("Loading model checkpoint:", CKPT_PATH)
    model = ModelClass.load_from_checkpoint(CKPT_PATH)
    model.to(DEVICE)
    model.eval()
    model.freeze()

    # try to find datamodule
    dm_mod, DMClass = find_datamodule_in_package(repo_root/"data", "data")
    dm = None
    if DMClass is not None:
        # try common constructor signatures
        try:
            dm = DMClass(data_csv=DATA_CSV, emb_dir=EMB_DIR, batch_size=BATCH_SIZE)
            print("Instantiated DataModule with (data_csv, emb_dir, batch_size).")
        except TypeError:
            try:
                dm = DMClass(DATA_CSV, EMB_DIR, batch_size=BATCH_SIZE)
                print("Instantiated DataModule with (DATA_CSV, EMB_DIR, batch_size).")
            except Exception as e:
                print("Could not instantiate DataModule automatically:", e)
                dm = None
    else:
        print("No LightningDataModule found in data/ — will fall back to iterating .npy files.")

    return model, dm

def extract_det_logits_from_model(model, batch_tensor):
    # Try to call model and extract detection logits using heuristics
    model_device = DEVICE
    model.to(model_device)
    with torch.no_grad():
        try:
            out = model(batch_tensor.to(model_device))
        except Exception:
            # maybe model expects (x, y, meta) signature
            try:
                out = model(batch_tensor.to(model_device), None, None)
            except Exception as e:
                raise RuntimeError("Model forward failed on a single batch. Inspect the model.forward signature.") from e

    if isinstance(out, dict):
        for key in ['det_logits','det_logit','det','logits','logit','detection_logits','detection']:
            if key in out and torch.is_tensor(out[key]):
                return out[key].detach().cpu().squeeze(-1)
        # search for any tensor-like value with shape (B,1) or (B,)
        for v in out.values():
            if torch.is_tensor(v):
                t = v.detach().cpu()
                if t.ndim == 2 and t.shape[1] == 1:
                    return t.squeeze(1)
                if t.ndim == 1:
                    return t
    # if model has encoder + det_head attributes, use them
    if hasattr(model, "encoder") and hasattr(model, "det_head"):
        x = batch_tensor
        with torch.no_grad():
            h = model.encoder(x.to(DEVICE))
            logits = model.det_head(h)
        return logits.detach().cpu().squeeze(-1)
    # if out is tensor
    if torch.is_tensor(out):
        return out.detach().cpu().squeeze(-1)
    raise RuntimeError("Could not extract detection logits from model output.")

def evaluate_with_datamodule(model, dm):
    # try test_dataloader() then predict_dataloader()
    loader = None
    for name in ("test_dataloader", "predict_dataloader", "val_dataloader", "train_dataloader"):
        if hasattr(dm, name):
            try:
                loader = getattr(dm, name)()
                if loader is not None:
                    print("Using datamodule loader:", name)
                    break
            except Exception:
                continue
    if loader is None:
        raise RuntimeError("Datamodule found but no usable dataloader method returned a loader.")
    all_probs = []
    all_targets = []
    metas = []
    for batch in tqdm(loader, desc="Evaluating dataloader"):
        # common pattern: (x, y, meta) or (x, y)
        if isinstance(batch, (list,tuple)):
            if len(batch) >= 2:
                x = batch[0]
                y = batch[1]
                meta = batch[2] if len(batch) > 2 else None
            else:
                x = batch[0]
                y = None
                meta = None
        else:
            x = batch
            y = None
            meta = None
        logits = extract_det_logits_from_model(model, x)
        probs = torch.sigmoid(torch.tensor(logits)).numpy().ravel().tolist()
        all_probs.extend(probs)
        if y is not None:
            if torch.is_tensor(y):
                all_targets.extend(y.cpu().numpy().ravel().tolist())
            else:
                all_targets.extend(np.array(y).ravel().tolist())
        if meta is not None:
            metas.extend(list(meta))
        else:
            metas.extend([""] * len(probs))
    df = pd.DataFrame({"meta": metas, "prob_det": all_probs})
    df.to_csv("predictions_binary_from_dataloader.csv", index=False)
    print("Wrote predictions_binary_from_dataloader.csv")
    metrics = None
    if len(all_targets) == len(all_probs) and len(all_targets) > 0:
        y = np.array(all_targets).astype(float)
        p = np.array(all_probs)
        auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
        preds = (p >= THRESHOLD).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
        cm = confusion_matrix(y, preds, labels=[0,1])
        metrics = {"auc": auc, "precision": float(prec), "recall": float(rec), "f1": float(f1), "confusion_matrix": cm.tolist()}
        print("Metrics:", metrics)
    else:
        print("No labels found / matched in dataloader; only probabilities written.")
    return df, metrics

def evaluate_on_npys(model, emb_dir):
    files = sorted(glob.glob(os.path.join(emb_dir, "*.npy")))
    rows = []
    for fn in tqdm(files, desc="Eval .npy"):
        emb = np.load(fn)
        x = torch.tensor(emb).unsqueeze(0).float()
        logits = extract_det_logits_from_model(model, x)
        p = float(torch.sigmoid(torch.tensor(logits)).cpu().numpy().ravel()[0])
        rows.append({"file": os.path.basename(fn), "prob_det": p})
    df = pd.DataFrame(rows)
    df.to_csv("predictions_binary_from_npys.csv", index=False)
    print("Wrote predictions_binary_from_npys.csv")
    return df

def main():
    model, dm = load_model_and_datamodule()
    if dm is not None:
        try:
            dm.setup(stage="predict")
        except Exception:
            try:
                dm.setup(stage="test")
            except Exception:
                pass
        evaluate_with_datamodule(model, dm)
    else:
        evaluate_on_npys(model, EMB_DIR)

if __name__ == "__main__":
    main()
