"""Precompute per-residue embeddings using HF models (e.g. ProtT5)."""
import argparse
import os
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

def seq_to_spaced(seq: str) -> str:
    return " ".join(list(seq.strip()))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="CSV with columns id,sequence")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model-name", default="Rostlab/prot_t5_xl_uniref50")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.input)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=False)
    model = AutoModel.from_pretrained(args.model_name)
    model = model.to(args.device)
    model.eval()

    for _, row in tqdm(df.iterrows(), total=len(df)):
        seq = row["sequence"]
        sid = row["id"]
        spaced = seq_to_spaced(seq)
        enc = tokenizer(spaced, return_tensors="pt")
        input_ids = enc["input_ids"].to(args.device)
        attention_mask = enc["attention_mask"].to(args.device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            emb = out.last_hidden_state.squeeze(0).cpu().numpy()
        np.save(os.path.join(args.out_dir, f"{sid}.npy"), emb)

if __name__ == "__main__":
    main()
