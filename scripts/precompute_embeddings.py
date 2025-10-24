"""Precompute per-residue embeddings using HF models (e.g., ProtT5)."""
import argparse
import os
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


def seq_to_spaced(seq: str) -> str:
    """Convert sequence to space-separated letters for ProtT5."""
    return " ".join(list(seq.strip()))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV with columns id,sequence")
    parser.add_argument("--out-dir", required=True, help="Directory to save .npy embeddings")
    parser.add_argument("--model-name", default="Rostlab/prot_t5_xl_uniref50", help="HF model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-length", type=int, default=1022, help="Truncate sequences longer than this")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.input)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModel.from_pretrained(args.model_name)
    model = model.to(args.device)
    model.eval()

    for row in tqdm(df.itertuples(index=False), total=len(df)):
        seq = getattr(row, "sequence")
        sid = str(getattr(row, "id"))

        # Truncate if sequence is too long
        if len(seq) > args.max_length:
            seq = seq[:args.max_length]

        spaced_seq = seq_to_spaced(seq)
        enc = tokenizer(spaced_seq, return_tensors="pt")
        input_ids = enc["input_ids"].to(args.device)
        attention_mask = enc["attention_mask"].to(args.device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            # handle possible tuple output
            if isinstance(out, tuple):
                emb = out[0].squeeze(0).cpu().numpy()
            else:
                emb = out.last_hidden_state.squeeze(0).cpu().numpy()

        # Save embeddings
        np.save(os.path.join(args.out_dir, f"{sid}.npy"), emb)

        # free memory
        del input_ids, attention_mask, out, emb
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
