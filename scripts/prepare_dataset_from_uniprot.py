#!/usr/bin/env python3
"""
scripts/prepare_dataset_from_uniprot.py

Convert UniProt TSVs (fragments + completes) into dataset.csv with columns:
  id,sequence,embedding_path,is_fragment,fragment_types

Conservative heuristics are used to extract N/C/I fragment types from the 'feature' column.
Optionally create a small fraction of synthetic fragments from complete sequences for
clean labeled positives to help training the multilabel head.

Usage:
  python scripts/prepare_dataset_from_uniprot.py \
    --frag-tsv data/uniprot/uniprot_fragments_*.tsv \
    --comp-tsv data/uniprot/uniprot_completes_*.tsv \
    --out dataset.csv --emb-dir data/embeddings --make-synthetic 0

Outputs:
  dataset.csv
"""
import argparse
import os
import re
import pandas as pd
import random

FRAG_TYPE_ORDER = ["N", "C", "I"]


def load_tsv(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    df = df.fillna("")
    return df


def parse_records(df):
    """
    Normalize columns. Return list of dicts with keys: id, sequence, feature_raw, fragment, organism, length
    """
    out = []
    for _, r in df.iterrows():
        # UniProt TSV headers can vary; try common names
        acc = r.get("Entry") or r.get("entry") or r.get("accession") or r.get("Accession")
        seq = r.get("Sequence") or r.get("sequence") or r.get("sequence (canonical)") or r.get("Sequence (canonical)")
        feat = r.get("Feature") or r.get("feature") or r.get("feature(FT)") or r.get("FT") or r.get("feature (FT)")
        frag = r.get("Fragment") or r.get("fragment") or r.get("fragment (yes/no)") or ""
        org = r.get("Organism") or r.get("organism_name") or r.get("Organism (common name)") or ""
        length = r.get("Length") or r.get("length") or ""
        if not acc or not seq:
            continue
        out.append({"id": str(acc).strip(), "sequence": str(seq).strip(), "feature_raw": str(feat), "fragment": str(frag), "organism": str(org), "length": length})
    return out


def heur_parse_types(feature_text, seq):
    """
    Conservative parse of feature text for N/C/I indicators.
    Returns semicolon-separated string (e.g. "N;I") or empty string.
    """
    if not feature_text or str(feature_text).strip() == "":
        return ""

    text = str(feature_text).lower()
    labels = set()

    # explicit keywords
    if any(k in text for k in ["n-terminal", "n terminal", "n-term", "non-terminal", "non_ter", "non ter"]):
        labels.add("N")
    if any(k in text for k in ["c-terminal", "c terminal", "c-term", "non-consecutive at c", "non_consecutive", "non consecutive"]):
        labels.add("C")
    if any(k in text for k in ["gap", "non-consecutive", "non consecutive", "internal", "missing internal", "internal gap"]):
        labels.add("I")

    # position ranges like "5-120" inside feature text: if start>1 -> N; if end < seq_len -> C
    pos_matches = re.findall(r"(\d+)\s*-\s*(\d+)", text)
    for a, b in pos_matches:
        try:
            a = int(a)
            b = int(b)
            if a > 1:
                labels.add("N")
            if b < len(seq):
                labels.add("C")
        except Exception:
            pass

    if not labels:
        return ""
    # order by FRAG_TYPE_ORDER
    ordered = sorted(labels, key=lambda x: FRAG_TYPE_ORDER.index(x))
    return ";".join(ordered)


def generate_dataset(frag_tsv, comp_tsv, out_csv, emb_dir=None, make_synthetic=0, synthetic_fraction=0.05, seed=42):
    random.seed(seed)
    frag_df = load_tsv(frag_tsv)
    comp_df = load_tsv(comp_tsv)
    fr_recs = parse_records(frag_df)
    co_recs = parse_records(comp_df)

    rows = []
    # fragments: is_fragment=1
    for r in fr_recs:
        types = heur_parse_types(r["feature_raw"], r["sequence"])
        emb_path = os.path.join(emb_dir, f"{r['id']}.npy") if emb_dir else ""
        rows.append({"id": r["id"], "sequence": r["sequence"], "embedding_path": emb_path, "is_fragment": 1, "fragment_types": types, "feature_raw": r["feature_raw"]})

    # completes: is_fragment=0
    for r in co_recs:
        emb_path = os.path.join(emb_dir, f"{r['id']}.npy") if emb_dir else ""
        rows.append({"id": r["id"], "sequence": r["sequence"], "embedding_path": emb_path, "is_fragment": 0, "fragment_types": "", "feature_raw": r.get("feature_raw", "")})

    # optional synthetic fragments from completes (small fraction)
    if make_synthetic:
        n_synth = int(len(co_recs) * synthetic_fraction)
        sample = random.sample(co_recs, min(n_synth, len(co_recs)))
        for r in sample:
            seq = r["sequence"]
            L = len(seq)
            if L < 20:
                continue
            # choose type and cut points
            ttype = random.choice(["N", "C", "I"])
            if ttype == "N":
                cut = random.randint(1, max(1, L // 4))
                new_seq = seq[cut:]
                types = "N"
            elif ttype == "C":
                cut = random.randint(L - max(1, L // 4), L - 1)
                new_seq = seq[:cut]
                types = "C"
            else:  # internal: remove a middle chunk
                a = random.randint(1, max(1, L // 4))
                b = random.randint(L - max(1, L // 4), L - 1)
                if a >= b:
                    continue
                new_seq = seq[:a] + seq[b:]
                types = "I"
            sid = r["id"] + "_synth"
            emb_path = ""  # recompute embeddings later for these
            rows.append({"id": sid, "sequence": new_seq, "embedding_path": emb_path, "is_fragment": 1, "fragment_types": types, "feature_raw": f"synth:{ttype}"})

    df_out = pd.DataFrame(rows)
    # Keep columns consistent
    df_out = df_out[["id", "sequence", "embedding_path", "is_fragment", "fragment_types", "feature_raw"]]
    df_out.to_csv(out_csv, index=False)
    print(f"Saved dataset to {out_csv} ({len(df_out)} records).")
    print("Note: fragment_types are heuristically parsed and will be noisy; keep feature_raw for auditing.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frag-tsv", required=True)
    p.add_argument("--comp-tsv", required=True)
    p.add_argument("--out", default="dataset.csv")
    p.add_argument("--emb-dir", default=None)
    p.add_argument("--make-synthetic", type=int, default=0, help="0 or 1")
    p.add_argument("--synthetic-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    generate_dataset(args.frag_tsv, args.comp_tsv, args.out, emb_dir=args.emb_dir, make_synthetic=args.make_synthetic, synthetic_fraction=args.synthetic_fraction, seed=args.seed)
