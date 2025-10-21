#!/usr/bin/env python3
# scripts/prepare_dataset_from_uniprot.py
"""
Convert UniProt TSV(s) into dataset.csv with columns:
id,sequence,embedding_path,is_fragment,fragment_types

Conservative heuristic parsing of 'feature' column for fragment-type labels N,C,I.
Optionally generate synthetic fragments by truncation (augmentation).

Usage:
  python scripts/prepare_dataset_from_uniprot.py --frag-tsv data/uniprot/uniprot_fragments_....tsv \
      --comp-tsv data/uniprot/uniprot_completes_....tsv \
      --out dataset.csv --emb-dir data/embeddings --make-synthetic 0
"""
import argparse, csv, os, re, pandas as pd
from collections import defaultdict
import random

FRAG_TYPE_ORDER = ["N", "C", "I"]

def parse_tsv_to_records(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    df = df.fillna("")
    records = []
    for _, r in df.iterrows():
        acc = r.get("Entry") or r.get("accession") or r.get("Entry name") or r.get("Entry")
        seq = r.get("Sequence") or r.get("sequence") or r.get("Sequence (canonical)") or r.get("sequence")
        feat = r.get("Feature") or r.get("feature") or r.get("feature(FT)") or r.get("FT")
        frag_flag = r.get("Fragment") or r.get("fragment") or ""
        org = r.get("Organism") or r.get("organism_name") or ""
        length = r.get("Length") or r.get("length") or ""
        if not acc or not seq:
            continue
        records.append({"id": acc, "sequence": seq.strip(), "feature_raw": feat, "fragment": frag_flag, "organism": org, "length": length})
    return records

def heur_parse_types(feature_text, seq):
    """
    Conservative heuristics: inspect feature_text for tokens that indicate N/C/internal gaps.
    Returns semicolon-separated string of labels among N,C,I, or empty string.
    """
    if not feature_text or str(feature_text).strip()=="":
        return ""
    text = str(feature_text).lower()
    labels = set()
    # N-terminal indicators
    if "n-terminal" in text or "n terminal" in text or "n-term" in text or "non-terminal" in text or "non_ter" in text:
        labels.add("N")
    # C-terminal
    if "c-terminal" in text or "c terminal" in text or "c-term" in text:
        labels.add("C")
    # internal gap / non-consecutive / gap / missing internal
    if "gap" in text or "non-consecutive" in text or "non consecutive" in text or "internal" in text or "missing internal" in text:
        labels.add("I")
    # some UniProt textual patterns include 'chain' 'fragment' w/ positions; try to detect position ranges like '5-120'
    # If we find explicit positions and they don't start at 1 -> possible N-terminal truncation; if they end before seq len -> possible C-terminal
    pos_matches = re.findall(r"(\d+)\s*-\s*(\d+)", text)
    if pos_matches:
        for a,b in pos_matches:
            a,b = int(a), int(b)
            if a > 1:
                labels.add("N")
            if b < len(seq):
                labels.add("C")
    return ";".join(sorted(labels, key=lambda x: FRAG_TYPE_ORDER.index(x))) if labels else ""

def generate_dataset(frag_tsv, comp_tsv, out_csv, emb_dir=None, make_synthetic=0, synthetic_fraction=0.05, seed=42):
    random.seed(seed)
    frag_recs = parse_tsv_to_records(frag_tsv)
    comp_recs = parse_tsv_to_records(comp_tsv)
    rows = []
    # fragments: is_fragment=1, try to parse types
    for r in frag_recs:
        types = heur_parse_types(r["feature_raw"], r["sequence"])
        emb_path = os.path.join(emb_dir, f"{r['id']}.npy") if emb_dir else ""
        rows.append({"id": r["id"], "sequence": r["sequence"], "embedding_path": emb_path, "is_fragment": 1, "fragment_types": types})
    # completes: is_fragment=0
    for r in comp_recs:
        emb_path = os.path.join(emb_dir, f"{r['id']}.npy") if emb_dir else ""
        rows.append({"id": r["id"], "sequence": r["sequence"], "embedding_path": emb_path, "is_fragment": 0, "fragment_types": ""})
    # optional synthetic fragments: randomly truncate some complete sequences to create controlled positives
    if make_synthetic:
        n_synth = int(len(comp_recs) * synthetic_fraction)
        sample = random.sample(comp_recs, min(n_synth, len(comp_recs)))
        for r in sample:
            seq = r["sequence"]
            L = len(seq)
            if L < 20:
                continue
            # choose type and cut points
            ttype = random.choice(["N", "C", "I"])
            if ttype == "N":
                cut = random.randint(1, max(1, L//3))
                new_seq = seq[cut:]
                types = "N"
            elif ttype == "C":
                cut = random.randint(L - max(1, L//3), L-1)
                new_seq = seq[:cut]
                types = "C"
            else:  # internal: remove a middle chunk and glue ends
                a = random.randint(1, max(1, L//3))
                b = random.randint(L - max(1, L//3), L-1)
                if a >= b:
                    continue
                new_seq = seq[:a] + seq[b:]
                types = "I"
            sid = r["id"] + "_synth"
            emb_path = ""  # embedding will need recomputation; leave blank or compute separately
            rows.append({"id": sid, "sequence": new_seq, "embedding_path": emb_path, "is_fragment": 1, "fragment_types": types})
    # write CSV
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("Saved dataset with", len(df), "rows to", out_csv)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frag-tsv", required=True)
    p.add_argument("--comp-tsv", required=True)
    p.add_argument("--out", default="dataset.csv")
    p.add_argument("--emb-dir", default=None)
    p.add_argument("--make-synthetic", type=int, default=0)
    args = p.parse_args()
    generate_dataset(args.frag_tsv, args.comp_tsv, args.out, emb_dir=args.emb_dir, make_synthetic=args.make_synthetic)
