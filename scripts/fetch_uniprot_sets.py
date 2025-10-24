#!/usr/bin/env python3
"""
scripts/fetch_uniprot_sets.py (fixed minimal fields)

Fetch reviewed UniProt sequences with fragment:true and fragment:false.
Handles pagination via Link header (rel="next"). Saves timestamped TSV files.

If --include-features is provided, the script will also fetch per-accession JSON
and extract the 'features' field for each accession (this is slower but reliable).

Usage:
  python scripts/fetch_uniprot_sets.py --out-dir data/uniprot --page-size 500
  python scripts/fetch_uniprot_sets.py --out-dir data/uniprot --page-size 500 --include-features 1
"""
import os
import time
import argparse
import requests
from datetime import datetime
from urllib.parse import urlparse, parse_qs

BASE = "https://rest.uniprot.org/uniprotkb/search"
ENTRY_JSON = "https://rest.uniprot.org/uniprotkb/{}.json"


def fetch_tsv(query, fields, size=500, sleep=0.2, max_pages=None):
    params = {"query": query, "fields": ",".join(fields), "format": "tsv", "size": size}
    url = BASE
    page = 0
    out_parts = []
    while True:
        r = requests.get(url, params=params, timeout=60)
        if r.status_code != 200:
            # print helpful debug information from UniProt
            print(f"HTTP {r.status_code} from UniProt for url: {r.url}")
            try:
                print("Response body:", r.text[:1000])
            except Exception:
                pass
            r.raise_for_status()
        txt = r.text
        if page == 0:
            out_parts.append(txt)
        else:
            # drop header on subsequent pages
            out_parts.append("\n".join(txt.splitlines()[1:]))
        page += 1

        link = r.headers.get("Link", "")
        next_url = None
        if link:
            # look for rel="next"
            for part in link.split(","):
                if 'rel="next"' in part:
                    if "<" in part and ">" in part:
                        next_url = part.split("<")[1].split(">")[0]
                        break
        if not next_url:
            break
        url = next_url
        params = None  # next_url contains params already
        time.sleep(sleep)
        if max_pages and page >= max_pages:
            break
    return "\n".join(out_parts)


def save_text(txt, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(txt)
    print("Saved:", path)


def fetch_features_for_accessions(accessions, sleep=0.05):
    """
    Fetch per-accession JSON and return a dict accession -> features_text (concatenated features).
    This is slow (one request per accession) but robust. Keep sleep small to avoid overloading the API.
    """
    out = {}
    for i, acc in enumerate(accessions):
        url = ENTRY_JSON.format(acc)
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            print(f"Warning: got HTTP {r.status_code} for accession {acc}; skipping features")
            out[acc] = ""
            continue
        try:
            j = r.json()
            feats = j.get("features", [])
            feat_lines = []
            for f in feats:
                t = f.get("type", "")
                desc = f.get("description", "")
                loc = f.get("location", {})
                pos_txt = ""
                if "begin" in loc and "end" in loc:
                    pos_txt = f"{loc['begin'].get('position','?')}-{loc['end'].get('position','?')}"
                elif "position" in loc:
                    pos_txt = str(loc["position"].get("position","?"))
                feat_lines.append("|".join([t, desc, pos_txt]))
            out[acc] = " ; ".join(feat_lines)
        except Exception as e:
            print(f"Failed to parse JSON for {acc}: {e}")
            out[acc] = ""
        time.sleep(sleep)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data/uniprot")
    p.add_argument("--page-size", type=int, default=500)
    p.add_argument("--max-pages", type=int, default=None)
    p.add_argument("--include-features", type=int, default=0, help="If 1, fetch per-accession JSON to extract features (slow)")
    args = p.parse_args()

    # Use a minimal set of safe return fields. 'entry_name' and some other names are not accepted by the TSV API.
    fields = ["accession", "sequence", "length"]
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    frag_q = "(reviewed:true) AND (fragment:true)"
    comp_q = "(reviewed:true) AND (fragment:false)"

    print("Fetching fragments (TSV)...")
    frag_txt = fetch_tsv(frag_q, fields, size=args.page_size, max_pages=args.max_pages)
    frag_path = os.path.join(args.out_dir, f"uniprot_fragments_{ts}.tsv")
    save_text(frag_txt, frag_path)

    print("Fetching completes (TSV)...")
    comp_txt = fetch_tsv(comp_q, fields, size=args.page_size, max_pages=args.max_pages)
    comp_path = os.path.join(args.out_dir, f"uniprot_completes_{ts}.tsv")
    save_text(comp_txt, comp_path)

    if args.include_features:
        import pandas as pd

        frag_df = pd.read_csv(frag_path, sep="\t", dtype=str).fillna("")
        comp_df = pd.read_csv(comp_path, sep="\t", dtype=str).fillna("")
        # accession column in TSV is usually named 'Entry' or 'accession' depending on API; try both
        acc_col = "Entry" if "Entry" in frag_df.columns else ("accession" if "accession" in frag_df.columns else frag_df.columns[0])
        accs = list(pd.concat([frag_df[acc_col].astype(str), comp_df[acc_col].astype(str)]).unique())
        print(f"Fetching features for {len(accs)} accessions (this will take time)...")
        feat_map = fetch_features_for_accessions(accs, sleep=0.05)

        def attach_features(df):
            df = df.copy()
            df["features_text"] = df[acc_col].map(lambda x: feat_map.get(str(x), ""))
            return df

        frag_df2 = attach_features(frag_df)
        comp_df2 = attach_features(comp_df)
        frag_path2 = os.path.join(args.out_dir, f"uniprot_fragments_{ts}_with_features.tsv")
        comp_path2 = os.path.join(args.out_dir, f"uniprot_completes_{ts}_with_features.tsv")
        frag_df2.to_csv(frag_path2, sep="\t", index=False)
        comp_df2.to_csv(comp_path2, sep="\t", index=False)
        print("Saved TSVs with features:", frag_path2, comp_path2)

    print("Done. Keep these raw TSVs for provenance and debugging.")


if __name__ == "__main__":
    main()
