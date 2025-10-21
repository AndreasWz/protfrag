#!/usr/bin/env python3
"""
scripts/fetch_uniprot_sets.py

Fetch reviewed UniProt sequences with fragment:true and fragment:false.
Handles pagination via Link header (rel="next"). Saves timestamped TSV files.

Usage:
  python scripts/fetch_uniprot_sets.py --out-dir data/uniprot --page-size 500

Notes:
- Keep raw TSVs for debugging and provenance.
"""
import os
import time
import argparse
import requests
from datetime import datetime

BASE = "https://rest.uniprot.org/uniprotkb/search"


def fetch_tsv(query, fields, size=500, sleep=0.2, max_pages=None):
    params = {"query": query, "fields": ",".join(fields), "format": "tsv", "size": size}
    url = BASE
    page = 0
    out_parts = []
    while True:
        r = requests.get(url, params=params, timeout=60)
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data/uniprot")
    p.add_argument("--page-size", type=int, default=500)
    p.add_argument("--max-pages", type=int, default=None)
    args = p.parse_args()

    fields = ["accession", "sequence", "feature", "fragment", "organism_name", "length"]
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    frag_q = "(reviewed:true) AND (fragment:true)"
    comp_q = "(reviewed:true) AND (fragment:false)"

    print("Fetching fragments...")
    frag_txt = fetch_tsv(frag_q, fields, size=args.page_size, max_pages=args.max_pages)
    frag_path = os.path.join(args.out_dir, f"uniprot_fragments_{ts}.tsv")
    save_text(frag_txt, frag_path)

    print("Fetching completes...")
    comp_txt = fetch_tsv(comp_q, fields, size=args.page_size, max_pages=args.max_pages)
    comp_path = os.path.join(args.out_dir, f"uniprot_completes_{ts}.tsv")
    save_text(comp_txt, comp_path)

    print("Done. Keep these raw TSVs for provenance and debugging.")


if __name__ == "__main__":
    main()
