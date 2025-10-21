#!/usr/bin/env python3
# scripts/fetch_uniprot_sets.py
"""
Fetch reviewed UniProt sequences for fragments and completes.
Saves TSVs: out_fragments.tsv, out_completes.tsv, and combined raw JSON for traceability.

Usage:
  python scripts/fetch_uniprot_sets.py --out-dir data/uniprot --page-size 500
"""
import requests, time, argparse, os
from urllib.parse import urlencode
from datetime import datetime

BASE = "https://rest.uniprot.org/uniprotkb/search"

def fetch_tsv(query, fields, size=500, sleep=0.2, max_pages=None):
    params = {"query": query, "fields": ",".join(fields), "format": "tsv", "size": size}
    url = BASE
    results = []
    page = 0
    while True:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        text = r.text
        if page == 0:
            results.append(text)
        else:
            # drop header on subsequent pages
            results.append("\n".join(text.splitlines()[1:]))
        page += 1

        # Try to parse Link header to find next page (UniProt uses Link: <...cursor=...>; rel="next")
        link = r.headers.get("Link", "")
        next_url = None
        if link:
            # Look for rel="next" portion
            parts = link.split(",")
            for p in parts:
                if 'rel="next"' in p:
                    # extract url between <>
                    if "<" in p and ">" in p:
                        next_url = p.split("<")[1].split(">")[0]
                        break
        if not next_url:
            break
        # follow next_url (it already contains full query + cursor)
        url = next_url
        params = None  # next_url includes params
        time.sleep(sleep)
        if max_pages and page >= max_pages:
            break
    return "\n".join(results)

def save_text(txt, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(txt)
    print("Saved", out_path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data/uniprot")
    p.add_argument("--page-size", type=int, default=500)
    p.add_argument("--max-pages", type=int, default=None)
    args = p.parse_args()

    fields = ["accession,sequence,feature,fragment,organism_name,length".replace(" ", "")]
    # fields as string list — UniProt expects the exact field names; we'll pass a simple list below:
    # but restful API wants comma separated; we pass explicit names:
    fields = ["accession", "sequence", "feature", "fragment", "organism_name", "length"]

    ts = datetime.utcnow().isoformat()
    print("Starting fetch:", ts)

    frag_q = "(reviewed:true) AND (fragment:true)"
    comp_q = "(reviewed:true) AND (fragment:false)"

    frag_txt = fetch_tsv(frag_q, fields, size=args.page_size, max_pages=args.max_pages)
    comp_txt = fetch_tsv(comp_q, fields, size=args.page_size, max_pages=args.max_pages)

    save_text(frag_txt, os.path.join(args.out_dir, f"uniprot_fragments_{ts}.tsv"))
    save_text(comp_txt, os.path.join(args.out_dir, f"uniprot_completes_{ts}.tsv"))
    print("Done.")

if __name__ == "__main__":
    main()
