import pandas as pd
import h5py
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

# --- KONFIGURATION: Pfade zu DEINEN Dateien ---

# 1. Liste deiner CSVs (Metadaten von deinem Kollegen)
CSV_FILES = [
    "data/uniprot/dataset1_metadata_sampled_12000.csv",
    "data/uniprot/dataset2_metadata_sampled_26000.csv",
    "data/uniprot/dataset3_metadata_sampled_12000.csv"
]

# 2. Liste deiner H5s (Embeddings - NUR DIE NEUEN)
H5_FILES = [
    "data/uniprot/bulk_embeddings/dataset1_embeddings_12000.h5",
    "data/uniprot/bulk_embeddings/dataset2_embeddings_26000.h5",
    "data/uniprot/bulk_embeddings/dataset3_embeddings_12000.h5"
]

# 3. Wo sollen die Ergebnisse hin?
OUTPUT_METADATA = "data/processed/metadata.csv"
OUTPUT_EMBEDDINGS_DIR = "data/embeddings"
# ----------------------------------------------

def map_frag_type(frag_type_str):
    """Wandelt die 'frag_type'-Spalte in unsere 3 Ziel-Label um."""
    s = str(frag_type_str).lower()
    return {
        'n_terminal': 1 if 'n' in s or 'both' in s else 0,
        'c_terminal': 1 if 'c' in s or 'both' in s else 0,
        'internal': 1 if 'internal' in s or 'mixed' in s else 0,
    }

def load_h5_handles(paths):
    handles = {}
    print("Loading H5 files...")
    for p in paths:
        try:
            h = h5py.File(p, 'r')
            handles[Path(p).name] = h
            print(f"  Loaded {p}")
        except Exception as e:
            print(f"  ERROR loading {p}: {e}")
    return handles

def find_embedding(handles, key):
    """Sucht einen Key (ID) in allen offenen H5-Dateien."""
    
    # Strategie: Wir suchen in 'orig_id' (Dataset) oder direkt als Key
    for name, h in handles.items():
        # Fall A: ID ist ein Key (falls die H5 so strukturiert ist)
        if key in h:
            return h[key][:]
        
        # Fall B: ID ist in einem 'orig_id' Dataset gespeichert (wahrscheinlich bei deinen neuen Files)
        if 'orig_id' in h:
            try:
                # Caching für Speed (wir lesen die IDs nur einmal pro Datei)
                if not hasattr(h, '_cached_ids'):
                    ids = h['orig_id'][:]
                    if ids.dtype == 'object': 
                        h._cached_ids = [s.decode('utf-8') for s in ids]
                    else:
                        h._cached_ids = list(ids)
                
                if key in h._cached_ids:
                    idx = h._cached_ids.index(key)
                    # Embedding holen (meistens unter 'embeddings')
                    if 'embeddings' in h:
                        return h['embeddings'][idx]
            except: pass
            
    return None

def main():
    Path(OUTPUT_EMBEDDINGS_DIR).mkdir(parents=True, exist_ok=True)
    
    # 1. H5s laden
    handles = load_h5_handles(H5_FILES)
    if not handles:
        print("Keine H5 Dateien geladen. Abbruch.")
        return

    # 2. CSVs laden und kombinieren
    print("Lade CSVs...")
    dfs = []
    for csv_path in CSV_FILES:
        try:
            df = pd.read_csv(csv_path)
            # Hier nehmen wir an, dass die CSVs schon Splits haben, oder wir weisen sie zu.
            # Wenn keine Split-Info da ist, müssen wir sie später erstellen.
            # Aber laut deiner Beschreibung SIND das schon Train/Val/Test sets.
            # Wir müssen wissen, welche Datei was ist.
            # Heuristik: Dateiname enthält 'dataset1' -> train, 'dataset2' -> val, 'dataset3' -> test?
            # ODER: Die CSVs haben eine 'split' Spalte.
            # Ich füge eine Logik ein, um Splits basierend auf Dateinamen zu raten, falls keine Spalte da ist.
            
            if 'split' not in df.columns:
                if 'dataset1' in str(csv_path): df['split'] = 'train'
                elif 'dataset2' in str(csv_path): df['split'] = 'val'
                elif 'dataset3' in str(csv_path): df['split'] = 'test'
                else: df['split'] = 'train' # Fallback

            dfs.append(df)
            print(f"  Geladen: {csv_path} ({len(df)} Zeilen)")
        except FileNotFoundError:
            print(f"  WARNUNG: Datei nicht gefunden: {csv_path}")
    
    if not dfs:
        print("Keine CSVs geladen. Abbruch.")
        return
        
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Gesamt geladen: {len(full_df)} Zeilen.")

    # 3. Verarbeiten
    print("Verarbeite Daten und entpacke Embeddings...")
    
    final_rows = []
    existing_embs = set(p.stem for p in Path(OUTPUT_EMBEDDINGS_DIR).glob("*.pt"))
    print(f"  Bereits vorhandene Embeddings: {len(existing_embs)}")
    
    missing_count = 0
    
    for _, row in tqdm(full_df.iterrows(), total=len(full_df)):
        # IDs bestimmen
        # Wir benutzen 'entry' als unsere finale ID.
        # Falls 'entry' fehlt, nutzen wir 'source_accession' + Index als Fallback
        entry_id = row.get('entry', row.get('source_accession', f"seq_{_}"))
        
        # WICHTIG: Wonach suchen wir im H5?
        # In deinen neuen CSVs heißt die Spalte 'orig_id' oder 'source_accession'.
        lookup_id = row.get('orig_id', row.get('source_accession', entry_id))

        # Embedding holen (nur wenn noch nicht da)
        if str(entry_id) not in existing_embs:
            emb_data = find_embedding(handles, lookup_id)
            
            if emb_data is not None:
                # Speichern
                t = torch.tensor(emb_data, dtype=torch.float32)
                if t.ndim > 1: t = t.squeeze()
                torch.save({'mean_pooled': t}, Path(OUTPUT_EMBEDDINGS_DIR) / f"{entry_id}.pt")
                existing_embs.add(str(entry_id))
            else:
                missing_count += 1
                # Kein Embedding -> Überspringen für Metadata!
                continue
        
        # Metadaten vorbereiten
        # Labels parsen
        frag_type = row.get('frag_type', row.get('type', '')) 
        labels = map_frag_type(frag_type)
        
        # Split übernehmen
        split_val = row.get('split', 'train')

        final_rows.append({
            'entry': entry_id,
            'split': split_val,
            'is_fragment': int(row.get('is_fragment', 0)),
            'n_terminal': labels['n_terminal'],
            'c_terminal': labels['c_terminal'],
            'internal': labels['internal'],
            'sequence_length': int(row.get('sequence_length', 0))
        })

    print(f"Verarbeitung abgeschlossen. Embeddings nicht gefunden: {missing_count}")

    # 4. DataFrame erstellen und speichern
    master_df = pd.DataFrame(final_rows)
    
    if master_df.empty:
        print("FEHLER: Keine gültigen Daten übrig geblieben.")
        return

    print(f"Speichere Metadata nach {OUTPUT_METADATA}...")
    master_df.to_csv(OUTPUT_METADATA, index=False)
    
    print("\nFertig! Finale Splits:")
    print(master_df['split'].value_counts())
    
    # Cleanup
    for h in handles.values(): h.close()

if __name__ == "__main__":
    main()