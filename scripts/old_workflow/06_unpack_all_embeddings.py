import h5py
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np

# --- ANPASSEN: Pfade zu ALLEN 5 H5-Dateien ---
# Echte (vollständige) Embeddings
REAL_H5_PATHS = [
    "data/uniprot/bulk_embeddings/swiss-prot_fragments.h5",
    "data/uniprot/bulk_embeddings/swiss-prot_complete.h5"
]
# Synthetische (50k) Embeddings
SYNTHETIC_H5_PATHS = [
    "data/uniprot/bulk_embeddings/dataset_1.h5",
    "data/uniprot/bulk_embeddings/dataset_2.h5",
    "data/uniprot/bulk_embeddings/dataset_3.h5"
]
# ---

def load_h5_database(h5_paths):
    """Lädt alle H5-Dateien in ein Dictionary für schnellen Zugriff."""
    db = {}
    print(f"Loading {len(h5_paths)} H5 files...")
    for h5_path in h5_paths:
        h5_path = Path(h5_path) # Stelle sicher, dass es ein Path-Objekt ist
        try:
            db[h5_path.name] = h5py.File(h5_path, 'r')
            # Lese den 'id_key', um zu prüfen, wie die IDs gespeichert sind
            id_key = 'orig_id' # Standard-Annahme
            if 'orig_id' not in db[h5_path.name]:
                 if 'ids' in db[h5_path.name]: id_key = 'ids'
                 elif 'entry_id' in db[h5_path.name]: id_key = 'entry_id'
                 else:
                     # Fallback: Benutze H5-Keys, wenn kein ID-Dataset gefunden wird
                     id_key = 'keys'
            
            if id_key == 'keys':
                print(f"  Loaded {h5_path.name} (contains {len(db[h5_path.name].keys())} keys)")
            else:
                print(f"  Loaded {h5_path.name} (dataset '{id_key}' contains {len(db[h5_path.name][id_key])} IDs)")
                
        except Exception as e:
            print(f"Warning: Could not load {h5_path.name}: {e}")
    return db

def find_embedding(h5_dbs, key):
    """
    Sucht in allen geladenen H5-DBs nach einem Key.
    Behandelt sowohl H5-Keys als auch Datasets (wie 'orig_id').
    """
    if key is None or pd.isna(key):
        return None
        
    for db_name, db in h5_dbs.items():
        # --- Fall 1: Der Key ist der H5-Key (Standard für ECHTE Daten) ---
        if key in db:
            return db[key][:] # Gibt die Daten als Numpy-Array zurück
            
        # --- Fall 2: Der Key ist in einem ID-Dataset (Standard für SYNTHETISCHE Daten) ---
        id_key = 'orig_id' # Standard-Annahme
        if 'orig_id' not in db:
             if 'ids' in db: id_key = 'ids'
             elif 'entry_id' in db: id_key = 'entry_id'
             else: id_key = None # Kein bekanntes ID-Dataset
        
        if id_key:
            try:
                # Lade die ID-Liste (dekodiere, falls nötig)
                # ACHTUNG: Wir lesen das Dataset hier direkt, um Speicher zu sparen,
                # aber das ist langsam, wenn wir es für jeden Key tun.
                # Besser wäre es, die IDs einmal zu laden (siehe unten).
                pass 
            except Exception:
                pass

    # Optimierte Suche: Wir gehen davon aus, dass wir wissen, welche DB wir durchsuchen müssen
    # (Das passiert im Main-Loop). Hier ist eine generische Suche als Fallback.
    return None

# Hilfsfunktion für optimierte Suche in synthetischen DBs
def find_synthetic_embedding(db, key, id_cache):
    """
    Sucht ein Embedding in einer spezifischen synthetischen DB, 
    unter Verwendung eines vorab geladenen ID-Caches.
    """
    if key in id_cache:
        idx = id_cache[key]
        if 'embeddings' in db:
            return db['embeddings'][idx]
    return None


def main(args):
    # Wichtig: Das Skript mit Sequenzen verwenden!
    master_list_path = Path(args.master_file) 
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Lade die finale Master-Liste (z.B. 430k Einträge)
    print(f"Loading master metadata from {master_list_path}...")
    try:
        df = pd.read_csv(master_list_path)
    except FileNotFoundError:
        print(f"Error: Master file not found at {master_list_path}.")
        print("Please run '05_combine_metadata.py' first.")
        return
    print(f"Found {len(df)} total entries to unpack.")

    # 2. Lade ALLE H5-Dateien (echte + synthetische)
    real_dbs = load_h5_database(REAL_H5_PATHS)
    synthetic_dbs = load_h5_database(SYNTHETIC_H5_PATHS)
    
    # 2b. Erstelle ID-Caches für synthetische DBs (für Speed)
    print("Building ID caches for synthetic databases...")
    synthetic_id_caches = {}
    for name, db in synthetic_dbs.items():
        id_key = 'orig_id' if 'orig_id' in db else 'ids' # Fallback
        if id_key in db:
            print(f"  Caching IDs from {name}...")
            ids = db[id_key][:]
            if ids.dtype == 'object':
                ids = [s.decode('utf-8') for s in ids]
            # Erstelle Mapping: ID -> Index
            synthetic_id_caches[name] = {id_val: i for i, id_val in enumerate(ids)}

    # 3. Scanne den Output-Ordner (für Resume-Logik)
    print(f"Scanning {output_dir} for already unpacked embeddings...")
    existing_files = set(p.stem for p in tqdm(output_dir.glob("*.pt"), desc="Scanning output dir"))
    print(f"{len(existing_files)} existing embeddings found.")

    # 4. Iteriere durch die Master-Liste und entpacke, was fehlt
    print("Unpacking embeddings...")
    unpacked_count = 0
    missing_in_h5 = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Unpacking"):
        entry_id = row['entry']
        
        # 1. Überspringen, wenn schon vorhanden
        if entry_id in existing_files:
            continue
            
        # 2. Bestimmen, wo gesucht werden muss
        h5_key = row.get('h5_key', entry_id) # 'h5_key' für synthetische, 'entry' für echte
        
        embedding = None
        
        # Unterscheide: Synthetische vs. Echte Suche
        # Wenn 'entry_id' anders ist als 'h5_key', ist es wahrscheinlich synthetisch
        if str(entry_id) != str(h5_key): 
            # Suche in allen synthetischen DBs mit Cache
            for name, db in synthetic_dbs.items():
                if name in synthetic_id_caches:
                    embedding = find_synthetic_embedding(db, h5_key, synthetic_id_caches[name])
                    if embedding is not None:
                        break
        else:
            # Suche in echten DBs (direkter Key-Zugriff)
            for db in real_dbs.values():
                if h5_key in db:
                    embedding = db[h5_key][:]
                    break
            
        # 3. Speichern, wenn gefunden
        if embedding is not None:
            # Stelle sicher, dass es ein 1D-Vektor ist (falls es [1, 1024] ist)
            if embedding.ndim > 1:
                embedding = embedding.squeeze()
                
            save_dict = {'mean_pooled': torch.tensor(embedding, dtype=torch.float32)}
            output_path = output_dir / f"{entry_id}.pt"
            torch.save(save_dict, output_path)
            unpacked_count += 1
        else:
            # Optional: Zeige Warnung nur beim ersten Mal oder am Ende
            # print(f"Warning: ID '{entry_id}' (H5 key: '{h5_key}') not found.")
            missing_in_h5 += 1

    # 5. Aufräumen
    for db in real_dbs.values():
        db.close()
    for db in synthetic_dbs.values():
        db.close()

    print("\n--- Unpacking Complete ---")
    print(f"Unpacked {unpacked_count} new embeddings.")
    if missing_in_h5 > 0:
        print(f"Warning: {missing_in_h5} IDs from metadata were not found in any H5 file.")
    final_count = len(list(output_dir.glob("*.pt")))
    print(f"The folder {output_dir} now contains {final_count} total embedding files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unpack all HDF5 embeddings into individual .pt files.")
    parser.add_argument(
        '--master-file', 
        type=str, 
        default='data/processed/metadata_FINAL_MASTER_with_seqs.csv', # Wichtig!
        help='Path to the combined master list of all sequences (real + synthetic) *with sequences*.'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='data/embeddings',
        help='Directory to save the individual .pt embedding files.'
    )
    args = parser.parse_args()
    main(args)