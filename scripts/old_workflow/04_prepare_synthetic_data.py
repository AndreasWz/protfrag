import pandas as pd
import argparse
from pathlib import Path
import glob
import numpy as np

# --- ANPASSEN: Pfade zu den riesigen synthetischen CSVs ---
# (z.B. die von Aeneas generierten)
SYNTHETIC_CSV_PATTERN = "data/uniprot/dataset*.csv" 
# ---

def map_frag_type(frag_type_str):
    """Mapping für Fragment-Typen."""
    s = str(frag_type_str).lower()
    return {
        'n_terminal': 1 if 'n' in s or 'both' in s else 0,
        'c_terminal': 1 if 'c' in s or 'both' in s else 0,
        'internal': 1 if 'internal' in s or 'mixed' in s else 0,
    }

def main(args):
    output_path = Path(args.output)
    real_splits_path = Path(args.real_splits)
    
    print(f"1. Loading REAL splits from {real_splits_path}...")
    df_real = pd.read_csv(real_splits_path)
    
    # Wir holen uns die IDs, die im TRAIN-Set sind.
    # Nur von diesen dürfen wir synthetische Daten nutzen!
    train_entries = set(df_real[df_real['split'] == 'train']['entry'])
    print(f"   Found {len(train_entries)} proteins in the REAL training set.")

    print(f"2. Loading synthetic candidates from {SYNTHETIC_CSV_PATTERN}...")
    csv_files = glob.glob(SYNTHETIC_CSV_PATTERN)
    if not csv_files:
        print("Error: No synthetic CSV files found.")
        return

    chunk_list = []
    total_synthetic = 0
    kept_synthetic = 0
    
    # Wir laden die CSVs und filtern sofort, um Speicher zu sparen
    for f in csv_files:
        print(f"   Processing {f}...")
        df_chunk = pd.read_csv(f)
        total_synthetic += len(df_chunk)
        
        # WICHTIG: Filtern! Behalte nur Fragmente, deren Original-ID im Train-Set ist
        # Aeneas' CSVs haben eine Spalte 'orig_id', die auf das Eltern-Protein zeigt
        if 'orig_id' not in df_chunk.columns:
            print(f"   Warning: 'orig_id' missing in {f}. Skipping.")
            continue
            
        df_filtered = df_chunk[df_chunk['orig_id'].isin(train_entries)].copy()
        
        if len(df_filtered) > 0:
            # Standardisierung
            type_df = df_filtered['frag_type'].apply(lambda x: pd.Series(map_frag_type(x)))
            df_filtered = pd.concat([df_filtered, type_df], axis=1)
            df_filtered['is_fragment'] = 1
            df_filtered['split'] = 'train'
            df_filtered['h5_key'] = df_filtered['entry'] # H5 Key ist die ID in der CSV
            
            # Wir benennen 'entry' um, um Kollisionen zu vermeiden, falls nötig
            # (Aber Aeneas hat schon 'datasetX_...' IDs vergeben, das sollte passen)
            
            kept_synthetic += len(df_filtered)
            chunk_list.append(df_filtered)

    if not chunk_list:
        print("No synthetic fragments matched the training set IDs.")
        return

    df_synthetic = pd.concat(chunk_list, ignore_index=True)
    print(f"   Filtered: Kept {kept_synthetic} of {total_synthetic} synthetic fragments (only those belonging to train set).")

    # --- SCHRITT 3: Das PAIRING sicherstellen ---
    # Wir wollen sicherstellen, dass für jedes Fragment auch das 'Complete' Elternteil dabei ist.
    # Das Elternteil ist ja 'orig_id'.
    # Da wir oben nach 'train_entries' gefiltert haben, WISSEN wir, dass das Elternteil
    # im df_real (Train split) existiert.
    
    # Wir müssen hier nichts tun! 
    # Das Skript 05 wird df_real (mit den Eltern) und df_synthetic (mit den Kindern) zusammenfügen.
    # Da beide im 'train' split landen, sieht das Modell beide.
    
    # Aufräumen der Spalten für den Merge
    final_cols = ['entry', 'h5_key', 'is_fragment', 'n_terminal', 'c_terminal', 'internal', 'sequence_length', 'split', 'sequence']
    # Falls sequence_length fehlt, berechnen
    if 'sequence_length' not in df_synthetic.columns:
        df_synthetic['sequence_length'] = df_synthetic['sequence'].str.len()
        
    df_synthetic = df_synthetic.reindex(columns=final_cols)
    
    print(f"3. Saving {len(df_synthetic)} validated synthetic training samples to {output_path}...")
    df_synthetic.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_splits', type=str, default='data/processed/metadata_real_splits.csv')
    parser.add_argument('--output', type=str, default='data/processed/metadata_synthetic_filtered.csv')
    args = parser.parse_args()
    main(args)