import h5py
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import argparse

# --- ANPASSEN ---
# Passe diese Pfade an, damit sie auf deine zwei großen H5-Dateien zeigen
FRAGMENT_H5_PATH = "data/embeddings_download/fragments.h5"
COMPLETE_H5_PATH = "data/embeddings_download/blocker_complete.h5"
# ---

def process_h5_file(filepath, rep_ids_set, output_dir, existing_files):
    """Liest eine H5-Datei und speichert die benötigten Embeddings einzeln."""
    print(f"Verarbeite Datei: {filepath}...")
    try:
        with h5py.File(filepath, 'r') as f:
            # Iteriere über alle Einträge in der H5-Datei
            for entry_id in tqdm(f.keys(), desc=f"Scanning {filepath.name}"):
                # 1. Nur IDs verarbeiten, die in unserem nicht-redundanten Set sind
                if entry_id in rep_ids_set:
                    # 2. Nur verarbeiten, wenn wir es nicht schon haben
                    if entry_id not in existing_files:
                        try:
                            # Embedding lesen
                            embedding = f[entry_id][:]
                            
                            # Als PyTorch-Tensor im erwarteten Diktat-Format speichern
                            save_dict = {'mean_pooled': torch.tensor(embedding, dtype=torch.float32)}
                            output_path = output_dir / f"{entry_id}.pt"
                            torch.save(save_dict, output_path)
                            
                            # Zur Resume-Logik hinzufügen
                            existing_files.add(entry_id)
                        except Exception as e:
                            print(f"Fehler beim Verarbeiten von {entry_id}: {e}")
                            
    except IOError as e:
        print(f"FEHLER: Konnte Datei nicht öffnen: {filepath}. {e}")
        print("Bitte überprüfe die Pfade FRAGMENT_H5_PATH und COMPLETE_H5_PATH oben im Skript.")
    except Exception as e:
        print(f"Ein allgemeiner Fehler ist aufgetreten: {e}")


def main(args):
    rep_ids_path = Path(args.rep_ids_file)
    output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Lade die Liste der IDs, die wir behalten wollen
    print(f"Lade representative IDs von {rep_ids_path}...")
    if not rep_ids_path.exists():
        print(f"FEHLER: {rep_ids_path} nicht gefunden. Bitte zuerst '02_run_mmseqs.sh' ausführen.")
        return
    rep_ids_df = pd.read_csv(rep_ids_path, header=None, names=['entry'])
    rep_ids_set = set(rep_ids_df['entry'])
    print(f"{len(rep_ids_set)} nicht-redundante IDs geladen.")
    
    # 2. Scanne den Output-Ordner (für Resume-Logik)
    print(f"Scanne {output_dir} nach bereits existierenden Embeddings...")
    existing_files = set(p.stem for p in tqdm(output_dir.glob("*.pt"), desc="Scanning output dir"))
    print(f"{len(existing_files)} existierende Embeddings gefunden.")
    
    # 3. Verarbeite beide H5-Dateien
    process_h5_file(Path(FRAGMENT_H5_PATH), rep_ids_set, output_dir, existing_files)
    process_h5_file(Path(COMPLETE_H5_PATH), rep_ids_set, output_dir, existing_files)
    
    print("\n--- Entpacken abgeschlossen ---")
    final_count = len(list(output_dir.glob("*.pt")))
    print(f"Der Ordner {output_dir} enthält jetzt {final_count} Embedding-Dateien.")
    
    if final_count < len(rep_ids_set):
        print(f"Warnung: Es fehlen noch {len(rep_ids_set) - final_count} Embeddings.")
        print("Möglicherweise waren einige IDs aus 'representative_ids.txt' nicht in den H5-Dateien enthalten.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unpack bulk HDF5 embeddings into individual .pt files.")
    parser.add_argument(
        '--rep-ids-file', 
        type=str, 
        default='data/processed/clustered/representative_ids.txt',
        help='Path to the file with representative sequence IDs from MMseqs2.'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='data/embeddings',
        help='Directory to save the individual .pt embedding files.'
    )
    args = parser.parse_args()
    main(args)