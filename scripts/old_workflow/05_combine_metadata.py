import pandas as pd
import argparse
from pathlib import Path

def main(args):
    real_data_path = Path(args.real_data)
    synthetic_data_path = Path(args.synthetic_data)
    output_path = Path(args.output)
    
    print(f"Loading REAL data with splits from {real_data_path}...")
    try:
        df_real = pd.read_csv(real_data_path)
        # Check if 'split' column exists
        if 'split' not in df_real.columns:
            print(f"ERROR: 'split' column missing in {real_data_path}.")
            print("Please re-run '03_create_splits_REAL.py' correctly.")
            return
    except FileNotFoundError:
        print(f"Error: File not found at {real_data_path}")
        return
        
    print(f"Loading SYNTHETIC data with splits from {synthetic_data_path}...")
    try:
        df_synthetic = pd.read_csv(synthetic_data_path)
        # Ensure synthetic data is marked as train
        if 'split' not in df_synthetic.columns:
            print("Adding 'split=train' to synthetic data...")
            df_synthetic['split'] = 'train'
    except FileNotFoundError:
        print(f"Error: File not found at {synthetic_data_path}")
        return

    # --- Erstelle 'h5_key' Spalte für ECHTE Daten ---
    # Bei echten Daten ist der 'entry' (z.B. P12345) auch der 'h5_key'
    df_real['h5_key'] = df_real['entry']

    # Definiere die Spalten, die wir für die Master-Datei brauchen
    # 'sequence' wird für das Embedding-Skript (06) benötigt
    master_cols = [
        'entry', 'h5_key', 'is_fragment', 'n_terminal', 'c_terminal', 'internal',
        'sequence_length', 'sequence', 'split'
    ]
    
    # Sicherstellen, dass beide DataFrames dieselben Spalten haben
    df_real_aligned = df_real.reindex(columns=master_cols)
    df_synthetic_aligned = df_synthetic.reindex(columns=master_cols)

    print("Combining real and synthetic metadata...")
    df_combined = pd.concat([df_real_aligned, df_synthetic_aligned], ignore_index=True)
    
    # WICHTIG: Speichere eine Version MIT Sequenzen für das Entpacken (Script 06)
    seq_output_path = output_path.parent / "metadata_FINAL_MASTER_with_seqs.csv"
    print(f"Saving master list *with sequences* for embedding to {seq_output_path}...")
    df_combined.to_csv(seq_output_path, index=False)

    # ...und eine Version OHNE Sequenzen für das Training (schneller laden)
    print(f"Saving combined master metadata file (for training) to {output_path}...")
    output_cols = [
        'entry', 'h5_key', 'is_fragment', 'n_terminal', 'c_terminal', 'internal',
        'sequence_length', 'split'
    ]
    df_combined.to_csv(output_path, columns=output_cols, index=False)
    
    print("\n--- Combining Complete ---")
    print(f"Real sequences:      {len(df_real):>10}")
    print(f"Synthetic sequences: {len(df_synthetic):>10}")
    print(f"Total sequences:     {len(df_combined):>10}")
    print("\nFinal split distribution:")
    print(df_combined['split'].value_counts())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine real and synthetic metadata files.")
    # WICHTIG: Hier default auf metadata_real_splits.csv gesetzt!
    parser.add_argument('--real_data', type=str, default='data/processed/metadata_real_splits.csv',
                        help="Input CSV of REAL data with splits")
    parser.add_argument('--synthetic_data', type=str, default='data/processed/metadata_synthetic_50k.csv',
                        help="Input CSV of SYNTHETIC data with splits")
    parser.add_argument('--output', type=str, default='data/processed/metadata_FINAL_MASTER.csv',
                        help="Output combined master CSV file (for training)")
    args = parser.parse_args()
    
    main(args)