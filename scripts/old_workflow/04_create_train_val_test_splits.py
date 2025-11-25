import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
import argparse
from pathlib import Path
from tqdm import tqdm # Importiere tqdm

def main(args):
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    print(f"Loading raw metadata from {input_path}...")
    df = pd.read_csv(input_path)
    original_count = len(df)
    print(f"Loaded {original_count} total sequences.")

    # --- Step 1: Filter by Representative IDs ---
    if args.rep_ids_file:
        rep_ids_path = Path(args.rep_ids_file)
        if not rep_ids_path.exists():
            print(f"Warning: Representative IDs file not found at {rep_ids_path}. Using all sequences.")
        else:
            print(f"Loading representative IDs from {rep_ids_path}...")
            rep_ids = pd.read_csv(rep_ids_path, header=None, names=['entry'])
            rep_ids_set = set(rep_ids['entry'])
            
            df = df[df['entry'].isin(rep_ids_set)]
            print(f"Filtered to non-redundant set: {len(df)} sequences.")
            
    # --- NEU: Step 2: Filter by existing embeddings ---
    # (Diese Logik stellt sicher, dass wir die 11 fehlenden IDs ignorieren)
    emb_dir = Path(args.embeddings_dir)
    if not emb_dir.exists():
        print(f"FATAL ERROR: Embeddings directory not found at {emb_dir}.")
        return
    else:
        print(f"Scanning {emb_dir} for existing embeddings (this may take a moment)...")
        existing_embeddings = set(p.stem for p in tqdm(emb_dir.glob("*.pt"), desc="Scanning embeddings"))
        
        if not existing_embeddings:
            print(f"FATAL ERROR: No .pt files found in {emb_dir}.")
            return
        else:
            pre_filter_count = len(df)
            df = df[df['entry'].isin(existing_embeddings)]
            post_filter_count = len(df)
            print(f"Filtered to existing embeddings: {pre_filter_count} -> {post_filter_count} sequences.")


    if len(df) == 0:
        print("Error: No sequences left to split after filtering. Exiting.")
        return

    # --- Step 3: Create length bins for stratification ---
    n_bins = args.n_length_bins
    try:
        df['length_bin'] = pd.qcut(df['sequence_length'], q=n_bins, labels=False, duplicates='drop')
    except ValueError as e:
        print(f"Warning: Could not create {n_bins} length bins (Error: {e}). Using simple stratification.")
        df['length_bin'] = 0 # Fallback

    # --- Step 4: Create a combined stratification column ---
    df['stratify_col'] = (
        df['is_fragment'].astype(str) + '_' + 
        df['length_bin'].astype(str)
    )
    
    # --- Step 5: Perform a 2-stage split ---
    test_val_size = args.val_ratio + args.test_ratio
    if test_val_size == 0: test_size_relative = 0
    elif test_val_size >= 1.0:
        print(f"Warning: val_ratio ({args.val_ratio}) + test_ratio ({args.test_ratio}) >= 1.0. Clamping to 0.98")
        test_val_size = 0.98
        
    if test_val_size > 0:
        test_size_relative = args.test_ratio / test_val_size
    else:
        test_size_relative = 0

    df['split'] = 'train'
    
    if test_val_size > 0:
        n_splits_1 = int(round(1 / test_val_size))
        if n_splits_1 < 2: n_splits_1 = 2

        vc = df['stratify_col'].value_counts()
        skf = None 
        if (vc < n_splits_1).any():
            print(f"Warning: Not all strata have {n_splits_1} members. Falling back to non-stratified split.")
            df['stratify_col'] = 0 
            vc_fallback = df['stratify_col'].value_counts()
            if (vc_fallback < n_splits_1).any().item():
                print("Error: Not enough samples to perform split. Check data.")
                n_splits_1 = 1
            else:
                 skf = StratifiedKFold(n_splits=n_splits_1, shuffle=True, random_state=args.seed)
        else:
             skf = StratifiedKFold(n_splits=n_splits_1, shuffle=True, random_state=args.seed)

        if n_splits_1 > 1 and skf is not None:
            try:
                train_idx, test_val_idx = next(skf.split(df, df['stratify_col']))
                df.iloc[test_val_idx, df.columns.get_loc('split')] = 'test_val_temp'
            except ValueError:
                print("Warning: Stratification failed. Splitting randomly.")
                test_val_size_float = test_val_size if test_val_size < 1.0 else (1.0 / n_splits_1)
                stratify_data = df['stratify_col'].values if (df['stratify_col'].value_counts() >= n_splits_1).all() else None
                train_idx, test_val_idx = train_test_split(df.index, test_size=test_val_size_float, random_state=args.seed, shuffle=True, stratify=stratify_data)
                df.loc[test_val_idx, 'split'] = 'test_val_temp'
        else:
             print("Not enough data to create test/val splits. All data will be marked as 'train'.")

        if test_size_relative > 0 and test_size_relative < 1:
            test_val_df = df[df['split'] == 'test_val_temp']
            if not test_val_df.empty:
                n_splits_2 = int(round(1 / test_size_relative))
                if n_splits_2 < 2: n_splits_2 = 2
                
                skf_test = None
                vc_2 = test_val_df['stratify_col'].value_counts()
                if (vc_2 < n_splits_2).any():
                    print(f"Warning: Not all strata have {n_splits_2} members for val/test split. Falling back to non-stratified.")
                    test_val_df.loc[:, 'stratify_col'] = 0
                    vc_2_fallback = test_val_df['stratify_col'].value_counts()
                    if (vc_2_fallback < n_splits_2).any().item():
                        print("Error: Not enough samples for val/test split. Marking all as 'val'.")
                        n_splits_2 = 1
                    else:
                        skf_test = StratifiedKFold(n_splits=n_splits_2, shuffle=True, random_state=args.seed)
                else:
                    skf_test = StratifiedKFold(n_splits=n_splits_2, shuffle=True, random_state=args.seed)

                if n_splits_2 > 1 and skf_test is not None:
                    try:
                        val_idx_rel, test_idx_rel = next(skf_test.split(test_val_df, test_val_df['stratify_col']))
                        val_idx_orig = test_val_df.iloc[val_idx_rel].index
                        test_idx_orig = test_val_df.iloc[test_idx_rel].index
                        df.loc[val_idx_orig, 'split'] = 'val'
                        df.loc[test_idx_orig, 'split'] = 'test'
                    except ValueError:
                        print("Warning: Val/Test stratification failed. Splitting randomly.")
                        test_size_float = test_size_relative if test_size_relative < 1.0 else 0.5
                        stratify_data_2 = test_val_df['stratify_col'].values if (test_val_df['stratify_col'].value_counts() >= n_splits_2).all() else None
                        val_idx, test_idx = train_test_split(test_val_df.index, test_size=test_size_float, random_state=args.seed, shuffle=True, stratify=stratify_data_2)
                        df.loc[val_idx, 'split'] = 'val'
                        df.loc[test_idx, 'split'] = 'test'
                elif n_splits_2 <= 1:
                    df.loc[df['split'] == 'test_val_temp', 'split'] = 'val'

        elif test_size_relative == 0 and test_val_size > 0:
             df.loc[df['split'] == 'test_val_temp', 'split'] = 'val'
        elif test_size_relative == 1 and test_val_size > 0:
             df.loc[df['split'] == 'test_val_temp', 'split'] = 'test'

    # --- Step 6: Clean and save ---
    final_df = df.drop(columns=['stratify_col', 'length_bin'], errors='ignore')
    final_df.to_csv(output_path, index=False)
    
    print(f"Successfully created final metadata splits for {len(final_df)} sequences at {output_path}")
    print("\n--- Final Split Summary ---")
    print(final_df['split'].value_counts())
    print(final_df['split'].value_counts(normalize=True).apply(lambda x: f"{x:.1%}"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create stratified train/val/test splits for REAL data.")
    parser.add_argument('--input', type=str, default='data/processed/metadata_raw.csv',
                        help='Input raw metadata CSV file')
    parser.add_argument('--output', type=str, default='data/processed/metadata.csv',
                        help="Output metadata CSV file with splits (for training)")
    
    parser.add_argument('--rep-ids-file', type=str, default='data/processed/clustered/representative_ids.txt',
                        help='Path to the file containing representative sequence IDs from MMseqs2. (Optional)')
    
    # --- NEUES ARGUMENT ---
    parser.add_argument('--embeddings-dir', type=str, default='data/embeddings',
                        help='Path to the embedding directory. If provided, will filter for existing embeddings. (Optional)')
    
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--n_length_bins', type=int, default=5,
                        help="Number of length bins for stratification")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(total_ratio, 1.0):
        print(f"Warning: Ratios do not sum to 1 (sum={total_ratio}). Normalizing...")
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio
        
    main(args)