import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import argparse
from pathlib import Path
from tqdm import tqdm

def main(args):
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    print(f"Loading raw metadata from {input_path}...")
    df = pd.read_csv(input_path)
    original_count = len(df)
    print(f"Loaded {original_count} total sequences.")

    # --- Step 1: Filter by Representative IDs (Optional) ---
    if args.rep_ids_file:
        rep_ids_path = Path(args.rep_ids_file)
        if not rep_ids_path.exists():
            print(f"Warning: Representative IDs file not found at {rep_ids_path}. Skipping redundancy reduction.")
        else:
            print(f"Loading representative IDs from {rep_ids_path}...")
            rep_ids = pd.read_csv(rep_ids_path, header=None, names=['entry'])
            rep_ids_set = set(rep_ids['entry'])
            
            df = df[df['entry'].isin(rep_ids_set)]
            print(f"Filtered to non-redundant set: {len(df)} sequences.")
    
    # --- Step 2: NEW - Filter by Existing Embeddings (Optional) ---
    if args.embeddings_dir:
        emb_dir = Path(args.embeddings_dir)
        if not emb_dir.exists():
            print(f"Warning: Embeddings directory not found at {emb_dir}. Skipping embedding filter.")
        else:
            print(f"Scanning {emb_dir} for existing embeddings (this may take a moment)...")
            # Use .stem to get filename without .pt
            existing_embeddings = set(p.stem for p in tqdm(emb_dir.glob("*.pt")))
            
            if not existing_embeddings:
                print(f"Warning: No .pt files found in {emb_dir}. Training will fail.")
            else:
                pre_filter_count = len(df)
                # Filter the DataFrame to only include entries with existing embeddings
                df = df[df['entry'].isin(existing_embeddings)]
                post_filter_count = len(df)
                print(f"Filtered to existing embeddings: {pre_filter_count} -> {post_filter_count} sequences.")

    if len(df) == 0:
        print("Error: No sequences left to split after filtering. Exiting.")
        return

    # 3. Create length bins for stratification
    # Adjust bins if we have very few samples
    n_bins = min(args.n_length_bins, len(df) // 100, 5) # Heuristic
    if n_bins < 2: n_bins = 2
    
    try:
        df['length_bin'] = pd.qcut(df['sequence_length'], q=n_bins, labels=False, duplicates='drop')
    except ValueError:
        print("Warning: Could not create length bins. Using simple stratification.")
        df['length_bin'] = 0 # Fallback

    # 4. Create a combined stratification column
    df['stratify_col'] = (
        df['is_fragment'].astype(str) + '_' + 
        df['length_bin'].astype(str)
    )
    
    # 5. Perform a 2-stage split
    test_val_size = args.val_ratio + args.test_ratio
    if test_val_size == 0: test_size_relative = 0
    else: test_size_relative = args.test_ratio / test_val_size

    df['split'] = 'train'
    
    if test_val_size > 0:
        n_splits_1 = int(round(1 / test_val_size))
        if n_splits_1 < 2: n_splits_1 = 2

        # Check if we have enough samples in each class for stratification
        vc = df['stratify_col'].value_counts()
        if (vc < n_splits_1).any():
            print(f"Warning: Not all strata have {n_splits_1} members. Falling back to non-stratified split.")
            # Create a dummy stratify column if stratification is impossible
            df['stratify_col'] = 0 
            skf = StratifiedKFold(n_splits=n_splits_1, shuffle=True, random_state=args.seed)
        else:
             skf = StratifiedKFold(n_splits=n_splits_1, shuffle=True, random_state=args.seed)

        try:
            train_idx, test_val_idx = next(skf.split(df, df['stratify_col']))
            df.iloc[test_val_idx, df.columns.get_loc('split')] = 'test_val_temp'
        except ValueError:
            print("Warning: Stratification failed. Splitting randomly.")
            from sklearn.model_selection import train_test_split
            train_idx, test_val_idx = train_test_split(df.index, test_size=test_val_size, random_state=args.seed)
            df.loc[test_val_idx, 'split'] = 'test_val_temp'
        
        # Split val and test
        if test_size_relative > 0 and test_size_relative < 1:
            test_val_df = df[df['split'] == 'test_val_temp']
            
            n_splits_2 = int(round(1 / test_size_relative))
            if n_splits_2 < 2: n_splits_2 = 2

            vc_2 = test_val_df['stratify_col'].value_counts()
            if (vc_2 < n_splits_2).any():
                 print(f"Warning: Not all strata have {n_splits_2} members for val/test split. Falling back to non-stratified.")
                 test_val_df['stratify_col'] = 0
                 skf_test = StratifiedKFold(n_splits=n_splits_2, shuffle=True, random_state=args.seed)
            else:
                 skf_test = StratifiedKFold(n_splits=n_splits_2, shuffle=True, random_state=args.seed)

            try:
                val_idx_rel, test_idx_rel = next(skf_test.split(test_val_df, test_val_df['stratify_col']))
                val_idx_orig = test_val_df.iloc[val_idx_rel].index
                test_idx_orig = test_val_df.iloc[test_idx_rel].index
                df.loc[val_idx_orig, 'split'] = 'val'
                df.loc[test_idx_orig, 'split'] = 'test'
            except ValueError:
                 print("Warning: Val/Test stratification failed. Splitting randomly.")
                 val_idx, test_idx = train_test_split(test_val_df.index, test_size=test_size_relative, random_state=args.seed)
                 df.loc[val_idx, 'split'] = 'val'
                 df.loc[test_idx, 'split'] = 'test'

        elif test_size_relative == 0:
             df.loc[df['split'] == 'test_val_temp', 'split'] = 'val'
        else:
             df.loc[df['split'] == 'test_val_temp', 'split'] = 'test'

    # 6. Clean and save
    final_df = df.drop(columns=['stratify_col', 'length_bin'], errors='ignore')
    final_df.to_csv(output_path, index=False)
    
    print(f"Successfully created splits for {len(final_df)} sequences and saved to {output_path}")
    print("\n--- Split Summary ---")
    print(final_df['split'].value_counts())
    print(final_df['split'].value_counts(normalize=True).apply(lambda x: f"{x:.1%}"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create stratified train/val/test splits.")
    parser.add_argument('--input', type=str, default='data/processed/metadata_raw.csv',
                        help='Input raw metadata CSV file')
    parser.add_argument('--output', type=str, default='data/processed/metadata.csv',
                        help='Output metadata CSV file with splits')
    
    parser.add_argument('--rep-ids-file', type=str, default='data/processed/clustered/representative_ids.txt',
                        help='Path to the file containing representative sequence IDs from MMseqs2. (Optional)')
    
    # --- NEW Argument ---
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
