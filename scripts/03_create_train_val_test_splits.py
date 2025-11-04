# scripts/03_create_train_val_test_splits.py
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import argparse
from pathlib import Path

def main(args):
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    print(f"Loading raw metadata from {input_path}...")
    df = pd.read_csv(input_path)
    original_count = len(df)
    print(f"Loaded {original_count} total sequences.")

    # --- NEW: Filter by Representative IDs ---
    if args.rep_ids_file:
        rep_ids_path = Path(args.rep_ids_file)
        if not rep_ids_path.exists():
            print(f"Warning: Representative IDs file not found at {rep_ids_path}. Skipping redundancy reduction.")
        else:
            print(f"Loading representative IDs from {rep_ids_path}...")
            # Load the file, which is a single column of IDs without a header
            rep_ids = pd.read_csv(rep_ids_path, header=None, names=['entry'])
            rep_ids_set = set(rep_ids['entry'])
            
            # Filter the main DataFrame
            df = df[df['entry'].isin(rep_ids_set)]
            new_count = len(df)
            print(f"Filtered to non-redundant set: {original_count} -> {new_count} sequences.")
    # --- End of NEW section ---

    # 1. Create length bins for stratification
    # We use quartiles for binning
    df['length_bin'] = pd.qcut(df['sequence_length'], q=args.n_length_bins, labels=False, duplicates='drop')
    
    # 2. Create a combined stratification column
    # This ensures our splits have a similar distribution of
    # fragment status AND sequence lengths.
    df['stratify_col'] = (
        df['is_fragment'].astype(str) + '_' + 
        df['length_bin'].astype(str)
    )
    
    # 3. Perform a 2-stage split (train/val+test, then val/test)
    
    # Calculate split sizes
    test_val_size = args.val_ratio + args.test_ratio
    # Adjust relative test size to avoid division by zero if test_val_size is 0
    if test_val_size == 0:
        test_size_relative = 0
    elif test_val_size > 1:
         print(f"Warning: val_ratio ({args.val_ratio}) + test_ratio ({args.test_ratio}) > 1. Clamping.")
         test_val_size = 1.0
         test_size_relative = args.test_ratio / test_val_size if test_val_size > 0 else 0
    else:
        test_size_relative = args.test_ratio / test_val_size

    df['split'] = 'train' # Default all to train
    
    # Only split if test_val_size > 0
    if test_val_size > 0:
        # Split off train
        n_splits_1 = int(round(1/test_val_size))
        if n_splits_1 < 2:
            print(f"Warning: test_val_size ({test_val_size}) is too large. Adjusting to 50/50 split.")
            n_splits_1 = 2

        skf = StratifiedKFold(n_splits=n_splits_1, shuffle=True, random_state=args.seed)
        
        try:
            train_idx, test_val_idx = next(skf.split(df, df['stratify_col']))
        except ValueError:
            print("Warning: Could not stratify perfectly. Using smaller n_splits (3).")
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed) # Fallback
            train_idx, test_val_idx = next(skf.split(df, df['stratify_col']))

        df.iloc[test_val_idx, df.columns.get_loc('split')] = 'test_val_temp'
        
        # 4. Split val and test
        if test_size_relative > 0 and test_size_relative < 1:
            test_val_df = df[df['split'] == 'test_val_temp']
            
            n_splits_2 = int(round(1/test_size_relative))
            if n_splits_2 < 2:
                n_splits_2 = 2 # Fallback to 50/50

            skf_test = StratifiedKFold(n_splits=n_splits_2, shuffle=True, random_state=args.seed)
            
            try:
                val_idx_rel, test_idx_rel = next(skf_test.split(test_val_df, test_val_df['stratify_col']))
            except ValueError:
                print("Warning: Could not stratify test/val perfectly. Using 50/50 split (n_splits=2).")
                skf_test = StratifiedKFold(n_splits=2, shuffle=True, random_state=args.seed) # Fallback
                val_idx_rel, test_idx_rel = next(skf_test.split(test_val_df, test_val_df['stratify_col']))

            # Get original indices
            val_idx_orig = test_val_df.iloc[val_idx_rel].index
            test_idx_orig = test_val_df.iloc[test_idx_rel].index
            
            df.loc[val_idx_orig, 'split'] = 'val'
            df.loc[test_idx_orig, 'split'] = 'test'
        elif test_size_relative == 0:
             df.loc[df['split'] == 'test_val_temp', 'split'] = 'val' # All become validation
        else: # test_size_relative == 1
             df.loc[df['split'] == 'test_val_temp', 'split'] = 'test' # All become test

    
    # 5. Clean and save
    final_df = df.drop(columns=['stratify_col', 'length_bin'], errors='ignore')
    final_df.to_csv(output_path, index=False)
    
    print(f"Successfully created splits and saved to {output_path}")
    print("\n--- Split Summary ---")
    print(final_df['split'].value_counts(normalize=True).apply(lambda x: f"{x:.1%}"))
    print("\n--- Training Set Fragment Distribution ---")
    print(final_df[final_df['split']=='train']['is_fragment'].value_counts(normalize=True).apply(lambda x: f"{x:.1%}"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create stratified train/val/test splits.")
    parser.add_argument('--input', type=str, default='data/processed/metadata_raw.csv',
                        help='Input raw metadata CSV file')
    parser.add_argument('--output', type=str, default='data/processed/metadata.csv',
                        help='Output metadata CSV file with splits')
    
    # --- NEW Argument ---
    parser.add_argument('--rep-ids-file', type=str, default='data/processed/clustered/representative_ids.txt',
                        help='Path to the file containing representative sequence IDs from MMseqs2. (Optional)')
    
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--n_length_bins', type=int, default=5,
                        help="Number of length bins for stratification")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Basic validation for ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(total_ratio, 1.0):
        print(f"Warning: Ratios do not sum to 1 (sum={total_ratio}). Normalizing...")
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio
        
    main(args)