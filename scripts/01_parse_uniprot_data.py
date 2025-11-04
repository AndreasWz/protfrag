# scripts/01_parse_uniprot_data.py
import pandas as pd
from Bio import SeqIO
from pathlib import Path
from tqdm import tqdm
import argparse

# Import the parser from our src/ package
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils.fragment_parser import FragmentAnnotationParser

def load_sequences(fasta_file: Path) -> dict:
    """Loads a FASTA file into a dictionary of {entry: sequence}."""
    sequences = {}
    print(f"Parsing sequences from {fasta_file.name}...")
    for record in tqdm(SeqIO.parse(fasta_file, "fasta")):
        entry = record.id.split('|')[1] # UniProt IDs are like sp|A0A068B6Q6|...
        sequences[entry] = str(record.seq)
    return sequences

def main(args):
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load all sequences (fragments and complete)
    fragment_seqs = load_sequences(raw_dir / "fragments.fasta")
    complete_seqs = load_sequences(raw_dir / "complete.fasta")
    all_seqs = {**fragment_seqs, **complete_seqs}
    
    # 2. Load fragment annotations
    print("Loading fragment annotations...")
    annot_df = pd.read_csv(
        raw_dir / "fragment_annotations.tsv",
        sep="\t",
        usecols=['Entry', 'Non-adjacent residues', 'Non-terminal residue', 'Fragment']
    )
    annot_df = annot_df.rename(columns={'Entry': 'entry'})
    
    # 3. Create DataFrame for complete sequences
    complete_df = pd.DataFrame(complete_seqs.keys(), columns=['entry'])
    complete_df['is_fragment'] = 0
    
    # 4. Create DataFrame for fragment sequences
    fragment_df = pd.DataFrame(fragment_seqs.keys(), columns=['entry'])
    fragment_df['is_fragment'] = 1
    
    # Merge fragment data with annotations
    fragment_df = pd.merge(fragment_df, annot_df, on='entry', how='left')
    
    # 5. Combine DataFrames
    combined_df = pd.concat([fragment_df, complete_df], ignore_index=True)
    
    # 6. Add sequence and length
    print("Adding sequences and lengths...")
    combined_df['sequence'] = combined_df['entry'].map(all_seqs)
    combined_df['sequence_length'] = combined_df['sequence'].apply(len)
    
    # Handle entries with missing sequences (should not happen if FASTA files are complete)
    combined_df = combined_df.dropna(subset=['sequence'])
    combined_df['sequence_length'] = combined_df['sequence_length'].astype(int)

    # 7. Initialize the parser
    parser = FragmentAnnotationParser(
        n_terminal_threshold=args.n_term_thresh,
        c_terminal_threshold=args.c_term_thresh
    )
    
    # 8. Apply parser to get multilabel columns
    print("Applying fragment type parser...")
    parsed_labels = []
    for _, row in tqdm(combined_df.iterrows(), total=len(combined_df)):
        if row['is_fragment'] == 1:
            non_ter_pos = parser.parse_non_ter(row.get('Non-terminal residue', ''))
            non_cons_gaps = parser.parse_non_cons(row.get('Non-adjacent residues', ''))
            labels = parser.classify_fragment(non_ter_pos, non_cons_gaps, row['sequence_length'])
        else:
            labels = {'n_terminal': False, 'c_terminal': False, 'internal': False}
        parsed_labels.append(labels)
        
    labels_df = pd.DataFrame(parsed_labels).astype(int)
    final_df = pd.concat([combined_df, labels_df], axis=1)
    
    # 9. Clean and save
    final_df = final_df[[
        'entry', 'is_fragment', 'n_terminal', 'c_terminal', 'internal',
        'sequence_length', 'sequence'
    ]]
    
    output_path = processed_dir / "metadata_raw.csv"
    final_df.to_csv(output_path, index=False)
    print(f"Successfully created raw metadata at {output_path}")
    print("\n--- Parsing Summary ---")
    print(f"Total sequences processed: {len(final_df)}")
    print(f"Complete sequences: {len(final_df[final_df['is_fragment'] == 0])}")
    print(f"Fragment sequences: {len(final_df[final_df['is_fragment'] == 1])}")
    print(f"\nFragment Type Distribution (non-exclusive):")
    print(final_df[final_df['is_fragment'] == 1][['n_terminal', 'c_terminal', 'internal']].sum())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse UniProt data into a clean metadata file.")
    parser.add_argument('--raw_dir', type=str, default='data/raw', help='Directory with raw UniProt files')
    parser.add_argument('--processed_dir', type=str, default='data/processed', help='Directory to save processed metadata')
    parser.add_argument('--n_term_thresh', type=int, default=10, help='Position threshold for N-terminal')
    parser.add_argument('--c_term_thresh', type=int, default=10, help='Position threshold for C-terminal')
    args = parser.parse_args()
    main(args)