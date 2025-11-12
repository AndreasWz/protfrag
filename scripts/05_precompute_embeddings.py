# scripts/04_precompute_embeddings.py
"""
Precompute ProtT5 embeddings for all sequences.
! only needed if not all seqs got loaded by 03_unpack_embeddings.py
"""
import argparse
from pathlib import Path
import pandas as pd
import torch
from transformers import T5Tokenizer, T5EncoderModel
from tqdm import tqdm
import re


def load_model(model_name: str, device: str):
    """Load ProtT5 model and tokenizer."""
    print(f"Loading model {model_name}...")
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return tokenizer, model


def preprocess_sequence(sequence: str) -> str:
    """
    Preprocess protein sequence for ProtT5.
    Add spaces between amino acids and replace rare amino acids.
    """
    # Add spaces between amino acids
    sequence = " ".join(list(sequence))
    # Replace rare amino acids
    sequence = re.sub(r"[UZOB]", "X", sequence)
    return sequence


def generate_embeddings_batch(
    sequences: list,
    tokenizer,
    model,
    device: str,
    max_length: int = 1024
) -> dict:
    """
    Generate embeddings for a batch of sequences.
    
    Returns:
        Dictionary with:
            - mean_pooled: Mean-pooled per-protein embeddings
            - per_residue: Per-residue embeddings (optional, memory intensive)
    """
    # Preprocess sequences
    sequences_processed = [preprocess_sequence(seq) for seq in sequences]
    
    # Tokenize
    ids = tokenizer(
        sequences_processed,
        add_special_tokens=True,
        padding="longest",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    input_ids = ids['input_ids'].to(device)
    attention_mask = ids['attention_mask'].to(device)
    
    # Generate embeddings
    with torch.no_grad():
        embeddings = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = embeddings.last_hidden_state  # (batch, seq_len, hidden_dim)
    
    # Remove padding and special tokens
    results = []
    for i, (emb, mask) in enumerate(zip(embeddings, attention_mask)):
        # Get valid positions (non-padding, excluding <pad>, </s>)
        valid_mask = mask.bool()
        valid_emb = emb[valid_mask]  # Remove padding
        
        # Remove start and end tokens
        valid_emb = valid_emb[1:-1]  # Remove <pad> and </s>
        
        # Mean pooling for per-protein representation
        mean_pooled = valid_emb.mean(dim=0)
        
        results.append({
            'mean_pooled': mean_pooled.cpu(),
            'per_residue': valid_emb.cpu(),  # Keep if needed for per-residue tasks
            'length': len(valid_emb)
        })
    
    return results


def generate_embeddings(
    metadata_path: str,
    output_dir: str,
    model_name: str = 'Rostlab/prot_t5_xl_uniref50',
    batch_size: int = 8,
    device: str = 'cuda',
    save_per_residue: bool = False,
    max_length: int = 1024
):
    """
    Generate and save embeddings for all sequences.
    
    Args:
        metadata_path: Path to metadata CSV with sequences
        output_dir: Directory to save embeddings
        model_name: HuggingFace model name
        batch_size: Batch size for embedding generation
        device: Device to use (cuda/cpu)
        save_per_residue: Whether to save per-residue embeddings
        max_length: Maximum sequence length
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    print(f"Loading metadata from {metadata_path}...")
    df = pd.read_csv(metadata_path)
    
    # Check if we have sequences
    if 'sequence' not in df.columns:
        raise ValueError("Metadata must contain 'sequence' column")
    
    print(f"Found {len(df)} sequences")
    
    # Load model
    device = device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    tokenizer, model = load_model(model_name, device)
    
    # Generate embeddings in batches
    print("\nGenerating embeddings...")
    n_batches = (len(df) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(df), batch_size), total=n_batches):
        batch_df = df.iloc[i:i+batch_size]
        batch_sequences = batch_df['sequence'].tolist()
        batch_entries = batch_df['entry'].tolist()
        
        # Skip if all embeddings already exist
        all_exist = all(
            (output_dir / f"{entry}.pt").exists()
            for entry in batch_entries
        )
        if all_exist:
            continue
        
        # Generate embeddings
        try:
            batch_embeddings = generate_embeddings_batch(
                batch_sequences,
                tokenizer,
                model,
                device,
                max_length=max_length
            )
            
            # Save individual embeddings
            for entry, emb_dict in zip(batch_entries, batch_embeddings):
                output_path = output_dir / f"{entry}.pt"
                
                # Prepare data to save
                save_dict = {'mean_pooled': emb_dict['mean_pooled']}
                if save_per_residue:
                    save_dict['per_residue'] = emb_dict['per_residue']
                
                torch.save(save_dict, output_path)
        
        except Exception as e:
            print(f"\nError processing batch starting at index {i}: {e}")
            print(f"Entries: {batch_entries}")
            continue
    
    print(f"\nEmbeddings saved to {output_dir}")
    
    # Verify all embeddings were generated
    missing = []
    for entry in df['entry']:
        if not (output_dir / f"{entry}.pt").exists():
            missing.append(entry)
    
    if missing:
        print(f"\nWarning: {len(missing)} embeddings were not generated:")
        print(missing[:10], "..." if len(missing) > 10 else "")
    else:
        print("\nAll embeddings generated successfully!")


def main():
    parser = argparse.ArgumentParser(description='Generate ProtT5 embeddings')
    parser.add_argument(
        '--metadata',
        type=str,
        required=True,
        help='Path to metadata CSV with sequences'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/embeddings',
        help='Output directory for embeddings'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='Rostlab/prot_t5_xl_uniref50',
        help='HuggingFace model name'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for embedding generation'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--save-per-residue',
        action='store_true',
        help='Save per-residue embeddings (memory intensive)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=1024,
        help='Maximum sequence length'
    )
    
    args = parser.parse_args()
    
    generate_embeddings(
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        save_per_residue=args.save_per_residue,
        max_length=args.max_length
    )


if __name__ == '__main__':
    main()