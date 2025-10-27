"""
train.py

Improved training script with:
 - Automatic pos_weight calculation from data
 - Reproducibility (seeding)
 - Test set evaluation
 - Better logging and checkpointing
 - Gradient clipping
 - Automatic num_types detection
"""
import argparse
import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from data.datamodule import ProtFragDataModule
from models.fragment_detector import FragmentDetector


def infer_emb_dim(emb_dir, sample_id=None):
    """Infer embedding dimension from a sample file."""
    if sample_id:
        arr = np.load(os.path.join(emb_dir, f"{sample_id}.npy"))
    else:
        files = [f for f in os.listdir(emb_dir) if f.endswith('.npy')]
        if not files:
            raise RuntimeError('No .npy found in emb_dir')
        arr = np.load(os.path.join(emb_dir, files[0]))
    
    # Handle pooling dimension
    return arr.shape[1] if len(arr.shape) > 1 else arr.shape[0]


def compute_pos_weights(csv_path, fragment_types_col="fragment_types"):
    """
    Compute pos_weight for detection and per-type classification.
    
    Returns:
        pos_weight_det: float, ratio of negatives to positives
        pos_weight_types: list, per-type ratios
        num_types: int, number of fragment types
    """
    df = pd.read_csv(csv_path)
    
    # Detection pos_weight
    n_fragments = df["is_fragment"].sum()
    n_non_fragments = len(df) - n_fragments
    
    if n_fragments == 0:
        print("⚠️  WARNING: No fragments found in dataset!")
        pos_weight_det = 1.0
    else:
        pos_weight_det = n_non_fragments / n_fragments
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Fragments: {n_fragments} ({n_fragments/len(df)*100:.1f}%)")
    print(f"  Non-fragments: {n_non_fragments} ({n_non_fragments/len(df)*100:.1f}%)")
    print(f"  Detection pos_weight: {pos_weight_det:.3f}")
    
    # Type pos_weights
    if fragment_types_col not in df.columns:
        print("⚠️  WARNING: fragment_types column not found, using uniform weights")
        return pos_weight_det, None, 3
    
    # Parse fragment types (assuming comma-separated binary string like "1,0,1")
    type_counts = []
    fragment_rows = df[df["is_fragment"] == 1]
    
    if len(fragment_rows) == 0:
        return pos_weight_det, None, 3
    
    for _, row in fragment_rows.iterrows():
        types_str = str(row[fragment_types_col])
        if types_str and types_str != "nan" and types_str != "":
            try:
                types = [int(x) for x in types_str.split(",")]
                if not type_counts:
                    type_counts = [0] * len(types)
                for i, val in enumerate(types):
                    type_counts[i] += val
            except:
                pass
    
    num_types = len(type_counts) if type_counts else 3
    
    # Compute per-type weights
    pos_weight_types = []
    if type_counts:
        print(f"\n  Fragment type distribution:")
        for i, count in enumerate(type_counts):
            if count == 0:
                weight = 1.0
                print(f"    Type {i}: 0 samples (weight: 1.0)")
            else:
                neg_count = len(fragment_rows) - count
                weight = neg_count / count if count > 0 else 1.0
                print(f"    Type {i}: {count} samples ({count/len(fragment_rows)*100:.1f}%, weight: {weight:.3f})")
            pos_weight_types.append(weight)
    else:
        pos_weight_types = [1.0] * num_types
    
    return pos_weight_det, pos_weight_types, num_types


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    import random
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CUDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"🌱 Random seed set to: {seed}")


def main():
    p = argparse.ArgumentParser(description="Train Fragment Detector")
    
    # Data
    p.add_argument("--data-csv", required=True, help="Path to CSV with annotations")
    p.add_argument("--emb-dir", required=True, help="Directory with .npy embeddings")
    
    # Model
    p.add_argument("--hidden-dim", default=512, type=int, help="Hidden dimension")
    p.add_argument("--dropout", default=0.2, type=float, help="Dropout rate")
    p.add_argument("--pool", default="mean", choices=["mean", "max", "mean+max"], 
                   help="Pooling strategy")
    
    # Training
    p.add_argument("--batch-size", default=64, type=int, help="Batch size")
    p.add_argument("--max-epochs", default=50, type=int, help="Maximum epochs")
    p.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
    p.add_argument("--type-loss-weight", default=1.0, type=float, 
                   help="Weight for type classification loss")
    
    # Hardware
    p.add_argument("--gpus", default=1, type=int, help="Number of GPUs (0 for CPU)")
    p.add_argument("--num-workers", default=4, type=int, help="DataLoader workers")
    
    # Experiment
    p.add_argument("--experiment-name", default="fragment_detector", 
                   help="Experiment name for logging")
    p.add_argument("--seed", default=42, type=int, help="Random seed")
    p.add_argument("--disable-pos-weight", action="store_true", 
                   help="Disable automatic pos_weight calculation")
    p.add_argument("--gradient-clip", default=1.0, type=float, 
                   help="Gradient clipping value (0 to disable)")
    
    args = p.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Compute pos_weights from data
    if args.disable_pos_weight:
        print("⚠️  pos_weight calculation disabled")
        pos_weight_det = None
        pos_weight_types = None
        num_types = 3
    else:
        pos_weight_det, pos_weight_types, num_types = compute_pos_weights(args.data_csv)
    
    # Infer embedding dimension
    emb_dim = infer_emb_dim(args.emb_dir)
    if args.pool == 'mean+max':
        emb_dim = emb_dim * 2
    print(f"\n🔢 Embedding dimension: {emb_dim}")
    
    # Setup data module
    dm = ProtFragDataModule(
        csv_path=args.data_csv,
        emb_dir=args.emb_dir,
        batch_size=args.batch_size,
        pool=args.pool,
        num_workers=args.num_workers,
        seed=args.seed
    )
    dm.setup()
    
    # Create model
    model = FragmentDetector(
        emb_dim=emb_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_types=num_types,
        lr=args.lr,
        pos_weight_det=pos_weight_det,
        pos_weight_types=pos_weight_types,
        type_loss_weight=args.type_loss_weight
    )
    
    print(f"\n🏗️  Model created:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss',
        mode='min',
        save_top_k=3,
        filename='epoch{epoch:02d}-loss{val/loss:.4f}',
        auto_insert_metric_name=False
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val/loss',
        patience=10,
        mode='min',
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Logger
    logger = TensorBoardLogger(
        save_dir="logs",
        name=args.experiment_name,
        default_hp_metric=False
    )
    
    # Trainer
    if args.gpus and args.gpus > 0:
        accelerator = 'gpu'
        devices = args.gpus
    else:
        accelerator = 'cpu'
        devices = 1
    
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        accelerator=accelerator,
        devices=devices,
        gradient_clip_val=args.gradient_clip if args.gradient_clip > 0 else None,
        log_every_n_steps=10,
        precision='16-mixed' if accelerator == 'gpu' else 32,
        deterministic=True
    )
    
    print(f"\n🚀 Starting training...")
    print(f"  Accelerator: {accelerator}")
    print(f"  Devices: {devices}")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Gradient clip: {args.gradient_clip if args.gradient_clip > 0 else 'disabled'}")
    print(f"  Logs: {logger.log_dir}\n")
    
    # Train
    trainer.fit(model, datamodule=dm)
    
    # Test evaluation
    print("\n📊 Evaluating on test set...")
    test_results = trainer.test(model, datamodule=dm, ckpt_path='best')
    
    print("\n✅ Training complete!")
    print(f"  Best model: {checkpoint_callback.best_model_path}")
    print(f"  Best val/loss: {checkpoint_callback.best_model_score:.4f}")


if __name__ == '__main__':
    main()