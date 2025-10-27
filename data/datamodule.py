"""
data/datamodule.py

Improved DataModule with:
 - Shuffled and stratified splits
 - Dataset statistics logging
 - Better error handling
 - Configurable num_workers
 - Validation of embeddings
"""
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Optional
from sklearn.model_selection import train_test_split

from .dataset import EmbeddingSequenceDataset


class ProtFragDataModule(pl.LightningDataModule):
    def __init__(
        self,
        csv_path: str,
        emb_dir: Optional[str] = None,
        batch_size: int = 64,
        pool: str = "mean",
        num_workers: int = 4,
        seed: int = 42,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        stratify: bool = True
    ):
        super().__init__()
        self.csv_path = csv_path
        self.emb_dir = emb_dir
        self.batch_size = batch_size
        self.pool = pool
        self.num_workers = num_workers
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.stratify = stratify

        # Will be set in setup()
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage=None):
        """Prepare data splits with proper shuffling and stratification."""
        
        print(f"\n📂 Loading data from {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        
        # Build records
        records = []
        missing_embeddings = []
        
        for idx, row in df.iterrows():
            emb_path = row.get("embedding_path")
            
            # Resolve embedding path
            if pd.isna(emb_path) and self.emb_dir:
                emb_path = os.path.join(self.emb_dir, f"{row['id']}.npy")
            elif self.emb_dir and emb_path and not os.path.isabs(emb_path):
                emb_path = os.path.join(self.emb_dir, emb_path)
            
            # Check if embedding exists
            if emb_path and not os.path.exists(emb_path):
                missing_embeddings.append((row.get("id"), emb_path))
                continue
            
            records.append({
                "id": row.get("id"),
                "sequence": row.get("sequence"),
                "embedding_path": emb_path,
                "is_fragment": row.get("is_fragment", 0),
                "fragment_types": row.get("fragment_types", ""),
            })
        
        if missing_embeddings:
            print(f"⚠️  WARNING: {len(missing_embeddings)} embeddings not found")
            if len(missing_embeddings) <= 5:
                for sample_id, path in missing_embeddings:
                    print(f"  - {sample_id}: {path}")
        
        if len(records) == 0:
            raise RuntimeError("No valid records found! Check your CSV and embedding paths.")
        
        print(f"✅ Loaded {len(records)} valid records")
        
        # Create labels for stratification
        labels = [r["is_fragment"] for r in records]
        
        # Shuffle records
        indices = np.arange(len(records))
        np.random.seed(self.seed)
        np.random.shuffle(indices)
        records = [records[i] for i in indices]
        labels = [labels[i] for i in indices]
        
        # Split data
        n = len(records)
        
        if self.stratify and len(set(labels)) > 1:
            # Stratified split
            try:
                # First split: train vs (val+test)
                train_indices, temp_indices = train_test_split(
                    range(n),
                    train_size=self.train_ratio,
                    stratify=labels,
                    random_state=self.seed
                )
                
                # Second split: val vs test
                temp_labels = [labels[i] for i in temp_indices]
                val_size = self.val_ratio / (self.val_ratio + self.test_ratio)
                
                val_indices, test_indices = train_test_split(
                    temp_indices,
                    train_size=val_size,
                    stratify=temp_labels,
                    random_state=self.seed
                )
                
                train = [records[i] for i in train_indices]
                val = [records[i] for i in val_indices]
                test = [records[i] for i in test_indices]
                
                print(f"✅ Stratified split created")
                
            except ValueError as e:
                print(f"⚠️  Stratification failed ({e}), using random split")
                self.stratify = False
        
        if not self.stratify:
            # Simple random split
            train_end = int(n * self.train_ratio)
            val_end = train_end + int(n * self.val_ratio)
            
            train = records[:train_end]
            val = records[train_end:val_end]
            test = records[val_end:]
        
        # Print split statistics
        self._print_split_stats("Train", train)
        self._print_split_stats("Val", val)
        self._print_split_stats("Test", test)
        
        # Create datasets
        self.train_ds = EmbeddingSequenceDataset(train, pool=self.pool)
        self.val_ds = EmbeddingSequenceDataset(val, pool=self.pool)
        self.test_ds = EmbeddingSequenceDataset(test, pool=self.pool)
        
        print(f"\n✅ Datasets created (pool={self.pool})")

    def _print_split_stats(self, split_name: str, records: list):
        """Print statistics for a data split."""
        n_fragments = sum(1 for r in records if r["is_fragment"] == 1)
        n_total = len(records)
        frag_pct = (n_fragments / n_total * 100) if n_total > 0 else 0
        
        # Analyze fragment types
        type_stats = {}
        for r in records:
            if r["is_fragment"] == 1:
                types_str = str(r["fragment_types"])
                if types_str and types_str != "nan" and types_str != "":
                    try:
                        types = [int(x) for x in types_str.split(",")]
                        for i, val in enumerate(types):
                            if val == 1:
                                type_stats[i] = type_stats.get(i, 0) + 1
                    except:
                        pass
        
        type_summary = ", ".join([f"T{i}:{cnt}" for i, cnt in sorted(type_stats.items())])
        
        print(f"\n  {split_name} split: {n_total} samples")
        print(f"    Fragments: {n_fragments} ({frag_pct:.1f}%)")
        if type_summary:
            print(f"    Type counts: {type_summary}")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if self.num_workers > 0 else False,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.num_workers > 0 else False,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.num_workers > 0 else False
        )