import os
import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Optional

from .dataset import EmbeddingSequenceDataset


class ProtFragDataModule(pl.LightningDataModule):
    def __init__(self, csv_path: str, emb_dir: Optional[str] = None, batch_size: int = 64, pool: str = "mean"):
        super().__init__()
        self.csv_path = csv_path
        self.emb_dir = emb_dir
        self.batch_size = batch_size
        self.pool = pool

    def setup(self, stage=None):
        df = pd.read_csv(self.csv_path)
        records = []
        for _, row in df.iterrows():
            emb_path = row.get("embedding_path")
            if pd.isna(emb_path) and self.emb_dir:
                emb_f = os.path.join(self.emb_dir, f"{row['id']}.npy")
                emb_path = emb_f
            elif self.emb_dir and emb_path and not os.path.isabs(emb_path):
                emb_path = os.path.join(self.emb_dir, emb_path)

            records.append(
                {
                    "id": row.get("id"),
                    "sequence": row.get("sequence"),
                    "embedding_path": emb_path,
                    "is_fragment": row.get("is_fragment", 0),
                    "fragment_types": row.get("fragment_types", ""),
                }
            )

        n = len(records)
        train = records[: int(n * 0.8)]
        val = records[int(n * 0.8) : int(n * 0.9)]
        test = records[int(n * 0.9) :]

        self.train_ds = EmbeddingSequenceDataset(train, pool=self.pool)
        self.val_ds = EmbeddingSequenceDataset(val, pool=self.pool)
        self.test_ds = EmbeddingSequenceDataset(test, pool=self.pool)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=2)
