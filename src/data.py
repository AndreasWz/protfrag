# src/data.py

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# (This class is probably fine, but included for completeness)
class ProteinFragmentDataset(Dataset):
    def __init__(self, metadata_df: pd.DataFrame, embedding_dir: Path, embedding_type: str = "mean_pooled"):
        self.metadata = metadata_df.reset_index(drop=True)
        self.embedding_dir = embedding_dir
        self.embedding_type = embedding_type
        self.label_type_columns = ['n_terminal', 'c_terminal', 'internal']

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        entry_id = row['entry']
        
        # Load pre-computed embedding
        emb_path = self.embedding_dir / f"{entry_id}.pt"
        try:
            # A2's embedding script saves a dictionary
            emb_dict = torch.load(emb_path, map_location='cpu')
            embedding = emb_dict[self.embedding_type]
        except FileNotFoundError:
            print(f"Warning: Embedding file not found for {entry_id}. Returning zero tensor.")
            embedding = torch.zeros(1024) # Fallback, adjust '1024' as needed
        except KeyError:
            print(f"Warning: Key '{self.embedding_type}' not in {emb_path}. Using first key.")
            emb_dict = torch.load(emb_path, map_location='cpu')
            embedding = emb_dict[list(emb_dict.keys())[0]]

        # Get binary label (is_fragment)
        label_binary = torch.tensor(row['is_fragment'], dtype=torch.float32)
        
        # Get multilabel target (fragment_types)
        label_type = torch.tensor(row[self.label_type_columns].values.astype(float), dtype=torch.float32)
        
        return {
            "entry": entry_id,
            "embedding": embedding.float(),
            "is_fragment": label_binary,
            "fragment_types": label_type
        }

#
# --- PASTE THIS ENTIRE CLASS ---
#

class FragmentDataModule(pl.LightningDataModule):
    def __init__(
        self,
        metadata_path: str,
        embedding_dir: str,
        embedding_type: str = "mean_pooled",
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.metadata_path = Path(self.hparams.metadata_path)
        # --- This is the fix from the 'str' vs 'Path' error ---
        # We save the Path object here for the dataloaders
        self.embedding_dir = Path(self.hparams.embedding_dir)
        # ---
        
        self.train_df = None
        self.val_df = None
        self.test_df = None

    def setup(self, stage: str = None):
        print(f"Loading metadata from {self.metadata_path}...")
        df = pd.read_csv(self.metadata_path)
        
        # Filter by split
        self.train_df = df[df['split'] == 'train']
        self.val_df = df[df['split'] == 'val']
        self.test_df = df[df['split'] == 'test']
        
        print(f"Train samples: {len(self.train_df)}")
        print(f"Validation samples: {len(self.val_df)}")
        print(f"Test samples: {len(self.test_df)}")

    def get_class_weights(self) -> dict:
        """Computes class weights for imbalanced datasets."""
        if self.train_df is None:
            self.setup('fit')
            
        # Binary weights
        binary_weights = compute_class_weight(
            'balanced',
            classes=np.array([0, 1]),
            y=self.train_df['is_fragment'].values
        )
        binary_weights_tensor = torch.tensor(binary_weights, dtype=torch.float32)
        
        # Multilabel weights (only on fragment data)
        fragment_df = self.train_df[self.train_df['is_fragment'] == 1]
        multilabel_weights = []
        for col in ['n_terminal', 'c_terminal', 'internal']:
            weights = compute_class_weight(
                'balanced',
                classes=np.array([0, 1]),
                y=fragment_df[col].values
            )
            # We need the weight for the positive class (1)
            multilabel_weights.append(weights[1])
            
        multilabel_weights_tensor = torch.tensor(multilabel_weights, dtype=torch.float32)

        return {
            "binary": binary_weights_tensor,
            "multilabel": multilabel_weights_tensor
        }

    def _create_dataset(self, df: pd.DataFrame) -> ProteinFragmentDataset:
        return ProteinFragmentDataset(
            metadata_df=df,
            # Use the Path object, not the hparams string
            embedding_dir=self.embedding_dir,
            embedding_type=self.hparams.embedding_type
        )

    def train_dataloader(self):
        return DataLoader(
            self._create_dataset(self.train_df),
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True if self.hparams.num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self._create_dataset(self.val_df),
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True if self.hparams.num_workers > 0 else False
        )

    def test_dataloader(self):
        # --- This was the fix for the previous error ---
        return DataLoader(
            self._create_dataset(self.test_df),
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True if self.hparams.num_workers > 0 else False
        )
    
    def predict_dataloader(self):
         return self.test_dataloader()