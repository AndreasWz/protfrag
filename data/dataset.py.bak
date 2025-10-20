import os
from typing import Optional, List
import numpy as np
import torch
from torch.utils.data import Dataset


class EmbeddingSequenceDataset(Dataset):
    """
    Expected CSV columns: id, sequence, embedding_path, is_fragment (0/1), fragment_types (semicolon-separated, e.g. "N;I")
    """

    FRAG_TYPE_ORDER = ["N", "C", "I"]

    def __init__(self, records: List[dict], pool: str = "mean"):
        self.records = records
        self.pool = pool

    def __len__(self):
        return len(self.records)

    def _load_emb(self, path: str) -> np.ndarray:
        if path.endswith(".npy"):
            arr = np.load(path)
        else:
            loaded = np.load(path)
            if "emb" in loaded:
                arr = loaded["emb"]
            elif "arr_0" in loaded:
                arr = loaded["arr_0"]
            else:
                arr = loaded[list(loaded.keys())[0]]
        return arr

    def _pool(self, arr: np.ndarray) -> np.ndarray:
        if self.pool == "mean":
            return arr.mean(axis=0)
        elif self.pool == "max":
            return arr.max(axis=0)
        elif self.pool == "mean+max":
            return np.concatenate([arr.mean(axis=0), arr.max(axis=0)], axis=0)
        else:
            raise ValueError(f"Unknown pool: {self.pool}")

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        emb_path = rec.get("embedding_path")
        if emb_path is None:
            raise RuntimeError("embedding_path must be provided in CSV for this dataset implementation")
        arr = self._load_emb(emb_path)
        pooled = self._pool(arr)
        emb = torch.from_numpy(pooled).float()

        is_fragment = float(rec.get("is_fragment", 0))
        frag_types_str = rec.get("fragment_types", "") or ""
        frag_types = [0.0] * len(self.FRAG_TYPE_ORDER)
        if frag_types_str:
            for lab in frag_types_str.split(";"):
                lab = lab.strip()
                if lab in self.FRAG_TYPE_ORDER:
                    frag_types[self.FRAG_TYPE_ORDER.index(lab)] = 1.0
        frag_types = torch.tensor(frag_types, dtype=torch.float32)

        return {
            "id": rec.get("id"),
            "sequence": rec.get("sequence"),
            "emb": emb,
            "is_fragment": torch.tensor(is_fragment, dtype=torch.float32),
            "fragment_types": frag_types,
        }
