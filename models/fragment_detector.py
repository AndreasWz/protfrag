"""
models/fragment_detector.py

Improved LightningModule:
 - BatchNorm layers in encoder
 - pos_weight support for BCEWithLogitsLoss on detection head and per-type weights for multilabel head
 - validation epoch aggregation computes MCC and logs AUROC
 - ReduceLROnPlateau scheduler
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC
from sklearn.metrics import matthews_corrcoef
import numpy as np

class FragmentDetector(pl.LightningModule):
    def __init__(self, emb_dim: int, hidden_dim: int = 512, dropout: float = 0.2, num_types: int = 3, lr: float = 1e-4, pos_weight_det: float = None, pos_weight_types=None):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.det_head = nn.Linear(hidden_dim, 1)
        self.type_head = nn.Linear(hidden_dim, num_types)

        # Loss functions with optional pos weights
        if pos_weight_det is not None:
            pw = torch.tensor(pos_weight_det, dtype=torch.float32)
            self.det_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            self.det_loss_fn = nn.BCEWithLogitsLoss()

        if pos_weight_types is not None:
            pw_types = torch.tensor(pos_weight_types, dtype=torch.float32)
            self.type_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw_types)
        else:
            self.type_loss_fn = nn.BCEWithLogitsLoss()

        self.lr = lr

        # metrics
        self.train_det_auc = BinaryAUROC()
        self.val_det_auc = BinaryAUROC()

        # store epoch preds
        self._val_probs = []
        self._val_targets = []

    def forward(self, emb):
        x = self.encoder(emb)
        det_logit = self.det_head(x).squeeze(-1)
        type_logits = self.type_head(x)
        return det_logit, type_logits

    def training_step(self, batch, batch_idx):
        emb = batch["emb"]
        det_logit, type_logits = self(emb)

        det_loss = self.det_loss_fn(det_logit, batch["is_fragment"])
        type_loss = self.type_loss_fn(type_logits, batch["fragment_types"])
        loss = det_loss + type_loss

        # update metric
        try:
            self.train_det_auc.update(torch.sigmoid(det_logit), batch["is_fragment"].int())
        except Exception:
            pass

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/det_loss", det_loss, on_step=True, on_epoch=True)
        self.log("train/type_loss", type_loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        emb = batch["emb"]
        det_logit, type_logits = self(emb)

        det_loss = self.det_loss_fn(det_logit, batch["is_fragment"])
        type_loss = self.type_loss_fn(type_logits, batch["fragment_types"])
        loss = det_loss + type_loss

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/det_loss", det_loss, on_epoch=True)
        self.log("val/type_loss", type_loss, on_epoch=True)

        probs = torch.sigmoid(det_logit).detach().cpu().numpy()
        types_prob = torch.sigmoid(type_logits).detach().cpu().numpy()

        self._val_probs.append(probs)
        self._val_targets.append(batch["is_fragment"].detach().cpu().numpy())

        try:
            self.val_det_auc.update(torch.sigmoid(det_logit), batch["is_fragment"].int())
        except Exception:
            pass

    def on_validation_epoch_end(self):
        # AUROC
        try:
            auc = self.val_det_auc.compute()
            self.log("val/det_auc", auc, prog_bar=True)
            self.val_det_auc.reset()
        except Exception:
            pass

        # MCC aggregated
        if len(self._val_targets) > 0:
            probs = np.concatenate(self._val_probs)
            y_true = np.concatenate(self._val_targets)
            y_pred = (probs >= 0.5).astype(int)
            try:
                mcc = matthews_corrcoef(y_true, y_pred)
                self.log("val/mcc", float(mcc), prog_bar=True)
            except Exception:
                pass

        # reset buffers
        self._val_probs = []
        self._val_targets = []

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"}}
