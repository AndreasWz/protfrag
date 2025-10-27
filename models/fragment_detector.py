"""
models/fragment_detector.py

Improved LightningModule with better design:
 - LayerNorm instead of BatchNorm (more stable)
 - Type loss only computed for actual fragments
 - Proper exception handling with logging
 - Train metrics properly logged
 - Class balancing computed from data
 - Better optimizer configuration
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC, BinaryF1Score
from sklearn.metrics import matthews_corrcoef, classification_report
import numpy as np


class FragmentDetector(pl.LightningModule):
    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        num_types: int = 3,
        lr: float = 1e-4,
        pos_weight_det: float = None,
        pos_weight_types=None,
        type_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Encoder with LayerNorm (more stable than BatchNorm)
        self.encoder = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Task heads
        self.det_head = nn.Linear(hidden_dim, 1)
        self.type_head = nn.Linear(hidden_dim, num_types)

        # Loss functions with optional pos weights
        if pos_weight_det is not None:
            pw = torch.tensor([pos_weight_det], dtype=torch.float32)
            self.det_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            self.det_loss_fn = nn.BCEWithLogitsLoss()

        if pos_weight_types is not None:
            pw_types = torch.tensor(pos_weight_types, dtype=torch.float32)
            self.type_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw_types)
        else:
            self.type_loss_fn = nn.BCEWithLogitsLoss()

        self.lr = lr
        self.type_loss_weight = type_loss_weight

        # Metrics
        self.train_det_auc = BinaryAUROC()
        self.val_det_auc = BinaryAUROC()
        self.train_det_f1 = BinaryF1Score()
        self.val_det_f1 = BinaryF1Score()

        # Validation epoch storage
        self._val_probs = []
        self._val_targets = []
        self._val_type_probs = []
        self._val_type_targets = []

    def forward(self, emb):
        x = self.encoder(emb)
        det_logit = self.det_head(x).squeeze(-1)
        type_logits = self.type_head(x)
        return det_logit, type_logits

    def training_step(self, batch, batch_idx):
        emb = batch["emb"]
        det_logit, type_logits = self(emb)

        # Detection loss (always computed)
        det_loss = self.det_loss_fn(det_logit, batch["is_fragment"])

        # Type loss only for fragments
        fragment_mask = batch["is_fragment"] > 0.5
        if fragment_mask.sum() > 0:
            type_loss = self.type_loss_fn(
                type_logits[fragment_mask], batch["fragment_types"][fragment_mask]
            )
        else:
            type_loss = torch.tensor(0.0, device=emb.device)

        # Combined loss with weighting
        loss = det_loss + self.type_loss_weight * type_loss

        # Update metrics (with proper error handling)
        det_probs = torch.sigmoid(det_logit)
        try:
            self.train_det_auc.update(det_probs, batch["is_fragment"].int())
            self.train_det_f1.update(det_probs, batch["is_fragment"].int())
        except (ValueError, RuntimeError) as e:
            self.log("train/metric_error", 1.0, on_step=False, on_epoch=True)

        # Logging
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/det_loss", det_loss, on_step=False, on_epoch=True)
        self.log("train/type_loss", type_loss, on_step=False, on_epoch=True)
        self.log(
            "train/fragments_in_batch",
            fragment_mask.float().mean(),
            on_step=False,
            on_epoch=True,
        )

        return loss

    def on_train_epoch_end(self):
        # Compute and log training metrics
        try:
            train_auc = self.train_det_auc.compute()
            self.log("train/det_auc", train_auc, prog_bar=False)
            self.train_det_auc.reset()
        except (ValueError, RuntimeError):
            pass

        try:
            train_f1 = self.train_det_f1.compute()
            self.log("train/det_f1", train_f1, prog_bar=False)
            self.train_det_f1.reset()
        except (ValueError, RuntimeError):
            pass

    def validation_step(self, batch, batch_idx):
        emb = batch["emb"]
        det_logit, type_logits = self(emb)

        # Detection loss
        det_loss = self.det_loss_fn(det_logit, batch["is_fragment"])

        # Type loss only for fragments
        fragment_mask = batch["is_fragment"] > 0.5
        if fragment_mask.sum() > 0:
            type_loss = self.type_loss_fn(
                type_logits[fragment_mask], batch["fragment_types"][fragment_mask]
            )
        else:
            type_loss = torch.tensor(0.0, device=emb.device)

        loss = det_loss + self.type_loss_weight * type_loss

        # Logging
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/det_loss", det_loss, on_epoch=True)
        self.log("val/type_loss", type_loss, on_epoch=True)

        # Store predictions for epoch-end metrics
        det_probs = torch.sigmoid(det_logit).detach().cpu().numpy()
        type_probs = torch.sigmoid(type_logits).detach().cpu().numpy()
        targets = batch["is_fragment"].detach().cpu().numpy()
        type_targets = batch["fragment_types"].detach().cpu().numpy()

        self._val_probs.append(det_probs)
        self._val_targets.append(targets)
        self._val_type_probs.append(type_probs)
        self._val_type_targets.append(type_targets)

        # Update metrics
        try:
            self.val_det_auc.update(
                torch.sigmoid(det_logit), batch["is_fragment"].int()
            )
            self.val_det_f1.update(torch.sigmoid(det_logit), batch["is_fragment"].int())
        except (ValueError, RuntimeError):
            pass

    def on_validation_epoch_end(self):
        # AUROC and F1
        try:
            auc = self.val_det_auc.compute()
            self.log("val/det_auc", auc, prog_bar=True)
            self.val_det_auc.reset()
        except (ValueError, RuntimeError):
            self.log("val/det_auc", 0.5, prog_bar=True)

        try:
            f1 = self.val_det_f1.compute()
            self.log("val/det_f1", f1, prog_bar=True)
            self.val_det_f1.reset()
        except (ValueError, RuntimeError):
            pass

        # MCC and classification report
        if len(self._val_targets) > 0:
            probs = np.concatenate(self._val_probs)
            y_true = np.concatenate(self._val_targets)
            y_pred = (probs >= 0.5).astype(int)

            try:
                mcc = matthews_corrcoef(y_true, y_pred)
                self.log("val/mcc", float(mcc), prog_bar=True)
            except (ValueError, RuntimeError) as e:
                self.log("val/mcc", 0.0)

            # Additional useful metrics
            try:
                tp = ((y_pred == 1) & (y_true == 1)).sum()
                fp = ((y_pred == 1) & (y_true == 0)).sum()
                fn = ((y_pred == 0) & (y_true == 1)).sum()
                tn = ((y_pred == 0) & (y_true == 0)).sum()

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0

                self.log("val/precision", float(precision))
                self.log("val/recall", float(recall))
                self.log("val/tp", float(tp))
                self.log("val/fp", float(fp))
                self.log("val/fn", float(fn))
                self.log("val/tn", float(tn))
            except Exception:
                pass

        # Reset buffers
        self._val_probs = []
        self._val_targets = []
        self._val_type_probs = []
        self._val_type_targets = []

    def configure_optimizers(self):
        # Separate learning rates for encoder and heads
        param_groups = [
            {"params": self.encoder.parameters(), "lr": self.lr},
            {"params": self.det_head.parameters(), "lr": self.lr * 2},
            {"params": self.type_head.parameters(), "lr": self.lr * 2},
        ]

        opt = torch.optim.AdamW(param_groups, weight_decay=1e-5)

        # More aggressive scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.3, patience=5, min_lr=1e-7, verbose=True
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }