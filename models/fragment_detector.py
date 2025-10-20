import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import AUROC, F1Score


class FragmentDetector(pl.LightningModule):
    def __init__(self, emb_dim: int, hidden_dim: int = 512, dropout: float = 0.2, num_types: int = 3, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.det_head = nn.Linear(hidden_dim, 1)
        self.type_head = nn.Linear(hidden_dim, num_types)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = lr

        self.train_det_auc = AUROC(task="binary")
        self.val_det_auc = AUROC(task="binary")
        self.val_f1 = F1Score(task="binary")

    def forward(self, emb):
        x = self.encoder(emb)
        det_logit = self.det_head(x).squeeze(-1)
        type_logits = self.type_head(x)
        return det_logit, type_logits

    def training_step(self, batch, batch_idx):
        emb = batch["emb"]
        det_logit, type_logits = self(emb)

        det_loss = self.loss_fn(det_logit, batch["is_fragment"])
        type_loss = self.loss_fn(type_logits, batch["fragment_types"])
        loss = det_loss + type_loss

        try:
            self.train_det_auc.update(torch.sigmoid(det_logit), batch["is_fragment"].long())
        except Exception:
            pass

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/det_loss", det_loss, on_step=True, on_epoch=True)
        self.log("train/type_loss", type_loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        emb = batch["emb"]
        det_logit, type_logits = self(emb)

        det_loss = self.loss_fn(det_logit, batch["is_fragment"])
        type_loss = self.loss_fn(type_logits, batch["fragment_types"])
        loss = det_loss + type_loss

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/det_loss", det_loss, on_epoch=True)
        self.log("val/type_loss", type_loss, on_epoch=True)

        try:
            self.val_det_auc.update(torch.sigmoid(det_logit), batch["is_fragment"].long())
        except Exception:
            pass

        return {"det_logits": det_logit, "is_fragment": batch["is_fragment"], "type_logits": type_logits}

    def on_validation_epoch_end(self) -> None:
        """Compatibility with Lightning v2+: compute and log validation AUROC here.
        We use the torchmetrics AUROC instance stored on the module.
        """
        try:
            auc = self.val_det_auc.compute()
            # log scalar (on_epoch True is default for this hook)
            self.log("val/det_auc", auc, prog_bar=True)
            # reset metric for next epoch
            self.val_det_auc.reset()
        except Exception:
            pass


    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return opt
