# src/model.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Optional
from torchmetrics import AUROC

# Import from our new local metrics file
from .metrics import MatthewsCorrCoef, MultilabelMetrics, PerClassMCC

class FragmentDetector(pl.LightningModule):
    """
    Multi-task model for protein fragment prediction.
    (Based on Approach 2's FragmentPredictor)
    """
    
    def __init__(
        self,
        embedding_dim: int = 1024,
        hidden_dims: list = [512, 256],
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        binary_loss_weight: float = 1.0,
        multilabel_loss_weight: float = 1.0,
        use_class_weights: bool = True,
        binary_class_weights: Optional[torch.Tensor] = None,
        multilabel_class_weights: Optional[torch.Tensor] = None,
        scheduler: str = 'cosine',
        warmup_epochs: int = 5
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Build shared encoder
        layers = []
        in_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        
        # Binary classification head
        self.binary_head = nn.Linear(in_dim, 1)
        
        # Multilabel classification head
        self.multilabel_head = nn.Linear(in_dim, 3)  # [N, C, Internal]
        
        # Loss functions
        pos_weight_binary = binary_class_weights[1:2] if use_class_weights and binary_class_weights is not None else None
        self.binary_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_binary)
        
        pos_weight_multi = multilabel_class_weights if use_class_weights and multilabel_class_weights is not None else None
        self.multilabel_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_multi)
        
        # Metrics
        self.train_binary_mcc = MatthewsCorrCoef()
        self.val_binary_mcc = MatthewsCorrCoef()
        self.test_binary_mcc = MatthewsCorrCoef()
        
        self.val_multilabel_metrics = MultilabelMetrics(num_classes=3)
        self.test_multilabel_metrics = MultilabelMetrics(num_classes=3)
        
        self.val_binary_auroc = AUROC(task='binary')
        self.test_binary_auroc = AUROC(task='binary')
        
        self.val_multilabel_mcc = PerClassMCC(num_classes=3)
        self.test_multilabel_mcc = PerClassMCC(num_classes=3)
    
    def forward(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.encoder(embedding)
        binary_logits = self.binary_head(features).squeeze(-1)
        multilabel_logits = self.multilabel_head(features)
        
        return {
            'binary_logits': binary_logits,
            'multilabel_logits': multilabel_logits
        }
    
    def _shared_step(self, batch: Dict, stage: str):
        embeddings = batch['embedding']
        is_fragment = batch['is_fragment']
        fragment_types = batch['fragment_types']
        
        outputs = self(embeddings)
        binary_logits = outputs['binary_logits']
        multilabel_logits = outputs['multilabel_logits']
        
        # Compute losses
        loss_b = self.binary_criterion(binary_logits, is_fragment)
        
        # Only compute multilabel loss for fragments
        fragment_mask = is_fragment == 1
        if fragment_mask.sum() > 0:
            loss_m = self.multilabel_criterion(
                multilabel_logits[fragment_mask],
                fragment_types[fragment_mask]
            )
        else:
            loss_m = torch.tensor(0.0, device=self.device)
        
        total_loss = (
            self.hparams.binary_loss_weight * loss_b +
            self.hparams.multilabel_loss_weight * loss_m
        )
        
        binary_probs = torch.sigmoid(binary_logits)
        multilabel_probs = torch.sigmoid(multilabel_logits)
        
        self.log(f'{stage}/loss_total', total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f'{stage}/loss_binary', loss_b, on_step=False, on_epoch=True)
        self.log(f'{stage}/loss_multilabel', loss_m, on_step=False, on_epoch=True)
        
        # Update metrics
        if stage == 'train':
            self.train_binary_mcc(binary_probs, is_fragment)
        elif stage == 'val':
            self.val_binary_mcc(binary_probs, is_fragment)
            self.val_binary_auroc(binary_probs, is_fragment.long())
            if fragment_mask.sum() > 0:
                self.val_multilabel_metrics(multilabel_probs[fragment_mask], fragment_types[fragment_mask])
                self.val_multilabel_mcc(multilabel_probs[fragment_mask], fragment_types[fragment_mask])
        elif stage == 'test':
            self.test_binary_mcc(binary_probs, is_fragment)
            self.test_binary_auroc(binary_probs, is_fragment.long())
            if fragment_mask.sum() > 0:
                self.test_multilabel_metrics(multilabel_probs[fragment_mask], fragment_types[fragment_mask])
                self.test_multilabel_mcc(multilabel_probs[fragment_mask], fragment_types[fragment_mask])
        
        return {'loss': total_loss}

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, 'val')
    
    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, 'test')
    
    def _log_epoch_end_metrics(self, stage: str):
        if stage == 'train':
            mcc = self.train_binary_mcc.compute()
            self.log('train/binary_mcc', mcc, prog_bar=True)
            self.train_binary_mcc.reset()
            return

        mcc = self.val_binary_mcc.compute() if stage == 'val' else self.test_binary_mcc.compute()
        auroc = self.val_binary_auroc.compute() if stage == 'val' else self.test_binary_auroc.compute()
        multi_metrics = self.val_multilabel_metrics.compute() if stage == 'val' else self.test_multilabel_metrics.compute()
        multi_mcc = self.val_multilabel_mcc.compute() if stage == 'val' else self.test_multilabel_mcc.compute()
        
        self.log(f'{stage}/binary_mcc', mcc, prog_bar=True)
        self.log(f'{stage}/binary_auroc', auroc)
        self.log(f'{stage}/multilabel_f1', multi_metrics['macro_f1'])
        self.log(f'{stage}/multilabel_mcc', multi_mcc['macro_mcc'])
        
        class_names = ['n_terminal', 'c_terminal', 'internal']
        for i, name in enumerate(class_names):
            self.log(f'{stage}/f1_{name}', multi_metrics['f1'][i])
            self.log(f'{stage}/mcc_{name}', multi_mcc['per_class_mcc'][i])
        
        if stage == 'val':
            self.val_binary_mcc.reset()
            self.val_binary_auroc.reset()
            self.val_multilabel_metrics.reset()
            self.val_multilabel_mcc.reset()
        else:
            self.test_binary_mcc.reset()
            self.test_binary_auroc.reset()
            self.test_multilabel_metrics.reset()
            self.test_multilabel_mcc.reset()

    def on_train_epoch_end(self):
        self._log_epoch_end_metrics('train')

    def on_validation_epoch_end(self):
        self._log_epoch_end_metrics('val')

    def on_test_epoch_end(self):
        self._log_epoch_end_metrics('test')
    
    def configure_optimizers(self):
        # (Copied from A2)
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        if self.hparams.scheduler == 'cosine':
            # This requires access to the datamodule, which is complex.
            # A simpler CosineAnnealingLR is often sufficient.
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.max_epochs - self.hparams.warmup_epochs
            )
            return [optimizer], [scheduler]
        elif self.hparams.scheduler == 'reduce_on_plateau':
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='max', factor=0.5, patience=5
                    ),
                    'monitor': 'val/binary_mcc',
                }
            }
        else:
            return optimizer
    
    def predict_step(self, batch, batch_idx):
        # (Copied from A2)
        embeddings = batch['embedding']
        outputs = self(embeddings)
        binary_probs = torch.sigmoid(outputs['binary_logits'])
        multilabel_probs = torch.sigmoid(outputs['multilabel_logits'])
        
        return {
            'entry': batch['entry'],
            'binary_probs': binary_probs,
            'multilabel_probs': multilabel_probs,
            'binary_preds': (binary_probs > 0.5).long(),
            'multilabel_preds': (multilabel_probs > 0.5).long()
        }