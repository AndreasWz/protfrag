# scripts/evaluate_model.py
import torch, pandas as pd, numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score, confusion_matrix

# load checkpoint
ckpt = "path/to/checkpoint.ckpt"
model = FragmentDetector.load_from_checkpoint(ckpt)
model.eval()

# dataloader
from data.datamodule import ProtFragDataModule
dm = ProtFragDataModule("dataset.csv", emb_dir="data/embeddings", batch_size=64)
dm.setup()
loader = dm.test_dataloader()

y_true, y_pred_prob = [], []
for batch in loader:
    with torch.no_grad():
        logits, _ = model(batch['emb'])
        probs = torch.sigmoid(logits).cpu().numpy()
    y_true.extend(batch['is_fragment'].cpu().numpy().tolist())
    y_pred_prob.extend(probs.tolist())

y_true = np.array(y_true); y_pred_prob = np.array(y_pred_prob)
y_pred = (y_pred_prob >= 0.5).astype(int)
print("MCC:", matthews_corrcoef(y_true, y_pred))
print("ROC-AUC:", roc_auc_score(y_true, y_pred_prob))
print(confusion_matrix(y_true, y_pred))
