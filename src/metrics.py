# src/metrics.py
import torch
from torchmetrics import Metric, F1Score, Precision, Recall, MatthewsCorrCoef as BinaryMatthewsCorrCoef

class MatthewsCorrCoef(BinaryMatthewsCorrCoef):
    """Wrapper for Binary MCC to handle edge cases."""
    def compute(self) -> torch.Tensor:
        try:
            return super().compute()
        except RuntimeError:
            # Handle "The total number of predictions ... is zero"
            return torch.tensor(0.0, device=self.device)

class MultilabelMetrics(Metric):
    """Calculates per-class and macro F1/Precision/Recall."""
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.precision = Precision(task='multilabel', num_labels=num_classes, average=None)
        self.recall = Recall(task='multilabel', num_labels=num_classes, average=None)
        self.f1 = F1Score(task='multilabel', num_labels=num_classes, average=None)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.precision.update(preds, target)
        self.recall.update(preds, target)
        self.f1.update(preds, target)

    def compute(self) -> dict:
        f1 = self.f1.compute()
        return {
            'f1': f1,
            'precision': self.precision.compute(),
            'recall': self.recall.compute(),
            'macro_f1': f1.mean()
        }

class PerClassMCC(Metric):
    """Calculates MCC for each class in a multilabel setting."""
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.mccs = torch.nn.ModuleList([BinaryMatthewsCorrCoef(task="binary") for _ in range(num_classes)])

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        for i in range(self.num_classes):
            self.mccs[i].update(preds[:, i], target[:, i])

    def compute(self) -> dict:
        per_class_mcc = torch.stack([mcc.compute() for mcc in self.mccs])
        return {
            'per_class_mcc': per_class_mcc,
            'macro_mcc': per_class_mcc.mean()
        }