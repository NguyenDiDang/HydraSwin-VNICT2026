"""
Inverse-Frequency Class-Weighted Loss Formulation
Mitigates gradient starvation for long-tail minority classes.
"""

import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

def compute_class_weights(series, epsilon=1e-6):
    """
    Computes inverse-frequency penalty scalar:
    W_c = N_total / (N_c + epsilon)
    """
    clean_series = series.dropna()
    counts = clean_series.value_counts()
    total = len(clean_series)
    weights = {cls: total / (count + epsilon) for cls, count in counts.items()}
    encoder = LabelEncoder().fit(clean_series)
    return [weights[cls] for cls in encoder.classes_]

class MultiTaskInverseFrequencyLoss(nn.Module):
    """
    Global optimization objective summing weighted cross-entropy losses:
    L_total = sum(lambda_t * L_t(W_{c,t}))
    """
    def __init__(self, class_weights_dict=None, task_weights=None):
        super(MultiTaskInverseFrequencyLoss, self).__init__()
        self.class_weights_dict = class_weights_dict or {}
        self.task_weights = task_weights or {
            "gender": 1.0,
            "master_cat": 1.0,
            "sub_cat": 1.0,
            "article_type": 1.0,
            "base_colour": 1.0,
            "season": 1.0
        }

    def forward(self, outputs, targets):
        total_loss = 0.0
        losses_by_task = {}
        for task, logit in outputs.items():
            if task not in targets:
                continue
            weight = self.class_weights_dict.get(task, None)
            if weight is not None and not isinstance(weight, torch.Tensor):
                weight = torch.tensor(weight, dtype=torch.float32, device=logit.device)
            criterion = nn.CrossEntropyLoss(weight=weight)
            task_loss = criterion(logit, targets[task])
            lambda_t = self.task_weights.get(task, 1.0)
            total_loss += lambda_t * task_loss
            losses_by_task[task] = task_loss.item()
        return total_loss, losses_by_task
