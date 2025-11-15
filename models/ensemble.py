import torch
import torch.nn as nn
from typing import List, Tuple

class ReadoutAdapter(nn.Module):
    """
    Adapter that turns a model's output into the correct training domain.
    - For classification: should output logits (no activation).
    - For regression: identity by default (unless you want to re-map).
    """
    def __init__(self, in_features: int, out_features: int, mode: str):
        super().__init__()
        self.mode = mode  # 'classification' or 'regression'
        if mode == 'classification':
            # Map to logits; identity if already logits-like (you can set weight to identity later if you want).
            self.readout = nn.Linear(in_features, out_features)
        else:
            # For regression, do nothing by default.
            self.readout = nn.Identity()

    def forward(self, x):
        return self.readout(x)

class ModelWithHead(nn.Module):
    """
    Wraps a base model with a readout head (needed if the base model doesn't output logits for CE).
    """
    def __init__(self, base: nn.Module, readout: nn.Module = None):
        super().__init__()
        self.base = base
        self.readout = readout if readout is not None else nn.Identity()

    def forward(self, x):
        return self.readout(self.base(x))

class EnsembleModel(nn.Module):
    """
    Simple weighted ensemble over multiple models.
    - For classification: averages **logits** before argmax.
    - For regression: averages outputs.
    """
    def __init__(self, members: List[nn.Module], weights: List[float] = None):
        super().__init__()
        self.members = nn.ModuleList(members)
        if weights is None:
            weights = [1.0] * len(members)
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, x):
        logits_or_outputs = []
        for m in self.members:
            logits_or_outputs.append(m(x))
        stacked = torch.stack(logits_or_outputs, dim=0)   # [M, B, D]
        w = self.weights / (self.weights.sum() + 1e-12)   # normalize
        w = w.view(-1, 1, 1)                              # [M,1,1]
        return (w * stacked).sum(dim=0)                   # [B, D]