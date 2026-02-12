"""
Custom loss functions for deadwood increase prediction.
Weighted MSE loss with configurable target-dependent weighting for imbalanced regression.
"""

import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self, exponent=1.0, max_gain=5.0):
        super().__init__()
        self.exponent = exponent
        self.max_gain = max_gain

    def forward(self, pred, target):
        # target is [0, 1]
        # if target=0, weight=1
        # if target=1, weight=1 + max_gain (e.g., 6.0)
        weights = 1.0 + torch.pow(target, self.exponent) * self.max_gain
        
        loss = weights * (pred - target) ** 2
        return loss.mean()