import torch
import torch.nn as nn
from assymetric_loss_opt import AsymmetricLossOptimized


class BCEPolyLoss(nn.Module):
    def __init__(self, eps=1.0):
        super(BCEPolyLoss, self).__init__()
        self.eps = eps
        self.BCE = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, labels):
        """"
        Parameters
        ----------
        logits: input logits
        labels: targets (multi-label binarized vector)
        """
        pt = torch.sum(labels * torch.sigmoid(logits), dim=-1)[0].item()
        loss = self.BCE(logits, labels)
        poly = loss + self.eps * (1.0 - pt)

        return poly


class FLPolyLoss(nn.Module):
    def __init__(self, eps=1.0, gamma=2.0):
        super(FLPolyLoss, self).__init__()
        self.eps = eps
        self.gamma = gamma
        self.FL = AsymmetricLossOptimized()
        self.target = None

    def forward(self, logits, labels):
        p = torch.sigmoid(logits)
        pt = labels * p + (1.0 - labels) * (1.0 - p)
        target = self.target
        fl = self.FL(pt, target)

        poly_loss = fl + self.eps * torch.pow(1.0 - pt, self.gamma + 1.0)

        return poly_loss