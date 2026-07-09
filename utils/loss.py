import torch
import torch.nn as nn


def build_loss():

    return nn.CrossEntropyLoss(
        label_smoothing=0.1
    )

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy with Label Smoothing
    """

    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()

        self.smoothing = smoothing
        self.weight = weight

    def forward(self, pred, target):

        num_classes = pred.size(1)

        log_probs = torch.log_softmax(pred, dim=1)

        with torch.no_grad():

            true_dist = torch.zeros_like(log_probs)

            true_dist.fill_(self.smoothing / (num_classes - 1))

            true_dist.scatter_(
                1,
                target.unsqueeze(1),
                1.0 - self.smoothing
            )

        loss = -true_dist * log_probs

        if self.weight is not None:

            class_weight = self.weight[target]

            loss = loss.sum(dim=1) * class_weight

        else:

            loss = loss.sum(dim=1)

        return loss.mean()