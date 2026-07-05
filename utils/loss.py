import torch.nn as nn


def build_loss():

    return nn.CrossEntropyLoss(
        label_smoothing=0.1
    )