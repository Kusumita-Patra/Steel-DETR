import torch
import numpy as np


def mixup_data(images, labels, alpha=0.2):
    """
    Perform MixUp augmentation.

    Returns:
        mixed_images
        labels_a
        labels_b
        lambda
    """

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = images.size(0)

    index = torch.randperm(batch_size).to(images.device)

    mixed_images = lam * images + (1 - lam) * images[index]

    labels_a = labels
    labels_b = labels[index]

    return mixed_images, labels_a, labels_b, lam

def mixup_loss(criterion, pred, y_a, y_b, lam):

    return (
        lam * criterion(pred, y_a)
        + (1 - lam) * criterion(pred, y_b)
    )