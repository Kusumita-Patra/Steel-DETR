from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import Counter
from torch.utils.data import WeightedRandomSampler
import torch

from config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
)

# -----------------------------
# Data Augmentation
# -----------------------------

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def get_dataloaders():
    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=train_transform,
    )

    val_dataset = datasets.ImageFolder(
        root=VAL_DIR,
        transform=val_transform,
    )

    test_dataset = datasets.ImageFolder(
        root=TEST_DIR,
        transform=val_transform,
    )
    #---------------------------
    # Compute Sample Weights
    #---------------------------
    targets = train_dataset.targets

    class_counts = Counter(targets)

    class_weights = {
        cls: 1.0 / count
        for cls, count in class_counts.items()
    }

    sample_weights = [
        class_weights[label]
        for label in targets
    ]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
      train_dataset,
      batch_size=BATCH_SIZE,
      sampler=sampler,
      num_workers=NUM_WORKERS,
      pin_memory=True
  )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader