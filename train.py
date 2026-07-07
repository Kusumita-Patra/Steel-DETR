import torch
import numpy as np
import torch.nn as nn

from sklearn.utils.class_weight import compute_class_weight

from config import (
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY
)

from backbone.resnet50 import ResNet50Backbone
from neck.bifpn import BiFPN
from models.classifier import SteelClassifier


from utils.seed import set_seed
from utils.checkpoint import save_checkpoint

from engine.trainer import Trainer
from engine.validator import Validator

from data.gc10_dataset import get_dataloaders

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


# ==========================================================
# Main
# ==========================================================

def main():

    set_seed(42)

    # ----------------------
    # Dataloaders
    # ----------------------
    train_loader, val_loader, test_loader = get_dataloaders()

    # ----------------------
    # Get labels from the training dataset
    # ----------------------
    labels = train_loader.dataset.targets

    # ----------------------
    # Compute balanced class weights
    # ----------------------
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )

    # ----------------------
    # Convert to tensor
    # ----------------------
    class_weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE
    )

    print("\nClass Weights:")
    for cls, w in zip(train_loader.dataset.classes, class_weights):
        print(f"{cls:20s}: {w:.4f}")

    # ----------------------
    # Model
    # ----------------------
    model = SteelClassifier(
        num_classes=10,
        pretrained=True
    ).to(DEVICE)

    # ----------------------
    # Loss
    # ----------------------
    criterion = nn.CrossEntropyLoss(
      weight=class_weights
  )

    # ----------------------
    # Optimizer
    # ----------------------
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # ----------------------
    # Scheduler
    # ----------------------
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS
    )

    # ----------------------
    # Trainer & Validator
    # ----------------------
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler
    )

    validator = Validator(
        model=model,
        val_loader=val_loader,
        criterion=criterion
    )

    # ======================================================
    # Training Loop
    # ======================================================

    best_acc = 0.0

    for epoch in range(NUM_EPOCHS):

        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")

        train_loss, train_acc = trainer.train_one_epoch()

        val_loss, val_acc, val_metrics = validator.validate()

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Acc: {val_acc:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | "
            f"Prec: {val_metrics['precision']:.4f} | "
            f"Rec: {val_metrics['recall']:.4f}"
        )
        

        # Save best model
        if val_acc > best_acc:

            best_acc = val_acc

            patience_counter = 0
          
            save_checkpoint(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc
                },
                "outputs/checkpoints/best_model.pth"
            )

            print("✅ Best model saved!")

            
        else:
            patience_counter += 1

        if patience_counter >= 8:
            print("Early stopping.")
            break


    print("Training complete!")


# ==========================================================
# Entry point
# ==========================================================

if __name__ == "__main__":
    main()