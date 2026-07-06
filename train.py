import torch

from config import (
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY
)

from backbone.resnet50 import ResNet50Backbone
from neck.bifpn import BiFPN
from models.classifier import SteelClassifier

from utils.loss import build_loss
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
    # Model
    # ----------------------
    model = SteelClassifier(
        num_classes=10,
        pretrained=True
    ).to(DEVICE)

    # ----------------------
    # Loss
    # ----------------------
    criterion = build_loss()

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

        val_loss, val_acc = validator.validate()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(
          f"Val Loss: {val_loss:.4f} | "
          f"Acc: {val_acc['accuracy']:.4f} | "
          f"F1: {val_acc['f1']:.4f} | "
          f"Prec: {val_acc['precision']:.4f} | "
          f"Rec: {val_acc['recall']:.4f}"
        )
        scheduler.step()

        # Save best model
        if val_acc > best_acc:

            best_acc = val_acc

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

    print("Training complete!")


# ==========================================================
# Entry point
# ==========================================================

if __name__ == "__main__":
    main()