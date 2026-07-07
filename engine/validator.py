import torch
from tqdm import tqdm

from config import DEVICE
from utils.metrics import compute_metrics


class Validator:

    def __init__(self, model, val_loader, criterion):

        self.model = model.to(DEVICE)

        self.val_loader = val_loader

        self.criterion = criterion

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        running_loss = 0.0

        all_preds = []

        all_labels = []

        progress = tqdm(
            self.val_loader,
            desc="Validation",
            leave=False
        )

        for images, labels in progress:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)

            avg_loss = running_loss / len(self.val_loader.dataset)

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())

            all_labels.extend(labels.cpu().numpy())

            progress.set_postfix({
                "Loss": f"{running_loss / (progress.n + 1):.4f}"
            })

      # -------------------------
      # Metrics
      # -------------------------
        avg_loss = running_loss / len(self.val_loader)

        metrics = compute_metrics(all_labels, all_preds)

        return avg_loss, metrics["accuracy"], metrics