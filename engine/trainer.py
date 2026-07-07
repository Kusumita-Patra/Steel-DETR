import torch
from tqdm import tqdm

from config import (
    DEVICE,
    USE_AMP,
    GRAD_CLIP
)


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        criterion,
        optimizer,
        scheduler=None,
    ):

        self.model = model.to(DEVICE)

        self.train_loader = train_loader

        self.criterion = criterion

        self.optimizer = optimizer

        self.scheduler = scheduler

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=USE_AMP
        )
    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]
    def train_one_epoch(self):

        self.model.train()

        running_loss = 0.0

        correct = 0

        total = 0

        progress = tqdm(
            self.train_loader,
            desc="Training",
            leave=False
        )

        for images, labels in progress:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            self.optimizer.zero_grad()

            with torch.amp.autocast(
                device_type="cuda",
                enabled=USE_AMP
            ):

                outputs = self.model(images)

                loss = self.criterion(
                    outputs,
                    labels
                )

            self.scaler.scale(loss).backward()

            # Gradient Clipping
            self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                GRAD_CLIP
            )

            self.scaler.step(self.optimizer)

            self.scaler.update()

            running_loss += loss.item() * labels.size(0)
            

            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()

            total += labels.size(0)

            progress.set_postfix({

              "Loss": f"{running_loss/(total/labels.size(0)):.4f}",

              "Acc": f"{100*correct/total:.2f}%",

              "LR": f"{self.get_lr():.6f}"

              })

        if self.scheduler is not None:

            self.scheduler.step()

        epoch_loss = running_loss / total

        epoch_acc = correct / total

        return epoch_loss, epoch_acc