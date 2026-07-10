import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from config import DEVICE

from models.classifier import SteelClassifier
from data.gc10_dataset import get_dataloaders

_, _, test_loader = get_dataloaders()

model = SteelClassifier(
    num_classes=10,
    pretrained=False
).to(DEVICE)

checkpoint = torch.load(
    "outputs/checkpoints/best_model.pth",
    map_location=DEVICE
)

model.load_state_dict(checkpoint["model"])

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        # Original prediction
        logits1 = model(images)

        # Horizontal Flip
        logits2 = model(TF.hflip(images))

        # Brightness slightly higher
        bright = torch.clamp(images * 1.1, 0, 1)
        logits3 = model(bright)

        # Brightness slightly lower
        dark = torch.clamp(images * 0.9, 0, 1)
        logits4 = model(dark)

        #Vertical Flip
        logits5 = model(TF.vflip(images))

        outputs = (
            logits1 +
            logits2 +
            logits3 +
            logits4 +
            logits5
        ) / 5

        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())

        all_labels.extend(labels.numpy())

#---------------------
# Classification Report
#---------------------
print("\nClassification Report\n")

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=test_loader.dataset.classes,
        digits=4
    )
)

#---------------------
# Confusion Matrix
#---------------------
cm = confusion_matrix(
    all_labels,
    all_preds
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=test_loader.dataset.classes
)

plt.figure(figsize=(10,10))

disp.plot(
    xticks_rotation=45,
    cmap="Blues",
    values_format="d"
)

plt.tight_layout()

plt.savefig(
    "outputs/predictions/confusion_matrix.png",
    dpi=300
)

plt.show()

#-------------------------
# Overall Metrics
#-------------------------
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print("\nOverall Performance")

print(f"Accuracy : {accuracy_score(all_labels, all_preds):.4f}")

print(f"Precision: {precision_score(all_labels, all_preds, average='macro'):.4f}")

print(f"Recall   : {recall_score(all_labels, all_preds, average='macro'):.4f}")

print(f"F1 Score : {f1_score(all_labels, all_preds, average='macro'):.4f}")


