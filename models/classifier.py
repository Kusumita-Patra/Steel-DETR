import torch
import torch.nn as nn

from backbone.resnet50 import ResNet50Backbone
from neck.bifpn import BiFPN


class SteelClassifier(nn.Module):
    """
    ResNet50 + BiFPN + Classification Head
    """

    def __init__(
        self,
        num_classes=10,
        bifpn_channels=256,
        dropout=0.3,
        pretrained=True,
    ):
        super().__init__()

        self.backbone = ResNet50Backbone(pretrained=pretrained)

        self.neck = BiFPN(out_channels=bifpn_channels)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            bifpn_channels * 4,
            num_classes,
        )

    def forward(self, x):

        # Backbone Features
        features = self.backbone(x)

        # BiFPN Features
        pyramid = self.neck(features)

        pooled = []

        for level in ["p2", "p3", "p4", "p5"]:

            feat = self.pool(pyramid[level])

            feat = feat.flatten(1)

            pooled.append(feat)

        x = torch.cat(pooled, dim=1)

        x = self.dropout(x)

        logits = self.classifier(x)

        return logits