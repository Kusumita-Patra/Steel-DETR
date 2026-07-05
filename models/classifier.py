import torch
import torch.nn as nn

from backbone.resnet50 import ResNet50Backbone
from neck.bifpn import BiFPN


class FeatureAttention(nn.Module):
    """
    Learns an importance weight for each BiFPN feature level.
    """

    def __init__(self, channels=256, num_levels=4):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(channels * num_levels, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_levels),
            nn.Softmax(dim=1)
        )

    def forward(self, features):

        x = torch.cat(features, dim=1)

        weights = self.attention(x)

        weighted = []

        for i in range(len(features)):
            weighted.append(features[i] * weights[:, i].unsqueeze(1))

        return torch.cat(weighted, dim=1)


class SteelClassifier(nn.Module):

    def __init__(
        self,
        num_classes=10,
        bifpn_channels=256,
        num_bifpn_layers=2,
        dropout=0.3,
        pretrained=True
    ):
        super().__init__()

        # Backbone
        self.backbone = ResNet50Backbone(pretrained=pretrained)

        # BiFPN
        self.neck = BiFPN(
            channels=bifpn_channels,
            num_layers=num_bifpn_layers
        )

        # Global Average Pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Feature Attention
        self.feature_attention = FeatureAttention(
            channels=bifpn_channels,
            num_levels=4
        )

        # Classification Head
        self.classifier = nn.Sequential(

            nn.Dropout(dropout),

            nn.Linear(bifpn_channels * 4, 512),

            nn.ReLU(inplace=True),

            nn.Dropout(dropout),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):

        # Backbone
        features = self.backbone(x)

        # BiFPN
        pyramid = self.neck(features)

        pooled = []

        for level in ["p2", "p3", "p4", "p5"]:

            feat = self.pool(pyramid[level])

            feat = feat.flatten(1)

            pooled.append(feat)

        # Attention-based fusion
        fused = self.feature_attention(pooled)

        # Classification
        logits = self.classifier(fused)

        return logits