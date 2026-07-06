import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        if pretrained:
          weights = ResNet50_Weights.IMAGENET1K_V2
        else:
          weights = None

        resnet = resnet50(weights=weights)

        # Remove fully connected + avgpool
        self.stage0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        self.stage1 = resnet.layer1  # C2
        self.stage2 = resnet.layer2  # C3
        self.stage3 = resnet.layer3  # C4
        self.stage4 = resnet.layer4  # C5

    def forward(self, x):
        x = self.stage0(x)

        c2 = self.stage1(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)

        return {
            "c2": c2,
            "c3": c3,
            "c4": c4,
            "c5": c5
        }