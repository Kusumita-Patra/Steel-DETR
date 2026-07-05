import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    """

    def __init__(self, channels, reduction=16):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        b, c, _, _ = x.shape

        y = self.pool(x).view(b, c)

        y = self.fc(y).view(b, c, 1, 1)

        return x * y


class SeparableConvBlock(nn.Module):
    """
    Depthwise Separable Convolution
    + BatchNorm
    + SiLU
    + SE Attention
    + Residual Connection
    """

    def __init__(self, channels):
        super().__init__()

        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False
        )

        self.pointwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False
        )

        self.bn = nn.BatchNorm2d(channels)

        self.act = nn.SiLU(inplace=True)

        self.se = SEBlock(channels)

    def forward(self, x):

        identity = x

        x = self.depthwise(x)

        x = self.pointwise(x)

        x = self.bn(x)

        x = self.act(x)

        x = self.se(x)

        x = x + identity

        return x

class WeightedFusion(nn.Module):
    """
    Learnable weighted feature fusion
    """

    def __init__(self, num_inputs):
        super().__init__()

        self.weights = nn.Parameter(torch.ones(num_inputs))

        self.eps = 1e-4

    def forward(self, features):

        weights = torch.softmax(self.weights, dim=0)

        weights = weights / (weights.sum() + self.eps)

        out = 0

        for w, f in zip(weights, features):
            out = out + w * f

        return sum(w * f for w, f in zip(weights, features))


class BiFPNBlock(nn.Module):

    def __init__(self, channels=256):
        super().__init__()

        # Projection
        self.c2_proj = nn.Conv2d(256, channels, 1)
        self.c3_proj = nn.Conv2d(512, channels, 1)
        self.c4_proj = nn.Conv2d(1024, channels, 1)
        self.c5_proj = nn.Conv2d(2048, channels, 1)

        # Fusion weights
        self.fuse_p4 = WeightedFusion(2)
        self.fuse_p3 = WeightedFusion(2)
        self.fuse_p2 = WeightedFusion(2)

        self.fuse_p3_out = WeightedFusion(2)
        self.fuse_p4_out = WeightedFusion(2)
        self.fuse_p5_out = WeightedFusion(2)

        # Convolutions
        self.conv_p2 = SeparableConvBlock(channels)
        self.conv_p3 = SeparableConvBlock(channels)
        self.conv_p4 = SeparableConvBlock(channels)
        self.conv_p5 = SeparableConvBlock(channels)

    def forward(self, features):

        c2 = self.c2_proj(features["c2"])
        c3 = self.c3_proj(features["c3"])
        c4 = self.c4_proj(features["c4"])
        c5 = self.c5_proj(features["c5"])

        # -------------------------
        # Top-Down Path
        # -------------------------

        p5 = c5

        p4 = self.fuse_p4([
            c4,
            F.interpolate(
                p5,
                size=c4.shape[-2:],
                mode="nearest"
            )
        ])
        p4 = self.conv_p4(p4)

        p3 = self.fuse_p3([
            c3,
            F.interpolate(
                p4,
                size=c3.shape[-2:],
                mode="nearest"
            )
        ])
        p3 = self.conv_p3(p3)

        p2 = self.fuse_p2([
            c2,
            F.interpolate(
                p3,
                size=c2.shape[-2:],
                mode="nearest"
            )
        ])
        p2 = self.conv_p2(p2)

        # -------------------------
        # Bottom-Up Path
        # -------------------------

        p3 = self.fuse_p3_out([
            p3,
            F.max_pool2d(p2, 2)
        ])
        p3 = self.conv_p3(p3)

        p4 = self.fuse_p4_out([
            p4,
            F.max_pool2d(p3, 2)
        ])
        p4 = self.conv_p4(p4)

        p5 = self.fuse_p5_out([
            p5,
            F.max_pool2d(p4, 2)
        ])
        p5 = self.conv_p5(p5)

        return {
            "p2": p2,
            "p3": p3,
            "p4": p4,
            "p5": p5,
        }


class BiFPN(nn.Module):
    """
    Stack multiple BiFPN blocks
    """

    def __init__(
        self,
        channels=256,
        num_layers=3
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(BiFPNBlock(channels))

        for _ in range(num_layers - 1):

            self.layers.append(
                FeatureBiFPN(channels)
            )

    def forward(self, features):

        x = features

        for layer in self.layers:
            x = layer(x)

        return x


class FeatureBiFPN(nn.Module):
    """
    Additional BiFPN blocks operating
    directly on P2-P5.
    """

    def __init__(self, channels):
        super().__init__()

        self.conv2 = SeparableConvBlock(channels)
        self.conv3 = SeparableConvBlock(channels)
        self.conv4 = SeparableConvBlock(channels)
        self.conv5 = SeparableConvBlock(channels)

    def forward(self, features):

        p2 = self.conv2(features["p2"]) + features["p2"]
        p3 = self.conv2(features["p3"]) + features["p3"]
        p4 = self.conv2(features["p4"]) + features["p4"]
        p5 = self.conv2(features["p5"]) + features["p5"]

        return {
            "p2": p2,
            "p3": p3,
            "p4": p4,
            "p5": p5,
        }