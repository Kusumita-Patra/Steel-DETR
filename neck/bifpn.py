import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# Squeeze-and-Excitation Block
# ==========================================================

class SEBlock(nn.Module):
    """
    Channel Attention using Squeeze-and-Excitation.
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

        b, c, _, _ = x.size()

        y = self.pool(x).view(b, c)

        y = self.fc(y).view(b, c, 1, 1)

        return x * y


# ==========================================================
# Depthwise Separable Convolution
# ==========================================================

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
            stride=1,
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


# ==========================================================
# Learnable Weighted Feature Fusion
# ==========================================================

class WeightedFusion(nn.Module):
    """
    Fast normalized learnable fusion.
    """

    def __init__(self, num_inputs):
        super().__init__()

        self.weights = nn.Parameter(
            torch.ones(num_inputs),
            requires_grad=True
        )

    def forward(self, features):

        weights = torch.softmax(self.weights, dim=0)

        fused = 0

        for w, f in zip(weights, features):
            fused += w * f

        return fused

# ==========================================================
# Backbone Feature Projection
# ==========================================================

class ProjectionLayer(nn.Module):
    """
    Projects ResNet backbone feature maps to a common channel dimension.

    Input:
        c2 : [B, 256, H/4,  W/4]
        c3 : [B, 512, H/8,  W/8]
        c4 : [B,1024, H/16, W/16]
        c5 : [B,2048, H/32, W/32]

    Output:
        p2, p3, p4, p5 : all with `channels` feature maps
    """

    def __init__(self, channels=256):
        super().__init__()

        self.p2_proj = nn.Sequential(
            nn.Conv2d(256, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )

        self.p3_proj = nn.Sequential(
            nn.Conv2d(512, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )

        self.p4_proj = nn.Sequential(
            nn.Conv2d(1024, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )

        self.p5_proj = nn.Sequential(
            nn.Conv2d(2048, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, features):

        return {
            "p2": self.p2_proj(features["c2"]),
            "p3": self.p3_proj(features["c3"]),
            "p4": self.p4_proj(features["c4"]),
            "p5": self.p5_proj(features["c5"]),
        }

# ==========================================================
# BiFPN Layer
# ==========================================================

class BiFPNLayer(nn.Module):
    """
    One complete BiFPN Layer.

    Input:
        p2, p3, p4, p5

    Performs:
        1. Top-down pathway
        2. Bottom-up pathway
        3. Weighted feature fusion
        4. Separable convolutions
    """

    def __init__(self, channels=256):
        super().__init__()

        # ---------- Top-Down Fusion ----------
        self.fuse_p4_td = WeightedFusion(2)
        self.fuse_p3_td = WeightedFusion(2)
        self.fuse_p2_td = WeightedFusion(2)

        # ---------- Bottom-Up Fusion ----------
        self.fuse_p3_out = WeightedFusion(3)
        self.fuse_p4_out = WeightedFusion(3)
        self.fuse_p5_out = WeightedFusion(2)

        # ---------- Convolutions ----------
        self.conv_p2 = SeparableConvBlock(channels)
        self.conv_p3 = SeparableConvBlock(channels)
        self.conv_p4 = SeparableConvBlock(channels)
        self.conv_p5 = SeparableConvBlock(channels)

    def forward(self, features):

        p2 = features["p2"]
        p3 = features["p3"]
        p4 = features["p4"]
        p5 = features["p5"]

        # =====================================================
        # Top-Down Pathway
        # =====================================================

        p5_td = p5

        p4_td = self.fuse_p4_td([
            p4,
            F.interpolate(
                p5_td,
                size=p4.shape[-2:],
                mode="nearest"
            )
        ])
        p4_td = self.conv_p4(p4_td)

        p3_td = self.fuse_p3_td([
            p3,
            F.interpolate(
                p4_td,
                size=p3.shape[-2:],
                mode="nearest"
            )
        ])
        p3_td = self.conv_p3(p3_td)

        p2_td = self.fuse_p2_td([
            p2,
            F.interpolate(
                p3_td,
                size=p2.shape[-2:],
                mode="nearest"
            )
        ])
        p2_td = self.conv_p2(p2_td)

        # =====================================================
        # Bottom-Up Pathway
        # =====================================================

        p3_out = self.fuse_p3_out([
            p3,
            p3_td,
            F.max_pool2d(p2_td, kernel_size=2)
        ])
        p3_out = self.conv_p3(p3_out)

        p4_out = self.fuse_p4_out([
            p4,
            p4_td,
            F.max_pool2d(p3_out, kernel_size=2)
        ])
        p4_out = self.conv_p4(p4_out)

        p5_out = self.fuse_p5_out([
            p5,
            F.max_pool2d(p4_out, kernel_size=2)
        ])
        p5_out = self.conv_p5(p5_out)

        return {
            "p2": p2_td,
            "p3": p3_out,
            "p4": p4_out,
            "p5": p5_out,
        }
# ==========================================================
# BiFPN Wrapper (Stacked Layers)
# ==========================================================


class BiFPN(nn.Module):

    def __init__(self, channels=256, num_layers=3):
        super().__init__()

        self.projection = ProjectionLayer(channels)

        self.layers = nn.ModuleList([
            BiFPNLayer(channels)
            for _ in range(num_layers)
        ])

    def forward(self, features):

        x = self.projection(features)

        for layer in self.layers:
            x = layer(x)

        return x


