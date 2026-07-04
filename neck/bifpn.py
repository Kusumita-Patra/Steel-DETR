import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """
    Conv -> BatchNorm -> SiLU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class BiFPN(nn.Module):

    def __init__(self, out_channels=256):
        super().__init__()

        # Project backbone features to same dimension
        self.c2_proj = nn.Conv2d(256, out_channels, 1)
        self.c3_proj = nn.Conv2d(512, out_channels, 1)
        self.c4_proj = nn.Conv2d(1024, out_channels, 1)
        self.c5_proj = nn.Conv2d(2048, out_channels, 1)

        # Top-down pathway
        self.p4_td = ConvBNAct(out_channels, out_channels)
        self.p3_td = ConvBNAct(out_channels, out_channels)
        self.p2_td = ConvBNAct(out_channels, out_channels)

        # Bottom-up pathway
        self.p3_out = ConvBNAct(out_channels, out_channels)
        self.p4_out = ConvBNAct(out_channels, out_channels)
        self.p5_out = ConvBNAct(out_channels, out_channels)

    def forward(self, features):

        c2 = self.c2_proj(features["c2"])
        c3 = self.c3_proj(features["c3"])
        c4 = self.c4_proj(features["c4"])
        c5 = self.c5_proj(features["c5"])

        # -------------------
        # Top-Down
        # -------------------

        p5 = c5

        p4 = self.p4_td(
            c4 + F.interpolate(
                p5,
                size=c4.shape[-2:],
                mode="nearest"
            )
        )

        p3 = self.p3_td(
            c3 + F.interpolate(
                p4,
                size=c3.shape[-2:],
                mode="nearest"
            )
        )

        p2 = self.p2_td(
            c2 + F.interpolate(
                p3,
                size=c2.shape[-2:],
                mode="nearest"
            )
        )

        # -------------------
        # Bottom-Up
        # -------------------

        p3 = self.p3_out(
            p3 +
            F.max_pool2d(p2, 2)
        )

        p4 = self.p4_out(
            p4 +
            F.max_pool2d(p3, 2)
        )

        p5 = self.p5_out(
            p5 +
            F.max_pool2d(p4, 2)
        )

        return {
            "p2": p2,
            "p3": p3,
            "p4": p4,
            "p5": p5
        }