"""Zero-DCE++ network loader.

Architecture compatible with Li-Chongyi/Zero-DCE_extension Zero-DCE++
pretrained snapshot `Epoch99.pth`.
Original project: https://github.com/Li-Chongyi/Zero-DCE_extension
License of original project: Attribution-NonCommercial 4.0 International.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwisePointwiseConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.depth_conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch)
        self.point_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.point_conv(self.depth_conv(x))


class ZeroDCEPP(nn.Module):
    def __init__(self, scale_factor: int = 1):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        nf = 32
        self.e_conv1 = DepthwisePointwiseConv(3, nf)
        self.e_conv2 = DepthwisePointwiseConv(nf, nf)
        self.e_conv3 = DepthwisePointwiseConv(nf, nf)
        self.e_conv4 = DepthwisePointwiseConv(nf, nf)
        self.e_conv5 = DepthwisePointwiseConv(nf * 2, nf)
        self.e_conv6 = DepthwisePointwiseConv(nf * 2, nf)
        self.e_conv7 = DepthwisePointwiseConv(nf * 2, 3)

    @staticmethod
    def enhance(x: torch.Tensor, curve: torch.Tensor) -> torch.Tensor:
        for _ in range(4):
            x = x + curve * (torch.pow(x, 2) - x)
        enhanced_once = x
        for _ in range(4):
            x = x + curve * (torch.pow(x, 2) - x)
        return x

    def forward(self, x: torch.Tensor):
        if self.scale_factor == 1:
            x_down = x
        else:
            x_down = F.interpolate(x, scale_factor=1 / self.scale_factor, mode="bilinear", align_corners=False)
        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        curve = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        if self.scale_factor != 1:
            curve = self.upsample(curve)
        return self.enhance(x, curve), curve
