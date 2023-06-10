import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from libs.fno_models import SpectralConv2d
import matplotlib.pyplot as plt


################################################################
# UNet
################################################################
""" UNET model: https://github.com/milesial/Pytorch-UNet """

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, use_spectral_conv=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, use_spectral_conv=True, modes=12):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            if use_spectral_conv:
                self.conv = SpectralConv2d(in_channels=in_channels, out_channels=out_channels, modes1=modes, modes2=modes)
            else:
                self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            if use_spectral_conv:
                self.conv = SpectralConv2d(in_channels=in_channels, out_channels=out_channels, modes1=modes, modes2=modes)
            else:
                self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_classes=1, bilinear=False, use_v_plane=False, use_spectral_conv=True):
        super(UNet, self).__init__()
        self.input_channel_num = 4 if use_v_plane else 3
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_spectral_conv = use_spectral_conv
        self.inc = nn.Linear(self.input_channel_num, 32)
        self.down1 = Down(32, 64, use_spectral_conv=False)
        self.down2 = Down(64, 128, use_spectral_conv=False)
        self.down3 = Down(128, 256, use_spectral_conv=False)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor, use_spectral_conv=False)
        self.up1 = Up(512, 256 // factor, bilinear, use_spectral_conv=False)
        self.up2 = Up(256, 128 // factor, bilinear, use_spectral_conv=False)
        self.up3 = Up(128, 64 // factor, bilinear, use_spectral_conv=False)
        self.up4 = Up(64, 32, bilinear, use_spectral_conv, modes=12)
        self.outc = nn.Linear(32, n_classes)

    def forward(self, p_plane, v_plane):
        grid = self.get_grid(p_plane.shape, p_plane.device)
        p_plane = torch.cat((p_plane, grid), dim=-1)
        x1 = self.inc(p_plane).permute(0,3,1,2)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        p_plane = self.up1(x5, x4)
        p_plane = self.up2(p_plane, x3)
        p_plane = self.up3(p_plane, x2)
        p_plane = self.up4(p_plane, x1)
        p_plane = p_plane.permute(0,2,3,1)
        p_plane = self.outc(p_plane)
        return p_plane

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
