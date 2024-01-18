import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from spatial_correlation_sampler import SpatialCorrelationSampler


def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, batch_norm = False, dropout=0):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias), 
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias), 
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )


class FastFlowNet(nn.Module):
    def __init__(self):
        super(FastFlowNet, self).__init__()

        self.pconv1_1 = convrelu(3, 16, kernel_size=3, stride=2, dropout=0.1)
        self.pconv1_2 = convrelu(16, 16, kernel_size=3, stride=1, dropout=0.1)
        self.pconv2_1 = convrelu(16, 32, kernel_size=3, stride=2, dropout=0.1)
        self.pconv2_2 = convrelu(32, 32, kernel_size=3, stride=1, dropout=0.1)
        self.pconv2_3 = convrelu(32, 32, kernel_size=3, stride=1, dropout=0.1)
        self.pconv3_1 = convrelu(32, 64, kernel_size=3, stride=2, dropout=0.2)
        self.pconv3_2 = convrelu(64, 64, kernel_size=3, stride=1, dropout=0.2)
        self.pconv3_3 = convrelu(64, 64, kernel_size=3, stride=1, dropout=0.2)

        self.dw_conv = convrelu(128, 256, kernel_size=1, stride=1, batch_norm=True, dropout=0.2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        img1 = x[:, :3, :, :]
        img2 = x[:, 3:6, :, :]
        f11 = self.pconv1_2(self.pconv1_1(img1))
        f21 = self.pconv1_2(self.pconv1_1(img2))
        f12 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f11)))
        f22 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f21)))
        f13 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f12)))
        f23 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f22)))
        f14 = F.avg_pool2d(f13, kernel_size=(2, 2), stride=(2, 2))
        f24 = F.avg_pool2d(f23, kernel_size=(2, 2), stride=(2, 2))
        f15 = F.avg_pool2d(f14, kernel_size=(2, 2), stride=(2, 2))
        f25 = F.avg_pool2d(f24, kernel_size=(2, 2), stride=(2, 2))
        f16 = F.avg_pool2d(f15, kernel_size=(2, 2), stride=(2, 2))
        f26 = F.avg_pool2d(f25, kernel_size=(2, 2), stride=(2, 2))      # (batch, 64, 4, 8)

        f_concat = torch.cat([f16, f26], dim=1)     # (batch, 128, 4, 8)
        f = self.dw_conv(f_concat)                  # (batch, 256, 4, 8)
        
        return f