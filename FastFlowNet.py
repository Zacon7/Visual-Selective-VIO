import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from spatial_correlation_sampler import SpatialCorrelationSampler


class Correlation(nn.Module):
    def __init__(self, max_displacement):
        super(Correlation, self).__init__()
        self.max_displacement = max_displacement
        self.kernel_size = 2 * max_displacement + 1
        self.corr = SpatialCorrelationSampler(1, self.kernel_size, 1, 0, 1)

    def forward(self, x, y):
        b, c, h, w = x.shape
        return self.corr(x, y).view(b, -1, h, w) / c    # Normalization, Makes the network more stable during training


def convrelu(in_channels, out_channels, kernel_size=3, stride=1,
             padding=1, dilation=1, groups=1, bias=True, batch_norm=False):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias),
            nn.LeakyReLU(0.1, inplace=True)
        )

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


class Decoder(nn.Module):
    def __init__(self, in_channels, groups):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.conv1 = convrelu(in_channels, 96, kernel_size=3, stride=1)
        self.conv2 = convrelu(96, 96, kernel_size=3, stride=1, groups=groups)
        self.conv3 = convrelu(96, 96, kernel_size=3, stride=1, groups=groups)
        self.conv4 = convrelu(96, 96, kernel_size=3, stride=1, groups=groups)
        self.conv5 = convrelu(96, 64, kernel_size=3, stride=1)
        self.conv6 = convrelu(64, 32, kernel_size=3, stride=1)
        self.conv7 = nn.Conv2d(32, 2, 3, 1, 1)

    def channel_shuffle(self, x, groups):
        b, c, h, w = x.size()
        channels_per_group = c // groups
        x = x.view(b, groups, channels_per_group, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(b, -1, h, w)
        return x

    def forward(self, x):
        if self.groups == 1:
            out = self.conv7(self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))))
        else:
            out = self.conv1(x)
            out = self.channel_shuffle(self.conv2(out), self.groups)
            out = self.channel_shuffle(self.conv3(out), self.groups)
            out = self.channel_shuffle(self.conv4(out), self.groups)    # (batch, 96, 4, 8)
            out = self.conv7(self.conv6(self.conv5(out)))
        return out


class FastFlowNet(nn.Module):
    def __init__(self, groups=3):
        super(FastFlowNet, self).__init__()

        self.groups = groups
        self.pconv1_1 = convrelu(3, 16, kernel_size=3, stride=2)
        self.pconv1_2 = convrelu(16, 16, kernel_size=3, stride=1)
        self.pconv2_1 = convrelu(16, 32, kernel_size=3, stride=2)
        self.pconv2_2 = convrelu(32, 32, kernel_size=3, stride=1)
        self.pconv2_3 = convrelu(32, 32, kernel_size=3, stride=1)
        self.pconv3_1 = convrelu(32, 64, kernel_size=3, stride=2)
        self.pconv3_2 = convrelu(64, 64, kernel_size=3, stride=1)
        self.pconv3_3 = convrelu(64, 64, kernel_size=3, stride=1)

        self.corr = Correlation(4)
        self.index = torch.tensor([0, 2, 4, 6, 8,
                                   10, 12, 14, 16,
                                   18, 20, 21, 22, 23, 24, 26,
                                   28, 29, 30, 31, 32, 33, 34,
                                   36, 38, 39, 40, 41, 42, 44,
                                   46, 47, 48, 49, 50, 51, 52,
                                   54, 56, 57, 58, 59, 60, 62,
                                   64, 66, 68, 70,
                                   72, 74, 76, 78, 80])

        self.rconv5 = convrelu(64, 32, 3, 1)
        self.rconv6 = convrelu(64, 32, kernel_size=3, stride=1)
        self.decoder5 = Decoder(87, groups)
        self.decoder6 = Decoder(87, groups)
        self.up6 = deconv(2, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def warp(self, x, flo):
        B, C, H, W = x.size()
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat([xx, yy], 1).to(x)
        vgrid = grid + flo
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W-1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H-1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)        
        output = F.grid_sample(x, vgrid, mode='bilinear', align_corners=True)
        return output

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

        flow7_up = torch.zeros(f16.size(0), 2, f16.size(2), f16.size(3)).to(f15)
        cv6 = torch.index_select(self.corr(f16, f26), dim=1, index=self.index.to(f16).long())
        r16 = self.rconv6(f16)
        cat6 = torch.cat([cv6, r16, flow7_up], 1)    # (batch, 87, 4, 8)
        flow6 = self.decoder6(cat6)                  # (batch, 96, 4, 8)

        flow6_up = self.up6(flow6)
        f25_w = self.warp(f25, flow6_up*0.625)
        cv5 = torch.index_select(self.corr(f15, f25_w), dim=1, index=self.index.to(f15).long())
        r15 = self.rconv5(f15)
        cat5 = torch.cat([cv5, r15, flow6_up], 1)
        flow5 = self.decoder5(cat5) + flow6_up

        return flow5
