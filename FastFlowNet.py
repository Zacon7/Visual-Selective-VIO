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
        return self.corr(x, y).view(b, -1, h, w) / c


def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias),
        nn.LeakyReLU(0.1, inplace=True)
    )


class Decoder(nn.Module):
    def __init__(self, in_channels, groups):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.conv1 = convrelu(in_channels, 96, 3, 1)
        self.conv2 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv3 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv4 = convrelu(96, 96, 3, 1, groups=groups)
        # self.conv5 = convrelu(96, 64, 3, 1)
        # self.conv6 = convrelu(64, 32, 3, 1)
        # self.conv7 = nn.Conv2d(32, 2, 3, 1, 1)

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
            # out = self.conv7(self.conv6(self.conv5(out)))
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

        self.rconv6 = convrelu(64, 32, 3, 1)
        self.decoder6 = Decoder(87, groups)

        # self.dw_conv = convrelu(128, 256, kernel_size=1, stride=1, batch_norm=True)

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

        # f_concat = torch.cat([f16, f26], dim=1)     # (batch, 128, 4, 8)
        # f = self.dw_conv(f_concat)                  # (batch, 128, 4, 8)

        # return f

        flow7_up = torch.zeros(f16.size(0), 2, f16.size(2), f16.size(3)).to(f15)
        cv6 = torch.index_select(self.corr(f16, f26), dim=1, index=self.index.to(f16).long())
        r16 = self.rconv6(f16)
        cat6 = torch.cat([cv6, r16, flow7_up], 1)    # (batch, 87, 4, 8)
        flow6 = self.decoder6(cat6)                  # (batch, 96, 4, 8)

        return flow6