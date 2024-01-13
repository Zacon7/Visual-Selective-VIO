import torch
import torch.nn as nn
import torch.nn.functional as F

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)  # , inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)  # , inplace=True)
        )

class CSFlow(nn.Module):

    def __init__(self):
        super(CSFlow, self).__init__()

        self.fnet = BasicEncoder(
            output_dim=256, norm_fn='instance')

        # self.cnet = BasicEncoder(
        #     output_dim=hdim + cdim,
        #     norm_fn='batch',
        #     dropout=self.dropout)

        self.strip_corr_block_v2 = StripCrossCorrMap_v2(
            in_chan=256, out_chan=256)
        
        # self.update_block = BasicUpdateBlock(
        #     self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img, dataset='KITTI', train_flag=True):
        """Flow is represented as difference between two coordinate grids.

        flow = coords1 - coords0, Modified by Hao
        """
        N, C, H, W = img.shape

        if dataset == 'KITTI' and not train_flag:
            coords0 = coords_grid(N, H // 8 + 1, W // 8 + 1, device=img.device)
            coords1 = coords_grid(N, H // 8 + 1, W // 8 + 1, device=img.device)
        else:
            coords0 = coords_grid(N, H // 8, W // 8, device=img.device)
            coords1 = coords_grid(N, H // 8, W // 8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def forward(self, images):
        """Estimate optical flow between pair of frames."""

        # Modified, take image pairs as input
        image1 = images[0]
        image2 = images[1]
        image1 = 2 * (image1 + 0.5) - 1.0
        image2 = 2 * (image2 + 0.5) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        # run the feature network
        fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()


        strip_coor_map, strip_corr_map_w, strip_corr_map_h = self.strip_corr_block_v2(
            [fmap1, fmap2]
        )
        
        corr_fn = CorrBlock_v2(fmap1, fmap2, strip_coor_map)

        coords0, coords1 = self.initialize_flow(image1)

        # init flow with regression before GRU iters
        corr_w_act = torch.nn.functional.softmax(
            strip_corr_map_w, dim=3)  # B H1 W1 1 W2
        corr_h_act = torch.nn.functional.softmax(
            strip_corr_map_h, dim=4)  # B H1 W1 H2 1

        flo_v = corr_w_act.mul(strip_corr_map_w)  # B H1 W1 1 W2
        flo_u = corr_h_act.mul(strip_corr_map_h)  # B H1 W1 H2 1

        flow_v = torch.sum(flo_v, dim=4).squeeze(dim=3)  # B H1 W1
        flow_u = torch.sum(flo_u, dim=3).squeeze(dim=3)  # B H1 W1

        corr_init = torch.stack((flow_u, flow_v), dim=1)  # B 2 H1 W1

        coords1 = coords1.detach()
        coords1 = coords1 + corr_init

        # add loss
        flow_up = upflow8(coords1 - coords0)
        
        # 表示每个样本中都有两个通道，分别对应 x 和 y 方向的光流场分量
        return flow_up    # B 2 H W



class StripCrossCorrMap_v2(nn.Module):
    """Strip Cross Corr Augmentation Module by Hao, version2.0"""

    def __init__(self, in_chan=256, out_chan=256, *args, **kwargs):
        super(StripCrossCorrMap_v2, self).__init__()
        self.conv1_1 = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1_2 = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv2_1 = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv2_2 = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)

        self.init_weight()

    def forward(self, x):
        fmap1, fmap2 = x

        # vertical query map
        fmap1_w = self.conv1_1(fmap1)  # B, 256, H, W
        batchsize, c_middle, h, w = fmap1_w.size()
        fmap1_w = fmap1_w.view(batchsize, c_middle, -1)     # B, C, H*W

        # horizontal query map
        fmap1_h = self.conv1_2(fmap1)  # B, 256, H, W
        batchsize, c_middle, h, w = fmap1_h.size()
        fmap1_h = fmap1_h.view(batchsize, c_middle, -1)     # B, C, H*W

        # vertical striping map
        fmap2_w = self.conv2_1(fmap2)  # B, 256, H, W
        fmap2_w = F.avg_pool2d(fmap2_w, [h, 1])
        fmap2_w = fmap2_w.view(batchsize, c_middle, -1).permute(0, 2, 1)    # B, W, C

        # horizontal striping map
        fmap2_h = self.conv2_2(fmap2)  # B, 256, H, W
        fmap2_h = F.avg_pool2d(fmap2_h, [1, w])
        fmap2_h = fmap2_h.view(batchsize, c_middle, -1).permute(0, 2, 1)    # B, H, C

        # cross strip corr map
        strip_corr_map_w = torch.bmm(fmap2_w, fmap1_w).\
            view(batchsize, w, h, w, 1).permute(0, 2, 3, 4, 1)      # B H1 W1 1 W2
        strip_corr_map_h = torch.bmm(fmap2_h, fmap1_h).\
            view(batchsize, h, h, w, 1).permute(0, 2, 3, 1, 4)      # B H1 W1 H2 1

        return (strip_corr_map_w + strip_corr_map_h).view(
            batchsize, h, w, 1, h, w), strip_corr_map_w, strip_corr_map_h

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, torch.nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)


def interp(x, dest):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[2:], **interp_args)

class ConvBNReLU(nn.Module):
    """Conv with BN and ReLU, used for Strip Corr Module"""

    def __init__(self,
                 in_chan,
                 out_chan,
                 ks=3,
                 stride=1,
                 padding=1,
                 *args,
                 **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=ks,
            stride=stride,
            padding=padding,
            bias=False)
        self.bn = torch.nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class BasicEncoder(nn.Module):

    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.25):
        from torch.nn.modules.utils import _pair
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m,
                            (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(
            self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)

        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


class CorrBlock_v2:
    """Corr Block, modified by Hao, concat SC with 4D corr"""

    def __init__(self,
                 fmap1,
                 fmap2,
                 strip_coor_map=None,
                 num_levels=4,
                 radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock_v2.corr(fmap1, fmap2)

        if strip_coor_map is not None:
            # strip correlation augmentation with concat
            corr = torch.cat((corr, strip_coor_map), dim=3)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """Wrapper for grid_sample, uses pixel coordinates."""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(
        torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(
        flow, size=new_size, mode=mode, align_corners=True)


class ResidualBlock(nn.Module):

    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(
                num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(
                num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(
                    num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)

