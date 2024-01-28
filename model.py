import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, orthogonal_
import numpy as np
import torch.nn.functional as F
from FastFlowNet import FastFlowNet


def conv(in_planes, out_planes, kernel_size=3, stride=1, dropout=0, batchNorm=True):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)  # , inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)  # , inplace=True)
        )


class InertialEncoder(nn.Module):
    '''The inertial encoder for raw imu data'''

    def __init__(self, opt):
        super(InertialEncoder, self).__init__()

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout)
        )
        self.proj = nn.Linear(256 * 11, opt.i_f_len)

    def forward(self, fi):
        '''
        input:
            fi: (batch, seq_len=10, 11, 6)
        return:
            out:(batch, seq_len=10, i_f_len=256)
        '''
        batch_size, seq_len = fi.shape[0], fi.shape[1]
        fi = fi.view(batch_size * seq_len, fi.size(2), fi.size(3))  # fi:  (batch *seq_len=10, 11, 6)
        fi = self.encoder_conv(fi.permute(0, 2, 1))                 # fi:  (batch *seq_len=10, 256, 11)
        out = self.proj(fi.view(fi.shape[0], -1))                   # out: (batch *seq_len=10, 256)
        return out.view(batch_size, seq_len, 256)                   # out: (batch, seq_len=10, 256)


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.opt = opt

        # define the OpticalFlow-Encoder based on opt.flow_encoder (flownet or fastflownet)
        if self.opt.flow_encoder == 'flownet':
            self.conv1 = conv(6, 64, kernel_size=7, stride=2, dropout=0.2, batchNorm=opt.flownetBN)
            self.conv2 = conv(64, 128, kernel_size=5, stride=2, dropout=0.2, batchNorm=opt.flownetBN)
            self.conv3 = conv(128, 256, kernel_size=5, stride=2, dropout=0.2, batchNorm=opt.flownetBN)
            self.conv3_1 = conv(256, 256, kernel_size=3, stride=1, dropout=0.2, batchNorm=opt.flownetBN)
            self.conv4 = conv(256, 512, kernel_size=3, stride=2, dropout=0.2, batchNorm=opt.flownetBN)
            self.conv4_1 = conv(512, 512, kernel_size=3, stride=1, dropout=0.2, batchNorm=opt.flownetBN)
            self.conv5 = conv(512, 512, kernel_size=3, stride=2, dropout=0.2, batchNorm=opt.flownetBN)
            self.conv5_1 = conv(512, 512, kernel_size=3, stride=1, dropout=0.2, batchNorm=opt.flownetBN)
            self.conv6 = conv(512, 1024, kernel_size=3, stride=2, dropout=0.5, batchNorm=opt.flownetBN)

            __tmp = Variable(torch.zeros(1, 6, opt.img_h, opt.img_w))
            __tmp = self.flownet(__tmp)
            self.visual_head = nn.Linear(int(np.prod(__tmp.size())), opt.v_f_len)

        elif self.opt.flow_encoder == 'fastflownet':
            self.fastflownet = FastFlowNet()

            __tmp = Variable(torch.zeros(1, 6, opt.img_h, opt.img_w))
            __tmp = self.fastflownet(__tmp)
            self.visual_head = nn.Linear(int(np.prod(__tmp.size())), opt.v_f_len)

        # define the IMU Encoder
        self.inertial_encoder = InertialEncoder(opt)

    def forward(self, imgs, imus):
        '''
        input:
            imgs: (batch, seq_len=11, 3, H, W)
            imus: (batch, 101, 6)
        return:
            fv: (batch, seq_len=10, v_f_len=512)
            fi: (batch, seq_len=10, i_f_len=256)
        '''

        # feed imgs into Flow Encoder
        fv = torch.cat((imgs[:, :-1], imgs[:, 1:]), dim=2)  # fv: (batch, seq_len=10, 6, H, W)
        batch_size, seq_len = fv.size(0), fv.size(1)
        fv = fv.view(batch_size * seq_len, fv.size(2), fv.size(3), fv.size(4))

        if self.opt.flow_encoder == 'flownet':
            fv = self.flownet(fv)                 # fv: (batch* seq_len=10, 1024, 4, 8)
        elif self.opt.flow_encoder == 'fastflownet':
            fv = self.fastflownet(fv)             # fv: (batch* seq_len=10, 128, 4, 8)

        fv = fv.view(batch_size, seq_len, -1)     # fv: (batch, seq_len=10, -1)
        fv = self.visual_head(fv)                 # fv: (batch, seq_len=10, v_f_len=512)

        # feed imus into IMU Encoder
        fi = torch.cat([imus[:, i * 10:i * 10 + 11, :].unsqueeze(1)
                       for i in range(10)], dim=1)  # fi: (batch, seq_len=10, 11, 6)
        fi = self.inertial_encoder(fi)              # fi: (batch, seq_len=10, i_f_len=256)

        return fv, fi

    def flownet(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6


class FusionModule(nn.Module):
    '''The fusion module'''

    def __init__(self, opt):
        super(FusionModule, self).__init__()
        self.fuse_method = opt.fuse_method
        self.f_len = opt.i_f_len + opt.v_f_len
        if self.fuse_method == 'soft':
            self.net = nn.Sequential(
                nn.Linear(self.f_len, self.f_len)
            )
        elif self.fuse_method == 'hard':
            self.net = nn.Sequential(
                nn.Linear(self.f_len, 2 * self.f_len)
            )

    def forward(self, v, i):
        if self.fuse_method == 'cat':
            return torch.cat((v, i), -1)
        elif self.fuse_method == 'soft':
            feat_cat = torch.cat((v, i), -1)
            weights = self.net(feat_cat)
            return feat_cat * weights
        elif self.fuse_method == 'hard':
            feat_cat = torch.cat((v, i), -1)
            weights = self.net(feat_cat)
            weights = weights.view(v.shape[0], v.shape[1], self.f_len, 2)
            mask = F.gumbel_softmax(weights, tau=1, hard=True, dim=-1)
            return feat_cat * mask[:, :, :, 0]


class PolicyNet(nn.Module):
    '''The policy network module'''

    def __init__(self, opt):
        super(PolicyNet, self).__init__()
        in_dim = opt.i_f_len + opt.rnn_hidden_size
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 2)
        )

    def forward(self, x, temp):
        # x = concat(fi, h^t-1), shape = (batch, 1, i_f_len + rnn_hidden_size)
        logits = self.net(x)    # logits: (batch, 1, 2)
        hard_mask = F.gumbel_softmax(logits, tau=temp, hard=True, dim=-1)   # hard_mask: (batch, 1, 2)
        return logits, hard_mask


class PoseRNN(nn.Module):
    '''The pose estimation network'''

    def __init__(self, opt):
        super(PoseRNN, self).__init__()

        # The main RNN network
        f_len = opt.v_f_len + opt.i_f_len
        self.rnn = nn.LSTM(
            input_size=f_len,
            hidden_size=opt.rnn_hidden_size,
            num_layers=2,
            dropout=opt.rnn_dropout_between,
            batch_first=True
        )

        self.fuse = FusionModule(opt)

        # The output networks
        self.rnn_drop_out = nn.Dropout(opt.rnn_dropout_out)
        self.regressor = nn.Sequential(
            nn.Linear(opt.rnn_hidden_size, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 6)
        )

    def forward(self, fv, fv_alter, fi, dec, prev=None):
        if prev is not None:
            prev = (prev[0].transpose(1, 0).contiguous(), prev[1].transpose(1, 0).contiguous())

        # Select between fv and fv_alter
        v_in = fv * dec[:, :, :1] + fv_alter * dec[:, :, -1:] if fv_alter is not None else fv
        fused = self.fuse(v_in, fi)

        # hc is a tuple that contains hidden state(hc[0]) and cell state(hc[1])
        # both two state have shape of (num_layers * num_directions, batch_size, hidden_size)
        out, hc = self.rnn(fused) if prev is None else self.rnn(fused, prev)
        out = self.rnn_drop_out(out)
        pose = self.regressor(out)

        # Make sure that hc order conforms to the shape of (batch_size, seq_len, hidden size)
        hc = (hc[0].transpose(1, 0).contiguous(), hc[1].transpose(1, 0).contiguous())
        return pose, hc


class DeepVIO(nn.Module):
    def __init__(self, opt):
        super(DeepVIO, self).__init__()

        self.Feature_net = Encoder(opt)
        self.Policy_net = PolicyNet(opt)
        self.Pose_net = PoseRNN(opt)
        self.opt = opt

        initialization(self)

    def forward(self, imgs, imus, is_first=True, hc=None, temp=5, selection='gumbel-softmax', p=0.2):
        
        fv, fi = self.Feature_net(imgs, imus)   # fv: (batch, seq_len=10, v_f_len=512)
                                                # fi: (batch, seq_len=10, i_f_len=256)
        batch_size, seq_len = fv.shape[0], fv.shape[1]

        poses, decisions, logits = [], [], []
        hidden = torch.zeros(batch_size, self.opt.rnn_hidden_size).to(
            fv.device) if hc is None else hc[0].contiguous()[:, -1, :]
        fv_alter = torch.zeros_like(fv)  # zero padding in the paper, can be replaced by other

        for i in range(seq_len):
            if i == 0 and is_first:
                # The first relative pose is estimated by both images and imu by default
                pose, hc = self.Pose_net(fv[:, i:i + 1, :], None, fi[:, i:i + 1, :], None, hc)
            else:
                if selection == 'gumbel-softmax':
                    # Otherwise, sample the decision from the policy network
                    p_in = torch.cat((fi[:, i, :], hidden), -1)
                    logit, decision = self.Policy_net(p_in.detach(), temp)
                    decision = decision.unsqueeze(1)
                    logit = logit.unsqueeze(1)
                    pose, hc = self.Pose_net(
                        fv[:, i: i + 1, :],
                        fv_alter[:, i: i + 1, :],
                        fi[:, i: i + 1, :],
                        decision, hc)
                    decisions.append(decision)
                    logits.append(logit)

                elif selection == 'random':
                    decision = (torch.rand(fv.shape[0], 1, 2) < p).float()
                    decision[:, :, 1] = 1 - decision[:, :, 0]
                    decision = decision.to(fv.device)
                    logit = 0.5 * torch.ones((fv.shape[0], 1, 2)).to(fv.device)
                    pose, hc = self.Pose_net(
                        fv[:, i: i + 1, :],
                        fv_alter[:, i: i + 1, :],
                        fi[:, i: i + 1, :],
                        decision, hc)
                    decisions.append(decision)
                    logits.append(logit)

            poses.append(pose)
            # The purpose of hc[0][:, -1,:] is to select the hidden state of the last layer for each sample.
            hidden = hc[0].contiguous()[:, -1, :]

        poses = torch.cat(poses, dim=1)
        decisions = torch.cat(decisions, dim=1)
        logits = torch.cat(logits, dim=1)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        return poses, decisions, probs, hc


def initialization(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or \
                isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.kaiming_normal_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(0)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
                    n = param.size(0)
                    start, end = n // 4, n // 2
                    param.data[start:end].fill_(1.)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
