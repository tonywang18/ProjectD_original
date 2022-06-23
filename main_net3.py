import torch
import torch.nn as nn
import torch.nn.functional as F


def ConvBnAct(in_ch, out_ch, ker_sz, stride, pad, act=nn.Identity(), group=1, dilation=1):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, ker_sz, stride, pad, groups=group, bias=False, dilation=dilation),
                         nn.BatchNorm2d(out_ch, eps=1e-8, momentum=0.9),
                         act)


def DenseBnAct(in_ch, out_ch, act=nn.Identity()):
    return nn.Sequential(nn.Linear(in_ch, out_ch, bias=False),
                         nn.BatchNorm1d(out_ch, eps=1e-8, momentum=0.9),
                         act)


class ResBlockA(nn.Module):
    def __init__(self, in_ch, out_ch, stride, act):
        super().__init__()
        self.use_skip = in_ch == out_ch and stride == 1
        inter_ch = in_ch * 2
        self.conv1 = ConvBnAct(in_ch, inter_ch, 1, 1, 0, act)
        self.conv2 = ConvBnAct(inter_ch, inter_ch, 5, stride, 2, act, group=inter_ch)
        self.conv3 = ConvBnAct(inter_ch, out_ch, 1, 1, 0)
        # self.conv1 = ConvBnAct(in_ch, in_ch*2, 3, stride, 1, act, group=in_ch)
        # self.conv2 = ConvBnAct(in_ch*2, out_ch, 3, 1, 1, act)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)

        y2 = x
        if y2.shape[2] != y.shape[2] or y2.shape[3] != y.shape[3]:
            y2 = F.interpolate(x, y.shape[2:4], mode='bilinear')
        if y2.shape[1] < y.shape[1]:
            pad_dim = y.shape[1] - y2.shape[1]
            y2 = F.pad(y2, [0, 0, 0, 0, 0, pad_dim], mode='constant')
        y = y2 + y
        return y


class ResBlockB(nn.Module):
    def __init__(self, in_ch, out_ch, stride, act):
        super().__init__()
        inter_ch = in_ch * 2
        self.conv1 = ConvBnAct(in_ch, inter_ch, 1, 1, 0, act)
        self.conv2 = ConvBnAct(inter_ch, inter_ch, 5, 1, 2, act, group=inter_ch)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear') if stride > 1 else nn.Identity()
        self.conv3 = ConvBnAct(inter_ch, out_ch, 1, 1, 0)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.up(y)
        y = self.conv3(y)

        y2 = x
        y2 = self.up(y2)
        if y2.shape[1] > y.shape[1]:
            y2 = y2[:, :y.shape[1]]
        y = y2 + y
        return y


class FAM(nn.Module):
    def __init__(self, in_ch, out_ch, act):
        super().__init__()
        self.conv_s2 = ConvBnAct(in_ch, in_ch, 3, 1, 1, act)
        self.conv_s4 = ConvBnAct(in_ch, in_ch, 3, 1, 1, act)
        self.conv_s8 = ConvBnAct(in_ch, in_ch, 3, 1, 1, act)
        self.conv = ConvBnAct(in_ch, out_ch, 3, 1, 1, act)

    def forward(self, x):
        y_s1 = x
        y_s2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        y_s4 = F.interpolate(x, scale_factor=0.25, mode='bilinear')
        y_s8 = F.interpolate(x, scale_factor=0.125, mode='bilinear')

        y_s2 = self.conv_s2(y_s2)
        y_s4 = self.conv_s4(y_s4)
        y_s8 = self.conv_s8(y_s8)

        y_s2 = F.interpolate(y_s2, y_s1.shape[2:], mode='bilinear')
        y_s4 = F.interpolate(y_s4, y_s1.shape[2:], mode='bilinear')
        y_s8 = F.interpolate(y_s8, y_s1.shape[2:], mode='bilinear')

        y = y_s1 + y_s2 + y_s4 + y_s8
        y = self.conv(y)
        return y


class group_block(nn.Module):
    def __init__(self, in_ch, out_ch, stride, act, block_type, blocks, **kwargs):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        layers = []
        for i in range(blocks):
            if i == 0:
                block = block_type(in_ch=in_ch, out_ch=out_ch, stride=stride, act=act, **kwargs)
            else:
                block = block_type(in_ch=out_ch, out_ch=out_ch, stride=1, act=act, **kwargs)
            layers.append(block)
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        outputs = self.layers(inputs)
        return outputs


class MainNet(nn.Module):
    model_id = 3

    def __init__(self):
        super().__init__()
        act = nn.LeakyReLU(0.02)
        # 128x
        self.conv1 = ConvBnAct(3, 24, 3, 1, 1)
        self.conv2 = ConvBnAct(24, 24, 3, 1, 1, act)
        # self.gb2 = group_block(16, 24, 1, act, ResBlockA, 3)
        # 128x
        self.gb3 = group_block(24, 48, 2, act, ResBlockA, 3)
        # 64x
        self.gb4 = group_block(48, 96, 2, act, ResBlockA, 4)
        # 32x
        self.gb5 = group_block(96, 192, 2, act, ResBlockA, 4)
        # 16x
        self.gb6 = group_block(192, 256, 2, act, ResBlockA, 4)
        # 8x
        self.gb7 = group_block(256, 320, 2, act, ResBlockA, 3)

        self.fam1 = FAM(256, 256, act)
        self.fam2 = FAM(192, 192, act)
        self.fam3 = FAM(96, 96, act)
        self.fam4 = FAM(48, 48, act)

        # blance
        # self.gb3_b = ConvBnAct(96, 128, 1, 1, 0, act)
        self.d_conv1 = ConvBnAct(320, 256, 1, 1, 0, act)
        self.d_conv2 = ConvBnAct(256, 192, 1, 1, 0, act)
        self.d_conv3 = ConvBnAct(192, 96, 1, 1, 0, act)
        self.d_conv4 = ConvBnAct(96, 48, 1, 1, 0, act)

        self.d2_conv1 = ConvBnAct(320, 256, 1, 1, 0, act)
        self.d2_conv2 = ConvBnAct(320, 192, 1, 1, 0, act)
        self.d2_conv3 = ConvBnAct(320, 96, 1, 1, 0, act)
        self.d2_conv4 = ConvBnAct(320, 48, 1, 1, 0, act)

        self.out_conv1 = ConvBnAct(48, 24, 1, 1, 0, act)
        self.out_conv2 = nn.Conv2d(24, 1, 1, 1, 0)

    def forward(self, x):
        # 128
        y = self.conv1(x)
        y = y2b = self.conv2(y)
        y = y3b = self.gb3(y)
        y = y4b = self.gb4(y)
        y = y5b = self.gb5(y)
        y = y6b = self.gb6(y)
        y = y7b = self.gb7(y)

        y3p = F.interpolate(y7b, y3b.shape[2:], mode='bilinear')
        y4p = F.interpolate(y7b, y4b.shape[2:], mode='bilinear')
        y5p = F.interpolate(y7b, y5b.shape[2:], mode='bilinear')
        y6p = F.interpolate(y7b, y6b.shape[2:], mode='bilinear')

        y7 = F.interpolate(y7b, size=y6b.shape[2:], mode='bilinear')
        y7 = self.d_conv1(y7) + y6b + self.d2_conv1(y6p)
        y6 = self.fam1(y7)
        y6 = F.interpolate(y6, size=y5b.shape[2:], mode='bilinear')
        y6 = self.d_conv2(y6) + y5b + self.d2_conv2(y5p)
        y5 = self.fam2(y6)
        y5 = F.interpolate(y5, size=y4b.shape[2:], mode='bilinear')
        y5 = self.d_conv3(y5) + y4b + self.d2_conv3(y4p)
        y4 = self.fam3(y5)
        y4 = F.interpolate(y4, size=y3b.shape[2:], mode='bilinear')
        y3 = self.d_conv4(y4) + y3b + self.d2_conv4(y3p)
        y2 = self.fam4(y3)
        y2 = F.interpolate(y2, size=y2b.shape[2:], mode='bilinear')

        y = self.out_conv1(y2)
        y = self.out_conv2(y)
        return y


if __name__ == '__main__':
    from model_utils_torch import *
    a = torch.zeros(6, 3, 128, 128).cuda(0)
    net = MainNet().cuda(0)
    print_params_size2(net)
    y = net(a)
    print(y.shape)
