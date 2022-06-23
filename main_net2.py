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
        inter_ch = in_ch * 3
        self.conv1 = ConvBnAct(in_ch, inter_ch, 1, 1, 0, act)
        self.conv2 = ConvBnAct(inter_ch, inter_ch, 5, stride, 2, act, group=inter_ch)
        self.conv3 = ConvBnAct(inter_ch, out_ch, 1, 1, 0)
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)

        y2 = x
        if y2.shape[2] != y.shape[2] or y2.shape[3] != y.shape[3]:
            y2 = F.interpolate(x, y.shape[2:4], mode='area')
        if y2.shape[1] < y.shape[1]:
            pad_dim = y.shape[1] - y2.shape[1]
            y2 = F.pad(y2, [0, 0, 0, 0, 0, pad_dim], mode='constant')
        y = y2 + y
        return y


class ResBlockB(nn.Module):
    def __init__(self, in_ch, out_ch, stride, act):
        super().__init__()
        inter_ch = in_ch * 3
        self.conv1 = ConvBnAct(in_ch, inter_ch, 1, 1, 0, act)
        self.conv2 = ConvBnAct(inter_ch, inter_ch, 5, 1, 2, act, group=inter_ch)
        self.up = nn.Upsample(scale_factor=2, mode='area') if stride > 1 else nn.Identity()
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

    # @torch.jit.script_method
    def forward(self, inputs):
        outputs = self.layers(inputs)
        return outputs


class MainNet(nn.Module):
    model_id = 2

    def __init__(self):
        super().__init__()
        act = nn.SELU()
        # 256x
        self.conv1 = ConvBnAct(3, 24, 5, 2, 2)
        self.gb2 = group_block(24, 48, 1, act, ResBlockA, 2)
        # 128x
        self.gb3 = group_block(48, 96, 2, act, ResBlockA, 2)
        # 64x
        self.gb4 = group_block(96, 192, 2, act, ResBlockA, 2)
        # 32x
        self.gb5 = group_block(192, 256, 2, act, ResBlockA, 2)
        # 16x
        self.gb6 = group_block(256, 320, 2, act, ResBlockA, 2)
        # 8x
        self.gb7 = group_block(320, 384, 2, act, ResBlockA, 2)
        # 4x

        # blance
        # self.gb3_b = ConvBnAct(96, 128, 1, 1, 0, act)
        self.gb4_b = ConvBnAct(192, 128, 1, 1, 0, act)
        self.gb5_b = ConvBnAct(256, 128, 1, 1, 0, act)
        self.gb6_b = ConvBnAct(320, 128, 1, 1, 0, act)
        self.gb7_b = ConvBnAct(384, 128, 1, 1, 0, act)

        # aspp
        self.aspp1 = ConvBnAct(512, 128, 3, 1, 1, dilation=1)
        self.aspp2 = ConvBnAct(512, 128, 3, 1, 3, dilation=3)
        self.aspp3 = ConvBnAct(512, 128, 3, 1, 5, dilation=5)
        self.aspp4 = ConvBnAct(512, 128, 3, 1, 7, dilation=7)

        self.aspp_merge = ConvBnAct(512, 512, 1, 1, 0, dilation=1)

        # 32x
        self.gb12 = group_block(512, 256, 2, act, ResBlockB, 2)
        # 64x
        self.gb13 = group_block(256, 128, 2, act, ResBlockB, 2)
        # 128x
        self.gb14 = group_block(128, 64, 2, act, ResBlockB, 2)

        self.out_conv1 = ConvBnAct(64, 32, 1, 1, 0, act)
        self.out_conv2 = nn.Conv2d(32, 1, 1, 1, 0)

    def forward(self, x):
        # 512
        y = self.conv1(x)
        y = self.gb2(y)
        y = self.gb3(y)
        y = y4b = self.gb4(y)
        y = y5b = self.gb5(y)
        y = y6b = self.gb6(y)
        y = y7b = self.gb7(y)

        # y3b = self.gb3_b(y3b)
        y4b = self.gb4_b(y4b)
        y5b = self.gb5_b(y5b)
        y6b = self.gb6_b(y6b)
        y7b = self.gb7_b(y7b)

        y5b = F.interpolate(y5b, y4b.shape[2:], mode='area')
        y6b = F.interpolate(y6b, y4b.shape[2:], mode='area')
        y7b = F.interpolate(y7b, y4b.shape[2:], mode='area')

        y = torch.cat([y4b, y5b, y6b, y7b], 1)

        asp1 = self.aspp1(y)
        asp2 = self.aspp2(y)
        asp3 = self.aspp3(y)
        asp4 = self.aspp4(y)

        y = torch.cat([asp1, asp2, asp3, asp4], 1)
        y = self.aspp_merge(y)

        y = self.gb12(y)
        y = self.gb13(y)
        y = self.gb14(y)

        y = self.out_conv1(y)
        y = self.out_conv2(y)
        return y


if __name__ == '__main__':
    from model_utils_torch import *
    a = torch.zeros(6, 3, 256, 256).cuda(0)
    net = MainNet(3).cuda(0)
    print_params_size2(net)
    y = net(a)
    print(y.shape)
