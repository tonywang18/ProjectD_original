import torch
import torch.nn as nn
import torch.nn.functional as F


class Act1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x*0.5+torch.sin(x*5)*0.5


class FRN(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, in_ch, 1, 1), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, in_ch, 1, 1), requires_grad=True)

    def forward(self, x):
        nu2 = torch.pow(x, 2).mean(dim=(2, 3), keepdim=True)
        y = x * torch.rsqrt(nu2 + 1e-8)
        return self.gamma * y + self.beta


class TLU(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.bias = nn.Parameter(torch.full([1, in_ch, 1, 1], fill_value=-1), requires_grad=True)

    def forward(self, x):
        y = torch.max(x, self.bias)
        return y


def ConvFrnTlu(in_ch, out_ch, ker_sz, stride, pad, act=nn.Identity(), group=1, dilation=1):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, ker_sz, stride, pad, groups=group, bias=False, dilation=dilation),
                         FRN(out_ch),
                         nn.LeakyReLU(0.02))


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

        # self.conv1 = ConvBnAct(in_ch, inter_ch, 1, 1, 0, act)
        # self.conv2 = ConvBnAct(inter_ch, inter_ch, 5, stride, 2, act, group=inter_ch)
        # self.conv3 = ConvBnAct(inter_ch, out_ch, 1, 1, 0)
        self.conv1 = ConvBnAct(in_ch, inter_ch, 5, 1, 2, act)
        self.conv2 = ConvBnAct(inter_ch, out_ch, 5, stride, 2, act)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        # y = self.conv3(y)

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
        inter_ch = in_ch * 3
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


class MainNet(nn.Module):
    model_id = 1

    def __init__(self, out_ch=1):
        super().__init__()
        act = nn.LeakyReLU(0.02)
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.rb1 = ResBlockA(16, 32, 1, act)
        self.rb2 = ResBlockA(32, 64, 1, act)
        self.rb3 = ResBlockA(64, 64, 1, act)
        self.fam1 = FAM(64, 64, act)
        self.out_conv1 = nn.Conv2d(64, out_ch, 1, 1, 0)
        self.rb4 = ResBlockA(64, 64, 1, act)
        self.rb5 = ResBlockA(64, 64, 1, act)
        self.fam2 = FAM(64, 64, act)
        self.out_conv2 = nn.Conv2d(64, out_ch, 1, 1, 0)
        self.rb6 = ResBlockA(64, 64, 1, act)
        self.rb7 = ResBlockA(64, 64, 1, act)
        self.fam3 = FAM(64, 64, act)
        self.out_conv3 = nn.Conv2d(64, out_ch, 1, 1, 0)
        # self.layer1 = ConvBnAct(3, 32, 5, 1, 2, act=act)
        # self.layer2 = ConvBnAct(32, 32, 3, 1, 1, act=act)
        # self.layer3 = ConvBnAct(32, 64, 3, 1, 1, act=act)
        # self.layer4 = ConvBnAct(64, 64, 3, 1, 1, act=act)
        # self.layer5 = ConvBnAct(64, 64, 3, 1, 1, act=act)
        # self.layer6 = ConvBnAct(64, 64, 3, 1, 1, act=act)

    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.rb1(y)
        y = self.rb2(y)
        y = self.rb3(y)
        y = self.fam1(y)
        oy1 = self.out_conv1(y)
        y = self.rb4(y)
        y = self.rb5(y)
        y = self.fam2(y)
        oy2 = self.out_conv2(y)
        y = self.rb6(y)
        y = self.rb7(y)
        y = self.fam3(y)
        oy3 = self.out_conv3(y)

        oy1 = (oy1*0.1).tanh() / 2 + 0.5
        oy2 = (oy2*0.1).tanh() / 2 + 0.5
        oy3 = (oy3*0.1).tanh() / 2 + 0.5
        return oy1, oy2, oy3


if __name__ == '__main__':
    from model_utils_torch import *

    a = torch.zeros(10, 3, 128, 128).cuda(0)
    net = MainNet().cuda(0)
    print_params_size2(net)
    y = net(a)
    print(y.shape)
