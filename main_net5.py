import torch
import torch.nn as nn
import torch.nn.functional as F
from rev_blocks import RevSequential, RevGroupBlock
from rev_utils import rev_sequential_backward_wrapper


def ConvBnAct(in_ch, out_ch, ker_sz, stride, pad, act=nn.Identity(), group=1, dilation=1):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, ker_sz, stride, pad, groups=group, bias=False, dilation=dilation),
                         nn.BatchNorm2d(out_ch, eps=1e-8, momentum=0.9),
                         act)


def BnActConv(in_ch, out_ch, ker_sz, stride, pad, act=nn.Identity(), group=1, dilation=1):
    return nn.Sequential(nn.BatchNorm2d(in_ch, eps=1e-8, momentum=0.9),
                         act,
                         nn.Conv2d(in_ch, out_ch, ker_sz, stride, pad, groups=group, bias=False, dilation=dilation))


def DeConvBnAct(in_ch, out_ch, ker_sz, stride, pad, act=nn.Identity(), group=1, dilation=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, ker_sz, stride, pad, groups=group, bias=False, dilation=dilation),
        nn.BatchNorm2d(out_ch, eps=1e-8, momentum=0.9),
        act)


def DenseBnAct(in_ch, out_ch, act=nn.Identity()):
    return nn.Sequential(nn.Linear(in_ch, out_ch, bias=False),
                         nn.BatchNorm1d(out_ch, eps=1e-8, momentum=0.9),
                         act)


class SEBlock(nn.Module):
    def __init__(self, in_ch, act):
        super().__init__()
        inter_ch = in_ch
        self.conv1 = ConvBnAct(in_ch, inter_ch, 1, 1, 0, act)
        self.conv2 = nn.Conv2d(inter_ch, in_ch, 1, 1, 0)

    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, 1)
        y = self.conv1(x)
        y = self.conv2(y)
        y = torch.softmax(y, 1)
        y = y * x
        return y


class ResBlockA(nn.Module):
    def __init__(self, in_ch, out_ch, stride, act):
        super().__init__()
        assert stride in (1, 2)
        if stride == 1 and in_ch == out_ch:
            self.skip = nn.Identity()
        else:
            skip_layer = []
            if stride > 1:
                skip_layer.append(nn.AvgPool2d(2, 2, 0, ceil_mode=True, count_include_pad=False))
            if in_ch != out_ch:
                skip_layer.append(nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False))
                skip_layer.append(nn.BatchNorm2d(out_ch, eps=1e-8, momentum=0.9))
            self.skip = nn.Sequential(*skip_layer)

        inter_ch = out_ch // 2
        self.conv1 = BnActConv(in_ch, inter_ch, ker_sz=3, stride=stride, pad=1, act=act)
        self.conv2 = BnActConv(inter_ch, out_ch, ker_sz=3, stride=1, pad=1, act=act)
        self.a = nn.Parameter(torch.zeros(1), True)

    def forward(self, x):
        skip_y = self.skip(x)
        y = self.conv1(x)
        y = self.conv2(y)
        y = skip_y + y * self.a
        return y


class ResBlockB(nn.Module):
    def __init__(self, in_ch, out_ch, stride, act):
        super().__init__()
        assert stride in (1, 2)
        if stride == 1 and in_ch == out_ch:
            self.skip = nn.Identity()
        else:
            skip_layer = []
            if stride > 1:
                skip_layer.append(nn.UpsamplingBilinear2d(scale_factor=2.))
            if in_ch != out_ch:
                skip_layer.append(nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False))
                skip_layer.append(nn.BatchNorm2d(out_ch, eps=1e-8, momentum=0.9))
            self.skip = nn.Sequential(*skip_layer)

        inter_ch = out_ch // 2

        if stride == 1:
            self.conv1 = BnActConv(in_ch, inter_ch, ker_sz=3, stride=stride, pad=1, act=act)
        else:
            # self.conv1 = DeConvBnAct(in_ch, inter_ch, 2, stride, 0, act)
            self.conv1 = nn.Sequential(BnActConv(in_ch, inter_ch, 3, 1, 1, act),
                                       nn.UpsamplingBilinear2d(scale_factor=2.))
        self.conv2 = BnActConv(inter_ch, out_ch, ker_sz=3, stride=1, pad=1, act=act)
        self.a = nn.Parameter(torch.zeros(1), True)

    def forward(self, x):
        skip_y = self.skip(x)
        y = self.conv1(x)
        y = self.conv2(y)
        y = skip_y + y * self.a
        return y


class RevBlockC(nn.Module):
    def __init__(self, in_ch, out_ch, stride, act, **kwargs):
        super().__init__()
        assert in_ch == out_ch
        assert stride == 1
        inter_ch = in_ch // 2
        self.conv1 = BnActConv(in_ch, inter_ch, ker_sz=3, stride=1, pad=1, act=act)
        self.conv2 = BnActConv(inter_ch, out_ch, ker_sz=3, stride=1, pad=1, act=act)
        self.se = SEBlock(in_ch, act)
        self.a = nn.Parameter(torch.zeros(1), True)

    def func(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.se(y)
        y = x + y * self.a
        return y

    def forward(self, x1, x2):
        y = x1 + self.func(x2)
        return x2, y

    def invert(self, y1, y2):
        x2, y = y1, y2
        x1 = y - self.func(x2)
        return x1, x2


class GroupBlock(nn.Module):
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
    model_id = 5

    def __init__(self, in_dim=3, out_dim=1):
        super().__init__()
        act = nn.LeakyReLU(0.02, inplace=True)
        # 192x
        self.conv1 = ConvBnAct(in_dim, 64, 3, 1, 1, act=act)
        self.gb1 = GroupBlock(64, 128, 2, act, ResBlockA, 1)
        self.rvb1 = RevGroupBlock(128, 128, 1, act, RevBlockC, 6)

        # 96x
        self.gb2 = GroupBlock(128, 256, 2, act, ResBlockA, 1)
        self.rvb2 = RevGroupBlock(256, 256, 1, act, RevBlockC, 12)
        # 48x

        # # 48x
        # self.gb3 = GroupBlock(128, 256, 1, act, ResBlockA, 1)
        # self.rvb3 = RevGroupBlock(256, 256, 1, act, RevBlockC, 6)

        # 48x
        self.gb13 = GroupBlock(256, 128, 2, act, ResBlockB, 1)
        self.rvb13 = RevGroupBlock(128, 128, 1, act, RevBlockC, 6)
        # 96x
        self.gb14 = GroupBlock(128, 64, 2, act, ResBlockB, 1)
        self.rvb14 = RevGroupBlock(64, 64, 1, act, RevBlockC, 6)
        # 192x

        self.out_conv1 = ConvBnAct(64, 64, 3, 1, 1, act)
        self.out_conv2 = ConvBnAct(64, 32, 3, 1, 1, act)
        self.out_conv3 = nn.Conv2d(32, out_dim, 1, 1, 0)
        self.out_conv3.bias.data[:] = 0.5

    def func2(self, rvb, x):
        if self.training:
            y1, y2 = rev_sequential_backward_wrapper(rvb, x, x, preserve_rng_state=False)
        else:
            y1, y2 = rvb(x, x)
        y = (y1 + y2) / 2
        return y

    def forward(self, x):
        # 512
        y = self.conv1(x)
        y = self.gb1(y)
        y = self.func2(self.rvb1, y)

        y = self.gb2(y)
        y = self.func2(self.rvb2, y)

        # y = self.gb3(y)
        # y = self.func2(self.rvb3, y)

        y = self.gb13(y)
        y = self.func2(self.rvb13, y)

        y = self.gb14(y)
        y = self.func2(self.rvb14, y)


        y = self.out_conv1(y)
        y = self.out_conv2(y)
        y = self.out_conv3(y)
        return y


if __name__ == '__main__':
    import model_utils_torch

    a = torch.zeros(8, 3, 192, 192).cuda(0)
    # a = torch.zeros(2, 512, 512, 3).cuda(0).permute(0, 3, 1, 2)
    net = MainNet(3, 1).cuda(0)
    model_utils_torch.print_params_size(net)
    y = net(a)
    y.sum().backward()
    print(y.shape)
