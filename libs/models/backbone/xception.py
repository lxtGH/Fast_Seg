from __future__ import print_function, division, absolute_import
import torch.nn as nn

from libs.core.operators import ConvBnRelu, SeparableConv2d
from libs.utils.tools import load_model

__all__ = ['Xception', 'Xception39','XceptionA']


class SeparableConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1,
                 has_relu=True, norm_layer=nn.BatchNorm2d):
        super(SeparableConvBnRelu, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                               padding, dilation, groups=in_channels,
                               bias=False)
        self.point_wise_cbr = ConvBnRelu(in_channels, out_channels, 1, 1, 0,
                                         has_bn=True, norm_layer=norm_layer,
                                         has_relu=has_relu, has_bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.point_wise_cbr(x)
        return x


class Block(nn.Module):
    expansion = 4

    def __init__(self, in_channels, mid_out_channels, has_proj, stride,
                 dilation=1, norm_layer=nn.BatchNorm2d):
        super(Block, self).__init__()
        self.has_proj = has_proj

        if has_proj:
            self.proj = SeparableConvBnRelu(in_channels,
                                            mid_out_channels * self.expansion,
                                            3, stride, 1,
                                            has_relu=False,
                                            norm_layer=norm_layer)

        self.residual_branch = nn.Sequential(
            SeparableConvBnRelu(in_channels, mid_out_channels,
                                3, stride, dilation, dilation,
                                has_relu=True, norm_layer=norm_layer),
            SeparableConvBnRelu(mid_out_channels, mid_out_channels, 3, 1, 1,
                                has_relu=True, norm_layer=norm_layer),
            SeparableConvBnRelu(mid_out_channels,
                                mid_out_channels * self.expansion, 3, 1, 1,
                                has_relu=False, norm_layer=norm_layer))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x
        if self.has_proj:
            shortcut = self.proj(x)

        residual = self.residual_branch(x)
        output = self.relu(shortcut + residual)

        return output


class Xception(nn.Module):
    def __init__(self, block, layers, channels, norm_layer=nn.BatchNorm2d):
        super(Xception, self).__init__()

        self.in_channels = 8
        self.conv1 = ConvBnRelu(3, self.in_channels, 3, 2, 1,
                                has_bn=True, norm_layer=norm_layer,
                                has_relu=True, has_bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, norm_layer,
                                       layers[0], channels[0], stride=2)
        self.layer2 = self._make_layer(block, norm_layer,
                                       layers[1], channels[1], stride=2)
        self.layer3 = self._make_layer(block, norm_layer,
                                       layers[2], channels[2], stride=2)

    def _make_layer(self, block, norm_layer, blocks,
                    mid_out_channels, stride=1):
        layers = []
        has_proj = True if stride > 1 else False
        layers.append(block(self.in_channels, mid_out_channels, has_proj,
                            stride=stride, norm_layer=norm_layer))
        self.in_channels = mid_out_channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, mid_out_channels,
                                has_proj=False, stride=1,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        blocks = []
        x = self.layer1(x)
        blocks.append(x)
        x = self.layer2(x)
        blocks.append(x)
        x = self.layer3(x)
        blocks.append(x)

        return blocks


"""
    Xception39 is used for BiSeg Network
"""
def Xception39(pretrained_model=None, **kwargs):
    model = Xception(Block, [4, 8, 4], [16, 32, 64], **kwargs)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model


class BlockA(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, norm_layer=nn.BatchNorm2d, start_with_relu=True):
        super(BlockA, self).__init__()
        if out_channels != in_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
            self.skipbn = norm_layer(out_channels)
        else:
            self.skip = None
        self.relu = nn.ReLU()
        rep = list()
        inter_channels = out_channels // 4

        if start_with_relu:
            rep.append(self.relu)
        rep.append(SeparableConv2d(in_channels, inter_channels, 3, 1, dilation, norm_layer=norm_layer))
        rep.append(norm_layer(inter_channels))

        rep.append(self.relu)
        rep.append(SeparableConv2d(inter_channels, inter_channels, 3, 1, dilation, norm_layer=norm_layer))
        rep.append(norm_layer(inter_channels))

        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inter_channels, out_channels, 3, stride, norm_layer=norm_layer))
            rep.append(norm_layer(out_channels))
        else:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inter_channels, out_channels, 3, 1, norm_layer=norm_layer))
            rep.append(norm_layer(out_channels))
        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        out = self.rep(x)
        if self.skip is not None:
            skip = self.skipbn(self.skip(x))
        else:
            skip = x
        out = out + skip
        return out


class Enc(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, norm_layer=nn.BatchNorm2d):
        super(Enc, self).__init__()
        block = list()
        block.append(BlockA(in_channels, out_channels, 2, norm_layer=norm_layer))
        for i in range(blocks - 1):
            block.append(BlockA(out_channels, out_channels, 1, norm_layer=norm_layer))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class FCAttention(nn.Module):
    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d):
        super(FCAttention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, 1000)
        self.conv = nn.Sequential(
            nn.Conv2d(1000, in_channels, 1, bias=False),
            norm_layer(in_channels),
            nn.ReLU())

    def forward(self, x):
        n, c, _, _ = x.size()
        att = self.avgpool(x).view(n, c)
        att = self.fc(att).view(n, 1000, 1, 1)
        att = self.conv(att)
        return x * att.expand_as(x)


"""
    XceptionA is used for DFANet  
"""

class XceptionA(nn.Module):
    def __init__(self, num_classes=1000, norm_layer=nn.BatchNorm2d):
        super(XceptionA, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 8, 3, 2, 1, bias=False),
                                   norm_layer(8),
                                   nn.ReLU())

        self.enc2 = Enc(8, 48, 4, norm_layer=norm_layer)
        self.enc3 = Enc(48, 96, 6, norm_layer=norm_layer)
        self.enc4 = Enc(96, 192, 4, norm_layer=norm_layer)

        self.fca = FCAttention(192, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.conv1(x)

        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.fca(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x