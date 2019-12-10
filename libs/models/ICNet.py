# Author: Xiangtai Li
# Email: lxtpku@pku.edu.cn
import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.core.operators import ConvBnRelu
from libs.models.PSPNet import PSPHead_res50


class CascadeFeatureFusion(nn.Module):
    """CFF Unit"""
    def __init__(self, low_channels, high_channels, out_channels, nclass, norm_layer=nn.BatchNorm2d):
        super(CascadeFeatureFusion, self).__init__()
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, 3, padding=2, dilation=2, bias=False),
            norm_layer(out_channels)
        )
        self.conv_high = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, 1, bias=False),
            norm_layer(out_channels)
        )
        self.conv_low_cls = nn.Conv2d(out_channels, nclass, 1, bias=False)

    def forward(self, x_low, x_high):
        x_low = F.interpolate(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)
        x_low = self.conv_low(x_low)
        x_high = self.conv_high(x_high)
        x = x_low + x_high
        x = F.relu(x, inplace=True)
        x_low_cls = self.conv_low_cls(x_low)

        return x, x_low_cls


class _ICHead(nn.Module):
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d):
        super(_ICHead, self).__init__()
        self.cff_12 = CascadeFeatureFusion(128, 64, 128, nclass, norm_layer)
        self.cff_24 = CascadeFeatureFusion(256, 256, 128, nclass, norm_layer)

        self.conv_cls = nn.Conv2d(128, nclass, 1, bias=False)

    def forward(self, x_sub1, x_sub2, x_sub4):
        outputs = list()
        x_cff_24, x_24_cls = self.cff_24(x_sub4, x_sub2)
        outputs.append(x_24_cls)
        x_cff_12, x_12_cls = self.cff_12(x_cff_24, x_sub1)
        outputs.append(x_12_cls)

        up_x2 = F.interpolate(x_cff_12, scale_factor=2, mode='bilinear', align_corners=True)
        up_x2 = self.conv_cls(up_x2)
        outputs.append(up_x2)
        up_x8 = F.interpolate(up_x2, scale_factor=4, mode='bilinear', align_corners=True)
        outputs.append(up_x8)
        # 1 -> 1/4 -> 1/8 -> 1/16
        outputs.reverse()
        return outputs


class ICNet(nn.Module):
    def __init__(self, nclass):
        super(ICNet, self).__init__()
        self.conv_sub1 = nn.Sequential(
            ConvBnRelu(3, 32, 3,2,1),
            ConvBnRelu(32, 32, 3, 2, 1),
            ConvBnRelu(32, 64, 3, 2, 1)
        )
        self.backbone = PSPHead_res50()
        self.head = _ICHead(nclass)

        self.conv_sub4 = ConvBnRelu(512, 256, 1)
        self.conv_sub2 = ConvBnRelu(512, 256, 1)

    def forward(self, x):

        # sub 1
        x_sub1_out = self.conv_sub1(x)

        # sub 2
        x_sub2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)

        x = self.backbone.relu1(self.backbone.bn1(self.backbone.conv1(x_sub2)))
        x = self.backbone.relu2(self.backbone.bn2(self.backbone.conv2(x)))
        x = self.backbone.relu3(self.backbone.bn3(self.backbone.conv3(x)))
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x_sub2_out = self.backbone.layer2(x)

        # sub 4
        x_sub4 = F.interpolate(x_sub2_out, scale_factor=0.5, mode='bilinear', align_corners=True)

        x = self.backbone.layer3(x_sub4)
        x = self.backbone.layer4(x)
        x_sub4_out = self.backbone.head(x)

        x_sub4_out = self.conv_sub4(x_sub4_out)
        x_sub2_out = self.conv_sub2(x_sub2_out)

        res = self.head(x_sub1_out, x_sub2_out, x_sub4_out)

        return res


def icnet(num_classes=19, data_set="cityscape"):
    return ICNet(num_classes)



if __name__ == '__main__':
    i = torch.Tensor(1,3,512,512).cuda()
    m = ICNet(19).cuda()
    m.eval()
    res= m(i)
    print("ICnet output length: ", len(res))
    for i in res:
        print(i.size())