# Author: Xiangtai Li
# Email: lxtpku@pku.edu.cn
# Pytorch Implementation of DongFeng SegNet:
# Partial Order Pruning: for Best Speed/Accuracy Trade-off in Neural Architecture Search.
# The backbone is pretrained on ImageNet

import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.core.operators import PSPModule, conv3x3, dsn
from libs.models.backbone.dfnet import dfnetv2, dfnetv1


class FusionNode(nn.Module):
    def __init__(self, inplane):
        super(FusionNode, self).__init__()
        self.fusion = conv3x3(inplane*2, inplane)

    def forward(self, x):
        x_h, x_l = x
        size = x_l.size()[2:]
        x_h = F.upsample(x_h, size, mode="bilinear", align_corners=True)
        res = self.fusion(torch.cat([x_h,x_l],dim=1))
        return res


class DFSeg(nn.Module):
    def __init__(self, nclass, type="dfv1"):
        super(DFSeg, self).__init__()

        if type == "dfv1":
            self.backbone = dfnetv1()
        else:
            self.backbone = dfnetv2()

        self.cc5 = nn.Conv2d(128,128,1)
        self.cc4 = nn.Conv2d(256,128,1)
        self.cc3 = nn.Conv2d(128,128,1)

        self.ppm = PSPModule(512,128)

        self.fn4 = FusionNode(128)
        self.fn3 = FusionNode(128)

        self.fc = dsn(128, nclass)

    def forward(self, x):
        x3,x4,x5 = self.backbone(x)
        x5 = self.ppm(x5)
        x5 = self.cc5(x5)
        x4 = self.cc4(x4)
        f4 = self.fn4([x5, x4])
        x3 = self.cc3(x3)
        out = self.fn3([f4, x3])
        out = self.fc(out)

        return [out]


def dfnetv1seg(num_classes=19, data_set="cityscapes"):
    return DFSeg(num_classes,type="dfv1")


def dfnetv2seg(num_classes=19, data_set="cityscapes"):
    return DFSeg(num_classes,type="dfv2")


if __name__ == '__main__':
    i = torch.Tensor(1,3,512,512).cuda()
    m = DFSeg(19,"dfv2").cuda()
    m.eval()
    o = m(i)
    print(o[0].size())