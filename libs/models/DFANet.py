# Author: Xiangtai Li
# Email: lxtpku@pku.edu.cn
"""
    Implementation of DFANet: a little different from the origin paper, I add more dsn loss for training.
    DFANet uses modified Xception backbone pretrained on ImageNet.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.models.backbone.xception import Enc, FCAttention, XceptionA
from libs.core.operators import ConvBnRelu, dsn


class DFANet(nn.Module):
    def __init__(self, nclass, **kwargs):
        super(DFANet, self).__init__()
        self.backbone = XceptionA()

        self.enc2_2 = Enc(240, 48, 4, **kwargs)
        self.enc3_2 = Enc(144, 96, 6, **kwargs)
        self.enc4_2 = Enc(288, 192, 4, **kwargs)
        self.fca_2 = FCAttention(192, **kwargs)

        self.enc2_3 = Enc(240, 48, 4, **kwargs)
        self.enc3_3 = Enc(144, 96, 6, **kwargs)
        self.enc3_4 = Enc(288, 192, 4, **kwargs)
        self.fca_3 = FCAttention(192, **kwargs)

        self.enc2_1_reduce = ConvBnRelu(48, 32, 1, **kwargs)
        self.enc2_2_reduce = ConvBnRelu(48, 32, 1, **kwargs)
        self.enc2_3_reduce = ConvBnRelu(48, 32, 1, **kwargs)
        self.conv_fusion = ConvBnRelu(32, 32, 1, **kwargs)

        self.fca_1_reduce = ConvBnRelu(192, 32, 1, **kwargs)
        self.fca_2_reduce = ConvBnRelu(192, 32, 1, **kwargs)
        self.fca_3_reduce = ConvBnRelu(192, 32, 1, **kwargs)
        self.conv_out = nn.Conv2d(32, nclass, 1)

        self.dsn1 = dsn(192, nclass)
        self.dsn2 = dsn(192, nclass)

        self.__setattr__('exclusive', ['enc2_2', 'enc3_2', 'enc4_2', 'fca_2', 'enc2_3', 'enc3_3', 'enc3_4', 'fca_3',
                                       'enc2_1_reduce', 'enc2_2_reduce', 'enc2_3_reduce', 'conv_fusion', 'fca_1_reduce',
                                       'fca_2_reduce', 'fca_3_reduce', 'conv_out'])

    def forward(self, x):
        # backbone
        stage1_conv1 = self.backbone.conv1(x)
        stage1_enc2 = self.backbone.enc2(stage1_conv1)
        stage1_enc3 = self.backbone.enc3(stage1_enc2)
        stage1_enc4 = self.backbone.enc4(stage1_enc3)
        stage1_fca = self.backbone.fca(stage1_enc4)
        stage1_out = F.interpolate(stage1_fca, scale_factor=4, mode='bilinear', align_corners=True)

        dsn1 = self.dsn1(stage1_out)
        # stage2
        stage2_enc2 = self.enc2_2(torch.cat([stage1_enc2, stage1_out], dim=1))
        stage2_enc3 = self.enc3_2(torch.cat([stage1_enc3, stage2_enc2], dim=1))
        stage2_enc4 = self.enc4_2(torch.cat([stage1_enc4, stage2_enc3], dim=1))
        stage2_fca = self.fca_2(stage2_enc4)
        stage2_out = F.interpolate(stage2_fca, scale_factor=4, mode='bilinear', align_corners=True)

        dsn2 = self.dsn2(stage2_out)

        # stage3
        stage3_enc2 = self.enc2_3(torch.cat([stage2_enc2, stage2_out], dim=1))
        stage3_enc3 = self.enc3_3(torch.cat([stage2_enc3, stage3_enc2], dim=1))
        stage3_enc4 = self.enc3_4(torch.cat([stage2_enc4, stage3_enc3], dim=1))
        stage3_fca = self.fca_3(stage3_enc4)


        stage1_enc2_decoder = self.enc2_1_reduce(stage1_enc2)
        stage2_enc2_docoder = F.interpolate(self.enc2_2_reduce(stage2_enc2), scale_factor=2,
                                            mode='bilinear', align_corners=True)
        stage3_enc2_decoder = F.interpolate(self.enc2_3_reduce(stage3_enc2), scale_factor=4,
                                            mode='bilinear', align_corners=True)
        fusion = stage1_enc2_decoder + stage2_enc2_docoder + stage3_enc2_decoder
        fusion = self.conv_fusion(fusion)

        stage1_fca_decoder = F.interpolate(self.fca_1_reduce(stage1_fca), scale_factor=4,
                                           mode='bilinear', align_corners=True)
        stage2_fca_decoder = F.interpolate(self.fca_2_reduce(stage2_fca), scale_factor=8,
                                           mode='bilinear', align_corners=True)
        stage3_fca_decoder = F.interpolate(self.fca_3_reduce(stage3_fca), scale_factor=16,
                                           mode='bilinear', align_corners=True)
        fusion = fusion + stage1_fca_decoder + stage2_fca_decoder + stage3_fca_decoder

        outputs = list()
        out = self.conv_out(fusion)
        outputs.append(out)
        outputs.append(dsn1)
        outputs.append(dsn2)
        return outputs

def dfanet(num_classes=19, data_set="cityscapes"):
    return DFANet(num_classes)


if __name__ == '__main__':
    i = torch.Tensor(1,3,512,512).cuda()
    m = DFANet(19).cuda()
    m.eval()
    o = m(i)
    print(o[0].size())
    print("output length: ", len(o))