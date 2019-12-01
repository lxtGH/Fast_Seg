# Author: Xiangtai Li
# Email: lxtpku@pku.edu.cn
# Pytorch Implementation Of MSFNet: Real-Time Semantic Segmentation via Multiply Spatial Fusion Network(face++)
# I didn't include the boundaries information

import torch
import torch.nn as nn



class MSFNet(nn.Module):
    def __init__(self):
        super(MSFNet, self).__init__()


    def forward(self, x):
        pass



if __name__ == '__main__':
    i = torch.Tensor(1, 3, 512, 512).cuda()
    m = MSFNet().cuda()
    m.eval()
    o = m(i)
    print(o[0].size())
    print("output length: ", len(o))