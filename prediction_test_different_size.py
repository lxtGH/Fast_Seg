import os
import argparse
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import datetime


import libs.models as models


N_CLASS = 19
color_list = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
color_map = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
             (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
             (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)]
up_kwargs = {'mode': 'bilinear', 'align_corners': True}


def transform(img):
    img = cv2.imread(img)
    IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)
    img = img - IMG_MEAN
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).cuda()
    return img

def transform_rgb(img):
    img = cv2.imread(img, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32)

    img /= 255
    IMG_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
    IMG_VARS = np.array((0.229, 0.224, 0.225), dtype=np.float32)

    img -= IMG_MEAN
    img /= IMG_VARS

    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).cuda()
    return img



def makeTestlist(dir,start=0,end=1525):
    out = []
    floder = os.listdir(dir)
    for f in floder:
        floder_dir = os.path.join(dir, f)
        for i in os.listdir(floder_dir):
            out.append(os.path.join(floder_dir, i))
    out.sort()
    return out[start:end]


def WholeTest(args, model, size=1.0):
    net = model.cuda()
    net.eval()
    saved_state_dict = torch.load(args.resume)
    net.load_state_dict(saved_state_dict)
    img_list = makeTestlist(args.input_dir)
    out_dir = args.output_dir
    for i in img_list:
        name = i
        with torch.no_grad():
            if args.rgb:
                img = transform_rgb(i)
            else:
                img = transform(i)
            _, _, origin_h, origin_w = img.size()
            h, w = int(origin_h*size), int(origin_w*size)
            img = F.upsample(img, size=(h, w), mode="bilinear", align_corners=True)
            out = net(img)[0]
            out = F.upsample(out, size=(origin_h, origin_w), mode='bilinear', align_corners=True)
            result = out.argmax(dim=1)[0]
            result = result.data.cpu().squeeze().numpy()
            row, col = result.shape
            dst = np.ones((row, col), dtype=np.uint8) * 255
            for i in range(19):
                dst[result == i] = color_list[i]
            print(name, " done!")
            save_name = os.path.join(out_dir, "/".join(name.split('/')[4:]))
            save_dir = "/".join(save_name.split("/")[:-1])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(save_name, dst)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch \
                Segmentation Crop Prediction')
    parser.add_argument('--input_dir', type=str,
                        default="/home/lxt/data/Cityscapes/leftImg8bit/test",
                        help='training dataset folder (default: \
                              $(HOME)/data)')
    parser.add_argument("--input_disp_dir", type=str, default=None)
    parser.add_argument('--output_dir', type=str, default="/home/lxt/debug/cgnl_ohem_crop_ms",
                        help='output directory of the model_test, for saving the seg_models')
    parser.add_argument("--resume", type=str, default="/home/lxt/Desktop/Seg_model_ZOO/CNL_net_4w_ohem/CS_scenes_40000.pth")
    parser.add_argument("--start",type=int,default=0,help="start index of crop test")
    parser.add_argument("--end",type=int,default=1525,help="end index of crop test")
    parser.add_argument("--gpu",type=str,default="0",help="which gpu to use")
    parser.add_argument("--arch",type=str,default=None, help="which network are used")
    parser.add_argument("--size",type=float,default=1.0,help="ratio of the input images")
    parser.add_argument("--rgb",type=int,default=0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    test_list = makeTestlist(args.input_dir,args.start, args.end)
    model= models.__dict__[args.arch](num_classes=19, data_set="cityscapes")
    WholeTest(args, model=model, size=args.size)