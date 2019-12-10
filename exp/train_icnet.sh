#!/usr/bin/env bash

# train the net (suppose 4 gpus)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node=4 train_distribute.py --data_set cityscapes \
--data_dir "/nas/dataset/CityScapes" \
--data_list "./data/cityscapes/train.txt" \
--arch icnet \
--restore_from "/nas/dataset/pretrained/resnet50-deep.pth" \
--input_size 832 \
--batch_size_per_gpu 4 \
--learning_rate 0.01 \
--num_steps 50000 \
--save_dir "./saveICNet" \
--rgb 0 \
--ohem 1 --ohem_thres 0.7 --ohem_keep 100000 \
--log_file "./log/ICNet.log"


# whole evaluation
python val.py --data_set cityscapes \
--data_dir "/nas/dataset/CityScapes" \
--data_list "./data/cityscapes/val.txt" \
--arch ICNet \
--rgb 0 \
--restore_from "./saveICNet/icnet_final.pth" \
--whole True \
--output_dir "./ICNet_vis"