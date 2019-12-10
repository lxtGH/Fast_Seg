#!/usr/bin/env bash

# train the net (suppose 4 gpus)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node=4 train_distribute.py --data_set cityscapes \
--data_dir "/nas/dataset/CityScapes" \
--data_list "./data/cityscapes/train.txt" \
--arch dfnetv1seg \
--restore_from "/nas/dataset/pretrained/df1_imagenet.pth" \
--input_size 832 \
--batch_size_per_gpu 4 \
--learning_rate 0.01 \
--num_steps 50000 \
--save_dir "./save/dfnetv1seg" \
--rgb 1 \
--ohem 1 --ohem_thres 0.7 --ohem_keep 100000 \
--log_file "./save/dfnetv1seg.log"


# whole evaluation
python val.py --data_set cityscapes \
--data_dir "/nas/dataset/CityScapes" \
--data_list "./data/cityscapes/val.txt" \
--arch dfnetv1seg \
--rgb 1 \
--restore_from "./save/dfnetv1seg/dfnetv1seg_final.pth" \
--whole True \
--output_dir "./ICNet_vis"