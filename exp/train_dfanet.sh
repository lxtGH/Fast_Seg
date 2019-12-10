#!/usr/bin/env bash

# train the net (suppose 4 gpus)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node=4 train_distribute.py --data_set cityscapes \
--data_dir "/nas/dataset/CityScapes" \
--data_list "./data/cityscapes/train.txt" \
--arch dfanet \
--restore_from "/nas/dataset/pretrained/xceptiona_imagenet.pth" \
--input_size 1024 \
--batch_size_per_gpu 4 \
--learning_rate 0.01 \
--num_steps 60000 \
--save_dir "./save/dfanet" \
--rgb 1 \
--ohem 1 --ohem_thres 0.7 --ohem_keep 100000 \
--log_file "./save/dfanet.log"


# whole evaluation
python val.py --data_set cityscapes \
--data_dir "/nas/dataset/CityScapes" \
--data_list "./data/cityscapes/val.txt" \
--arch dfanet \
--rgb 1 \
--restore_from "./save/dfnetv1seg/dfanet_final.pth" \
--whole True \
--output_dir "./dfanet_out"