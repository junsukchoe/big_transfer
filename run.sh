#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py \
      --name imagenet_test \
      --model BiT-M-R50x1
      --logdir train_log/ \
      --dataset imagenet2012