#!/bin/bash

export API_KEY="085788de93539cf40689d3714b6dc50a54f20a86"

CUDA_VISIBLE_DEVICES=0,1,2 python recover_non_dist.py \
    --batch-size 100 \
    --category-aware "global" \
    --lr 0.05 \
    --drop-rate 0.4 \
    --training-momentum 0.99 \
    --train-data-path /data/sunny/DD/SRe2L/SRe2L/*small_dataset/tiny-IN/data/ \
    --r-loss 0.01 \
    --initial-img-dir /data/sunny/DD/SRe2L/SRe2L/*small_dataset/tiny-IN/data/tiny-imagenet-200/ \
    --verifier --store-best-images \
    --batch_weight 1 \
    --conv_weight 1 \
    --aux-teacher ResNet18 MobileNetV2 ShuffleNetV2_0_5 efficientnet_b0 \
    --iteration 2000 \
    --ipc-number 50 \
    --flatness \
    --gpu-id "0,1,2" \
    --gpu 2 \
    --first-multiplier \



