CUDA_VISIBLE_DEVICES=0,1,2 python train_file_adapted_from_priyam.py \
    --batch-size 100 \
    --wandb-api-key "085788de93539cf40689d3714b6dc50a54f20a86" \
    --ls-type cos2 \
    --loss-type "mse_gt" \
    --ce-weight 0.025 \
    -T 20 \
    --adamw-lr 0.001 \
    --gpu-id 0,1,2 \
    -j 4 \
    --st 2 \
    --ema-dr 0.99 \
    --mix-type 'cutmix'\
    --weight-decay 0.0005 \
    --lr-warmup-epochs 5 \
    --lr-warmup-method 'linear' \
    --lr-warmup-decay 0.01 \
    --epochs 300 \
    --model ResNet101 \
    --ipc 10 \
    --gpu 0 \
    --train-dir /data/sunny/EDC/Branch_Tiny_ImageNet/recover/syn_data_jpeg/d_tinyim_it_0_ipc_10_flat_True_bs_100_lr_0.05_ca_global_tea_[ResMobShueff]_conw_1_batw_1_mom_0.99_fmul_True \
    --val-dir ../dataset/data/ \
    --teacher-name ResNet18 MobileNetV2 ShuffleNetV2_0_5 efficientnet_b0  \    
    #ResNet18 ConvNetW128 MobileNetV2 WRN_16_2 ShuffleNetV2_0_5 \
    #ResNet18 MobileNetV2 ShuffleNetV2_0_5 efficientnet_b0
    #soft-augmenttaion
    #ResNet18 ConvNetW128 MobileNetV2 WRN_16_2 ShuffleNetV2_0_5 \