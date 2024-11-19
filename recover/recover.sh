
CUDA_VISIBLE_DEVICES=0 python recover.py \
    --arch-name "resnet18" \
    --exp-name "EDC_original_Tiny_ImageNet_Recover_IPC_1" \
    --batch-size 100 \
    --category-aware "global" \
    --lr 0.05 \
    --drop-rate 0.4 \
    --ipc-number 1 \
    --training-momentum 0.8 \
    --iteration 2000 \
    --train-data-path /data/sunny/DD/SRe2L/SRe2L/*small_dataset/tiny-IN/data/tiny-imagenet-200/ \
    --r-loss 0.01\
     --initial-img-dir /data/sunny/DD/SRe2L/SRe2L/*small_dataset/tiny-IN/data/tiny-imagenet-200/ \
    --verifier --store-best-images --gpu-id 0,1,2