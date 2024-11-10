python train.py --model "ResNet50" \
                --wandb-key "085788de93539cf40689d3714b6dc50a54f20a86" \
                --data_path /data/sunny/DD/SRe2L/SRe2L/*small_dataset/tiny-IN/data/ \
                --batch-size 128 \
                --train-epochs 51 \
                --opt 'sgd' \
                --lr-teacher 0.045 \
                --momentum 0.9 \
                --wandb-key "085788de93539cf40689d3714b6dc50a54f20a86" \
                --weight-decay 0.0005 \
                --lr-scheduler 'cosineannealinglr' \
                --lr-warmup-epochs 5 \
                --lr-warmup-method 'linear' \
                --lr-warmup-decay 0.01 \
                --gpu-device "1" \

