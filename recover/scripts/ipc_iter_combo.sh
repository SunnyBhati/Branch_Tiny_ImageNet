#!/bin/bash
cd ..
export API_KEY="085788de93539cf40689d3714b6dc50a54f20a86"
export CUDA_VISIBLE_DEVICES = "0,1,2"
# Define different values for iterations, IPC, and GPU IDs
iterations_list=(0 1 10 100 500 1000 2000)
ipc_list=(1 10 50)
gpu_ids=(0 1 2)  # List of GPU IDs available for parallel execution

# Keep track of the total number of experiments
experiment_count=0

# Loop over each combination of iterations and ipc
for iterations in "${iterations_list[@]}"; do
    for ipc in "${ipc_list[@]}"; do
        # Get the GPU ID for this experiment (cycle through available GPUs)
        gpu_id=${gpu_ids[experiment_count % ${#gpu_ids[@]}]}
        
        echo "Running experiment with iterations=$iterations, ipc=$ipc on GPU $gpu_id"
        
        # Run the Python script on the selected GPU in the background
         python recover_non_dist.py --batch-size 100 \
                                    --category-aware "global" \
                                    --lr 0.05 \
                                    --drop-rate 0.4 \
                                    --training-momentum 0.99 \
                                    --train-data-path /data/sunny/DD/SRe2L/SRe2L/*small_dataset/tiny-IN/data/ \
                                    --r-loss 0.01 \
                                    --initial-img-dir /data/sunny/DD/SRe2L/SRe2L/*small_dataset/tiny-IN/data/tiny-imagenet-200/ \
                                    --verifier --store-best-images \
                                    --gpu "$gpu_id" \
                                    --batch_weight 1 \
                                    --conv_weight 1 \
                                    --aux-teacher ResNet18 MobileNetV2 ShuffleNetV2_0_5 efficientnet_b0 \
                                    --iteration "$iterations" \
                                    --ipc-number "$ipc" \
                                    --flatness \
                                    --first-multiplier \
                                    &
        
        # Increment the experiment count
        experiment_count=$((experiment_count + 1))
    done
done

# Wait for all background jobs to finish before exiting
wait